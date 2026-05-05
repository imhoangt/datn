"""
src/training/losses.py  +  src/training/trainer.py
Loss functions and training loop for CHARM-Net v7.1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ============================================================================
# losses.py
# ============================================================================

def focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss for multi-class classification.
    pred:   [N, C] probabilities (not logits)
    target: [N]    int64 class indices
    """
    idx   = torch.arange(len(target), device=pred.device)
    log_p = torch.log(pred.clamp(min=1e-8))
    nll   = -log_p[idx, target]
    p_t   = pred[idx, target]
    return ((1.0 - p_t) ** gamma * nll).mean()


def compute_loss(
    outputs      : Dict[str, torch.Tensor],
    y_8class     : torch.Tensor,
    class_weights: torch.Tensor,
    global_epoch : int,
    cfg          : dict,
) -> Dict:
    """
    Combined hierarchical loss.

    L_total = L_main + w_aux * (0.3*(L_occ + L_dyn + L_act) + 0.05*L_cons)

    L_main: label-smoothed cross-entropy on P_8class (weighted)
    L_occ:  BCE for presence detection
    L_dyn:  BCE for dynamic vs static (occupied samples only)
    L_act:  Focal loss for 6 dynamic action classes
    L_cons: receiver representation consistency

    FIX v6.5: L_dyn / L_act always tensors → always .item()
    """
    P          = outputs['p_8class']          # [B, 8]
    p_occ      = outputs['p_occ']             # [B, 1]
    p_dyn      = outputs['p_dyn']             # [B, 1]
    p_act      = outputs['p_act']             # [B, 6]
    z_per_recv = outputs['z_per_recv']        # [B, 3, D]

    device = P.device

    y_occ  = (y_8class != 7).float().unsqueeze(1)   # [B, 1]
    y_dyn  = (y_8class  < 6).float().unsqueeze(1)   # [B, 1]
    y_act  = y_8class.clamp(0, 5)                   # [B]

    # ── L_main: label smoothing ε=0.05 + per-sample class weighting ──────
    # FIX-M5: use logit_unified via F.log_softmax (numerically stable log-sum-exp)
    logit_unified = outputs['logit_unified']             # [B, 8]
    eps  = cfg.get('label_smoothing', 0.05)
    n_cl = logit_unified.shape[-1]
    soft = (1 - eps) * F.one_hot(y_8class, n_cl).float() + eps / n_cl
    w    = class_weights[y_8class]
    log_probs_main = F.log_softmax(logit_unified, dim=-1)
    L_main = ((-(soft * log_probs_main).sum(-1)) * w).mean()

    # ── L_occ: BCE all samples ─────────────────────────────────────────
    L_occ = F.binary_cross_entropy(p_occ.squeeze(1), y_occ.squeeze(1))

    # ── L_dyn: BCE occupied only ───────────────────────────────────────
    occ_mask = y_occ.squeeze(1).bool()
    if occ_mask.any():
        L_dyn = F.binary_cross_entropy(
            p_dyn[occ_mask].squeeze(1), y_dyn[occ_mask].squeeze(1)
        )
    else:
        L_dyn = torch.tensor(0.0, device=device)

    # ── L_act: Focal γ=2, dynamic only (classes 0..5) ──────────────────
    dyn_mask = (y_8class < 6)
    if dyn_mask.any():
        L_act = focal_loss(p_act[dyn_mask], y_act[dyn_mask],
                           gamma=cfg.get('focal_gamma', 2.0))
    else:
        L_act = torch.tensor(0.0, device=device)

    # ── L_cons: receiver consistency ────────────────────────────────────
    z_mean = z_per_recv.mean(dim=1, keepdim=True)
    L_cons = ((z_per_recv - z_mean) ** 2).mean()

    # ── Aux loss ramp 0→1 over first w_aux_ramp epochs ─────────────────
    ramp_ep = cfg.get('w_aux_ramp_epochs', 5)
    w_aux   = float(min(global_epoch / max(ramp_ep, 1), 1.0))
    w_cfg   = cfg.get('w_aux', 0.3)
    w_cons  = cfg.get('w_cons', 0.05)

    total = L_main + w_aux * (w_cfg * (L_occ + L_dyn + L_act) + w_cons * L_cons)

    return {
        'total'  : total,
        'L_main' : L_main.item(),
        'L_occ'  : L_occ.item(),
        'L_dyn'  : L_dyn.item(),   # always tensor → always .item()  FIX v6.5
        'L_act'  : L_act.item(),
        'L_cons' : L_cons.item(),
        'w_aux'  : w_aux,
    }


# ============================================================================
# trainer.py
# ============================================================================

def get_paired_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple:
    """Shuffle within batch for H1b CutMix paired samples."""
    X_amp, X_dfs, labels = batch
    j = torch.randperm(X_amp.size(0), device=X_amp.device)
    return X_amp, X_dfs, labels, X_amp[j], X_dfs[j], labels[j]


def apply_cutmix_time(
    X_amp       : torch.Tensor,
    X_amp_j     : torch.Tensor,
    global_epoch: int,
    cfg         : dict,
) -> torch.Tensor:
    """
    H1b CutMix-Time: replace a random time segment with paired sample.
    Applied in training loop AFTER DataLoader, using get_paired_batch.
    FIX v6.6 BUG#56: applied here (training loop), NOT inside Dataset.

    X_amp:   [B, T=350, F=52, M=3, A=4]
    X_amp_j: [B, T=350, F=52, M=3, A=4]  (shuffled batch)
    """
    start_ep   = cfg.get('cutmix_start_epoch', 8)
    prob       = cfg.get('cutmix_prob', 0.3)
    max_ratio  = cfg.get('cutmix_max_ratio', 0.2)

    if global_epoch < start_ep:
        return X_amp

    T = X_amp.shape[1]
    X_out = X_amp.clone()
    for b in range(X_amp.size(0)):
        if np.random.rand() < prob:
            cut_len = int(np.random.randint(1, max(2, int(T * max_ratio) + 1)))
            t1      = int(np.random.randint(0, T - cut_len))
            X_out[b, t1:t1 + cut_len] = X_amp_j[b, t1:t1 + cut_len]
    return X_out


def train_one_epoch(
    model        : nn.Module,
    loader,
    optimizer    : torch.optim.Optimizer,
    scaler,
    ema,
    global_epoch : int,
    class_weights: torch.Tensor,
    cfg          : dict,
    device       : torch.device,
    aug_level    : str = 'full',   # 'minimal' | 'light' | 'full'
) -> Dict:
    """Main training loop for one epoch."""
    model.train()
    optimizer.zero_grad()

    grad_accum = cfg.get('grad_accum', 4)
    max_norm   = cfg.get('max_grad_norm', 1.0)
    cw         = class_weights.to(device)

    sum_loss  = {k: 0.0 for k in ['total','L_main','L_occ','L_dyn','L_act','L_cons']}
    n_batches = 0

    for i, batch in enumerate(loader):
        # Paired batch for H1b CutMix
        X_amp, X_dfs, labels, X_amp_j, _, _ = get_paired_batch(batch)
        X_amp  = X_amp.to(device, non_blocking=True)
        X_dfs  = X_dfs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        X_amp_j = X_amp_j.to(device, non_blocking=True)

        # H1b CutMix-Time (FIX v6.6 BUG#56)
        if aug_level == 'full':
            X_amp = apply_cutmix_time(X_amp, X_amp_j, global_epoch, cfg)

        with torch.autocast(device_type=device.type):
            out  = model(X_amp, X_dfs)
            loss_dict = compute_loss(out, labels, cw, global_epoch, cfg)
            loss = loss_dict['total'] / grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update()

        for k in sum_loss:
            sum_loss[k] += loss_dict.get(k, 0.0)
        n_batches += 1

    # Final accumulation step if leftover
    if n_batches % grad_accum != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update()

    return {k: v / max(n_batches, 1) for k, v in sum_loss.items()}


@torch.no_grad()
def evaluate_with_ema(
    model  : nn.Module,
    ema,
    loader,
    device : torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model using EMA weights (FIX v6.5).
    Returns: macro_f1, all_probs [N,8], all_labels [N]
    """
    from sklearn.metrics import f1_score

    model.eval()
    all_probs, all_labels = [], []

    with ema.average_parameters():
        for X_amp, X_dfs, labels in loader:
            out = model(X_amp.to(device), X_dfs.to(device))
            all_probs.append(out['p_8class'].cpu().numpy())
            all_labels.append(labels.numpy())

    probs  = np.concatenate(all_probs)    # [N, 8]
    labels = np.concatenate(all_labels)   # [N]
    preds  = probs.argmax(-1)
    f1     = float(f1_score(labels, preds, average='macro', zero_division=0))
    return f1, probs, labels
