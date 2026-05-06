"""
src/training/losses.py  +  src/training/trainer.py
Loss functions and training loop for CHARM-Net v7.2.
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

    B      = y_8class.shape[0]
    device = P.device

    # ── Dataset-specific auxiliary loss hierarchy ──────────────────────
    aux_cfg   = cfg.get('aux_loss', {})
    use_occ   = aux_cfg.get('use_occ', True)
    use_dyn   = aux_cfg.get('use_dyn', True)
    use_act   = aux_cfg.get('use_act', True)
    empty_idx = aux_cfg.get('empty_class_idx', 7)
    dyn_max   = aux_cfg.get('dynamic_class_max_idx', 5)
    n_act     = aux_cfg.get('n_act_classes', 6)

    if use_occ and empty_idx is not None:
        y_occ = (y_8class != empty_idx).float().unsqueeze(1)
    else:
        y_occ = torch.ones(B, 1, device=device)

    y_dyn = (y_8class <= dyn_max).float().unsqueeze(1)
    y_act = y_8class.clamp(0, n_act - 1)

    # ── L_main: label smoothing ε=0.05 + per-sample class weighting ──────
    logit_unified = outputs['logit_unified']             # [B, 8]
    eps  = cfg.get('label_smoothing', 0.05)
    n_cl = logit_unified.shape[-1]
    soft = (1 - eps) * F.one_hot(y_8class, n_cl).float() + eps / n_cl
    w    = class_weights[y_8class]
    log_probs_main = F.log_softmax(logit_unified, dim=-1)
    L_main = ((-(soft * log_probs_main).sum(-1)) * w).mean()

    # ── L_occ: BCE all samples ─────────────────────────────────────────
    if use_occ:
        L_occ = F.binary_cross_entropy(p_occ.squeeze(1), y_occ.squeeze(1))
    else:
        L_occ = torch.tensor(0.0, device=device)

    # ── L_dyn: BCE occupied only ───────────────────────────────────────
    occ_mask = y_occ.squeeze(1).bool()
    if use_dyn and occ_mask.any():
        L_dyn = F.binary_cross_entropy(
            p_dyn[occ_mask].squeeze(1), y_dyn[occ_mask].squeeze(1)
        )
    else:
        L_dyn = torch.tensor(0.0, device=device)

    # ── L_act: Focal γ=2, dynamic only ────────────────────────────────
    if use_act:
        dyn_mask = y_8class <= dyn_max
        if dyn_mask.any():
            L_act = focal_loss(p_act[dyn_mask], y_act[dyn_mask],
                               gamma=cfg.get('focal_gamma', 2.0))
        else:
            L_act = torch.tensor(0.0, device=device)
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
        X_amp, X_dfs, labels = batch
        X_amp  = X_amp.to(device, non_blocking=True)
        X_dfs  = X_dfs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
    model              : nn.Module,
    ema,
    loader,
    device             : torch.device,
    temperature_scaler = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model using EMA weights.
    If temperature_scaler is provided, applies temperature scaling to logit_unified
    instead of using raw p_8class — use this for final test evaluation after calibration.
    Returns: macro_f1, all_probs [N,8], all_labels [N]
    """
    from sklearn.metrics import f1_score

    model.eval()
    all_probs, all_labels = [], []

    with ema.average_parameters():
        for X_amp, X_dfs, labels in loader:
            out = model(X_amp.to(device), X_dfs.to(device))
            if temperature_scaler is not None:
                probs = temperature_scaler(out['logit_unified'])
            else:
                probs = out['p_8class']
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    probs  = np.concatenate(all_probs)    # [N, 8]
    labels = np.concatenate(all_labels)   # [N]
    preds  = probs.argmax(-1)
    f1     = float(f1_score(labels, preds, average='macro', zero_division=0))
    return f1, probs, labels
