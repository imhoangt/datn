"""
src/models/heads.py
HierarchicalHeads: factored prediction P(occ) × P(dyn|occ) × P(action|dyn,occ)
                   + unified direct head for main cross-entropy loss (FIX-M5)
TemperatureScaler: post-training calibration via LBFGS on exact logits

Class hierarchy:
    empty(7)       = 1 - P(occupied)
    lying_still(6) = P(occ) × (1 - P(dynamic|occ))
    actions(0-5)   = P(occ) × P(action_k|dyn,occ)

Design notes (v7.2):
- head_unified = Linear(D, 8): exact logits for numerically stable L_main (FIX-M5)
- p_8class = softmax(logit_unified): proper probability vector for inference
- Factored heads (occ/dyn/act) kept as auxiliary supervision only
- TemperatureScaler uses logit_unified directly — no log(p) approximation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalHeads(nn.Module):
    """
    Four-head classifier: three factored aux heads + one unified main head.

    Heads:
        head_occ    : D → 1   (occupied vs empty, aux)
        head_dyn    : D → 1   (dynamic vs static, conditioned on occupied, aux)
        head_act    : D → 6   (6 dynamic actions, aux)
        head_unified: D → 8   (direct 8-class logits, main output — FIX-M5)

    Class assignment (factored, for aux losses):
        P_factored[0:6] = P(occ) × P(dyn|occ) × P(action_k|...)
        P_factored[6]   = P(occ) × (1 - P(dyn|occ))       → lying_still
        P_factored[7]   = 1 - P(occ)                        → empty

    Main output (for L_main and inference):
        p_8class = softmax(logit_unified)
    """

    def __init__(self, D: int = 80):
        super().__init__()
        self.head_occ     = nn.Linear(D, 1)   # 81 params
        self.head_dyn     = nn.Linear(D, 1)   # 81 params
        self.head_act     = nn.Linear(D, 6)   # 486 params
        self.head_unified = nn.Linear(D, 8)   # 648 params (FIX-M5)
        self._init_weights()

    def _init_weights(self):
        for head in [self.head_occ, self.head_dyn, self.head_act, self.head_unified]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, z: torch.Tensor) -> dict:
        """
        z: [B, D]
        returns dict with:
            p_8class      [B, 8]  — class probabilities from unified head (main)
            logit_unified [B, 8]  — raw logits for L_main + TemperatureScaler
            p_factored    [B, 8]  — factored hierarchical probabilities (aux interpretation)
            p_occ         [B, 1]  — P(occupied)
            p_dyn         [B, 1]  — P(dynamic | occupied)
            p_act         [B, 6]  — P(action | dynamic, occupied)
            logit_occ     [B, 1]  — raw logits for aux calibration
            logit_dyn     [B, 1]
            logit_act     [B, 6]
        """
        lo = self.head_occ(z)   # [B, 1]
        ld = self.head_dyn(z)   # [B, 1]
        la = self.head_act(z)   # [B, 6]

        p_occ = torch.sigmoid(lo)                # [B, 1]
        p_dyn = torch.sigmoid(ld)                # [B, 1]
        p_act = torch.softmax(la, dim=-1)        # [B, 6], sums to 1

        # Factored composition (for aux interpretation and L_occ/L_dyn/L_act)
        p_actions     = p_occ * p_dyn * p_act               # [B, 6]
        p_lying_still = p_occ * (1.0 - p_dyn)               # [B, 1]
        p_empty       = 1.0 - p_occ                          # [B, 1]
        p_factored    = torch.cat([p_actions, p_lying_still, p_empty], dim=-1)  # [B, 8]

        # Unified head — main output (FIX-M5)
        logit_unified = self.head_unified(z)                  # [B, 8]
        p_8class      = torch.softmax(logit_unified, dim=-1)  # [B, 8], exact softmax

        return {
            'p_8class'      : p_8class,
            'logit_unified' : logit_unified,
            'p_factored'    : p_factored,
            'p_occ'         : p_occ,
            'p_dyn'         : p_dyn,
            'p_act'         : p_act,
            'logit_occ'     : lo,
            'logit_dyn'     : ld,
            'logit_act'     : la,
        }


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling calibration [Guo et al., ICML 2017].

    v7.2: Uses logit_unified directly (exact logits) instead of log(p_8class)
    approximation. LBFGS on val set (held out from training).
    """

    def __init__(self, init_temp: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(init_temp)))

    def calibrate(
        self,
        model: nn.Module,
        val_loader,
        device: torch.device,
    ) -> float:
        """
        Fit temperature T on val_loader using LBFGS.
        Returns the optimized temperature value.

        @no_grad must NOT wrap the entire function — LBFGS needs gradients to
        flow to self.temperature. Only data collection is no_grad.
        """
        model.eval()
        self.to(device)

        logit_list, lab_list = [], []
        with torch.no_grad():   # no_grad for data collection only
            for X_amp, X_dfs, labels in val_loader:
                out = model(X_amp.to(device), X_dfs.to(device))
                logit_list.append(out['logit_unified'].cpu())
                lab_list.append(labels)

        logits = torch.cat(logit_list).to(device)   # [N, 8]
        labels = torch.cat(lab_list).to(device)     # [N]

        # Require grad only for temperature
        self.temperature.requires_grad_(True)
        opt = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)

        def closure():
            opt.zero_grad()
            scaled = logits / self.temperature.clamp(min=0.05)
            loss   = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature.requires_grad_(False)

        t_val = self.temperature.item()
        print(f"[TemperatureScaler] Calibrated T = {t_val:.4f}")
        return t_val

    def forward(self, logit_unified: torch.Tensor) -> torch.Tensor:
        """
        logit_unified: [B, 8] raw unified logits
        returns:       [B, 8] calibrated probabilities
        """
        T = self.temperature.clamp(min=0.05)
        return torch.softmax(logit_unified / T, dim=-1)
