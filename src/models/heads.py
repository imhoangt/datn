"""
src/models/heads.py
HierarchicalHeads: factored prediction P(occ) × P(dyn|occ) × P(action|dyn,occ)
TemperatureScaler: post-training calibration via LBFGS

Class hierarchy:
    empty(7)       = 1 - P(occupied)
    lying_still(6) = P(occ) × (1 - P(dynamic|occ))
    actions(0-5)   = P(occ) × P(dyn|occ) × P(action_k|dyn,occ)

Sum verification: P(occ)×P(dyn)×1 + P(occ)×(1-P(dyn)) + (1-P(occ)) = 1 ✓

Design notes (v6.6):
- Expose raw logits for TemperatureScaler
- Known limitation: P_8class = product sigmoid×softmax → approx calibration only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalHeads(nn.Module):
    """
    Three-head classifier with hierarchical factorization.

    Heads:
        head_occ: D → 1   (occupied vs empty)
        head_dyn: D → 1   (dynamic vs static, conditioned on occupied)
        head_act: D → 6   (6 dynamic actions: walk/run/stand_up/sit_down/bend/fall)

    Class assignment:
        P_8class[0:6] = P(occ) × P(dyn|occ) × P(action_k|...)
        P_8class[6]   = P(occ) × (1 - P(dyn|occ))       → lying_still
        P_8class[7]   = 1 - P(occ)                        → empty
    """

    def __init__(self, D: int = 80):
        super().__init__()
        self.head_occ = nn.Linear(D, 1)   # 81 params
        self.head_dyn = nn.Linear(D, 1)   # 81 params
        self.head_act = nn.Linear(D, 6)   # 486 params
        self._init_weights()

    def _init_weights(self):
        for head in [self.head_occ, self.head_dyn, self.head_act]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, z: torch.Tensor) -> dict:
        """
        z: [B, D]
        returns dict with:
            p_8class   [B, 8]  — final class probabilities (sums to 1)
            p_occ      [B, 1]  — P(occupied)
            p_dyn      [B, 1]  — P(dynamic | occupied)
            p_act      [B, 6]  — P(action | dynamic, occupied)
            logit_occ  [B, 1]  — raw logits for calibration
            logit_dyn  [B, 1]
            logit_act  [B, 6]
        """
        lo = self.head_occ(z)   # [B, 1]
        ld = self.head_dyn(z)   # [B, 1]
        la = self.head_act(z)   # [B, 6]

        p_occ = torch.sigmoid(lo)                # [B, 1]
        p_dyn = torch.sigmoid(ld)                # [B, 1]
        p_act = torch.softmax(la, dim=-1)        # [B, 6], sums to 1

        # Hierarchical composition
        # actions:      [B, 6] = p_occ * p_dyn * p_act
        # lying_still:  [B, 1] = p_occ * (1 - p_dyn)
        # empty:        [B, 1] = 1 - p_occ
        p_actions     = p_occ * p_dyn * p_act               # [B, 6]
        p_lying_still = p_occ * (1.0 - p_dyn)               # [B, 1]
        p_empty       = 1.0 - p_occ                          # [B, 1]

        p_8class = torch.cat([p_actions, p_lying_still, p_empty], dim=-1)  # [B, 8]

        return {
            'p_8class'  : p_8class,
            'p_occ'     : p_occ,
            'p_dyn'     : p_dyn,
            'p_act'     : p_act,
            'logit_occ' : lo,
            'logit_dyn' : ld,
            'logit_act' : la,
        }


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling calibration [Guo et al., ICML 2017].

    Known limitation: P_8class is a product of sigmoid×softmax activations,
    not a single logit vector. We approximate by treating log(P_8class) as
    pseudo-logits and scaling by T. This is an approximation but better than
    no calibration.

    Optimization: LBFGS on val set (held out from training).
    """

    def __init__(self, init_temp: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(init_temp)))

    @torch.no_grad()
    def calibrate(
        self,
        model: nn.Module,
        val_loader,
        device: torch.device,
    ) -> float:
        """
        Fit temperature T on val_loader using LBFGS.
        Returns the optimized temperature value.
        """
        model.eval()
        self.to(device)

        lp_list, lab_list = [], []
        for X_amp, X_dfs, labels in val_loader:
            out = model(X_amp.to(device), X_dfs.to(device))
            lp_list.append(torch.log(out['p_8class'].clamp(min=1e-7)).detach())
            lab_list.append(labels.to(device))

        log_probs = torch.cat(lp_list)   # [N, 8]
        labels    = torch.cat(lab_list)  # [N]

        # Require grad only for temperature
        self.temperature.requires_grad_(True)
        opt = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=100)

        def closure():
            opt.zero_grad()
            scaled = log_probs / self.temperature.clamp(min=0.05)
            loss   = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature.requires_grad_(False)

        t_val = self.temperature.item()
        print(f"[TemperatureScaler] Calibrated T = {t_val:.4f}")
        return t_val

    def forward(self, p_8class: torch.Tensor) -> torch.Tensor:
        """
        p_8class: [B, 8] raw hierarchical probabilities
        returns:  [B, 8] calibrated probabilities
        """
        T      = self.temperature.clamp(min=0.05)
        scaled = torch.log(p_8class.clamp(min=1e-7)) / T
        return torch.softmax(scaled, dim=-1)
