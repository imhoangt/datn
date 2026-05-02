"""
src/models/fusion.py
CrossModalFusion: gated cross-attention (Q=Z_amp, K/V=Z_dfs)
ReceiverFusion:   learnable weighted sum of M receiver representations

Design notes (v6.6):
- CrossModalFusion: self.norm_out is named module (FIX v6.0); gate init=-2.0
- ReceiverFusion:   z_per_recv stacked [B,3,D] for L_cons (FIX v6.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """
    Fuses amplitude encoder output (Z_amp) with DFS encoder output (Z_dfs)
    via cross-attention, gated by a learnable scalar.

        Q  = norm_amp(Z_amp)   [B, T_amp=350, D]
        K  = norm_dfs(Z_dfs)   [B, T_dfs=28,  D]
        V  = norm_dfs(Z_dfs)   [B, T_dfs=28,  D]
        Z_fused = Z_amp + sigmoid(gate) * cross_attn(Q, K, V)
        output  = norm_out(Z_fused)

    Gate init=-2.0 → sigmoid(-2)≈0.12: conservative start so noisy DFS
    doesn't destabilize amplitude branch at beginning of training.
    """

    def __init__(self, D: int = 80, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_amp   = nn.LayerNorm(D)
        self.norm_dfs   = nn.LayerNorm(D)
        self.cross_attn = nn.MultiheadAttention(
            D, n_heads, dropout=dropout, batch_first=True
        )
        # Named module — NOT created inline per forward call (FIX v6.0)
        self.norm_out   = nn.LayerNorm(D)
        # Learnable gate: init=-2.0 → sigmoid≈0.12
        self.gate       = nn.Parameter(torch.full((1,), -2.0))

    def forward(
        self, Z_amp: torch.Tensor, Z_dfs: torch.Tensor
    ) -> torch.Tensor:
        """
        Z_amp: [B, T_amp=350, D]
        Z_dfs: [B, T_dfs=28,  D]
        return: [B, T_amp=350, D]
        """
        Q  = self.norm_amp(Z_amp)
        KV = self.norm_dfs(Z_dfs)
        attn_out, _ = self.cross_attn(Q, KV, KV)           # [B, T_amp, D]
        Z_fused = Z_amp + torch.sigmoid(self.gate) * attn_out
        return self.norm_out(Z_fused)                        # [B, T_amp, D]


class ReceiverFusion(nn.Module):
    """
    Combines M=3 per-receiver encoded representations into one global representation.

    For each receiver m, we pool Z_fused_m over time to get a [B, D] descriptor.
    A linear layer maps the concatenated descriptors [B, D*M] to M attention weights α.
    The output is a weighted temporal sum across receivers.

    Also returns z_per_recv [B, M, D] for L_cons (receiver consistency loss).
    """

    def __init__(self, D: int = 80, M: int = 3):
        super().__init__()
        self.M = M
        # 240 → 3 = 723 params
        self.weight_proj = nn.Linear(D * M, M)

    def forward(
        self, Z_fused_list: list
    ) -> tuple:
        """
        Z_fused_list: list of M tensors, each [B, T=350, D]
        returns:
            Z_global   [B, T=350, D]   — weighted global representation
            z_per_recv [B, M, D]       — per-receiver global descriptors
        """
        # Time-pool each receiver: [B, D]
        z_means = [Z.mean(dim=1) for Z in Z_fused_list]       # list of [B, D]

        # Stack for L_cons: [B, M, D]
        z_per_recv = torch.stack(z_means, dim=1)

        # Compute attention weights: [B, M]
        alpha = torch.softmax(
            self.weight_proj(torch.cat(z_means, dim=-1)), dim=-1
        )

        # Weighted temporal sum: [B, T, D]
        Z_global = sum(
            alpha[:, m].unsqueeze(1).unsqueeze(2) * Z_fused_list[m]
            for m in range(self.M)
        )

        return Z_global, z_per_recv
