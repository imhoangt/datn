"""
src/models/fusion.py
CrossModalFusion: gated cross-attention (Q=Z_amp, K/V=Z_dfs)
ReceiverFusion:   learnable weighted sum of M receiver representations
ReceiverContextFusion: lightweight cross-receiver context sharing

Design notes (v7.0):
- CrossModalFusion: self.norm_out is named module (FIX v6.0); gate init=-2.0
- ReceiverFusion: weight_proj uses concat(mean,var) → Linear(D*M*2, M) (FIX-M2)
  Variance term makes alpha sensitive to locally noisy receivers
- ReceiverContextFusion: FiLM-style residual cross-RX context; gate init=0 (FIX-M4)
  Safe warm-start: tanh(0)=0 → no effect at epoch 0, learns to activate gradually
"""

import torch
import torch.nn as nn


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

    For each receiver m, we pool Z_fused_m over time to get mean [B,D] and variance [B,D].
    A linear layer maps the concatenated mean+var descriptors [B, D*M*2] to M weights α.
    Variance term detects locally noisy receivers (FIX-M2).

    Also returns z_per_recv [B, M, D] for L_cons (receiver consistency loss).
    """

    def __init__(self, D: int = 80, M: int = 3):
        super().__init__()
        self.M = M
        # D*M*2 → M: mean+var concat (FIX-M2); was D*M → M
        self.weight_proj = nn.Linear(D * M * 2, M)

    def forward(
        self, Z_fused_list: list
    ) -> tuple:
        """
        Z_fused_list: list of M tensors, each [B, T=350, D]
        returns:
            Z_global   [B, T=350, D]   — weighted global representation
            z_per_recv [B, M, D]       — per-receiver global descriptors
        """
        # Time-pool each receiver: mean [B, D] and variance [B, D]
        z_means = [Z.mean(dim=1) for Z in Z_fused_list]                       # M × [B, D]
        z_vars  = [Z.var(dim=1, unbiased=False) for Z in Z_fused_list]        # M × [B, D]

        # Stack for L_cons: [B, M, D]
        z_per_recv = torch.stack(z_means, dim=1)

        # Compute attention weights using mean+var concat (FIX-M2)
        # Python list concat (not tensor addition): 2*M items → [B, D*M*2]
        feat  = torch.cat(z_means + z_vars, dim=-1)
        alpha = torch.softmax(self.weight_proj(feat), dim=-1)   # [B, M]

        # Weighted temporal sum: [B, T, D]
        Z_global = sum(
            alpha[:, m].unsqueeze(1).unsqueeze(2) * Z_fused_list[m]
            for m in range(self.M)
        )

        return Z_global, z_per_recv


class ReceiverContextFusion(nn.Module):
    """
    Lightweight cross-receiver context sharing (FIX-M4).

    Each receiver receives a residual update from the mean of the other receivers'
    temporal representations. Gate initialized to 0 → tanh(0)=0 → safe warm-start
    (no effect at epoch 0, learns to activate gradually).

    Params: D×D + D + 1 = 6481 for D=80.
    """

    def __init__(self, D: int = 80):
        super().__init__()
        self.context_proj = nn.Linear(D, D)
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

    def forward(self, Z_list: list) -> list:
        """
        Z_list: M × [B, T, D]
        returns: M × [B, T, D] with cross-receiver residual context added
        """
        M = len(Z_list)
        if M <= 1:
            return Z_list
        z_means = [Z.mean(dim=1) for Z in Z_list]      # M × [B, D]
        Z_out = []
        for m in range(M):
            # Context = mean of all other receivers' temporal representations
            others = [z_means[k] for k in range(M) if k != m]
            ctx = torch.stack(others, dim=0).mean(dim=0)    # [B, D]
            ctx = self.context_proj(ctx).unsqueeze(1)        # [B, 1, D]
            Z_out.append(Z_list[m] + torch.tanh(self.gate) * ctx)
        return Z_out
