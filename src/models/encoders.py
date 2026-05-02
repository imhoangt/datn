"""
src/models/encoders.py
DilatedConvTransformerBlock (DCT-Block): Pre-LN, dilated depthwise conv + self-attn + FFN
AmpEncoder: 2× DCT-Block (k=7, d=[2,4]) with gradient checkpointing
DfsEncoder: 2× DCT-Block (k=5, d=[1,2]) with gradient checkpointing

Design notes (v6.6):
- Pre-LN for training stability [Xiong et al., ICML 2020]
- PE injected upstream (Stem); blocks add no internal PE
- Gradient checkpointing on ALL blocks, only during training [v6.4]
- Weight sharing: one encoder instance is called 3× (M=3 receivers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


class DilatedConvTransformerBlock(nn.Module):
    """
    Pre-LN block with 3 sub-layers:
        1. Dilated depthwise conv → pointwise conv (local context)
        2. Multi-head self-attention (global context, leverages upstream PE)
        3. Feed-forward network

    Args:
        D          : model dimension
        kernel_size: depthwise conv kernel
        dilation   : dilation factor for depthwise conv
        n_heads    : MHA heads (D must be divisible by n_heads)
        ffn_ratio  : FFN hidden dim = D × ffn_ratio
        dropout    : dropout probability
    """

    def __init__(
        self,
        D: int,
        kernel_size: int,
        dilation: int,
        n_heads: int = 4,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert D % n_heads == 0, f"D={D} must be divisible by n_heads={n_heads}"

        self.norm1   = nn.LayerNorm(D)
        self.norm2   = nn.LayerNorm(D)
        self.norm3   = nn.LayerNorm(D)

        # Dilated depthwise conv (temporal local context)
        self.dw_conv = nn.Conv1d(
            D, D, kernel_size=kernel_size,
            dilation=dilation, padding='same', groups=D
        )
        # Pointwise mixing
        self.pw_conv = nn.Conv1d(D, D, kernel_size=1)

        # Multi-head self-attention (global context)
        self.attn = nn.MultiheadAttention(
            D, n_heads, dropout=dropout, batch_first=True
        )

        # FFN
        hidden_dim = D * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, D),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.dw_conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.dw_conv.bias)
        nn.init.xavier_uniform_(self.pw_conv.weight)
        nn.init.zeros_(self.pw_conv.bias)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, T, D]"""
        # --- Branch 1: Dilated depthwise conv (local) ---
        h = self.norm1(x).transpose(1, 2)          # [B, D, T]
        h = self.pw_conv(self.dw_conv(h)).transpose(1, 2)  # [B, T, D]
        x = x + h

        # --- Branch 2: Self-attention (global) ---
        h_n = self.norm2(x)                         # cache pre-norm [B, T, D]
        h, _ = self.attn(h_n, h_n, h_n)
        x = x + h

        # --- Branch 3: FFN ---
        x = x + self.ffn(self.norm3(x))

        return x


class AmpEncoder(nn.Module):
    """
    2× DilatedConvTransformerBlock for amplitude branch.
    Block 1: kernel=7, dilation=2 → RF = 14 packets (40ms at 350Hz)
    Block 2: kernel=7, dilation=4 → RF = 28 packets (80ms at 350Hz)
    + Self-attention: global T=350 (1 second)

    One instance is reused for M=3 receivers → weight sharing.
    """

    def __init__(self, D: int = 80, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            DilatedConvTransformerBlock(D, kernel_size=7, dilation=2, n_heads=4),
            DilatedConvTransformerBlock(D, kernel_size=7, dilation=4, n_heads=4),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T=350, D] → [B, T=350, D]"""
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = grad_ckpt(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


class DfsEncoder(nn.Module):
    """
    2× DilatedConvTransformerBlock for DFS branch.
    Block 1: kernel=5, dilation=1 → RF = 5 frames
    Block 2: kernel=5, dilation=2 → RF = 10 frames
    T_dfs=28 frames → attn covers full 28-frame DFS spectrogram.

    One instance is reused for M=3 receivers → weight sharing.
    """

    def __init__(self, D: int = 80, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            DilatedConvTransformerBlock(D, kernel_size=5, dilation=1, n_heads=4),
            DilatedConvTransformerBlock(D, kernel_size=5, dilation=2, n_heads=4),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T_dfs=28, D] → [B, T_dfs=28, D]"""
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = grad_ckpt(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x
