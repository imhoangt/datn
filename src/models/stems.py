"""
src/models/stems.py
AmpStem: joint F×A projection → D → inject TemporalPE + FiLM receiver conditioning
DfsStem: 2D conv on spectrogram → project to D → inject TemporalPE + FiLM receiver conditioning

Design notes (v7.2):
- AmpStem: joint_proj Linear(F*A→D) replaces sequential ant_mlp+freq_proj (FIX-M1)
  Enables model to learn cross-(subcarrier, antenna) interactions
- DfsStem: T_dfs=28, V_dfs=128 fixed by STFT params
- Learnable Temporal PE (fixed-length T) [ref: PE survey arXiv:2502]
- FiLM receiver conditioning: scale×h + shift instead of additive h+embed (Perez 2018)
  scale init=1 (identity), shift init=0 (no shift) → safe warm-start
  Additive-only embed cannot modulate amplitude differences across hardware
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AmpStem(nn.Module):
    """
    Input:  x [B, T, F_sub, A]   (amplitude features, after F1-F4 preprocessing)
    Output: z [B, T, D]

    Steps:
        1. joint_proj + LayerNorm: [B,T,F*A=208] → [B,T,D]   joint F×A projection
        2. + TemporalPE [T,D]                                  inject temporal order
        3. FiLM: scale[rx_id] × h + shift[rx_id]              modulate by receiver identity
    """

    def __init__(self, D: int = 80, A: int = 4, F_sub: int = 52, T: int = 350):
        super().__init__()
        self.D     = D
        self.T     = T
        self.F_sub = F_sub
        self.A     = A

        # Joint F×A projection: flatten F and A then project (FIX-M1)
        self.joint_proj  = nn.Linear(F_sub * A, D)
        self.norm        = nn.LayerNorm(D)
        # Positional encoding (learnable, T tokens)
        self.temporal_pe = nn.Embedding(T, D)
        # FiLM receiver conditioning (Perez et al. 2018): scale×h + shift
        # scale init=1 → identity; shift init=0 → no offset (safe warm-start)
        self.recv_scale  = nn.Embedding(3, D)
        self.recv_shift  = nn.Embedding(3, D)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.temporal_pe.weight, std=0.02)
        nn.init.ones_(self.recv_scale.weight)    # identity scale at init
        nn.init.zeros_(self.recv_shift.weight)   # zero shift at init
        nn.init.xavier_uniform_(self.joint_proj.weight)
        nn.init.zeros_(self.joint_proj.bias)

    def forward(self, x: torch.Tensor, rx_id: int) -> torch.Tensor:
        """
        x:     [B, T, F_sub, A]
        rx_id: int in {0,1,2}
        return [B, T, D]
        """
        # 1. Joint F×A projection: flatten last two dims, then project + LN
        B, T, F, A = x.shape
        h = x.reshape(B, T, F * A)               # [B, T, F*A=208] — reshape (not view) for non-contiguous slices
        h = self.norm(self.joint_proj(h))         # [B, T, D]

        # 2. Temporal positional encoding
        positions = torch.arange(h.size(1), device=h.device)         # [T]
        h = h + self.temporal_pe(positions)                           # broadcast [T,D]

        # 3. FiLM receiver modulation: scale × h + shift (direct weight indexing)
        scale = self.recv_scale.weight[rx_id]     # [D]
        shift = self.recv_shift.weight[rx_id]     # [D]
        h = h * scale + shift                     # [B, T, D] broadcasts with [D]

        return h   # [B, T, D]


class DfsStem(nn.Module):
    """
    Input:  x [B, T_dfs, V_dfs]   (log-normalized Doppler spectrogram, single RX)
    Output: z [B, T_dfs, D]

    Steps:
        1. Unsqueeze channel dim → [B, 1, T_dfs, V_dfs]
        2. 3 × Conv2d (GroupNorm + GELU) with spatial-stride to reduce V
        3. Reshape: [B, C, T2, V2] → [B, T2, C*V2]
        4. Linear projection + LayerNorm: C*V2 → D
        5. + TemporalPE [T2, D]
        6. + RecvEmbed [D]

    Conv channels: 1 → 16 → 32 → 48
    V reduction:  128 → 128 → 64 → 32  (stride=(1,2) on conv2+conv3)
    T preserved:  28 → 28 throughout
    Final flatten: 48 × 32 = 1536 → D=80
    """

    def __init__(self, D: int = 80, T_dfs: int = 28, V_dfs: int = 128):
        super().__init__()
        self.D     = D
        self.T_dfs = T_dfs
        self.V_dfs = V_dfs

        # Conv block 1: preserve spatial dims (stride=1×1)
        self.conv1 = nn.Conv2d(1,  16, kernel_size=(3, 5), padding=(1, 2))
        self.gn1   = nn.GroupNorm(4, 16)    # 16/4=4 channels/group ✓

        # Conv block 2: halve V (stride=1×2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.gn2   = nn.GroupNorm(8, 32)    # 32/8=4 channels/group ✓

        # Conv block 3: halve V again (stride=1×2)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.gn3   = nn.GroupNorm(8, 48)    # 48/8=6 channels/group ✓

        # Compute V after convolutions: 128 → 64 → 32
        V_after_convs = V_dfs // 4   # 128 // 4 = 32
        flatten_dim   = 48 * V_after_convs   # 48 × 32 = 1536

        # Projection: 1536 → D (dominant param layer: 1536×80 ≈ 123K params)
        self.proj        = nn.Linear(flatten_dim, D)
        self.norm        = nn.LayerNorm(D)
        # Temporal PE for T_dfs=28 frames
        self.temporal_pe = nn.Embedding(T_dfs, D)
        # FiLM receiver conditioning: scale×h + shift (same as AmpStem)
        self.recv_scale  = nn.Embedding(3, D)
        self.recv_shift  = nn.Embedding(3, D)

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.temporal_pe.weight, std=0.02)
        nn.init.ones_(self.recv_scale.weight)    # identity scale at init
        nn.init.zeros_(self.recv_shift.weight)   # zero shift at init
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, rx_id: int) -> torch.Tensor:
        """
        x:     [B, T_dfs=28, V_dfs=128]
        rx_id: int in {0,1,2}
        return [B, T_dfs, D]
        """
        # 1. Add channel dim
        h = x.unsqueeze(1)                                            # [B, 1, 28, 128]

        # 2. Conv blocks
        h = F.gelu(self.gn1(self.conv1(h)))                          # [B, 16, 28, 128]
        h = F.gelu(self.gn2(self.conv2(h)))                          # [B, 32, 28,  64]
        h = F.gelu(self.gn3(self.conv3(h)))                          # [B, 48, 28,  32]

        # 3. Reshape: [B, C, T2, V2] → [B, T2, C*V2]
        B, C, T2, V2 = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, T2, C * V2)            # [B, 28, 1536]

        # 4. Project + LN
        h = self.norm(self.proj(h))                                   # [B, 28, D]

        # 5. Temporal PE
        positions = torch.arange(T2, device=h.device)
        h = h + self.temporal_pe(positions)                           # [B,28,D]+[28,D]

        # 6. FiLM receiver modulation
        scale = self.recv_scale.weight[rx_id]     # [D]
        shift = self.recv_shift.weight[rx_id]     # [D]
        h = h * scale + shift                     # [B, T_dfs, D] broadcasts with [D]

        return h   # [B, T_dfs, D]
