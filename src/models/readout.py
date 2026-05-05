"""
src/models/readout.py
DualScaleReadout: combines fast (v_mean, v_energy) and slow (BlurPool) temporal features.

Design notes (v7.1):
- BlurPool: Binomial[1,4,6,4,1]/16 anti-aliasing filter, stride=4, padding=0
  → output length = (350-5)//4+1 = 87 frames ✓
- groups=D for depthwise slow conv (separable)
- MLP: 240→160→80
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualScaleReadout(nn.Module):
    """
    Aggregates temporal sequence [B, T=350, D] into a single vector [B, D].

    Three parallel readout streams:
        v_mean   [B, D]: global temporal mean (fast, all scales)
        v_energy [B, D]: global temporal energy (fast, amplitude-aware)
        v_slow   [B, D]: slow-scale features via BlurPool stride=4 → depthwise conv → mean

    Concatenated → MLP: [B, 3D=240] → [B, 2D=160] → [B, D=80]

    BlurPool: Binomial filter [1,4,6,4,1]/16, stride=4, padding=0
        (350 - 5) // 4 + 1 = 87 frames  ← verified
    """

    def __init__(self, D: int = 80):
        super().__init__()
        self.D = D

        # Binomial blur kernel [1,4,6,4,1]/16, shape [D, 1, 5] for grouped conv
        blur_k = torch.tensor([1., 4., 6., 4., 1.]) / 16.
        # expand: [1,1,5] → [D,1,5]
        self.register_buffer(
            'blur_kernel', blur_k.view(1, 1, 5).expand(D, 1, 5).contiguous()
        )

        # Depthwise slow-scale conv on blurred features
        self.dw_slow = nn.Conv1d(D, D, kernel_size=5, padding=2, groups=D)

        # Projection MLP: 3D → 2D → D
        self.mlp = nn.Sequential(
            nn.Linear(D * 3, D * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(D * 2, D),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.dw_slow.weight, nonlinearity='relu')
        nn.init.zeros_(self.dw_slow.bias)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, Z_global: torch.Tensor) -> torch.Tensor:
        """
        Z_global: [B, T=350, D]
        return:   [B, D]
        """
        # Fast streams
        v_mean   = Z_global.mean(dim=1)                    # [B, D]
        v_energy = (Z_global ** 2).mean(dim=1)             # [B, D]

        # Slow stream: BlurPool + depthwise conv + mean
        Z_t    = Z_global.transpose(1, 2)                  # [B, D, T=350]
        Z_blur = F.conv1d(
            Z_t, self.blur_kernel,
            stride=4, padding=0, groups=self.D
        )                                                   # [B, D, 87]
        v_slow = self.dw_slow(Z_blur).mean(dim=-1)         # [B, D]

        # Concatenate and project
        z = torch.cat([v_mean, v_energy, v_slow], dim=-1)  # [B, 3D=240]
        return self.mlp(z)                                  # [B, D=80]
