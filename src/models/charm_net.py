"""
src/models/charm_net.py
CHARMNet v7.0: Channel, Hardware, Activity Recognition Multi-stream Network

Architecture:
    Per-RX (shared weights):
        AmpStem → AmpEncoder → ┐
                                CrossModalFusion → Z_fused_m [B,350,D]
        DfsStem → DfsEncoder → ┘

    ReceiverContextFusion([Z_0, Z_1, Z_2]) → cross-RX context (FIX-M4)
    ReceiverFusion([Z_0, Z_1, Z_2]) → Z_global [B,350,D], z_per_recv [B,3,D]
    DualScaleReadout(Z_global) → z [B,D]
    HierarchicalHeads(z) → p_8class [B,8], logit_unified [B,8]

Design notes (v7.0):
    - F_sub and A are configurable for multi-dataset support
    - All datasets normalized to F_sub=52, A=4 in preprocessing
    - drop_path_rate passed to both encoders (FIX-T3)
    - recv_context provides cross-receiver information sharing (FIX-M4)
    - param_count() utility for quick verification
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .stems    import AmpStem, DfsStem
from .encoders import AmpEncoder, DfsEncoder
from .fusion   import CrossModalFusion, ReceiverFusion, ReceiverContextFusion
from .readout  import DualScaleReadout
from .heads    import HierarchicalHeads, TemperatureScaler


class CHARMNet(nn.Module):
    """
    Full CHARM-Net v7.0.

    Input shapes:
        X_amp: [B, T=350, F=52, M=3, A=4]  amplitude features (float32)
        X_dfs: [B, T_dfs=28, V=128, M=3]   DFS spectrogram features (float32)

    Output dict:
        p_8class      [B, 8]   — class probabilities (softmax of logit_unified)
        logit_unified [B, 8]   — raw logits for L_main + TemperatureScaler
        p_factored    [B, 8]   — factored hierarchical probabilities (aux)
        p_occ         [B, 1]
        p_dyn         [B, 1]
        p_act         [B, 6]
        logit_occ     [B, 1]
        logit_dyn     [B, 1]
        logit_act     [B, 6]
        z_per_recv    [B, 3, D]  — per-receiver descriptors for L_cons
    """

    def __init__(
        self,
        D: int = 80,
        A: int = 4,
        F_sub: int = 52,
        T: int = 350,
        T_dfs: int = 28,
        V_dfs: int = 128,
        M: int = 3,
        n_heads: int = 4,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.D    = D
        self.M    = M
        self.F_sub = F_sub
        self.A     = A

        # Stems (shared across M receivers)
        self.amp_stem = AmpStem(D=D, A=A, F_sub=F_sub, T=T)
        self.dfs_stem = DfsStem(D=D, T_dfs=T_dfs, V_dfs=V_dfs)

        # Encoders (shared across M receivers; DropPath for stochastic depth — FIX-T3)
        self.amp_encoder = AmpEncoder(D=D, use_checkpoint=use_checkpoint,
                                       drop_path_rate=drop_path_rate)
        self.dfs_encoder = DfsEncoder(D=D, use_checkpoint=use_checkpoint,
                                       drop_path_rate=drop_path_rate)

        # Cross-modal fusion (shared across M receivers)
        self.cross_fusion = CrossModalFusion(D=D, n_heads=n_heads, dropout=dropout)

        # Cross-receiver context sharing (FIX-M4)
        self.recv_context = ReceiverContextFusion(D=D)

        # Receiver fusion (mean+var alpha — FIX-M2)
        self.recv_fusion = ReceiverFusion(D=D, M=M)

        # Readout
        self.readout = DualScaleReadout(D=D)

        # Classification heads (unified + factored aux — FIX-M5)
        self.heads = HierarchicalHeads(D=D)

    def encode_receiver(
        self,
        X_amp_m: torch.Tensor,
        X_dfs_m: torch.Tensor,
        rx_id: int,
    ) -> torch.Tensor:
        """
        Process one receiver m.

        X_amp_m: [B, T=350, F=52, A=4]
        X_dfs_m: [B, T_dfs=28, V=128]
        return:  [B, T=350, D]
        """
        Z_amp = self.amp_encoder(self.amp_stem(X_amp_m, rx_id))   # [B, 350, D]
        Z_dfs = self.dfs_encoder(self.dfs_stem(X_dfs_m, rx_id))   # [B,  28, D]
        return self.cross_fusion(Z_amp, Z_dfs)                      # [B, 350, D]

    def forward(
        self,
        X_amp: torch.Tensor,
        X_dfs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        X_amp: [B, T=350, F=52, M=3, A=4]   float32
        X_dfs: [B, T_dfs=28, V=128, M=3]    float32
        """
        # Process each receiver with shared weights
        Z_list = [
            self.encode_receiver(
                X_amp[:, :, :, m, :],   # [B, 350, 52, 4]
                X_dfs[:, :, :, m],      # [B,  28, 128]
                rx_id=m,
            )
            for m in range(self.M)
        ]

        # Cross-receiver context sharing (FIX-M4)
        Z_list = self.recv_context(Z_list)

        # Fuse receivers (mean+var alpha — FIX-M2)
        Z_global, z_per_recv = self.recv_fusion(Z_list)   # [B,350,D], [B,3,D]

        # Temporal readout
        z = self.readout(Z_global)                         # [B, D]

        # Classification (unified + factored aux — FIX-M5)
        out = self.heads(z)
        out['z_per_recv'] = z_per_recv

        return out

    def param_count(self) -> Dict[str, int]:
        """Returns parameter counts per component for verification."""
        def count(mod):
            return sum(p.numel() for p in mod.parameters())

        return {
            'amp_stem'      : count(self.amp_stem),
            'dfs_stem'      : count(self.dfs_stem),
            'amp_encoder'   : count(self.amp_encoder),
            'dfs_encoder'   : count(self.dfs_encoder),
            'cross_fusion'  : count(self.cross_fusion),
            'recv_context'  : count(self.recv_context),
            'recv_fusion'   : count(self.recv_fusion),
            'readout'       : count(self.readout),
            'heads'         : count(self.heads),
            'total'         : count(self),
        }


def build_model(cfg: dict, device: torch.device) -> CHARMNet:
    """
    Factory function: build CHARMNet from config dict.

    cfg: subset of config with keys from model section.
    """
    model = CHARMNet(
        D              = cfg.get('D', 80),
        A              = cfg.get('A', 4),
        F_sub          = cfg.get('F_sub', 52),
        T              = cfg.get('T_window', 350),
        T_dfs          = cfg.get('T_dfs', 28),
        V_dfs          = cfg.get('V_dfs', 128),
        M              = cfg.get('M', 3),
        n_heads        = cfg.get('n_heads', 4),
        ffn_ratio      = cfg.get('ffn_ratio', 2),
        dropout        = cfg.get('dropout', 0.1),
        drop_path_rate = cfg.get('drop_path_rate', 0.0),
        use_checkpoint = cfg.get('use_checkpoint', True),
    ).to(device)

    # Print parameter counts
    counts = model.param_count()
    print("[CHARMNet v7.0] Parameter counts:")
    for name, n in counts.items():
        print(f"  {name:20s}: {n:>8,}")

    return model
