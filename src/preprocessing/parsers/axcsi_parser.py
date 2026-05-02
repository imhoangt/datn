"""
src/preprocessing/parsers/axcsi_parser.py
Parse AX-CSI data for 'Exposing the CSI' dataset (Broadcom 43684, 802.11ax).

Dataset details (Cominelli et al., PerCom 2023):
  - Hardware: AX-CSI (Broadcom 43684, 802.11ax / Wi-Fi 6)
  - Channel: 160 MHz, 5 GHz band
  - Subcarriers: 2048 OFDM subcarriers per antenna per spatial stream
  - Sampling rate: ~150 frames/sec (150 fps injection rate)
  - Setup: 3 individuals, 3 environments, 12 activities
  - 80 GB dataset (~2 hours of CSI data)
  - Format: typically .npy or .npz files from their public repo

Subcarrier normalization strategy (2048→52):
  - Uniform selection: select 52 evenly spaced subcarriers from 2048
  - This retains broad spectral coverage while matching our model's F=52
  - Fisher selection (G5) further reduces to 13 within pipeline

GitHub: ansresearch/exposing-the-csi
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# AX-CSI dataset parameters
AXCSI_N_SUBS    = 2048   # 160MHz → 2048 subcarriers
AXCSI_FS        = 150.0  # ~150 frames/sec
AXCSI_N_STREAMS = 4      # up to 4 spatial streams
TARGET_N_SUBS   = 52


def load_axcsi_file(
    file_path: str | Path,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load AX-CSI numpy array from 'Exposing the CSI' dataset.

    Expected file formats:
        .npy:  array of shape [T, 2048, streams] or [T, 2048] complex
        .npz:  dict with 'csi' key
        .mat:  MATLAB file with 'csi' variable

    Returns:
        H_raw      [T, 2048, A] complex64  (None on error)
        timestamps [T] float64
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None, None

    try:
        suffix = file_path.suffix.lower()

        if suffix == '.npy':
            data = np.load(file_path, allow_pickle=False)

        elif suffix == '.npz':
            npz  = np.load(file_path, allow_pickle=False)
            key  = 'csi' if 'csi' in npz else list(npz.keys())[0]
            data = npz[key]

        elif suffix == '.mat':
            import scipy.io as sio
            mat  = sio.loadmat(str(file_path))
            for k in ('csi', 'CSI', 'data'):
                if k in mat:
                    data = mat[k]
                    break
            else:
                data = mat[[k for k in mat if not k.startswith('_')][0]]

        else:
            logger.error(f"Unsupported extension: {suffix}")
            return None, None

    except Exception as e:
        logger.error(f"Load error {file_path}: {e}")
        return None, None

    data = np.array(data)

    # Normalize shape to [T, F, A]
    if data.ndim == 2:
        # [T, F]: single antenna
        data = data[:, :, np.newaxis]
    elif data.ndim == 3:
        # Could be [T, F, A] or [T, A, F]
        if data.shape[2] > data.shape[1]:
            # [T, A, F] → transpose
            data = data.transpose(0, 2, 1)
    elif data.ndim == 4:
        # [T, streams, F, antennas] or similar → flatten streams×antennas
        T = data.shape[0]
        data = data.reshape(T, -1, data.shape[-1])
        if data.shape[1] != AXCSI_N_SUBS:
            data = data.transpose(0, 2, 1)

    # Convert to complex64
    if not np.iscomplexobj(data):
        logger.warning(f"{file_path}: real-valued data, treating as amplitude only")
        data = data.astype(np.float32).astype(np.complex64)
    else:
        data = data.astype(np.complex64)

    T_raw = data.shape[0]
    timestamps = np.arange(T_raw, dtype=np.float64) / AXCSI_FS

    logger.info(f"AX-CSI loaded: {file_path.name} → {data.shape}")
    return data, timestamps


def uniform_select_subcarriers(
    H_raw   : np.ndarray,
    n_select: int = 52,
) -> np.ndarray:
    """
    Uniformly select n_select subcarriers from H_raw.

    H_raw:    [T, F_raw=2048, A] complex64
    n_select: target number of subcarriers (default 52)

    Uses numpy linspace to get evenly spaced indices across the full bandwidth,
    preserving spectral coverage without aliasing.

    Returns: [T, n_select, A] complex64
    """
    F_raw   = H_raw.shape[1]
    indices = np.round(np.linspace(0, F_raw - 1, n_select)).astype(int)
    indices = np.clip(indices, 0, F_raw - 1)
    # Ensure unique indices (linspace may round to same integer for small F_raw)
    _, unique_mask = np.unique(indices, return_index=True)
    if len(unique_mask) < n_select:
        # Fallback: use first n_select unique
        indices = np.unique(indices)[:n_select]
        if len(indices) < n_select:
            pad = np.arange(len(indices), n_select)
            indices = np.concatenate([indices, indices[-1] + 1 + pad])
            indices = np.clip(indices, 0, F_raw - 1)

    return H_raw[:, indices].astype(np.complex64)


def normalize_axcsi_to_uniform(
    H_raw   : np.ndarray,
    target_F: int = 52,
    target_A: int = 4,
) -> np.ndarray:
    """
    Normalize AX-CSI [T, F_raw, A_raw] to unified format [T, 52, M=1, 4].

    NOTE: 'Exposing the CSI' dataset has 1 RX device (3 collectors = 3 files).
    We treat each file as 1 receiver. Multiple RX files are stacked in pipeline.

    Steps:
        1. Select 52 subcarriers (uniform)
        2. Pad/crop antennas to A=4

    Returns: [T, 52, 4] complex64 (single receiver, no M dimension yet)
    """
    # 1. Subcarrier selection
    if H_raw.shape[1] != target_F:
        H_sel = uniform_select_subcarriers(H_raw, n_select=target_F)
    else:
        H_sel = H_raw

    T, F, A_in = H_sel.shape

    # 2. Antenna padding/cropping
    if A_in < target_A:
        pad = np.zeros((T, F, target_A - A_in), dtype=np.complex64)
        H_out = np.concatenate([H_sel, pad], axis=2)
    else:
        H_out = H_sel[:, :, :target_A]

    return H_out   # [T, 52, 4]
