"""
src/preprocessing/parsers/intel5300_parser.py
Parse Intel 5300 NIC CSI data for XRF55 dataset.

Intel 5300 / linux-80211n-csitool format:
  - 30 subcarrier groups (each group = 2 adjacent subs at 20MHz)
  - Complex int8 (signed 8-bit real + 8-bit imag)
  - 3 antennas per device
  - Sampling rate: ~200 Hz (varies, typically stored in .dat or .mat files)
  - Data shape from XRF55: (200*t) × 1 × 3 × 3 × 30 where t = duration in seconds
    → for our pipeline: T × M × A × F = T × 3 × 3 × 30

XRF55 stores WiFi CSI as numpy .npy or .mat files.
We normalize: [T, 30, 3] complex64 (single RX, 3 antennas, 30 subcarriers)
Then pad subcarriers 30→52 (zero-pad) and antennas 3→4 (zero-pad) for uniform model input.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# XRF55 nominal sampling rate (packets/sec)
XRF55_FS_NOMINAL = 200.0
# Intel 5300 subcarrier count at 20MHz
INTEL5300_N_SUBS = 30


def parse_xrf55_csi(
    file_path: str | Path,
    file_format: str = 'auto',
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load XRF55 WiFi CSI from numpy/mat file.

    XRF55 WiFi CSI shape from paper: (200t × 1 × 3 × 3 × 30)
        dim0: T samples (200 Hz × t seconds)
        dim1: 1 (TX antennas, single TX)
        dim2: 3 (receivers / RX devices)
        dim3: 3 (RX antennas per device)
        dim4: 30 subcarriers

    We reshape to: [T, 30, M=3, A=3] complex64 per receiver grouping.

    Returns:
        H_raw      [T, 30, M=3, A=3] complex64 (None if error)
        timestamps [T] float64 (synthesized at 200Hz if not available)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None, None

    # Determine format
    suffix = file_path.suffix.lower()
    if file_format == 'auto':
        file_format = suffix.lstrip('.')

    try:
        if file_format in ('npy', 'npz'):
            data = np.load(file_path, allow_pickle=True)
            if isinstance(data, np.ndarray):
                csi_raw = data
            else:
                # .npz: try 'csi' or first key
                key = 'csi' if 'csi' in data else list(data.keys())[0]
                csi_raw = data[key]

        elif file_format == 'mat':
            try:
                import scipy.io as sio
                mat = sio.loadmat(str(file_path))
                # Try common keys in XRF55 mat files
                for key in ('csi_trace', 'csi', 'data', 'WiFiCSI'):
                    if key in mat:
                        csi_raw = mat[key]
                        break
                else:
                    # Use first non-metadata key
                    keys = [k for k in mat.keys() if not k.startswith('_')]
                    csi_raw = mat[keys[0]]
            except ImportError:
                logger.error("scipy.io not available for .mat parsing")
                return None, None

        else:
            logger.error(f"Unsupported format: {file_format}")
            return None, None

    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None, None

    # Parse shape
    csi_raw = np.array(csi_raw)
    if csi_raw.ndim == 5:
        # (T, 1, M=3, A=3, F=30) → rearrange to (T, F=30, M=3, A=3)
        T, _, M, A, F = csi_raw.shape
        H_raw = csi_raw[:, 0, :, :, :]          # [T, M, A, F]
        H_raw = H_raw.transpose(0, 3, 1, 2)     # [T, F, M, A]
    elif csi_raw.ndim == 4:
        # Assume [T, F, M, A] or [T, M, A, F]
        if csi_raw.shape[-1] == 30:
            # [T, M, A, F=30] → [T, F, M, A]
            H_raw = csi_raw.transpose(0, 3, 1, 2)
        elif csi_raw.shape[1] == 30:
            # [T, F=30, M, A]
            H_raw = csi_raw
        else:
            logger.error(f"Unexpected shape {csi_raw.shape}")
            return None, None
    elif csi_raw.ndim == 2:
        # Flat [T, F*M*A] - rare
        T = csi_raw.shape[0]
        H_raw = csi_raw.reshape(T, 30, 3, 3)
    else:
        logger.error(f"Cannot parse shape {csi_raw.shape}")
        return None, None

    # Convert to complex64
    if not np.iscomplexobj(H_raw):
        # Some datasets store amplitude only → treat as real with zero imag
        H_raw = H_raw.astype(np.float32).view(np.float32) + 0j
        H_raw = H_raw.astype(np.complex64)
    else:
        H_raw = H_raw.astype(np.complex64)

    T_raw = H_raw.shape[0]

    # Synthesize timestamps at nominal fs (XRF55 doesn't always store timestamps)
    timestamps = np.arange(T_raw, dtype=np.float64) / XRF55_FS_NOMINAL

    logger.info(f"XRF55 loaded: {file_path.name} → shape {H_raw.shape}")
    return H_raw, timestamps


def normalize_intel5300_to_uniform(
    H_raw   : np.ndarray,
    target_F: Optional[int] = None,
    target_A: int = 4,
) -> np.ndarray:
    """
    Normalize Intel 5300 CSI [T, F_in, M, A_in] for pipeline consumption.

    Args:
        target_F: if given, zero-pad (or trim) subcarriers to target_F.
                  If None (default), keep original F_in subcarriers unchanged.
        target_A: zero-pad (or trim) antenna dim to this count.

    H_raw: [T, F_in, M, A_in] complex64
    Returns: [T, F_out, M, target_A] complex64
             where F_out = target_F if target_F is not None, else F_in
    """
    T, F_in, M, A_in = H_raw.shape

    # Subcarrier dimension
    if target_F is None:
        H_padF = H_raw
    elif F_in < target_F:
        pad_F  = target_F - F_in
        H_padF = np.concatenate(
            [H_raw, np.zeros((T, pad_F, M, A_in), dtype=np.complex64)],
            axis=1,
        )
    else:
        H_padF = H_raw[:, :target_F]

    F_out = H_padF.shape[1]

    # Antenna dimension
    if A_in < target_A:
        pad_A  = target_A - A_in
        H_padA = np.concatenate(
            [H_padF, np.zeros((T, F_out, M, pad_A), dtype=np.complex64)],
            axis=3,
        )
    else:
        H_padA = H_padF[:, :, :, :target_A]

    return H_padA
