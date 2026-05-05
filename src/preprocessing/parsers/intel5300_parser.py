"""
src/preprocessing/parsers/intel5300_parser.py
Parse Intel 5300 CSI Tool binary .dat files (XRF55 dataset).

Binary record format (linux-80211n-csitool):
    [2B field_len big-endian] [1B code=0xBB] [20B header] [len_csi bytes CSI]

Header layout (20 bytes):
    timestamp(4) + bfee_count(2) + reserved(2) + Nrx(1) + Ntx(1) +
    rssi_a(1) + rssi_b(1) + rssi_c(1) + noise(1) + agc(1) +
    antenna_sel(1) + len_csi(2 LE) + flags(2) = 20 bytes

CSI data: 10-bit packed signed integers, order [sub][Tx][Rx] × (imag, real).
Signed: val >= 512 → val - 1024.

Output per file: [T, n_sub=30, Nrx] complex64
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

XRF55_FS_NOMINAL = 200.0   # nominal sampling rate (packets/sec)
INTEL5300_N_SUBS = 30       # Intel 5300: 30 subcarrier groups at 20MHz


def parse_intel5300_dat(
    filepath: str | Path,
    n_sub: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse Intel 5300 CSI Tool binary .dat file.

    Args:
        filepath: path to .dat file
        n_sub:    number of subcarriers (default 30 for Intel 5300 20MHz)

    Returns:
        H         [T, n_sub, Nrx] complex64
        timestamps [T] float64 (synthesized at XRF55_FS_NOMINAL if not extracted)

    Raises:
        FileNotFoundError: if file does not exist (raised by open())
        ValueError: if file contains 0 valid CSI records
    """
    records = []

    with open(filepath, 'rb') as f:
        while True:
            # Each record begins with 2-byte big-endian field_len + 1-byte code
            header_bytes = f.read(3)
            if len(header_bytes) < 3:
                break   # EOF

            field_len = int.from_bytes(header_bytes[:2], 'big')
            code      = header_bytes[2]

            if code != 0xBB:
                # Not a CSI record — skip
                remaining = field_len - 1   # already consumed 1 byte (code)
                if remaining > 0:
                    f.seek(remaining, 1)
                continue

            # Read 20-byte header
            hdr = f.read(20)
            if len(hdr) < 20:
                break

            Nrx     = hdr[8]
            Ntx     = hdr[9]
            len_csi = int.from_bytes(hdr[14:16], 'little')

            if Nrx < 1 or Ntx < 1:
                f.seek(len_csi, 1)
                continue

            csi_bytes = f.read(len_csi)
            if len(csi_bytes) < len_csi:
                break

            H = _extract_csi(csi_bytes, Nrx, Ntx, n_sub)   # [n_sub, Ntx, Nrx]
            records.append(H[:, 0, :])                        # Ntx=1 → [n_sub, Nrx]

    if not records:
        raise ValueError(f"No valid CSI records found in {filepath}")

    H_all     = np.stack(records, axis=0).astype(np.complex64)   # [T, n_sub, Nrx]
    T         = H_all.shape[0]
    timestamps = np.arange(T, dtype=np.float64) / XRF55_FS_NOMINAL

    return H_all, timestamps


def _extract_csi(
    csi_bytes: bytes,
    Nrx: int,
    Ntx: int,
    n_sub: int,
) -> np.ndarray:
    """
    Unpack 10-bit packed signed CSI integers from raw bytes.

    Intel 5300 CSI order: for each sub → for each Tx → for each Rx → (imag, real).
    Signed 10-bit: values >= 512 are negative (val - 1024).

    Returns: [n_sub, Ntx, Nrx] complex64
    """
    H = np.zeros((n_sub, Ntx, Nrx), dtype=np.complex64)
    data = np.frombuffer(csi_bytes, dtype=np.uint8).astype(np.int32)

    bits_left = 0
    buf       = 0
    byte_idx  = 0

    def read10() -> int:
        nonlocal bits_left, buf, byte_idx
        while bits_left < 10:
            buf       |= int(data[byte_idx]) << bits_left
            byte_idx  += 1
            bits_left += 8
        val       = buf & 0x3FF
        buf      >>= 10
        bits_left -= 10
        return val - 1024 if val >= 512 else val

    for k in range(n_sub):
        for j in range(Ntx):
            for i in range(Nrx):
                imag = read10()
                real = read10()
                H[k, j, i] = complex(real, imag)

    return H


def normalize_intel5300_to_uniform(
    H_raw   : np.ndarray,
    target_F: Optional[int] = None,
    target_A: int = 4,
) -> np.ndarray:
    """
    Pad/trim Intel 5300 CSI [T, F_in, A_in] to uniform shape for pipeline.

    Args:
        H_raw:    [T, F_in, A_in] complex64  (single receiver site)
        target_F: if given, zero-pad (or trim) subcarriers; None = keep F_in
        target_A: zero-pad (or trim) antenna dim

    Returns: [T, F_out, target_A] complex64
    """
    T, F_in, A_in = H_raw.shape

    # Subcarrier dimension
    if target_F is None or target_F == F_in:
        H_padF = H_raw
    elif F_in < target_F:
        pad = np.zeros((T, target_F - F_in, A_in), dtype=np.complex64)
        H_padF = np.concatenate([H_raw, pad], axis=1)
    else:
        H_padF = H_raw[:, :target_F, :]

    F_out = H_padF.shape[1]

    # Antenna dimension
    if A_in < target_A:
        pad = np.zeros((T, F_out, target_A - A_in), dtype=np.complex64)
        H_padA = np.concatenate([H_padF, pad], axis=2)
    elif A_in > target_A:
        H_padA = H_padF[:, :, :target_A]
    else:
        H_padA = H_padF

    return H_padA
