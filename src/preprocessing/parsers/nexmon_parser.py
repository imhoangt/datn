"""
src/preprocessing/parsers/nexmon_parser.py
Parse Nexmon CSI .pcap files from ASUS RT-AC86U (BCM4366c0).

BCM4366c0 format [Nexmon CSI GitHub]:
  - Custom floating-point: 1-bit sign + 9-bit mantissa + 6-bit exponent
    per real/imag component. Total 4 bytes/complex.
  - NOT int16 (that is BCM4339/BCM43455c0)
  - Libraries (csiread, CSIKit) handle unpacking automatically → complex64

Output per file: H_raw [T_raw, 64, 4] complex64, timestamps [T_raw] float64

Notes:
  - 64 FFT bins for 20MHz; subcarrier selection done in pipeline_own.py (A3)
  - A=4: 4 receive cores/antennas
  - Timestamps are derived from pcap packet timestamps (seconds since epoch)
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _unpack_float_bcm4366(data: bytes, n_complex: int = 64) -> np.ndarray:
    """
    Unpack BCM4366c0 custom floating-point format (vectorized via numpy).

    Each complex value = 4 bytes (little-endian uint32):
        bits [31]    : sign of real
        bits [30:22] : 9-bit mantissa of real (unsigned)
        bits [21:16] : 6-bit exponent of real
        bits [15]    : sign of imag
        bits [14:6]  : 9-bit mantissa of imag (unsigned)
        bits [5:0]   : 6-bit exponent of imag

    When sign=1: value = mantissa - 512  (two's complement over 9+1 bits)
    Scaled value: (mantissa_signed) * 2^(exponent - 31)

    Reference: seemoo-lab/nexmon_csi unpack_float.c
    Vectorized with numpy frombuffer + bitwise ops: ~50× faster than Python loop.
    """
    raw = np.frombuffer(data, dtype='<u4', count=n_complex)  # [n_complex] uint32

    # ── Real part ──────────────────────────────────────────────────────────
    e_re = (raw >> 16) & 0x3F          # 6-bit exponent  [0, 63]
    m_re = ((raw >> 22) & 0x1FF).astype(np.int32)  # 9-bit mantissa [0, 511]
    s_re = (raw >> 31) & 0x1           # sign bit
    # Two's complement: if sign=1 → mantissa = mantissa - 512
    m_re = np.where(s_re.astype(bool), m_re - 512, m_re).astype(np.float32)
    re   = m_re * np.exp2(e_re.astype(np.float32) - 31.0)

    # ── Imaginary part ─────────────────────────────────────────────────────
    e_im = raw & 0x3F                  # 6-bit exponent
    m_im = ((raw >> 6) & 0x1FF).astype(np.int32)   # 9-bit mantissa
    s_im = (raw >> 15) & 0x1          # sign bit
    m_im = np.where(s_im.astype(bool), m_im - 512, m_im).astype(np.float32)
    im   = m_im * np.exp2(e_im.astype(np.float32) - 31.0)

    return (re + 1j * im).astype(np.complex64)


def parse_nexmon_pcap(
    pcap_path: str | Path,
    src_mac  : Optional[str] = None,
    bw_mhz   : int = 20,
    n_sub    : int = 64,
    n_ant    : int = 4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parse a Nexmon CSI .pcap file.

    Args:
        pcap_path: path to .pcap file
        src_mac:   filter by source MAC (AP MAC). None = accept all.
        bw_mhz:    expected bandwidth (20/40/80 MHz)
        n_sub:     number of FFT subcarriers (64 for 20MHz)
        n_ant:     number of antennas (4 for BCM4366c0)

    Returns:
        H_raw      [T_raw, n_sub, n_ant] complex64  (None if parse fails)
        timestamps [T_raw] float64 seconds           (None if parse fails)

    Nexmon CSI UDP payload structure:
        4 bytes: magic 0x11111111 (or 2 bytes 0x1111 in newer firmware)
        6 bytes: source MAC
        2 bytes: sequence number
        2 bytes: core and spatial stream (core=bits[2:0], ss=bits[5:3])
        2 bytes: chanspec
        2 bytes: chip version
        n_sub * 4 bytes: CSI data (BCM4366c0 float format, per-core)
    """
    import dpkt

    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        logger.error(f"File not found: {pcap_path}")
        return None, None

    csi_list  : list = []
    ts_list   : list = []
    seen_cores: set  = set()

    try:
        with open(pcap_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                try:
                    # Parse Ethernet + IP + UDP layers
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    ip = eth.data
                    if not isinstance(ip.data, dpkt.udp.UDP):
                        continue
                    payload = ip.data.data

                    # Detect magic prefix
                    if len(payload) < 4:
                        continue
                    if payload[:4] == b'\x11\x11\x11\x11':
                        hdr_offset = 4
                    elif payload[:2] == b'\x11\x11':
                        hdr_offset = 2
                    else:
                        continue

                    # Parse header
                    if len(payload) < hdr_offset + 14:
                        continue

                    mac_bytes = payload[hdr_offset:hdr_offset + 6]
                    if src_mac is not None:
                        mac_str = ':'.join(f'{b:02x}' for b in mac_bytes)
                        if mac_str.lower() != src_mac.lower():
                            continue

                    core_ss  = struct.unpack_from('<H', payload, hdr_offset + 8)[0]
                    core     = core_ss & 0x7

                    # CSI data starts after header (hdr_offset + 14 bytes)
                    data_offset    = hdr_offset + 14
                    expected_bytes = n_sub * 4   # 4 bytes per complex
                    if len(payload) < data_offset + expected_bytes:
                        continue

                    csi_bytes = payload[data_offset:data_offset + expected_bytes]
                    csi_vec   = _unpack_float_bcm4366(csi_bytes, n_complex=n_sub)

                    seen_cores.add(core)
                    csi_list.append((ts, core, csi_vec))

                except (struct.error, AttributeError):
                    continue

    except Exception as e:
        logger.error(f"Failed to parse {pcap_path}: {e}")
        return None, None

    if not csi_list:
        logger.warning(f"No valid CSI packets in {pcap_path}")
        return None, None

    # Sort by timestamp to handle out-of-order packets
    csi_list.sort(key=lambda x: x[0])

    # Group by timestamp: collect all n_ant antenna readings for same packet.
    # BCM4366c0: 4 cores transmit sequentially for same Wi-Fi frame.
    # Group packets within MERGE_THRESH_S as one "sample".
    MERGE_THRESH_S = 0.001  # 1ms
    groups: list = []
    current_group: dict = {}
    current_ts: float = -1.0
    partial_count = 0

    for ts, core, csi_vec in csi_list:
        if current_ts < 0 or abs(ts - current_ts) > MERGE_THRESH_S:
            if current_group:
                if len(current_group) == n_ant:
                    groups.append((current_ts, current_group.copy()))
                else:
                    partial_count += 1
            current_group = {core: csi_vec}
            current_ts    = float(ts)
        else:
            current_group[core] = csi_vec
    # Flush last group
    if current_group:
        if len(current_group) == n_ant:
            groups.append((current_ts, current_group.copy()))
        else:
            partial_count += 1

    if partial_count > 0:
        logger.warning(
            f"{pcap_path.name}: {partial_count} partial groups (missing cores) discarded"
        )

    if not groups:
        # Fallback: treat each packet as independent (single antenna mode)
        logger.warning(f"{pcap_path}: grouping failed, using per-packet CSI")
        all_ts  = np.array([x[0] for x in csi_list], dtype=np.float64)
        all_csi = np.zeros((len(csi_list), n_sub, n_ant), dtype=np.complex64)
        for i, (ts, core, vec) in enumerate(csi_list):
            ant_idx = core % n_ant
            all_csi[i, :, ant_idx] = vec
        return all_csi, all_ts

    T_raw      = len(groups)
    H_raw      = np.zeros((T_raw, n_sub, n_ant), dtype=np.complex64)
    timestamps = np.zeros(T_raw, dtype=np.float64)

    for i, (ts, group) in enumerate(groups):
        timestamps[i] = ts
        for core, vec in group.items():
            ant_idx = core % n_ant
            H_raw[i, :, ant_idx] = vec

    logger.info(
        f"Parsed {pcap_path.name}: T={T_raw}, cores seen={sorted(seen_cores)}"
    )
    return H_raw, timestamps


def validate_and_clean_packets(
    H_raw     : np.ndarray,
    timestamps: np.ndarray,
    duration_s: float,
    fs_nominal: float = 350.0,
    min_packets: int = 300,
    max_loss   : float = 0.15,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    A2 packet validation and cleaning.

    Steps:
        1. Remove NaN/Inf packets
        2. Remove duplicate timestamps (keep first)
        3. Sort ascending by timestamp
        4. Check minimum packet count
        5. Check packet loss rate

    Returns: (H_clean, ts_clean) or (None, None) if reject
    """
    T = H_raw.shape[0]
    if T == 0:
        return None, None

    # 1. Remove NaN/Inf
    valid_mask = np.all(np.isfinite(H_raw.real), axis=(1, 2)) & \
                 np.all(np.isfinite(H_raw.imag), axis=(1, 2))
    H_raw      = H_raw[valid_mask]
    timestamps = timestamps[valid_mask]

    # 2. Remove duplicate timestamps (keep first occurrence)
    _, unique_idx = np.unique(timestamps, return_index=True)
    H_raw      = H_raw[unique_idx]
    timestamps = timestamps[unique_idx]

    # 3. Sort ascending
    sort_idx   = np.argsort(timestamps)
    H_raw      = H_raw[sort_idx]
    timestamps = timestamps[sort_idx]

    T_remaining = len(timestamps)

    # 4. Minimum packet check
    if T_remaining < min_packets:
        logger.warning(f"Too few packets: {T_remaining} < {min_packets}")
        return None, None

    # 5. Packet loss rate
    expected  = duration_s * fs_nominal
    loss_rate = 1.0 - T_remaining / expected
    if loss_rate > max_loss:
        logger.warning(f"Packet loss {loss_rate*100:.1f}% > {max_loss*100:.0f}% → reject")
        return None, None

    return H_raw.astype(np.complex64), timestamps.astype(np.float64)


def select_subcarriers_bcm4366c0(
    H_raw: np.ndarray,
) -> np.ndarray:
    """
    A3: Select 52 occupied subcarriers from 64-bin FFT output.

    BCM4366c0 / 802.11n/ac / 20MHz (numpy order):
        Bin 0         = DC                   → EXCLUDE
        Bins 1..26    = +1..+26 (26 bins)   → KEEP
        Bins 27..37   = edge + guard + Nyquist → EXCLUDE
        Bins 38..63   = -26..-1  (26 bins)  → KEEP

    Pilot subcarriers (±7, ±21) are KEPT for sensing (≠ data decoding).

    H_raw: [T, 64, A] complex64
    Returns: H_sub [T, 52, A] complex64
    """
    POS_BINS = slice(1, 27)    # +1..+26
    NEG_BINS = slice(38, 64)   # -26..-1
    return np.concatenate([H_raw[:, POS_BINS], H_raw[:, NEG_BINS]], axis=1)


def detect_ipi_gaps(
    timestamps    : np.ndarray,
    duration_s    : float,
    fs_nominal    : float = 350.0,
    large_gap_ms  : float = 100.0,
    medium_gap_ms : float = 30.0,
    small_factor  : float = 3.0,
    min_seg_len   : int   = 350,
    max_loss      : float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A4: Detect inter-packet interval gaps.

    Returns:
        large_gap:    [T-1] bool — gap > 100ms (should split recording)
        gap_flag_mask:[T-1] bool — gap warranting PCHIP repair (small/medium)
    """
    IPI        = np.diff(timestamps)
    median_IPI = float(np.median(IPI))

    small_gap  = IPI > small_factor * median_IPI
    medium_gap = IPI > medium_gap_ms * 1e-3
    large_gap  = IPI > large_gap_ms  * 1e-3

    gap_flag_mask = small_gap | medium_gap

    return large_gap, gap_flag_mask
