"""
src/preprocessing/common.py
Shared preprocessing utilities for ALL dataset pipelines.

Contains:
  - Precomputed SOS filter coefficients  (OPT v6.6: no recomputation per window)
  - agc_normalize         B1
  - hampel_detect         B2  (vectorized via scipy.ndimage.median_filter)
  - hampel_repair_complex B2  (vectorized: all F subcarriers in one PCHIP fit)
  - align_rx_timestamps   B3
  - pchip_resample_rx     B4  (vectorized boundary fill via shared NaN pattern)
  - select_ref_antenna    G1
  - bandpass_complex      G4
  - fit_preprocessing_stats  Stats Pass 1 (Welford + Fisher, SINGLE PASS)
  - build_windows         Stage D

All numpy, no torch.  Called from pipeline_*.py files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import binary_dilation, median_filter
from scipy.signal import butter, sosfiltfilt, stft as sp_stft

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Precomputed SOS filter coefficients (module-level constants)
# Computed once at import time — avoids recomputing per window (OPT v6.6 #58)
# ---------------------------------------------------------------------------
_FS     = 350.0
_SOS_LP = butter(4, 30.0,        btype='low',      fs=_FS, output='sos')   # Branch α F2
_SOS_BP = butter(4, [0.5, 80.0], btype='bandpass', fs=_FS, output='sos')   # Branch β G4
# Note: butter(4, bandpass) → 8th-order filter; sosfiltfilt zero-phase → 16th-order attenuation


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Window:
    """Single preprocessed window ready for model input."""
    X_amp          : np.ndarray          # [T=350, F, M=3, A=4] float32
    X_dfs          : np.ndarray          # [T_dfs=28, V=128, M=3]  float32
    label          : int                 # class id 0..7
    recording_id   : str
    person_id      : str
    room_id        : str
    window_idx     : int
    ref_antenna_idx: int                 # 0..3


@dataclass
class Recording:
    """Container for one preprocessed recording (all stages A-B done)."""
    recording_id      : str
    person_id         : str
    room_id           : str
    activity_label    : str
    activity_label_int: int
    H_uniform         : Optional[np.ndarray] = None   # [T_out,F,M,A] complex64
    metadata          : dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# B1 — AGC Normalization
# ---------------------------------------------------------------------------
def agc_normalize(
    H_rx: np.ndarray,
    noise_floor_ratio: float = 1e-3,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-packet, per-antenna AGC normalization.

    H_rx: [T, F, A] complex64  (single RX)
    Returns: H_norm [T,F,A] complex64, gain [T,A] float32
    """
    gain        = np.abs(H_rx).mean(axis=1)                      # [T, A]
    noise_floor = float(gain.max()) * noise_floor_ratio
    gain_safe   = np.where(gain > noise_floor, gain, 1.0)        # [T, A]
    H_norm      = H_rx / (gain_safe[:, np.newaxis, :] + eps)     # [T, F, A]
    return H_norm.astype(np.complex64), gain.astype(np.float32)


# ---------------------------------------------------------------------------
# B2 — Hampel Outlier Detection & PCHIP Cartesian Repair
# ---------------------------------------------------------------------------
def hampel_detect(
    track: np.ndarray,
    window_half: int = 7,
    k: float = 3.0,
) -> np.ndarray:
    """
    Vectorized Hampel identifier on a 1-D real time series.

    Uses scipy.ndimage.median_filter for O(T) complexity instead of O(T×window)
    Python loop. The rolling-MAD approximation uses:
        med(t)      = median_filter(track, size)
        abs_dev(t)  = |track(t) - med(t)|
        mad(t)      ≈ median_filter(abs_dev, size)   ← rolling window MAD approx

    This approximation is standard in signal processing and gives virtually
    identical outlier detection results to the exact rolling Hampel.

    track: [T] real
    Returns: bad_mask [T] bool
    """
    size     = 2 * window_half + 1
    med      = median_filter(track.astype(np.float64), size=size, mode='nearest')
    abs_dev  = np.abs(track - med)
    mad_roll = median_filter(abs_dev, size=size, mode='nearest')
    return abs_dev > k * 1.4826 * mad_roll


def _fill_boundary_nan(
    result      : np.ndarray,
    valid_values: np.ndarray,
    good_idx    : np.ndarray,
    bad_idx     : np.ndarray,
) -> np.ndarray:
    """
    Fill NaN values at boundaries of interpolated array with nearest valid value.
    Uses boolean masks (FIX v6.3) — not packet indices as array indices.

    result:       [len(bad_idx)]  — interpolated, may contain NaN at boundaries
    valid_values: [len(good_idx)] — values at good positions
    good_idx, bad_idx: index arrays into the ORIGINAL track
    """
    if not np.isnan(result).any():
        return result
    before = bad_idx < good_idx[0]
    after  = bad_idx > good_idx[-1]
    if before.any():
        result[before] = valid_values[0]
    if after.any():
        result[after]  = valid_values[-1]
    return result


def hampel_repair_complex(
    H_fma   : np.ndarray,
    bad_idx : np.ndarray,
    good_idx: np.ndarray,
) -> np.ndarray:
    """
    PCHIP Cartesian repair for outlier positions.

    H_fma:    [T, F] complex64 — per-antenna track for all subcarriers
    bad_idx:  indices of bad packets
    good_idx: indices of good packets

    Vectorized: 2 PchipInterpolator calls total (re and im), each fitting
    all F subcarrier columns simultaneously using SciPy's N-D y support.
    PchipInterpolator(x, y) with y=[n_good, F] fits each column independently.
    Boundary fill is also vectorized (NaN pattern identical for all F columns
    because bad_idx/good_idx are shared across subcarriers).
    """
    H_fixed = H_fma.copy()
    if len(good_idx) < 4:
        return H_fixed   # not enough points to interpolate safely

    # Vectorized: fit all F columns at once using N-D y support
    ri  = PchipInterpolator(good_idx, H_fma[good_idx].real, extrapolate=False)
    ii_ = PchipInterpolator(good_idx, H_fma[good_idx].imag, extrapolate=False)
    fre = ri(bad_idx)    # [n_bad, F]
    fim = ii_(bad_idx)   # [n_bad, F]

    # Vectorized boundary fill (extrapolate=False → NaN at out-of-range positions)
    # NaN pattern is identical across all F columns (shared bad_idx/good_idx)
    before = bad_idx < good_idx[0]
    after  = bad_idx > good_idx[-1]
    if before.any():
        fre[before] = H_fma[good_idx[0]].real   # [F] broadcast to [n_before, F]
        fim[before] = H_fma[good_idx[0]].imag
    if after.any():
        fre[after]  = H_fma[good_idx[-1]].real
        fim[after]  = H_fma[good_idx[-1]].imag

    H_fixed[bad_idx] = (fre + 1j * fim).astype(np.complex64)
    return H_fixed


def hampel_repair_rx(
    H_norm  : np.ndarray,
    rec_id  : str,
    rx_idx  : int,
    window_half: int = 7,
    k_factor   : float = 3.0,
) -> Optional[np.ndarray]:
    """
    Run Hampel detection + PCHIP repair for a single RX.

    H_norm: [T, F, A] complex64
    Returns: repaired H_norm or None if too many outliers (recording rejected).
    """
    H_amp_avg = np.abs(H_norm).mean(axis=1)   # [T, A] — envelope per antenna

    for a in range(H_norm.shape[2]):
        track    = H_amp_avg[:, a]
        bad_mask = hampel_detect(track, window_half=window_half, k=k_factor)
        bad_mask = binary_dilation(bad_mask, structure=np.ones(3))  # expand ±1

        bad_idx  = np.where(bad_mask)[0]
        good_idx = np.where(~bad_mask)[0]
        if len(bad_idx) == 0:
            continue

        frac = bad_mask.mean()
        if frac > 0.50:
            logger.error(
                f"[{rec_id}] RX{rx_idx} antenna {a}: {frac*100:.1f}% outliers → reject"
            )
            return None
        if frac > 0.20:
            logger.warning(
                f"[{rec_id}] RX{rx_idx} antenna {a}: {frac*100:.1f}% outliers"
            )

        H_norm[:, :, a] = hampel_repair_complex(H_norm[:, :, a], bad_idx, good_idx)

    return H_norm


# ---------------------------------------------------------------------------
# B3 — Multi-RX Timestamp Alignment
# ---------------------------------------------------------------------------
def align_rx_timestamps(
    H_norm_list    : List[np.ndarray],
    timestamps_list: List[np.ndarray],
    rec_id         : str,
    ref_rx         : int = 0,
    threshold_s    : float = 0.5,
    warn_s         : float = 0.1,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Estimate and apply per-RX clock offsets using cross-correlation of
    amplitude envelopes on top-5 high-variance subcarriers.

    H_norm_list:     list of M arrays [T_rx, F, A] complex64
    timestamps_list: list of M arrays [T_rx] float64 (seconds)
    ref_rx:          index of reference RX (unchanged)

    Returns:
        timestamps_list: updated with shifts applied
        shifts:         [M] float  (shift in seconds applied to each RX)

    FIX v6.6 BUG#54: common time grid computed AFTER this function,
    from ref_rx timestamps. This function only shifts; B4 resamples on grid.
    """
    ref_H   = H_norm_list[ref_rx]
    skip    = min(350, len(ref_H) // 5)   # skip first ~1s (settling transient)

    # Compute amplitude envelope for ref_rx once (reuse: FIX v6.5)
    amp_ref     = np.abs(ref_H[skip:])              # [T', F, A]
    var_per_sub = amp_ref.var(axis=(0, 2))           # [F]
    top5        = np.argsort(var_per_sub)[-5:]       # top-5 highest variance subs
    ref_env     = amp_ref[:, top5, :].mean(axis=(1, 2))  # [T']
    ref_env    -= ref_env.mean()

    shifts = [0.0] * len(H_norm_list)

    for rx in range(len(H_norm_list)):
        if rx == ref_rx:
            continue

        skip_o    = min(skip, len(H_norm_list[rx]) // 5)
        amp_other = np.abs(H_norm_list[rx][skip_o:])           # [T'', F, A]
        other_env = amp_other[:, top5, :].mean(axis=(1, 2))    # [T'']
        other_env -= other_env.mean()

        corr        = np.correlate(other_env, ref_env, mode='full')
        lag_packets = int(corr.argmax()) - (len(ref_env) - 1)
        lag_seconds = lag_packets / _FS

        if abs(lag_seconds) > threshold_s:
            raise ValueError(
                f"[{rec_id}] RX{rx} lag={lag_seconds:.3f}s > {threshold_s}s threshold → REJECTED"
            )
        if abs(lag_seconds) > warn_s:
            logger.warning(f"[{rec_id}] RX{rx} large lag={lag_seconds:.3f}s")

        shifts[rx]             = lag_seconds
        timestamps_list[rx]    = timestamps_list[rx] + lag_seconds

    return timestamps_list, shifts


# ---------------------------------------------------------------------------
# B4 — PCHIP Resampling onto Common Time Grid
# ---------------------------------------------------------------------------
def _fill_boundary_1d(arr: np.ndarray) -> np.ndarray:
    """Fill NaN at start/end of 1-D array with nearest valid value (in-place)."""
    mask = ~np.isnan(arr)
    if not mask.any():
        arr[:] = 0.0
        return arr
    fv = int(np.where(mask)[0][0])
    lv = int(np.where(mask)[0][-1])
    arr[:fv]   = arr[fv]
    arr[lv+1:] = arr[lv]
    return arr


def pchip_resample_rx(
    H_clean_rx  : np.ndarray,
    timestamps_rx: np.ndarray,
    t_common    : np.ndarray,
) -> np.ndarray:
    """
    Resample one RX onto the common time grid via PCHIP.

    H_clean_rx:   [T_raw, F, A] complex64
    timestamps_rx: [T_raw] float64  (shifted by B3)
    t_common:     [T_out] float64   — SINGLE shared grid from ref_rx (FIX v6.6 BUG#54)

    Returns: H_uni_rx [T_out, F, A] complex64

    Vectorized: 2 PchipInterpolator fits per antenna (re/im of all F columns
    simultaneously). Boundary fill is also vectorized: since timestamps_rx is
    shared across all F columns of one antenna, NaN positions are identical for
    all columns — only one boundary scan needed per antenna.
    """
    T_out    = len(t_common)
    F        = H_clean_rx.shape[1]
    A        = H_clean_rx.shape[2]
    H_uni_rx = np.zeros((T_out, F, A), dtype=np.complex64)

    for a in range(A):
        tracks_re = H_clean_rx[:, :, a].real   # [T_raw, F]
        tracks_im = H_clean_rx[:, :, a].imag

        ri  = PchipInterpolator(timestamps_rx, tracks_re, extrapolate=False)
        ii_ = PchipInterpolator(timestamps_rx, tracks_im, extrapolate=False)

        h_re = ri(t_common)    # [T_out, F]
        h_im = ii_(t_common)

        # Vectorized boundary fill:
        # extrapolate=False → NaN only at t_common points outside timestamps_rx range.
        # Since timestamps_rx is shared across all F columns of this antenna,
        # NaN positions are the same for every column — scan column 0 once.
        nan_col = np.isnan(h_re[:, 0])
        if nan_col.any():
            valid_idx = np.where(~nan_col)[0]
            if len(valid_idx) == 0:
                h_re[:] = 0.0
                h_im[:] = 0.0
            else:
                fv, lv = valid_idx[0], valid_idx[-1]
                if fv > 0:
                    h_re[:fv] = h_re[fv]    # [F] broadcast to [fv, F]
                    h_im[:fv] = h_im[fv]
                if lv + 1 < T_out:
                    h_re[lv + 1:] = h_re[lv]
                    h_im[lv + 1:] = h_im[lv]

        H_uni_rx[:, :, a] = (h_re + 1j * h_im).astype(np.complex64)

    return H_uni_rx


def resample_and_stack_receivers(
    H_norm_list    : List[np.ndarray],
    timestamps_list: List[np.ndarray],
    ref_rx         : int = 0,
    fs_out         : float = 350.0,
) -> np.ndarray:
    """
    B4 + COMBINE step.

    1. Compute single common grid from ref_rx timestamps (FIX v6.6 BUG#54)
    2. Resample each RX onto common grid
    3. Stack: [T_out, F, M, A]

    H_norm_list:     list of M arrays [T_rx, F, A] complex64 (after B3 alignment)
    timestamps_list: list of M shifted timestamp arrays [T_rx] float64

    Returns: H_uniform [T_out, F, M, A] complex64
    """
    # Common time grid from reference RX (FIX v6.6)
    ts_ref   = timestamps_list[ref_rx]
    t_common = np.arange(ts_ref[0], ts_ref[-1], 1.0 / fs_out)

    H_uni_list = [
        pchip_resample_rx(H_norm_list[k], timestamps_list[k], t_common)
        for k in range(len(H_norm_list))
    ]
    # Stack on M dimension: list of [T_out, F, A] → [T_out, F, M, A]
    return np.stack(H_uni_list, axis=2)


# ---------------------------------------------------------------------------
# G1 — Reference Antenna Selection
# ---------------------------------------------------------------------------
def select_ref_antenna(
    H_uniform         : np.ndarray,
    noise_floor_ratio : float = 0.10,
) -> int:
    """
    Select the reference antenna for conjugate multiplication (G2).
    Criterion: argmin(temporal variance) among valid antennas [WiDance/DoRF].
    The most stable antenna → C = H_other * conj(H_ref) shows clearest Doppler.

    H_uniform: [T, F, M, A] complex64
    Returns: ref_antenna_idx int 0..A-1 (global for entire recording)
    """
    amp_per_ant = np.abs(H_uniform).mean(axis=(0, 1, 2))   # [A] over T,F,M
    amp_full    = np.abs(H_uniform).mean(axis=(1, 2))       # [T, A] over F,M
    var_per_ant = amp_full.var(axis=0)                      # [A]

    noise_floor    = float(amp_per_ant.max()) * noise_floor_ratio
    valid_antennas = np.where(amp_per_ant > noise_floor)[0]
    if len(valid_antennas) == 0:
        valid_antennas = np.arange(H_uniform.shape[-1])

    return int(valid_antennas[np.argmin(var_per_ant[valid_antennas])])


# ---------------------------------------------------------------------------
# G4 — Bandpass Filter
# ---------------------------------------------------------------------------
def bandpass_complex(C_dyn: np.ndarray) -> np.ndarray:
    """
    Zero-phase bandpass filter [0.5, 80] Hz on complex array.

    Uses precomputed _SOS_BP (butter(4, [0.5,80], bandpass) = 8th-order filter).
    sosfiltfilt NEVER accepts complex → separate re/im always.

    C_dyn: [..., T=350, ...] complex, axis=0 is time
    """
    C_re = sosfiltfilt(_SOS_BP, C_dyn.real, axis=0)
    C_im = sosfiltfilt(_SOS_BP, C_dyn.imag, axis=0)
    return (C_re + 1j * C_im).astype(np.complex64)


# ---------------------------------------------------------------------------
# Stats Pass 1 — Welford Z-score + Fisher Score (SINGLE PASS)
# ---------------------------------------------------------------------------
def _lowpass_stats(X_abs_f64: np.ndarray) -> np.ndarray:
    """Low-pass filter keeping float64 (for Welford precision). Axis=0 = time."""
    return sosfiltfilt(_SOS_LP, X_abs_f64, axis=0)   # float64 preserved


def fit_preprocessing_stats(
    train_recordings: List[Recording],
    fold_id         : int,
    save_path       : str,
    T_window        : int = 350,
    stride          : int = 175,
    n_groups        : int = 13,
) -> dict:
    """
    SINGLE PASS over train windows: simultaneously fit
      1. Welford online Z-score statistics (F4 normalization)
      2. Fisher discriminant subcarrier scores (G5 selection)

    FIX v6.3 BUG#1: Welford avoids ~120GB RAM of np.stack.
    FIX v6.5: float64 throughout; precomputed _SOS_LP.
    FIX v6.5: non_ref precomputed per-recording (not per-window).
    FIX v6.7: F/M/A inferred from data (no hardcoded 52); Fisher groups via
              np.array_split → unequal sizes when F % n_groups != 0 (e.g. XRF55 F=30).

    Returns: stats dict saved to save_path.npz
    """
    if not train_recordings:
        logger.error("fit_preprocessing_stats: no training recordings")
        return {}

    # ── Infer dimensions from first recording ───────────────────────────
    _ref  = train_recordings[0].H_uniform   # [T_total, F, M, A]
    F     = int(_ref.shape[1])
    M     = int(_ref.shape[2])
    A     = int(_ref.shape[3])

    # ── Welford accumulators (float64) ──────────────────────────────────
    n_total  = 0
    mean_acc = np.zeros((F, M, A), dtype=np.float64)
    M2_acc   = np.zeros((F, M, A), dtype=np.float64)

    # ── Fisher accumulators ─────────────────────────────────────────────
    x_features: List[np.ndarray] = []
    y_labels  : List[int]        = []

    total_wins = 0

    for rec in train_recordings:
        ref_idx = rec.metadata['ref_antenna_idx']
        # FIX v6.5: precompute non_ref once per recording
        non_ref = [a for a in range(A) if a != ref_idx]

        windows = build_windows(rec, T=T_window, stride=stride)

        for win_H in windows:   # win_H: [T=350, F, M, A] complex64
            total_wins += 1

            # ── WELFORD: F1-F3 in float64 ────────────────────────────────
            X_abs = np.abs(win_H).astype(np.float64)
            X_lp  = _lowpass_stats(X_abs)                      # float64
            X_cen = X_lp - np.median(X_lp, axis=0, keepdims=True)

            m        = T_window                                 # 350
            b_mean   = X_cen.mean(axis=0)                      # [F, M, A]
            b_M2     = X_cen.var(axis=0) * m
            combined = n_total + m
            delta    = b_mean - mean_acc
            mean_acc = mean_acc + delta * (m / combined)
            M2_acc   = M2_acc + b_M2 + delta**2 * (n_total * m / combined)
            n_total  = combined

            # ── FISHER: G2-G4 ─────────────────────────────────────────────
            H_ref   = win_H[:, :, :, ref_idx]                   # [350, F, M]
            H_other = win_H[:, :, :, non_ref]                   # [350, F, M, n_non_ref]
            C       = H_other * np.conj(H_ref[:, :, :, None])
            C_dyn   = C - C.mean(axis=0, keepdims=True)
            C_bp    = bandpass_complex(C_dyn)                   # [350, F, M, n_non_ref]
            # x(w,f) = mean_{m,p}(Var_t(|C_bp[:,f,m,p]|))
            x_wf    = np.abs(C_bp).var(axis=0).mean(axis=(1, 2))  # [F]

            x_features.append(x_wf)
            y_labels.append(rec.activity_label_int)

    # ── Finalize Welford ─────────────────────────────────────────────────
    zscore_mu    = mean_acc.astype(np.float32)
    zscore_sigma = (np.sqrt(M2_acc / n_total) + 1e-8).astype(np.float32)

    # ── Finalize Fisher ──────────────────────────────────────────────────
    x_mat     = np.stack(x_features)   # [N_wins, F]
    y_arr     = np.array(y_labels)
    mu_global = x_mat.mean(axis=0)     # [F]

    between_var = np.zeros(F, dtype=np.float64)
    within_var  = np.zeros(F, dtype=np.float64)
    for c in np.unique(y_arr):
        mask = (y_arr == c)
        n_c  = int(mask.sum())
        x_c  = x_mat[mask]
        mu_c = x_c.mean(axis=0)
        between_var += n_c * (mu_c - mu_global) ** 2
        within_var  += n_c * x_c.var(axis=0)    # ddof=0 (population variance)

    fisher_scores = between_var / (within_var + 1e-12)   # [F]

    # Divide F subcarriers into n_groups using array_split.
    # Unequal group sizes when F % n_groups != 0 (e.g. F=30, n_groups=13
    # → 4 groups of 3 + 9 groups of 2). argmax Fisher within each group.
    sub_groups = np.array_split(np.arange(F), n_groups)
    selected   = np.array(
        [int(grp[np.argmax(fisher_scores[grp])]) for grp in sub_groups],
        dtype=np.int32,
    )

    stats = {
        'zscore_mu'            : zscore_mu,
        'zscore_sigma'         : zscore_sigma,
        'fisher_scores'        : fisher_scores.astype(np.float32),
        'selected_subcarriers' : selected,
        'fold_id'              : fold_id,
        'n_timesteps'          : n_total,
        'n_windows'            : total_wins,
    }

    np.savez(save_path, **stats)
    logger.info(
        f"[Fold {fold_id}] Stats fit: {n_total:,} timesteps / {total_wins:,} windows"
        f" | F={F}, groups={[len(g) for g in sub_groups]}"
    )
    logger.info(f"[Fold {fold_id}] Fisher-selected: {selected.tolist()}")
    return stats


# ---------------------------------------------------------------------------
# Stage D — Sliding Window
# ---------------------------------------------------------------------------
def build_windows(
    rec    : Recording,
    T      : int = 350,
    stride : int = 175,
) -> List[np.ndarray]:
    """
    Slide a window of size T with given stride over rec.H_uniform.
    Skip windows containing NaN.

    Returns: list of [T, F, M, A] complex64 arrays (raw CSI windows)
    """
    H = rec.H_uniform   # [T_total, F, M, A]
    windows = []
    start   = 0
    while start + T <= H.shape[0]:
        win_H = H[start:start + T]
        if not np.any(np.isnan(win_H)):
            windows.append(win_H)
        start += stride
    return windows


# ---------------------------------------------------------------------------
# Branch α — Amplitude Pipeline (per window, inference time)
# ---------------------------------------------------------------------------
def process_amp_branch(
    win_H : np.ndarray,
    stats : dict,
) -> np.ndarray:
    """
    F1-F4 amplitude feature extraction.

    win_H: [T=350, F, M=3, A=4] complex64
    stats: output of fit_preprocessing_stats (zscore_mu, zscore_sigma)

    Returns: X_amp [T=350, F, M=3, A=4] float32
    """
    # F1: Magnitude (float32 for inference)
    X_abs = np.abs(win_H).astype(np.float32)

    # F2: Low-pass 30Hz (precomputed _SOS_LP)
    X_lp = sosfiltfilt(_SOS_LP, X_abs, axis=0).astype(np.float32)

    # F3: Per-window static removal (no leakage)
    X_centered = X_lp - np.median(X_lp, axis=0, keepdims=True)

    # F4: Z-score normalization (apply fixed Welford stats)
    X_norm = (X_centered - stats['zscore_mu']) / stats['zscore_sigma']

    return X_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# Branch β — DFS Pipeline (per window, inference time)
# ---------------------------------------------------------------------------
def process_dfs_branch(
    win_H  : np.ndarray,
    stats  : dict,
    ref_idx: int,
) -> np.ndarray:
    """
    G2-G7 DFS feature extraction.

    win_H:   [T=350, F, M=3, A=4] complex64
    stats:   output of fit_preprocessing_stats (selected_subcarriers)
    ref_idx: reference antenna index for conjugate multiplication

    Returns: X_dfs [T_dfs=28, V=128, M=3] float32

    STFT parameters:
        nperseg=128, hop=8 → T_dfs = (350-128)//8 + 1 = 28 ✓
        nfft=128, two-sided → V=128 bins covering ±175 Hz
        Hann window, no padding (boundary=None, padded=False)

    sp_stft imported at module level (avoids repeated import inside hot loop).
    """
    non_ref = [a for a in range(win_H.shape[-1]) if a != ref_idx]

    # G2: Conjugate multiplication — cancels common phase offset
    H_ref   = win_H[:, :, :, ref_idx]                    # [350, 52, M]
    H_other = win_H[:, :, :, non_ref]                    # [350, 52, M, 3]
    C       = H_other * np.conj(H_ref[:, :, :, None])    # [350, 52, M, 3]

    # G3: Static path removal (per-window mean)
    C_dyn = C - C.mean(axis=0, keepdims=True)

    # G4: Bandpass [0.5, 80] Hz (precomputed _SOS_BP)
    C_bp = bandpass_complex(C_dyn)                        # [350, 52, M, 3]

    # G5: Fisher subcarrier selection
    sel      = stats['selected_subcarriers']              # [13]
    C_bp_sel = C_bp[:, sel, :, :]                         # [350, 13, M, 3]
    C_avg    = C_bp_sel.mean(axis=3)                      # [350, 13, M]

    M_rx = C_avg.shape[2]
    spectrograms = []

    for m in range(M_rx):
        # G6: STFT per subcarrier, incoherent power average
        P_subs = []
        for s in range(len(sel)):
            _, _, Zxx = sp_stft(
                C_avg[:, s, m],
                fs=350, window='hann', nperseg=128,
                noverlap=120, nfft=128,
                return_onesided=False,
                boundary=None, padded=False,
            )                                             # [128, 28]
            P_subs.append(np.abs(Zxx) ** 2)

        P_m       = np.stack(P_subs).mean(axis=0)        # [128, 28]
        P_shifted = np.fft.fftshift(P_m, axes=0)         # DC → center
        spectrograms.append(P_shifted.T)                  # [28, 128]

    X_stft = np.stack(spectrograms, axis=-1)              # [28, 128, M]

    # G7: Log power + per-window robust normalization
    S = 10.0 * np.log10(X_stft + 1e-10).astype(np.float32)   # [28, 128, M]
    for rx in range(M_rx):
        s     = S[:, :, rx]
        med   = float(np.median(s))
        scale = float((np.percentile(s, 95) - np.percentile(s, 5)) / 2.0) + 1e-8
        S[:, :, rx] = (s - med) / scale

    return S   # [28, 128, M] float32


# ---------------------------------------------------------------------------
# Augmentation utilities
# ---------------------------------------------------------------------------
def augment_joint(
    H_window        : np.ndarray,
    ref_antenna_idx : int,
) -> np.ndarray:
    """
    Joint augmentation on raw complex CSI window (Stages E1-E5).
    Applied TRAIN ONLY, on-the-fly.

    H_window: [T=350, F, M=3, A=4] complex64
    Returns:  augmented H, same shape
    """
    H = H_window.copy()

    # E1: Temporal shift ±30 packets (edge padding)
    if np.random.rand() < 0.5:
        sh = int(np.random.randint(-30, 31))
        if sh > 0:
            H = np.concatenate([H[:1].repeat(sh, axis=0), H[:-sh]], axis=0)
        elif sh < 0:
            H = np.concatenate([H[-sh:], H[-1:].repeat(-sh, axis=0)], axis=0)

    # E2: Amplitude scale (p=0.5)
    if np.random.rand() < 0.5:
        H = H * np.random.uniform(0.6, 1.4)

    # E3: Antenna dropout (p=0.2) — never drop reference antenna
    if np.random.rand() < 0.2:
        non_ref  = [a for a in range(H.shape[-1]) if a != ref_antenna_idx]
        drop_ant = int(np.random.choice(non_ref))
        noise    = np.random.randn(*H[:, :, :, drop_ant].shape) + \
                   1j * np.random.randn(*H[:, :, :, drop_ant].shape)
        H[:, :, :, drop_ant] = (noise * 0.01).astype(np.complex64)

    # E4: Receiver dropout (p=0.10)
    if np.random.rand() < 0.10:
        drop_rx = int(np.random.randint(0, H.shape[2]))
        noise   = np.random.randn(*H[:, :, drop_rx, :].shape) + \
                  1j * np.random.randn(*H[:, :, drop_rx, :].shape)
        H[:, :, drop_rx, :] = (noise * 0.01).astype(np.complex64)

    # E5: Complex Gaussian noise (p=0.3)
    if np.random.rand() < 0.3:
        H = H + (0.01 * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
                 ).astype(np.complex64)

    return H


def augment_amp(
    X_amp       : np.ndarray,
    X_amp_j     : Optional[np.ndarray],
    global_epoch: int,
    cfg         : dict,
) -> np.ndarray:
    """
    Branch α augmentation (H1a subcarrier masking, H1b CutMix-Time).
    Applied after F4.

    X_amp:   [T=350, F, M=3, A=4] float32
    X_amp_j: paired sample for CutMix, same shape (or None)
    """
    # H1a: Subcarrier masking
    p_mask     = cfg.get('subcarrier_mask_prob', 0.4)
    mask_min   = cfg.get('subcarrier_mask_min', 3)
    mask_max   = cfg.get('subcarrier_mask_max', 9)
    if np.random.rand() < p_mask:
        n_mask  = int(np.random.randint(mask_min, mask_max + 1))
        f_start = int(np.random.randint(0, X_amp.shape[1] - n_mask))
        X_amp = X_amp.copy()
        X_amp[:, f_start:f_start + n_mask, :, :] = 0.0

    # H1b: CutMix-Time (epoch >= cutmix_start, prob=cutmix_prob)
    cutmix_start = cfg.get('cutmix_start_epoch', 8)
    cutmix_prob  = cfg.get('cutmix_prob', 0.3)
    cutmix_ratio = cfg.get('cutmix_max_ratio', 0.2)
    if (global_epoch >= cutmix_start
            and X_amp_j is not None
            and np.random.rand() < cutmix_prob):
        T       = X_amp.shape[0]
        cut_len = int(np.random.randint(1, int(T * cutmix_ratio) + 1))
        t1      = int(np.random.randint(0, T - cut_len))
        X_amp   = X_amp.copy()
        X_amp[t1:t1 + cut_len] = X_amp_j[t1:t1 + cut_len]

    return X_amp


def augment_dfs(X_dfs: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Branch β on-the-fly augmentation (H2a Doppler masking, H2b time masking).
    Kept for backwards-compat; offline equivalent is augment_features_offline.
    fill=0.0 = background level after G7 median-centering.
    """
    # H2a: Doppler-bin masking
    if np.random.rand() < cfg.get('doppler_mask_prob', 0.3):
        n_m = int(np.random.randint(2, 7))
        v_s = int(np.random.randint(0, X_dfs.shape[1] - n_m))
        X_dfs = X_dfs.copy()
        X_dfs[:, v_s:v_s + n_m, :] = 0.0

    # H2b: Time-frame masking
    if np.random.rand() < cfg.get('time_mask_prob', 0.3):
        t_m = int(np.random.randint(1, 4))
        t_s = int(np.random.randint(0, X_dfs.shape[0] - t_m))
        X_dfs = X_dfs.copy()
        X_dfs[t_s:t_s + t_m, :, :] = 0.0

    return X_dfs


# ---------------------------------------------------------------------------
# Offline augmentation — deterministic, reproducible, stored in HDF5
#
# Design (verified against literature):
#   • Seeded np.random.RandomState → reproducible across runs / machines
#   • Physics-motivated E-stage ordering:
#       E1 temporal shift → E6 phase jitter → E2 amplitude scale
#       → E3 antenna dropout → E4 receiver dropout → E5 additive noise
#   • SpecAugment-motivated H-stage ordering:
#       H1a subcarrier (freq) mask → H2a Doppler (freq) mask → H2b time mask
#   • Probabilities raised vs. on-the-fly; diversity via parameter sampling
#   • E-stage and H-stage use different prime multipliers → independent RNG draws
# ---------------------------------------------------------------------------

def _window_seed(recording_id: str, window_idx: int) -> int:
    """
    Deterministic 31-bit seed for one (recording, window) pair.
    SHA-256 avoids collisions across large datasets.
    """
    import hashlib
    key = f"{recording_id}_{window_idx}".encode('utf-8')
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % (2 ** 31 - 1)


def augment_joint_offline(
    H_window       : np.ndarray,
    ref_antenna_idx: int,
    copy_id        : int,
    base_seed      : int,
) -> np.ndarray:
    """
    E-stage offline augmentation on raw complex CSI window.

    Ordering (physics-motivated):
      E1  Temporal shift   — geometry first; edge-pad preserves shape [T,F,M,A]
      E6  Phase jitter     — per-antenna oscillator offset (spectrally flat)
      E2  Amplitude scale  — amplifier gain fluctuation (after phase)
      E3  Antenna dropout  — hardware failure (never drops ref antenna)
      E4  Receiver dropout — receiver failure
      E5  Gaussian noise   — additive measurement noise last

    H_window : [T=350, F, M=3, A=4] complex64
    copy_id  : 1-based (1..N_aug)
    base_seed: from _window_seed(recording_id, window_idx)
    Returns  : same shape, complex64
    """
    rng = np.random.RandomState((base_seed * 7 + copy_id * 1009) % (2 ** 31 - 1))
    H   = H_window.copy()
    T, F, M, A = H.shape

    # E1: Temporal shift ±30 packets (p=0.90) — first, preserves temporal structure
    if rng.rand() < 0.90:
        sh = int(rng.randint(-30, 31))
        if sh > 0:
            H = np.concatenate([H[:1].repeat(sh, axis=0), H[:-sh]], axis=0)
        elif sh < 0:
            H = np.concatenate([H[-sh:], H[-1:].repeat(-sh, axis=0)], axis=0)

    # E6 (NEW): Per-antenna phase jitter (p=0.70)
    # Physical basis: oscillator drift is spectrally flat → same θ for all F subs,
    # independent across antennas. Range ±π/4 keeps conjugate-mult coherent.
    if rng.rand() < 0.70:
        for a in range(A):
            theta = rng.uniform(-np.pi / 4.0, np.pi / 4.0)
            H[:, :, :, a] = (H[:, :, :, a] * np.exp(1j * theta)).astype(np.complex64)

    # E2: Amplitude scale (p=0.90) — conservative [0.75,1.25] for transformer
    if rng.rand() < 0.90:
        H = H * np.float32(rng.uniform(0.75, 1.25))

    # E3: Antenna dropout (p=0.40) — never drop reference antenna
    if rng.rand() < 0.40:
        non_ref  = [a for a in range(A) if a != ref_antenna_idx]
        drop_ant = int(rng.choice(non_ref))
        noise    = (rng.randn(T, F, M) + 1j * rng.randn(T, F, M))
        H[:, :, :, drop_ant] = (noise * 0.01).astype(np.complex64)

    # E4: Receiver dropout (p=0.15)
    if rng.rand() < 0.15:
        drop_rx = int(rng.randint(0, M))
        noise   = (rng.randn(T, F, A) + 1j * rng.randn(T, F, A))
        H[:, :, drop_rx, :] = (noise * 0.01).astype(np.complex64)

    # E5: Complex Gaussian noise (p=0.70) — additive, applied last
    # Adaptive sigma [0.005, 0.02] for richer diversity across copies.
    if rng.rand() < 0.70:
        sigma = rng.uniform(0.005, 0.02)
        H = H + (sigma * (rng.randn(T, F, M, A) +
                          1j * rng.randn(T, F, M, A))).astype(np.complex64)

    return H.astype(np.complex64)


def augment_features_offline(
    X_amp    : np.ndarray,
    X_dfs    : np.ndarray,
    copy_id  : int,
    base_seed: int,
    cfg      : dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    H-stage offline augmentation on extracted features.

    Ordering (SpecAugment-motivated: frequency masking > time masking):
      H1a  Subcarrier masking  — freq mask on amplitude branch
      H2a  Doppler masking     — freq mask on DFS branch (freq before time)
      H2b  Time-frame masking  — time mask on DFS branch

    fill=0.0 is correct: G7 median-centers → 0 ≈ background.

    X_amp : [T=350, F, M=3, A=4] float32
    X_dfs : [T_dfs=28, V=128, M=3] float32
    cfg   : training section of base.yaml
    Returns: (X_amp_aug, X_dfs_aug), same shapes, float32
    """
    # Different prime multipliers from E-stage → statistically independent draws
    rng = np.random.RandomState(
        (base_seed * 13 + copy_id * 997 + 31337) % (2 ** 31 - 1)
    )
    F    = X_amp.shape[1]
    V    = X_dfs.shape[1]
    Tdfs = X_dfs.shape[0]

    # H1a: Subcarrier masking — frequency masking first (SpecAugment ordering)
    if rng.rand() < cfg.get('subcarrier_mask_prob_offline', 0.80):
        mask_min = cfg.get('subcarrier_mask_min', 3)
        mask_max = cfg.get('subcarrier_mask_max', 9)
        n_mask   = int(rng.randint(mask_min, mask_max + 1))
        f_start  = int(rng.randint(0, max(1, F - n_mask)))
        X_amp    = X_amp.copy()
        X_amp[:, f_start:f_start + n_mask, :, :] = 0.0

    # H2a: Doppler-bin masking — frequency masking before time masking
    if rng.rand() < cfg.get('doppler_mask_prob_offline', 0.60):
        n_m = int(rng.randint(2, min(8, V)))
        v_s = int(rng.randint(0, max(1, V - n_m)))
        X_dfs = X_dfs.copy()
        X_dfs[:, v_s:v_s + n_m, :] = 0.0

    # H2b: Time-frame masking — time masking second
    if rng.rand() < cfg.get('time_mask_prob_offline', 0.60):
        t_m = int(rng.randint(1, min(5, Tdfs)))
        t_s = int(rng.randint(0, max(1, Tdfs - t_m)))
        if not X_dfs.flags['OWNDATA']:
            X_dfs = X_dfs.copy()
        X_dfs[t_s:t_s + t_m, :, :] = 0.0

    return X_amp, X_dfs
