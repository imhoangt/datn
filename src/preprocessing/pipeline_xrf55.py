"""
src/preprocessing/pipeline_xrf55.py
Preprocessing pipeline adapter for XRF55 dataset (Intel 5300, 30 subs, ~200Hz).

Key differences from pipeline_own.py:
- Parser: load from .npy/.mat instead of .pcap
- Subcarriers: 30 → zero-pad to 52
- Antennas: 3 → zero-pad to 4
- Sampling rate: ~200Hz → 350Hz via PCHIP resampling
- Single device (no multi-RX alignment needed for pre-aligned data)
- Class mapping: 55 XRF55 classes → 8 target classes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from .common import (
    Recording,
    agc_normalize,
    hampel_repair_rx,
    pchip_resample_rx,
    select_ref_antenna,
    fit_preprocessing_stats,
    build_windows,
    process_amp_branch,
    process_dfs_branch,
    augment_joint,
    augment_dfs,
    _FS,
)
from .parsers.intel5300_parser import (
    parse_xrf55_csi,
    normalize_intel5300_to_uniform,
    XRF55_FS_NOMINAL,
)

logger = logging.getLogger(__name__)


# XRF55 activity class names (subset for our 8-class mapping)
XRF55_CLASS_MAP = {
    # XRF55 name (lowercase) → our label int
    'walking'       : 0,
    'walk'          : 0,
    'running'       : 1,
    'run'           : 1,
    'standing up'   : 2,
    'standup'       : 2,
    'stand_up'      : 2,
    'sitting down'  : 3,
    'sitdown'       : 3,
    'sit_down'      : 3,
    'picking up'    : 4,
    'pickup'        : 4,
    'pick_up'       : 4,
    'falling down'  : 5,
    'fall'          : 5,
    'falling'       : 5,
    'fallen'        : 5,
    'lying down'    : 6,
    'lying'         : 6,
    'lying_still'   : 6,
    'sleeping'      : 6,
    'no person'     : 7,
    'empty'         : 7,
    'background'    : 7,
}


def _map_xrf55_class(class_name: str) -> Optional[int]:
    """Map XRF55 activity name to our 8-class label."""
    name_lower = class_name.lower().strip()
    if name_lower in XRF55_CLASS_MAP:
        return XRF55_CLASS_MAP[name_lower]
    # Fuzzy match
    for key, val in XRF55_CLASS_MAP.items():
        if key in name_lower or name_lower in key:
            return val
    return None


def preprocess_xrf55_recording(
    file_path : str,
    label_int : int,
    rec_id    : str,
    cfg_pre   : dict,
) -> Optional[np.ndarray]:
    """
    Process one XRF55 recording file to H_uniform [T_out, F=30, M=3, A=4].

    XRF55 has pre-synchronized multi-receiver data in (T × M × A × F) format.
    We apply: load → normalize antennas (3→4, NO sub padding) → AGC → Hampel
              → resample 200→350Hz.
    No B3 alignment needed (XRF55 devices are synchronized).
    Subcarriers kept at F=30 (native Intel 5300); model built with F_sub=30.
    """
    H_raw, timestamps = parse_xrf55_csi(file_path)
    if H_raw is None:
        return None

    # H_raw: [T, F=30, M=3, A=3] complex64
    # Normalize to [T, F=30, M=3, A=4] — keep 30 subs, pad antennas 3→4
    T, F_in, M_in, A_in = H_raw.shape

    # Process per receiver
    H_norm_list  = []
    ts_list_each = []

    for m in range(M_in):
        H_rx = H_raw[:, :, m, :]   # [T, F=30, A=3]

        # Pad antennas 3→4 only; subcarriers kept at 30
        H_rx_padded = normalize_intel5300_to_uniform(
            H_rx,
            target_F=None,
            target_A=4,
        )   # [T, 30, 4]

        # B1: AGC
        H_norm, _ = agc_normalize(H_rx_padded)

        # B2: Hampel
        H_norm = hampel_repair_rx(H_norm, rec_id=rec_id, rx_idx=m)
        if H_norm is None:
            return None

        H_norm_list.append(H_norm)
        ts_list_each.append(timestamps.copy())

    # B4: Resample 200→350Hz using common grid (FIX v6.6)
    # All receivers already synchronized → use first receiver's timestamps
    t_common = np.arange(ts_list_each[0][0], ts_list_each[0][-1],
                         1.0 / cfg_pre.get('fs_out', 350.0))

    H_uni_list = [
        pchip_resample_rx(H_norm_list[m], ts_list_each[m], t_common)
        for m in range(M_in)
    ]

    # COMBINE: [T_out, F=30, M=3, A=4]
    H_uniform = np.stack(H_uni_list, axis=2)
    return H_uniform


def scan_xrf55_dataset(
    raw_dir    : Path,
    class_names: List[str],
) -> List[Recording]:
    """
    Scan XRF55 raw directory for CSI files.

    Expected structure:
        raw_dir/
          {activity}/
            {subject}_{trial}.npy   or similar
    """
    recordings = []

    # Try activity-based folder structure
    for act_dir in sorted(raw_dir.iterdir()):
        if not act_dir.is_dir():
            continue
        act_name = act_dir.name.lower()
        label_int = _map_xrf55_class(act_name)
        if label_int is None:
            logger.debug(f"Skipping unmapped class: {act_name}")
            continue

        for file_path in sorted(act_dir.glob('*.npy')) + sorted(act_dir.glob('*.mat')):
            # Extract subject ID from filename
            stem      = file_path.stem
            # Parse subject: typically first part before underscore
            parts     = stem.split('_')
            person_id = parts[0] if parts else 'S0'

            rec = Recording(
                recording_id       = stem,
                person_id          = person_id,
                room_id            = 'default',
                activity_label     = class_names[label_int],
                activity_label_int = label_int,
                metadata           = {'file_path': str(file_path)},
            )
            recordings.append(rec)

    # Fallback: flat structure with label in filename
    if not recordings:
        for file_path in sorted(raw_dir.glob('**/*.npy')) + \
                         sorted(raw_dir.glob('**/*.mat')):
            stem      = file_path.stem
            label_int = None
            for key, val in XRF55_CLASS_MAP.items():
                if key.replace(' ', '_') in stem.lower():
                    label_int = val
                    break
            if label_int is None:
                continue
            parts     = stem.split('_')
            person_id = parts[0] if parts else 'S0'
            rec = Recording(
                recording_id       = stem,
                person_id          = person_id,
                room_id            = 'default',
                activity_label     = class_names[label_int],
                activity_label_int = label_int,
                metadata           = {'file_path': str(file_path)},
            )
            recordings.append(rec)

    logger.info(f"XRF55: found {len(recordings)} recordings in {raw_dir}")
    return recordings


def preprocess_xrf55_dataset(cfg: dict, fold_id: int) -> Optional[str]:
    """Main entry point for XRF55 preprocessing. Returns HDF5 path."""
    from .pipeline_own import _loso_split, _extract_windows_for_split, _write_hdf5

    cfg_ds  = cfg.get('dataset', {})
    cfg_pre = cfg.get('preprocessing', {})

    raw_dir       = Path(cfg_ds.get('raw_dir', 'data/raw/xrf55'))
    processed_dir = Path(cfg_ds.get('processed_dir', 'data/processed/xrf55'))
    processed_dir.mkdir(parents=True, exist_ok=True)

    class_names = cfg_ds.get('class_names',
        ['walk','run','stand_up','sit_down','bend','fall','lying_still','empty'])

    out_path = processed_dir / f"fold_{fold_id:02d}.h5"
    if out_path.exists():
        return str(out_path)

    all_recordings = scan_xrf55_dataset(raw_dir, class_names)
    if not all_recordings:
        logger.error(f"No recordings found in {raw_dir}")
        return None

    # Preprocess each recording
    valid = []
    for rec in tqdm(all_recordings, desc='XRF55 preprocessing'):
        H_uniform = preprocess_xrf55_recording(
            rec.metadata['file_path'], rec.activity_label_int,
            rec.recording_id, cfg_pre,
        )
        if H_uniform is not None:
            rec.H_uniform = H_uniform
            valid.append(rec)

    if not valid:
        logger.error("No valid XRF55 recordings")
        return None

    # G1: select ref antenna
    for rec in valid:
        rec.metadata['ref_antenna_idx'] = select_ref_antenna(rec.H_uniform)

    # LOSO split
    train_recs, val_recs, test_recs = _loso_split(valid, fold_id)

    # Stats Pass 1
    stats_path = str(processed_dir / f"fold_{fold_id:02d}_stats.npz")
    stats = fit_preprocessing_stats(train_recs, fold_id, stats_path,
                                    cfg_pre.get('T_window', 350),
                                    cfg_pre.get('stride', 175))

    _write_hdf5(str(out_path), train_recs, val_recs, test_recs,
                stats, cfg_pre, fold_id, 'xrf55')

    logger.info(f"XRF55 saved: {out_path}")
    return str(out_path)
