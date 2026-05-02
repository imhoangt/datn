"""
src/preprocessing/pipeline_exposing.py
Preprocessing pipeline adapter for 'Exposing the CSI' dataset.
(Broadcom 43684, 802.11ax, 2048 subs, ~150Hz)

Key differences:
- 2048 subcarriers → uniform select 52
- ~150Hz → 350Hz via PCHIP
- 3 collector devices = 3 RX files per session
- 12 activities → 8-class mapping
- 3 subjects only → LOSO 3-fold
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .common import (
    Recording,
    agc_normalize,
    hampel_repair_rx,
    pchip_resample_rx,
    select_ref_antenna,
    fit_preprocessing_stats,
)
from .parsers.axcsi_parser import (
    load_axcsi_file,
    normalize_axcsi_to_uniform,
    AXCSI_FS,
)

logger = logging.getLogger(__name__)

EXPOSING_CLASS_MAP = {
    'walking'   : 0, 'walk'        : 0,
    'running'   : 1, 'run'         : 1,
    'standing'  : 2, 'stand'       : 2, 'standing up': 2,
    'sitting'   : 3, 'sitting down': 3, 'sit'        : 3,
    'bending'   : 4, 'bend'        : 4, 'picking'    : 4,
    'falling'   : 5, 'fall'        : 5,
    'lying'     : 6, 'lie'         : 6, 'lying still': 6,
    'empty'     : 7, 'no one'      : 7, 'background' : 7,
    'gesturing' : 4, 'other'       : 7,
}


def _map_exposing_class(name: str) -> Optional[int]:
    n = name.lower().strip()
    if n in EXPOSING_CLASS_MAP:
        return EXPOSING_CLASS_MAP[n]
    for key, val in EXPOSING_CLASS_MAP.items():
        if key in n or n in key:
            return val
    return None


def preprocess_exposing_recording(
    rx_files : List[str],
    rec_id   : str,
    cfg_pre  : dict,
) -> Optional[np.ndarray]:
    """
    Process one session (3 collector files) → H_uniform [T_out, 52, 3, 4].
    """
    H_norm_list, ts_list = [], []

    for m, fpath in enumerate(rx_files):
        H_raw, timestamps = load_axcsi_file(fpath)
        if H_raw is None:
            return None

        H_rx = normalize_axcsi_to_uniform(H_raw, target_F=52, target_A=4)  # [T,52,4]
        H_norm, _ = agc_normalize(H_rx)
        H_norm = hampel_repair_rx(H_norm, rec_id=rec_id, rx_idx=m)
        if H_norm is None:
            return None

        H_norm_list.append(H_norm)
        ts_list.append(timestamps.copy())

    # Common grid from ref_rx (FIX v6.6)
    t_common = np.arange(ts_list[0][0], ts_list[0][-1],
                         1.0 / cfg_pre.get('fs_out', 350.0))

    H_uni_list = [
        pchip_resample_rx(H_norm_list[m], ts_list[m], t_common)
        for m in range(len(rx_files))
    ]
    return np.stack(H_uni_list, axis=2)   # [T_out, 52, 3, 4]


def scan_exposing_dataset(raw_dir: Path, class_names: List[str]) -> List[Recording]:
    """Scan directory structure for Exposing the CSI data."""
    recordings = []

    for act_dir in sorted(raw_dir.iterdir()):
        if not act_dir.is_dir():
            continue
        label_int = _map_exposing_class(act_dir.name)
        if label_int is None:
            continue

        # Each session has 3 receiver files: *_rx0.*, *_rx1.*, *_rx2.*
        rx0_files = sorted(act_dir.glob('*_rx0*'))
        for rx0 in rx0_files:
            stem = rx0.stem.replace('_rx0', '')
            rx1  = act_dir / f"{stem}_rx1{rx0.suffix}"
            rx2  = act_dir / f"{stem}_rx2{rx0.suffix}"
            if not rx1.exists():
                # Single-receiver fallback: duplicate rx0 for all 3
                rx1 = rx0; rx2 = rx0

            parts     = stem.split('_')
            person_id = parts[0] if parts else 'S0'
            rec = Recording(
                recording_id       = stem,
                person_id          = person_id,
                room_id            = 'default',
                activity_label     = class_names[label_int],
                activity_label_int = label_int,
                metadata           = {
                    'rx_files': [str(rx0), str(rx1), str(rx2)]
                },
            )
            recordings.append(rec)

    logger.info(f"Exposing CSI: found {len(recordings)} recordings in {raw_dir}")
    return recordings


def preprocess_exposing_dataset(cfg: dict, fold_id: int) -> Optional[str]:
    """Main entry point for Exposing the CSI preprocessing."""
    from .pipeline_own import _loso_split, _write_hdf5

    cfg_ds  = cfg.get('dataset', {})
    cfg_pre = cfg.get('preprocessing', {})

    raw_dir       = Path(cfg_ds.get('raw_dir',       'data/raw/exposing'))
    processed_dir = Path(cfg_ds.get('processed_dir', 'data/processed/exposing'))
    processed_dir.mkdir(parents=True, exist_ok=True)

    class_names = cfg_ds.get('class_names',
        ['walk','run','stand_up','sit_down','bend','fall','lying_still','empty'])

    out_path = processed_dir / f"fold_{fold_id:02d}.h5"
    if out_path.exists():
        return str(out_path)

    all_recordings = scan_exposing_dataset(raw_dir, class_names)
    if not all_recordings:
        logger.error(f"No recordings found in {raw_dir}")
        return None

    valid = []
    for rec in tqdm(all_recordings, desc='Exposing preprocessing'):
        H_uniform = preprocess_exposing_recording(
            rec.metadata['rx_files'], rec.recording_id, cfg_pre)
        if H_uniform is not None:
            rec.H_uniform = H_uniform
            valid.append(rec)

    if not valid:
        logger.error("No valid Exposing CSI recordings")
        return None

    for rec in valid:
        rec.metadata['ref_antenna_idx'] = select_ref_antenna(rec.H_uniform)

    train_recs, val_recs, test_recs = _loso_split(valid, fold_id)

    stats_path = str(processed_dir / f"fold_{fold_id:02d}_stats.npz")
    stats = fit_preprocessing_stats(
        train_recs, fold_id, stats_path,
        cfg_pre.get('T_window', 350), cfg_pre.get('stride', 175),
    )

    _write_hdf5(str(out_path), train_recs, val_recs, test_recs,
                stats, cfg_pre, fold_id, 'exposing')

    logger.info(f"Exposing CSI saved: {out_path}")
    return str(out_path)
