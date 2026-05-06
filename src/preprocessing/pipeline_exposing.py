"""
src/preprocessing/pipeline_exposing.py
Preprocessing pipeline for 'Exposing the CSI' dataset.
(Broadcom 43684, AX-CSI, 802.11ax, 2048 subs, ~150Hz, 1 volunteer)

Directory structure:
    data/raw/expose_csi/
        rx_01/S7a_{X}.mat    X = activity code letter (A,B,C,D,K,I,H,E)
        rx_02/S7b_{X}.mat
        rx_03/S7c_{X}.mat

File shape: [12000, 2048, 4] complex (150fps × 80s = 12000 frames).
Select 52 subcarriers uniformly from 2048; no antenna padding needed (A=4).

Class mapping (activity code letter → our 8-class label):
    A=walk, B=run, C=jump, D=sit_still, K=squat, I=lay_down, H=clap, E=empty

Protocol: blocked chronological 5-fold (single volunteer, no LOSO possible).
For each activity recording (~66 windows), windows are split into 5 contiguous
time blocks. fold_id: test=block[fold_id], val=block[(fold_id-1)%5], train=rest.
All 8 activities appear in every split (class-complete by construction).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .common import (
    Recording,
    agc_normalize,
    hampel_repair_rx,
    pchip_resample_rx,
    select_ref_antenna,
    build_windows,
    fit_preprocessing_stats_from_windows,
)
from .parsers.axcsi_parser import (
    load_axcsi_file,
    normalize_axcsi_to_uniform,
    AXCSI_FS,
)

logger = logging.getLogger(__name__)

# Activity code letter → our 8-class label
EXPOSING_CLASS_CODE_MAP: Dict[str, int] = {
    'A': 0,   # walk
    'B': 1,   # run
    'C': 2,   # jump
    'D': 3,   # sit_still
    'K': 4,   # squat
    'I': 5,   # lay_down
    'H': 6,   # clap
    'E': 7,   # empty
}

EXPOSING_CLASS_NAMES = [
    'walk', 'run', 'jump', 'sit_still', 'squat', 'lay_down', 'clap', 'empty'
]

_RX_PREFIX = {0: 'S7a', 1: 'S7b', 2: 'S7c'}
_RX_DIR    = {0: 'rx_01', 1: 'rx_02', 2: 'rx_03'}


def _get_class_code_map(cfg_ds: dict) -> Dict[str, int]:
    raw = cfg_ds.get('class_code_map', {})
    if raw:
        return {str(k): int(v) for k, v in raw.items()}
    return EXPOSING_CLASS_CODE_MAP


def scan_exposing_dataset(
    raw_dir        : Path,
    class_code_map : Dict[str, int],
    class_names    : List[str],
) -> List[Recording]:
    """
    Scan expose_csi directory for matched file triples (rx_01/rx_02/rx_03).

    Expected: rx_01/S7a_{X}.mat, rx_02/S7b_{X}.mat, rx_03/S7c_{X}.mat
    where X is an activity code letter.

    Warns and skips if rx_02 or rx_03 file is missing (no silent fill).
    Raises:
        FileNotFoundError: if rx_01 directory doesn't exist
    """
    rx01_dir = raw_dir / 'rx_01'
    if not rx01_dir.exists():
        raise FileNotFoundError(f"rx_01 directory not found: {rx01_dir}")

    recordings = []

    for mat_file in sorted(rx01_dir.glob('S7a_*.mat')):
        stem = mat_file.stem          # e.g. 'S7a_A'
        code = stem.replace('S7a_', '')  # activity code, e.g. 'A'

        label_int = class_code_map.get(code)
        if label_int is None:
            logger.debug(f"Unmapped activity code '{code}' in {mat_file.name}")
            continue

        rx02_file = raw_dir / 'rx_02' / f'S7b_{code}.mat'
        rx03_file = raw_dir / 'rx_03' / f'S7c_{code}.mat'

        if not rx02_file.exists():
            logger.warning(f"Missing rx_02 file S7b_{code}.mat — skipping {code}")
            continue
        if not rx03_file.exists():
            logger.warning(f"Missing rx_03 file S7c_{code}.mat — skipping {code}")
            continue

        rec = Recording(
            recording_id       = f'expose_{code}',
            person_id          = 'volunteer_01',
            room_id            = 'expose_csi',
            activity_label     = class_names[label_int],
            activity_label_int = label_int,
            metadata           = {
                'rx_files'     : [str(mat_file), str(rx02_file), str(rx03_file)],
                'activity_code': code,
            },
        )
        recordings.append(rec)

    logger.info(f"Exposing CSI: found {len(recordings)} activity files in {raw_dir}")
    if not recordings:
        logger.error(
            "No recordings found. Check directory structure: "
            f"{raw_dir}/rx_01/S7a_{{X}}.mat  rx_02/S7b_{{X}}.mat  rx_03/S7c_{{X}}.mat"
        )
    return recordings


def preprocess_exposing_recording(
    rx_files: List[str],
    rec_id  : str,
    cfg_pre : dict,
) -> Optional[np.ndarray]:
    """
    Process one Exposing CSI session (3 receiver files) → H_uniform [T_out, 52, 3, 4].

    Steps:
        1. Load each .mat → [T=12000, 2048, 4] complex
        2. Uniform subcarrier select: 2048 → 52
        3. AGC + Hampel per receiver
        4. PCHIP resample 150Hz → 350Hz
        5. Stack → [T_out, 52, M=3, A=4]
    """
    H_norm_list = []
    ts_list     = []

    for m, fpath in enumerate(rx_files):
        H_raw, timestamps = load_axcsi_file(fpath)
        if H_raw is None:
            logger.error(f"Failed to load {fpath}")
            return None

        H_rx   = normalize_axcsi_to_uniform(H_raw, target_F=52, target_A=4)  # [T,52,4]
        H_norm, _ = agc_normalize(H_rx)
        H_norm = hampel_repair_rx(H_norm, rec_id=rec_id, rx_idx=m)
        if H_norm is None:
            return None

        H_norm_list.append(H_norm)
        ts_list.append(timestamps.copy())

    fs_out   = cfg_pre.get('fs_out', 350.0)
    t_common = np.arange(ts_list[0][0], ts_list[0][-1], 1.0 / fs_out)

    H_uni_list = [
        pchip_resample_rx(H_norm_list[m], ts_list[m], t_common)
        for m in range(len(rx_files))
    ]
    return np.stack(H_uni_list, axis=2)   # [T_out, 52, M=3, A=4]


def _write_hdf5_exposing_blocked(
    out_path  : str,
    all_recs  : List[Recording],
    fold_id   : int,
    n_folds   : int,
    cfg_pre   : dict,
    stats_path: str,
    n_aug     : int,
    cfg_aug   : dict,
) -> None:
    """
    Blocked 5-fold HDF5 writer for Exposing CSI (window-level split).

    For each recording (one per activity, ~66 windows after sliding window):
        blocks = np.array_split(windows, n_folds)  # 5 contiguous time blocks
        test  += blocks[fold_id]           from ALL activities
        val   += blocks[(fold_id-1)%5]     from ALL activities
        train += remaining 3 blocks        from ALL activities

    All 8 activities appear in every split (class-complete by construction).
    Stats are fit ONLY on train windows (no val/test leakage).
    Uses _write_hdf5_core — no duplicated HDF5 write logic.
    """
    from .pipeline_own import _write_hdf5_core

    T_window  = cfg_pre.get('T_window', 350)
    stride    = cfg_pre.get('stride', 175)
    n_classes = 8

    split_windows: Dict[str, List] = {'train': [], 'val': [], 'test': []}

    for rec in all_recs:
        ref_idx = rec.metadata['ref_antenna_idx']
        windows = list(build_windows(rec, T=T_window, stride=stride))
        if not windows:
            logger.warning(f"No windows for {rec.recording_id}, skipping")
            continue

        blocks    = np.array_split(np.arange(len(windows)), n_folds)
        test_idxs = set(blocks[fold_id].tolist())
        val_idxs  = set(blocks[(fold_id - 1) % n_folds].tolist())

        for i, win_H in enumerate(windows):
            entry = (
                win_H,
                rec.activity_label_int,
                ref_idx,
                rec.recording_id,
                rec.person_id,
                rec.room_id,
                i,
            )
            if i in test_idxs:
                split_windows['test'].append(entry)
            elif i in val_idxs:
                split_windows['val'].append(entry)
            else:
                split_windows['train'].append(entry)

    # Class completeness assertion — every split must contain all 8 activities
    for split_name, entries in split_windows.items():
        labels_present = {label_int for _, label_int, *_ in entries}
        expected       = set(range(n_classes))
        assert labels_present == expected, (
            f"Exposing CSI blocked split: '{split_name}' missing activities "
            f"{expected - labels_present}. Check that all recordings have "
            f"enough windows for {n_folds}-fold split."
        )

    # Fit stats on TRAIN windows only (no val/test leakage)
    train_entries_for_stats = [
        (win_H, label_int, ref_idx)
        for win_H, label_int, ref_idx, *_ in split_windows['train']
    ]
    stats = fit_preprocessing_stats_from_windows(
        train_entries=train_entries_for_stats,
        fold_id=fold_id,
        save_path=stats_path,
        n_groups=cfg_pre.get('n_fisher_groups', 13),
    )

    _write_hdf5_core(
        out_path=out_path,
        split_windows=split_windows,
        stats=stats,
        cfg_pre=cfg_pre,
        fold_id=fold_id,
        dataset_name='exposing',
        protocol='blocked_5fold',
        n_aug_offline=n_aug,
        cfg_aug=cfg_aug,
    )

    logger.info(
        f"Exposing CSI fold {fold_id}: "
        f"train={len(split_windows['train'])}, "
        f"val={len(split_windows['val'])}, "
        f"test={len(split_windows['test'])} windows"
    )


def preprocess_exposing_dataset(cfg: dict, fold_id: int) -> Optional[str]:
    """
    Main entry point for Exposing CSI preprocessing. Returns HDF5 path or None.

    Uses blocked 5-fold chronological split at window level (single volunteer).
    Stats are fit on train windows only. All splits contain all 8 activities.
    """
    cfg_ds  = cfg.get('dataset', {})
    cfg_pre = cfg.get('preprocessing', {})

    raw_dir       = Path(cfg_ds.get('raw_dir', 'data/raw/expose_csi'))
    processed_dir = Path(cfg_ds.get('processed_dir', 'data/processed/expose_csi'))
    processed_dir.mkdir(parents=True, exist_ok=True)

    class_code_map = _get_class_code_map(cfg_ds)
    class_names    = cfg_ds.get('class_names', EXPOSING_CLASS_NAMES)
    n_folds        = cfg.get('evaluation', {}).get('n_folds', 5)

    out_path = processed_dir / f"fold_{fold_id:02d}.h5"
    if out_path.exists():
        logger.info(f"Exposing CSI fold {fold_id} already exists: {out_path}")
        return str(out_path)

    all_recordings = scan_exposing_dataset(raw_dir, class_code_map, class_names)
    if not all_recordings:
        return None

    valid = []
    for rec in tqdm(all_recordings, desc=f'Exposing CSI preprocessing fold {fold_id}'):
        H_uniform = preprocess_exposing_recording(
            rec.metadata['rx_files'], rec.recording_id, cfg_pre,
        )
        if H_uniform is not None:
            rec.H_uniform = H_uniform
            valid.append(rec)

    if not valid:
        logger.error("No valid Exposing CSI recordings after preprocessing")
        return None

    for rec in valid:
        rec.metadata['ref_antenna_idx'] = select_ref_antenna(rec.H_uniform)

    stats_path    = str(processed_dir / f"fold_{fold_id:02d}_stats.npz")
    n_aug_offline = cfg.get('training', {}).get('n_aug_offline', 3)

    _write_hdf5_exposing_blocked(
        out_path   = str(out_path),
        all_recs   = valid,
        fold_id    = fold_id,
        n_folds    = n_folds,
        cfg_pre    = cfg_pre,
        stats_path = stats_path,
        n_aug      = n_aug_offline,
        cfg_aug    = cfg.get('training', {}),
    )

    logger.info(f"Exposing CSI fold {fold_id} saved: {out_path}")
    return str(out_path)
