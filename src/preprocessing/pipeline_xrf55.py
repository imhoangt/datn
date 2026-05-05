"""
src/preprocessing/pipeline_xrf55.py
Preprocessing pipeline for XRF55 dataset (Intel 5300, 30 subs, ~200Hz).

Directory structure:
    data/raw/xrf55/
        rx_01/{volunteer_id}/{class_id}_{volunteer_id}_{rep}.dat
        rx_02/{volunteer_id}/{class_id}_{volunteer_id}_{rep}.dat
        rx_03/{volunteer_id}/{class_id}_{volunteer_id}_{rep}.dat

Each .dat file: Intel 5300 binary (10-bit packed), single receiver site.
Per-file output: [T=1000, F=30, A=3] complex64 at ~200Hz.

Pipeline:
    load 3 .dat (rx_01/rx_02/rx_03) → pad A:3→4 → AGC → Hampel
    → resample 200→350Hz → stack → [T_out, 30, M=3, A=4]

Class mapping (XRF55 numeric activity ID → our 8-class label):
    39=walk, 35=run, 34=jump, 37=stand_up, 36=sit_down,
    9=bend_to_pick, 33=fall, 32=clap
    (verify against XRF55 dataset documentation)

Protocol: LOSO-person, 8 folds (1 volunteer per fold as test).
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
)
from .parsers.intel5300_parser import (
    parse_intel5300_dat,
    normalize_intel5300_to_uniform,
    XRF55_FS_NOMINAL,
)

logger = logging.getLogger(__name__)

# XRF55 numeric activity ID → our 8-class label (0-indexed)
# Mapping must be verified against XRF55 dataset documentation
XRF55_CLASS_ID_MAP: Dict[int, int] = {
    39: 0,   # walk
    35: 1,   # run
    34: 2,   # jump
    37: 3,   # stand_up
    36: 4,   # sit_down
    9:  5,   # bend_to_pick
    33: 6,   # fall
    32: 7,   # clap
}

XRF55_CLASS_NAMES = [
    'walk', 'run', 'jump', 'stand_up', 'sit_down', 'bend_to_pick', 'fall', 'clap'
]

# Override from config if provided
def _get_class_id_map(cfg_ds: dict) -> Dict[int, int]:
    raw = cfg_ds.get('class_id_map', {})
    if raw:
        return {int(k): int(v) for k, v in raw.items()}
    return XRF55_CLASS_ID_MAP


def scan_xrf55_dataset(
    raw_dir: Path,
    class_id_map: Dict[int, int],
    class_names: List[str],
) -> List[Recording]:
    """
    Scan XRF55 directory for matched triples (rx_01/rx_02/rx_03).

    Filename convention: {class_id}_{volunteer_id}_{rep}.dat
    Raises:
        FileNotFoundError: if rx_01 directory doesn't exist
    """
    rx01_dir = raw_dir / 'rx_01'
    rx02_dir = raw_dir / 'rx_02'
    rx03_dir = raw_dir / 'rx_03'

    if not rx01_dir.exists():
        raise FileNotFoundError(f"rx_01 directory not found: {rx01_dir}")

    recordings = []

    for vol_dir in sorted(rx01_dir.iterdir()):
        if not vol_dir.is_dir():
            continue
        volunteer_id = vol_dir.name

        for dat_file in sorted(vol_dir.glob('*.dat')):
            stem  = dat_file.stem
            parts = stem.split('_')
            if len(parts) < 3:
                logger.warning(f"Skipping unexpected filename: {dat_file.name}")
                continue

            try:
                class_id_raw = int(parts[0])
                rep          = parts[2]
            except (ValueError, IndexError):
                logger.warning(f"Cannot parse class/rep from: {dat_file.name}")
                continue

            label_int = class_id_map.get(class_id_raw)
            if label_int is None:
                logger.debug(f"Unmapped class_id {class_id_raw} in {dat_file.name}")
                continue

            # Find matching files in rx_02, rx_03
            rx02_file = rx02_dir / volunteer_id / dat_file.name
            rx03_file = rx03_dir / volunteer_id / dat_file.name

            if not rx02_file.exists():
                logger.warning(f"Missing rx_02 for {dat_file.name}, skipping")
                continue
            if not rx03_file.exists():
                logger.warning(f"Missing rx_03 for {dat_file.name}, skipping")
                continue

            rec = Recording(
                recording_id       = stem,
                person_id          = volunteer_id,
                room_id            = 'xrf55',
                activity_label     = class_names[label_int],
                activity_label_int = label_int,
                metadata           = {
                    'rx_files': [str(dat_file), str(rx02_file), str(rx03_file)],
                    'class_id_raw': class_id_raw,
                    'rep': rep,
                },
            )
            recordings.append(rec)

    logger.info(f"XRF55: found {len(recordings)} matched recordings in {raw_dir}")
    if not recordings:
        logger.error(
            f"No recordings found. Check directory structure: "
            f"{raw_dir}/rx_01/{{volunteer}}/{{class_id}}_{{vol}}_{{rep}}.dat"
        )
    return recordings


def preprocess_xrf55_recording(
    rx_files: List[str],
    rec_id  : str,
    cfg_pre : dict,
) -> Optional[np.ndarray]:
    """
    Process one XRF55 recording (3 receiver sites) → H_uniform [T_out, 30, M=3, A=4].

    Steps:
        1. Parse each .dat (binary Intel 5300) → [T, 30, A=3] complex64
        2. Pad A: 3→4 (zero-pad 4th antenna)
        3. AGC normalization per receiver
        4. Hampel outlier repair per receiver
        5. PCHIP resample 200Hz → 350Hz on common time grid
        6. Stack 3 receivers → [T_out, 30, 3, 4]
    """
    H_norm_list = []
    ts_list     = []

    for m, fpath in enumerate(rx_files):
        try:
            H_raw, timestamps = parse_intel5300_dat(fpath)
        except (ValueError, OSError) as e:
            logger.error(f"Failed to parse {fpath}: {e}")
            return None

        # H_raw: [T, 30, Nrx=3]
        # normalize_intel5300_to_uniform expects [T, F, A]; here Nrx acts as A
        H_rx_padded = normalize_intel5300_to_uniform(H_raw, target_F=None, target_A=4)
        # → [T, 30, 4]

        H_norm, _ = agc_normalize(H_rx_padded)
        H_norm    = hampel_repair_rx(H_norm, rec_id=rec_id, rx_idx=m)
        if H_norm is None:
            return None

        H_norm_list.append(H_norm)
        ts_list.append(timestamps.copy())

    # Common time grid: use first receiver's time range, step=1/fs_out
    fs_out   = cfg_pre.get('fs_out', 350.0)
    t_common = np.arange(ts_list[0][0], ts_list[0][-1], 1.0 / fs_out)

    H_uni_list = [
        pchip_resample_rx(H_norm_list[m], ts_list[m], t_common)
        for m in range(len(rx_files))
    ]

    # Stack: [T_out, 30, M=3, A=4]
    return np.stack(H_uni_list, axis=2)


def preprocess_xrf55_dataset(cfg: dict, fold_id: int) -> Optional[str]:
    """
    Main entry point for XRF55 preprocessing. Returns HDF5 path or None on failure.

    Performs LOSO split by volunteer_id (fold_id = held-out volunteer).
    """
    from .pipeline_own import _loso_split, _write_hdf5

    cfg_ds  = cfg.get('dataset', {})
    cfg_pre = cfg.get('preprocessing', {})

    raw_dir       = Path(cfg_ds.get('raw_dir', 'data/raw/xrf55'))
    processed_dir = Path(cfg_ds.get('processed_dir', 'data/processed/xrf55'))
    processed_dir.mkdir(parents=True, exist_ok=True)

    class_id_map = _get_class_id_map(cfg_ds)
    class_names  = cfg_ds.get('class_names', XRF55_CLASS_NAMES)

    out_path = processed_dir / f"fold_{fold_id:02d}.h5"
    if out_path.exists():
        logger.info(f"XRF55 fold {fold_id} already exists: {out_path}")
        return str(out_path)

    all_recordings = scan_xrf55_dataset(raw_dir, class_id_map, class_names)
    if not all_recordings:
        return None

    valid = []
    for rec in tqdm(all_recordings, desc=f'XRF55 preprocessing fold {fold_id}'):
        H_uniform = preprocess_xrf55_recording(
            rec.metadata['rx_files'], rec.recording_id, cfg_pre,
        )
        if H_uniform is not None:
            rec.H_uniform = H_uniform
            valid.append(rec)

    if not valid:
        logger.error("No valid XRF55 recordings after preprocessing")
        return None

    for rec in valid:
        rec.metadata['ref_antenna_idx'] = select_ref_antenna(rec.H_uniform)

    train_recs, val_recs, test_recs = _loso_split(valid, fold_id)

    stats_path = str(processed_dir / f"fold_{fold_id:02d}_stats.npz")
    stats = fit_preprocessing_stats(
        train_recs, fold_id, stats_path,
        cfg_pre.get('T_window', 350),
        cfg_pre.get('stride', 175),
    )

    n_aug_offline = cfg.get('training', {}).get('n_aug_offline', 3)
    _write_hdf5(
        str(out_path), train_recs, val_recs, test_recs,
        stats, cfg_pre, fold_id, 'xrf55',
        n_aug_offline=n_aug_offline,
        cfg_aug=cfg.get('training', {}),
    )

    logger.info(f"XRF55 fold {fold_id} saved: {out_path}")
    return str(out_path)
