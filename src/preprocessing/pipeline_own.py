"""
src/preprocessing/pipeline_own.py
Full preprocessing pipeline for own-collected data (BCM4366c0 / Nexmon CSI).

Implements CHARM-Net v6.6 pipeline exactly:
    Stage A: Parse → validate → select 52 subs → detect gaps
    Stage B: B1 AGC → B2 Hampel+PCHIP → B3 align → B4 common-grid resample → COMBINE
    Pre-split G1: select_ref_antenna for ALL recordings
    Stage C: LOSO split + anti-leakage assertions
    Stats Pass 1: Welford Z-score + Fisher (single pass, clean train only)
    Stage D: Sliding windows
    → Returns: dict of {split: [(X_amp, X_dfs, label, meta), ...]}

Usage:
    from src.preprocessing.pipeline_own import preprocess_own_dataset
    split_data = preprocess_own_dataset(cfg, fold_id)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .common import (
    Recording,
    agc_normalize,
    hampel_repair_rx,
    align_rx_timestamps,
    resample_and_stack_receivers,
    select_ref_antenna,
    fit_preprocessing_stats,
    build_windows,
    process_amp_branch,
    process_dfs_branch,
    # Offline augmentation (v6.7)
    _window_seed,
    augment_joint_offline,
    augment_features_offline,
)
from .parsers.nexmon_parser import (
    parse_nexmon_pcap,
    validate_and_clean_packets,
    select_subcarriers_bcm4366c0,
    detect_ipi_gaps,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single recording preprocessing (A → B → combine)
# ---------------------------------------------------------------------------
def preprocess_one_recording(
    rec_id      : str,
    pcap_files  : List[str],
    duration_s  : float,
    src_mac     : Optional[str],
    cfg_pre     : dict,
) -> Optional[np.ndarray]:
    """
    Run Stages A and B for one recording (3 .pcap files → H_uniform).

    Returns: H_uniform [T_out, 52, M=3, A=4] complex64, or None if rejected.
    """
    H_norm_list: List[np.ndarray] = []
    ts_list    : List[np.ndarray] = []

    n_rx  = len(pcap_files)
    fs_in = cfg_pre.get('fs_out', 350.0)   # already at 350Hz if nominal

    for rx_idx, pcap_path in enumerate(pcap_files):
        # A1: Parse
        H_raw, timestamps = parse_nexmon_pcap(
            pcap_path,
            src_mac=src_mac,
        )
        if H_raw is None:
            logger.warning(f"[{rec_id}] RX{rx_idx}: parse failed")
            return None

        # A2: Validate & clean
        H_raw, timestamps = validate_and_clean_packets(
            H_raw, timestamps,
            duration_s  = duration_s,
            fs_nominal  = cfg_pre.get('fs_out', 350.0),
            min_packets = cfg_pre.get('min_packets', 300),
            max_loss    = cfg_pre.get('packet_loss_max', 0.15),
        )
        if H_raw is None:
            logger.warning(f"[{rec_id}] RX{rx_idx}: validation failed")
            return None

        # A3: Subcarrier selection (64→52)
        H_sub = select_subcarriers_bcm4366c0(H_raw)   # [T, 52, 4]

        # A4: IPI gap detection — reject if any gap > 100ms
        # PCHIP would interpolate through long gaps producing spurious signal.
        large_gap, _ = detect_ipi_gaps(
            timestamps,
            duration_s,
            fs_nominal   = cfg_pre.get('fs_out', 350.0),
            large_gap_ms = cfg_pre.get('gap_large_ms', 100.0),
        )
        if large_gap.any():
            logger.warning(f"[{rec_id}] RX{rx_idx}: large IPI gap detected, rejecting")
            return None

        # B1: AGC normalization
        H_norm, _ = agc_normalize(H_sub)              # [T, 52, 4]

        # B2: Hampel outlier detection + PCHIP repair
        H_norm = hampel_repair_rx(
            H_norm,
            rec_id   = rec_id,
            rx_idx   = rx_idx,
            window_half = cfg_pre.get('hampel_half', 7),
            k_factor    = cfg_pre.get('hampel_k', 3.0),
        )
        if H_norm is None:
            return None

        H_norm_list.append(H_norm)
        ts_list.append(timestamps)

    # B3: Multi-RX timestamp alignment
    try:
        ts_list, shifts = align_rx_timestamps(
            H_norm_list      = H_norm_list,
            timestamps_list  = ts_list,
            rec_id           = rec_id,
            ref_rx           = cfg_pre.get('ref_rx', 0),
            threshold_s      = cfg_pre.get('align_threshold_s', 0.5),
        )
    except ValueError as e:
        logger.warning(str(e))
        return None

    # B4: PCHIP resample onto common grid + COMBINE (FIX v6.6 BUG#54)
    H_uniform = resample_and_stack_receivers(
        H_norm_list     = H_norm_list,
        timestamps_list = ts_list,
        ref_rx          = cfg_pre.get('ref_rx', 0),
        fs_out          = cfg_pre.get('fs_out', 350.0),
    )
    # H_uniform: [T_out, 52, M=3, A=4] complex64

    return H_uniform


# ---------------------------------------------------------------------------
# Dataset-level preprocessing
# ---------------------------------------------------------------------------
def _build_recording_list(
    raw_dir  : Path,
    class_names: List[str],
) -> List[Recording]:
    """
    Scan raw_dir for .pcap files and build Recording list.

    Expected directory structure:
        raw_dir/
          person_P0/
            room_01/
              setup_01/
                trajectory_Tr0/
                  walk_rep00_rx0.pcap
                  walk_rep00_rx1.pcap
                  walk_rep00_rx2.pcap
                  walk_rep01_rx0.pcap
                  ...

    Alternatively flat structure with naming convention:
        P0_room01_setup01_Tr0_walk_rep00_rx0.pcap
    """
    recordings: List[Recording] = []

    # Support both flat and nested structures
    all_pcaps = sorted(raw_dir.rglob('*_rx0.pcap'))

    if not all_pcaps:
        # Try flat naming
        all_pcaps = sorted(raw_dir.glob('*rx0*'))

    for pcap_rx0 in all_pcaps:
        stem     = pcap_rx0.stem.replace('_rx0', '')
        pcap_rx1 = pcap_rx0.parent / f"{stem}_rx1.pcap"
        pcap_rx2 = pcap_rx0.parent / f"{stem}_rx2.pcap"

        if not pcap_rx1.exists() or not pcap_rx2.exists():
            logger.warning(f"Missing RX1/RX2 for {stem}, skipping")
            continue

        # Parse metadata from filename or path
        meta = _parse_recording_metadata(stem, pcap_rx0, class_names)
        if meta is None:
            continue

        rec = Recording(
            recording_id       = stem,
            person_id          = meta['person_id'],
            room_id            = meta['room_id'],
            activity_label     = meta['activity'],
            activity_label_int = meta['label_int'],
            metadata           = {
                'pcap_files': [str(pcap_rx0), str(pcap_rx1), str(pcap_rx2)],
                'stem'      : stem,
            },
        )
        recordings.append(rec)

    logger.info(f"Found {len(recordings)} recordings in {raw_dir}")
    return recordings


def _parse_recording_metadata(
    stem       : str,
    pcap_path  : Path,
    class_names: List[str],
) -> Optional[dict]:
    """
    Extract person_id, room_id, activity from filename stem.

    Supports format: P{n}_room{r}_setup{s}_Tr{t}_{activity}_rep{rep}
    or path-based extraction.
    """
    # Try path-based extraction first
    parts = pcap_path.parts
    person_id = room_id = activity = None

    for part in parts:
        p = part.lower()
        if p.startswith('person_') or p.startswith('p') and p[1:].isdigit():
            person_id = part
        if 'room' in p:
            room_id = part
        for act in class_names:
            if act.replace('_', '') in p.replace('_', ''):
                activity = act

    # Fallback: filename parsing
    if person_id is None:
        for token in stem.split('_'):
            if token.upper().startswith('P') and token[1:].isdigit():
                person_id = token
                break
        if person_id is None:
            person_id = 'unknown'

    if room_id is None:
        for token in stem.split('_'):
            if 'room' in token.lower():
                room_id = token
                break
        if room_id is None:
            room_id = 'room_unknown'

    if activity is None:
        stem_lower = stem.lower()
        for act in class_names:
            if act.replace('_', '') in stem_lower.replace('_', ''):
                activity = act
                break
        if activity is None:
            logger.warning(f"Cannot determine activity from: {stem}")
            return None

    label_int = class_names.index(activity) if activity in class_names else -1
    if label_int < 0:
        return None

    return {
        'person_id' : person_id,
        'room_id'   : room_id,
        'activity'  : activity,
        'label_int' : label_int,
    }


def _loso_split(
    recordings : List[Recording],
    fold_id    : int,
) -> Tuple[List[Recording], List[Recording], List[Recording]]:
    """
    LOSO-Person 8-fold split.

    fold k: test=person[k], val=person[(k+1)%8], train=rest (6 persons)
    Anti-leakage: no recording in >1 split; no person in >1 split (for LOSO).
    """
    persons = sorted(set(r.person_id for r in recordings))
    n       = len(persons)

    test_pid = persons[fold_id % n]
    val_pid  = persons[(fold_id + 1) % n]
    train_pids = set(persons) - {test_pid, val_pid}

    train = [r for r in recordings if r.person_id in train_pids]
    val   = [r for r in recordings if r.person_id == val_pid]
    test  = [r for r in recordings if r.person_id == test_pid]

    # Anti-leakage assertions (FIX v6.5 verified)
    train_ids = {r.recording_id for r in train}
    val_ids   = {r.recording_id for r in val}
    test_ids  = {r.recording_id for r in test}
    assert train_ids & test_ids == set(), "LEAKAGE: train-test recording overlap!"
    assert train_ids & val_ids  == set(), "LEAKAGE: train-val recording overlap!"
    assert val_ids   & test_ids == set(), "LEAKAGE: val-test recording overlap!"

    train_pids_set = {r.person_id for r in train}
    val_pids_set   = {r.person_id for r in val}
    test_pids_set  = {r.person_id for r in test}
    assert train_pids_set & test_pids_set == set(), "LEAKAGE: person train-test!"
    assert train_pids_set & val_pids_set  == set(), "LEAKAGE: person train-val!"

    logger.info(
        f"[Fold {fold_id}] train={len(train)}, val={len(val)}, test={len(test)}"
    )
    return train, val, test


def preprocess_own_dataset(
    cfg    : dict,
    fold_id: int,
    n_jobs : int = -1,
) -> Optional[str]:
    """
    Full preprocessing pipeline for own-collected dataset.

    Steps:
        1. Scan raw_dir for recordings
        2. Parallel preprocessing: A+B per recording
        3. G1: select_ref_antenna for all recordings
        4. LOSO split (fold_id)
        5. Stats Pass 1 (Welford + Fisher) on train recordings
        6. Build windows + feature extraction
        7. Write HDF5 to processed_dir

    Returns: path to output HDF5 file, or None on failure.
    """
    cfg_ds  = cfg.get('dataset', {})
    cfg_pre = cfg.get('preprocessing', {})
    cfg_tr  = cfg.get('training', {})

    raw_dir       = Path(cfg_ds.get('raw_dir', 'data/raw/own'))
    processed_dir = Path(cfg_ds.get('processed_dir', 'data/processed/own'))
    processed_dir.mkdir(parents=True, exist_ok=True)

    class_names   = cfg_ds.get('class_names',
        ['walk','run','stand_up','sit_down','bend','fall','lying_still','empty'])
    duration_s    = float(cfg_ds.get('duration_s', 5.0))
    src_mac       = cfg_ds.get('src_mac', None)

    out_path = processed_dir / f"fold_{fold_id:02d}.h5"
    if out_path.exists():
        logger.info(f"HDF5 already exists: {out_path}, skipping preprocessing")
        return str(out_path)

    # 1. Build recording list
    all_recordings = _build_recording_list(raw_dir, class_names)
    if not all_recordings:
        logger.error(f"No recordings found in {raw_dir}")
        return None

    # 2. Parallel preprocessing (A → B → combine)
    def _process(rec: Recording) -> Optional[Recording]:
        H_uniform = preprocess_one_recording(
            rec_id     = rec.recording_id,
            pcap_files = rec.metadata['pcap_files'],
            duration_s = duration_s,
            src_mac    = src_mac,
            cfg_pre    = cfg_pre,
        )
        if H_uniform is not None:
            rec.H_uniform = H_uniform
            return rec
        return None

    logger.info(f"Preprocessing {len(all_recordings)} recordings (n_jobs={n_jobs})...")
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_process)(rec) for rec in tqdm(all_recordings, desc='Preprocessing')
    )
    valid_recordings = [r for r in results if r is not None]
    logger.info(f"Valid recordings: {len(valid_recordings)}/{len(all_recordings)}")

    if not valid_recordings:
        logger.error("No valid recordings after preprocessing")
        return None

    # 3. G1: Select reference antenna for ALL recordings (before split!)
    for rec in valid_recordings:
        rec.metadata['ref_antenna_idx'] = select_ref_antenna(
            rec.H_uniform,
            noise_floor_ratio=cfg_pre.get('agc_noise_floor_ratio', 0.10),
        )

    # 4. LOSO split
    train_recs, val_recs, test_recs = _loso_split(valid_recordings, fold_id)

    # 5. Stats Pass 1 (Welford + Fisher) — clean train only
    stats_path = str(processed_dir / f"fold_{fold_id:02d}_stats.npz")
    stats = fit_preprocessing_stats(
        train_recordings = train_recs,
        fold_id          = fold_id,
        save_path        = stats_path,
        T_window         = cfg_pre.get('T_window', 350),
        stride           = cfg_pre.get('stride', 175),
        n_groups         = cfg_pre.get('n_fisher_groups', 13),
    )

    # 6. Feature extraction and HDF5 write
    n_aug_offline = cfg_tr.get('n_aug_offline', 3)
    _write_hdf5(
        out_path      = str(out_path),
        train_recs    = train_recs,
        val_recs      = val_recs,
        test_recs     = test_recs,
        stats         = stats,
        cfg_pre       = cfg_pre,
        fold_id       = fold_id,
        dataset_name  = 'own',
        protocol      = 'LOSO_person',
        n_aug_offline = n_aug_offline,
        cfg_aug       = cfg_tr,
    )

    logger.info(f"Saved: {out_path}")
    return str(out_path)


def _write_hdf5_core(
    out_path     : str,
    split_windows: Dict[str, List],
    stats        : dict,
    cfg_pre      : dict,
    fold_id      : int,
    dataset_name : str,
    protocol     : str = 'LOSO_person',
    n_aug_offline: int = 0,
    cfg_aug      : Optional[dict] = None,
):
    """
    Core HDF5 writer shared by all dataset pipelines.

    split_windows[split] = list of 7-tuples:
        (win_H [T,F,M,A] complex64, label_int, ref_idx,
         rec_id, person_id, room_id, win_orig_idx)
    win_orig_idx: window index within its recording (for RNG seeding).
    X_dfs shape is inferred from the first train window (not hardcoded).
    """
    cfg_aug  = cfg_aug or {}
    T_window = cfg_pre.get('T_window', 350)
    F_sub    = len(stats['fisher_scores'])

    FLUSH_EVERY = 512

    # Probe X_amp / X_dfs shapes from first train window
    train_entries = split_windows.get('train', [])
    if train_entries:
        first_win_H, _, first_ref, *_ = train_entries[0]
        X_amp_probe = process_amp_branch(first_win_H, stats)
        X_dfs_probe = process_dfs_branch(first_win_H, stats, first_ref)
        _, _, M_rx, A    = X_amp_probe.shape      # e.g. 3, 4
        T_dfs, V_dfs, _  = X_dfs_probe.shape      # e.g. 28, 128
    else:
        M_rx  = cfg_pre.get('n_receivers', 3)
        A     = cfg_pre.get('n_antennas', 4)
        T_dfs = cfg_pre.get('T_dfs', 28)
        V_dfs = cfg_pre.get('V_dfs', 128)

    split_counts: dict = {}

    with h5py.File(out_path, 'w', libver='latest') as f:

        # ── Stats (written once, before splits) ─────────────────────────────
        stats_grp = f.create_group('stats')
        for k, v in stats.items():
            if isinstance(v, np.ndarray):
                stats_grp.create_dataset(k, data=v)
            else:
                stats_grp.attrs[k] = v

        for split_name, entries, n_aug in [
            ('train', split_windows.get('train', []), n_aug_offline),
            ('val',   split_windows.get('val',   []), 0),
            ('test',  split_windows.get('test',  []), 0),
        ]:
            grp = f.create_group(split_name)

            ds_amp = grp.create_dataset(
                'X_amp',
                shape=(0, T_window, F_sub, M_rx, A),
                maxshape=(None, T_window, F_sub, M_rx, A),
                dtype=np.float32,
                compression='gzip', compression_opts=4,
                chunks=(1, T_window, F_sub, M_rx, A),
            )
            ds_dfs = grp.create_dataset(
                'X_dfs',
                shape=(0, T_dfs, V_dfs, M_rx),
                maxshape=(None, T_dfs, V_dfs, M_rx),
                dtype=np.float32,
                compression='gzip', compression_opts=4,
                chunks=(1, T_dfs, V_dfs, M_rx),
            )
            ds_lbl = grp.create_dataset(
                'labels', shape=(0,), maxshape=(None,), dtype=np.int8,
            )

            meta_grp = grp.create_group('metadata')

            buf_amp: List[np.ndarray] = []
            buf_dfs: List[np.ndarray] = []
            buf_lbl: List[int]        = []
            write_cursor = 0

            m_rec_ids: List[str] = []
            m_per_ids: List[str] = []
            m_rom_ids: List[str] = []
            m_win_ids: List[int] = []
            m_ref_ids: List[int] = []
            m_aug_id : List[int] = []
            m_orig_wi: List[int] = []

            n_rejected = 0

            def _flush() -> None:
                nonlocal write_cursor
                if not buf_amp:
                    return
                n_new   = len(buf_amp)
                new_end = write_cursor + n_new
                ds_amp.resize(new_end, axis=0)
                ds_dfs.resize(new_end, axis=0)
                ds_lbl.resize(new_end, axis=0)
                ds_amp[write_cursor:new_end] = np.stack(buf_amp)
                ds_dfs[write_cursor:new_end] = np.stack(buf_dfs)
                ds_lbl[write_cursor:new_end] = np.array(buf_lbl, dtype=np.int8)
                write_cursor += n_new
                buf_amp.clear(); buf_dfs.clear(); buf_lbl.clear()

            for global_idx, entry in enumerate(
                tqdm(entries, desc=f'  [{split_name}]', leave=False)
            ):
                win_H, label_int, ref_idx, rec_id, person_id, room_id, win_orig_idx = entry
                seed = _window_seed(rec_id, win_orig_idx)

                X_amp_o = process_amp_branch(win_H, stats)
                X_dfs_o = process_dfs_branch(win_H, stats, ref_idx)

                if (X_amp_o.shape != (T_window, F_sub, M_rx, A)
                        or X_dfs_o.shape != (T_dfs, V_dfs, M_rx)
                        or np.any(np.isnan(X_amp_o))
                        or np.any(np.isnan(X_dfs_o))):
                    n_rejected += 1
                    continue

                buf_amp.append(X_amp_o); buf_dfs.append(X_dfs_o)
                buf_lbl.append(label_int)
                m_rec_ids.append(rec_id)
                m_per_ids.append(person_id)
                m_rom_ids.append(room_id)
                m_win_ids.append(win_orig_idx)
                m_ref_ids.append(ref_idx)
                m_aug_id.append(0)
                m_orig_wi.append(global_idx)

                for cid in range(1, n_aug + 1):
                    win_H_aug = augment_joint_offline(win_H, ref_idx, cid, seed)
                    X_amp_a   = process_amp_branch(win_H_aug, stats)
                    X_dfs_a   = process_dfs_branch(win_H_aug, stats, ref_idx)

                    if np.any(np.isnan(X_amp_a)) or np.any(np.isnan(X_dfs_a)):
                        X_amp_a = X_amp_o.copy()
                        X_dfs_a = X_dfs_o.copy()

                    X_amp_a, X_dfs_a = augment_features_offline(
                        X_amp_a, X_dfs_a, cid, seed, cfg_aug,
                    )

                    buf_amp.append(X_amp_a); buf_dfs.append(X_dfs_a)
                    buf_lbl.append(label_int)
                    m_rec_ids.append(rec_id)
                    m_per_ids.append(person_id)
                    m_rom_ids.append(room_id)
                    m_win_ids.append(win_orig_idx)
                    m_ref_ids.append(ref_idx)
                    m_aug_id.append(cid)
                    m_orig_wi.append(global_idx)

                if len(buf_amp) >= FLUSH_EVERY:
                    _flush()

            _flush()

            if n_rejected:
                logger.warning(
                    f"[{split_name}] skipped {n_rejected} windows (NaN/shape mismatch)"
                )

            N  = write_cursor
            dt = h5py.string_dtype()
            if N > 0:
                meta_grp.create_dataset('recording_ids', dtype=dt,
                    data=np.array(m_rec_ids, dtype=object))
                meta_grp.create_dataset('person_ids', dtype=dt,
                    data=np.array(m_per_ids, dtype=object))
                meta_grp.create_dataset('room_ids', dtype=dt,
                    data=np.array(m_rom_ids, dtype=object))
                meta_grp.create_dataset('window_indices',
                    data=np.array(m_win_ids, dtype=np.int32))
                meta_grp.create_dataset('ref_antenna_idxs',
                    data=np.array(m_ref_ids, dtype=np.int8))
                meta_grp.create_dataset('aug_copy_id',
                    data=np.array(m_aug_id, dtype=np.int8))
                meta_grp.create_dataset('orig_win_idx',
                    data=np.array(m_orig_wi, dtype=np.int32))
            else:
                for name in ('recording_ids', 'person_ids', 'room_ids'):
                    meta_grp.create_dataset(name, shape=(0,), dtype=dt)
                for name in ('window_indices', 'orig_win_idx'):
                    meta_grp.create_dataset(name, shape=(0,), dtype=np.int32)
                meta_grp.create_dataset('ref_antenna_idxs', shape=(0,), dtype=np.int8)
                meta_grp.create_dataset('aug_copy_id',       shape=(0,), dtype=np.int8)

            split_counts[split_name] = N
            n_orig = len(entries) - n_rejected
            logger.info(
                f"  [{split_name}] {N:,} samples ({n_orig:,} orig × {1 + n_aug} copies)"
            )

        f.attrs['fold_id']          = fold_id
        f.attrs['dataset']          = dataset_name
        f.attrs['protocol']         = protocol
        f.attrs['pipeline_version'] = 'v7.2'
        f.attrs['fs']               = 350
        f.attrs['n_subcarriers']    = F_sub
        f.attrs['n_classes']        = 8
        f.attrs['n_aug_offline']    = n_aug_offline
        f.attrs['n_train']          = split_counts.get('train', 0)
        f.attrs['n_val']            = split_counts.get('val',   0)
        f.attrs['n_test']           = split_counts.get('test',  0)


def _write_hdf5(
    out_path     : str,
    train_recs   : List[Recording],
    val_recs     : List[Recording],
    test_recs    : List[Recording],
    stats        : dict,
    cfg_pre      : dict,
    fold_id      : int,
    dataset_name : str,
    protocol     : str = 'LOSO_person',
    n_aug_offline: int = 0,
    cfg_aug      : Optional[dict] = None,
):
    """Assemble split_windows from Recording objects, then call _write_hdf5_core."""
    T_window = cfg_pre.get('T_window', 350)
    stride   = cfg_pre.get('stride', 175)

    split_windows: Dict[str, List] = {'train': [], 'val': [], 'test': []}
    for split_name, recs in [
        ('train', train_recs), ('val', val_recs), ('test', test_recs)
    ]:
        for rec in recs:
            ref_idx = rec.metadata['ref_antenna_idx']
            for win_idx, win_H in enumerate(
                build_windows(rec, T=T_window, stride=stride)
            ):
                split_windows[split_name].append((
                    win_H,
                    rec.activity_label_int,
                    ref_idx,
                    rec.recording_id,
                    rec.person_id,
                    rec.room_id,
                    win_idx,
                ))

    _write_hdf5_core(
        out_path=out_path,
        split_windows=split_windows,
        stats=stats,
        cfg_pre=cfg_pre,
        fold_id=fold_id,
        dataset_name=dataset_name,
        protocol=protocol,
        n_aug_offline=n_aug_offline,
        cfg_aug=cfg_aug,
    )
