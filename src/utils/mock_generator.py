"""
src/utils/mock_generator.py
Generate synthetic CSI data for testing the full pipeline end-to-end
before real data is available.

Generates realistic (but synthetic) CSI tensors with:
- Per-class amplitude patterns (e.g. walk=oscillating, fall=spike)
- Realistic noise, AGC variation, and packet gaps
- Correct tensor shapes for all dataset types
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES = ['walk', 'run', 'stand_up', 'sit_down', 'bend', 'fall', 'lying_still', 'empty']

# Rough DFS frequency signatures per class (Hz) for synthetic DFS
CLASS_DFS_FREQ = {
    0: 1.5,   # walk  ~1.5Hz Doppler
    1: 5.0,   # run   ~5Hz
    2: 1.0,   # stand_up transient
    3: 1.0,   # sit_down transient
    4: 0.8,   # bend
    5: 3.0,   # fall (broadband, brief)
    6: 0.3,   # lying_still (breathing)
    7: 0.05,  # empty (thermal noise only)
}


def _make_synthetic_h_uniform(
    label     : int,
    T_out     : int = 1749,
    F         : int = 52,
    M         : int = 3,
    A         : int = 4,
    fs        : float = 350.0,
    snr_db    : float = 20.0,
    rng       : Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate synthetic H_uniform [T_out, F, M, A] complex64 for one recording.

    Simulates activity-dependent Doppler by modulating amplitude with
    a sinusoid at the class-specific DFS frequency.
    """
    if rng is None:
        rng = np.random.default_rng()

    t    = np.arange(T_out) / fs
    f_d  = CLASS_DFS_FREQ.get(label, 0.5)

    # Base amplitude: static + activity Doppler modulation
    if label == 7:  # empty
        amp_env = 0.1 * np.ones(T_out)
    elif label == 5:  # fall: spike in first 0.5s then quiet
        amp_env = np.zeros(T_out)
        fall_len = int(0.4 * fs)
        amp_env[:fall_len] = np.hanning(fall_len) * 2.0
    else:
        amp_env = 0.5 + 0.3 * np.sin(2 * np.pi * f_d * t)

    H = np.zeros((T_out, F, M, A), dtype=np.complex64)
    noise_std = 10.0 ** (-snr_db / 20.0)

    for m in range(M):
        for a in range(A):
            # Per-path phase offset (static multipath)
            phase_offset = rng.uniform(0, 2 * np.pi, F)
            for f_idx in range(F):
                phase = 2 * np.pi * f_d * t + phase_offset[f_idx]
                real  = amp_env * np.cos(phase) + noise_std * rng.standard_normal(T_out)
                imag  = amp_env * np.sin(phase) + noise_std * rng.standard_normal(T_out)
                H[:, f_idx, m, a] = (real + 1j * imag).astype(np.complex64)

    # Add AGC variation: scale per-packet
    agc_scale = rng.uniform(0.8, 1.2, (T_out, 1, 1, 1)).astype(np.float32)
    H = (H * agc_scale).astype(np.complex64)

    return H


def generate_mock_dataset_hdf5(
    out_path     : str,
    n_persons    : int = 8,
    n_recs_per   : int = 8,    # per person per class
    n_classes    : int = 8,
    T_out        : int = 1749,
    fold_id      : int = 0,
    seed         : int = 42,
) -> str:
    """
    Generate a mock HDF5 dataset for testing.

    Structure matches pipeline_own.py output exactly.
    LOSO split: test=person[fold_id], val=person[(fold_id+1)%n_persons]

    Returns: path to HDF5 file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Build recording metadata
    records = []
    for p in range(n_persons):
        for c in range(n_classes):
            for r in range(n_recs_per):
                records.append({
                    'person_id'   : f'P{p}',
                    'label'       : c,
                    'rec_id'      : f'P{p}_{CLASS_NAMES[c]}_rep{r:02d}',
                })

    # LOSO split
    test_pid  = f'P{fold_id % n_persons}'
    val_pid   = f'P{(fold_id + 1) % n_persons}'
    train_pids = {f'P{p}' for p in range(n_persons)} - {test_pid, val_pid}

    splits = {
        'train': [r for r in records if r['person_id'] in train_pids],
        'val'  : [r for r in records if r['person_id'] == val_pid],
        'test' : [r for r in records if r['person_id'] == test_pid],
    }

    # Stats (dummy but correct shape)
    dummy_stats = {
        'zscore_mu'           : np.zeros((52, 3, 4), dtype=np.float32),
        'zscore_sigma'        : np.ones( (52, 3, 4), dtype=np.float32),
        'fisher_scores'       : rng.random(52).astype(np.float32),
        'selected_subcarriers': np.array(
            [g*4 + rng.integers(0,4) for g in range(13)], dtype=np.int32
        ),
    }

    T_window = 350
    stride   = 175
    n_wins_per_rec = (T_out - T_window) // stride + 1  # = 9

    with h5py.File(out_path, 'w', libver='latest') as f:
        for split_name, recs in splits.items():
            n_wins = len(recs) * n_wins_per_rec

            X_amp_all = np.zeros((n_wins, T_window, 52, 3, 4), dtype=np.float32)
            X_dfs_all = np.zeros((n_wins, 28, 128, 3), dtype=np.float32)
            labels    = np.zeros(n_wins, dtype=np.int8)
            rec_ids   = []
            per_ids   = []

            idx = 0
            for rec in records:
                if rec not in recs:
                    continue
                H_uni = _make_synthetic_h_uniform(
                    rec['label'], T_out=T_out, rng=rng
                )

                # Slide windows
                start = 0
                T_uni = H_uni.shape[0]
                while start + T_window <= T_uni:
                    win_H = H_uni[start:start + T_window]
                    # Simple feature extraction (no filter, just abs for mock)
                    X_amp_all[idx] = np.abs(win_H).astype(np.float32)
                    # Mock DFS: random spectrogram
                    X_dfs_all[idx] = rng.standard_normal((28, 128, 3)).astype(np.float32)
                    labels[idx]    = rec['label']
                    rec_ids.append(rec['rec_id'])
                    per_ids.append(rec['person_id'])
                    idx   += 1
                    start += stride

            X_amp_all = X_amp_all[:idx]
            X_dfs_all = X_dfs_all[:idx]
            labels    = labels[:idx]

            grp = f.create_group(split_name)
            grp.create_dataset('X_amp',  data=X_amp_all, compression='gzip',
                               compression_opts=1, chunks=(1,350,52,3,4))
            grp.create_dataset('X_dfs',  data=X_dfs_all, compression='gzip',
                               compression_opts=1, chunks=(1,28,128,3))
            grp.create_dataset('labels', data=labels, dtype=np.int8)
            meta_grp = grp.create_group('metadata')
            dt = h5py.string_dtype()
            meta_grp.create_dataset('recording_ids',
                data=np.array(rec_ids[:idx], dtype=object), dtype=dt)
            meta_grp.create_dataset('person_ids',
                data=np.array(per_ids[:idx], dtype=object), dtype=dt)
            meta_grp.create_dataset('ref_antenna_idxs',
                data=np.zeros(idx, dtype=np.int8))

        # Stats
        sg = f.create_group('stats')
        for k, v in dummy_stats.items():
            sg.create_dataset(k, data=v)

        f.attrs.update({
            'fold_id': fold_id, 'dataset': 'mock', 'protocol': 'LOSO_person',
            'pipeline_version': 'v6.6', 'fs': 350, 'n_subcarriers': 52,
            'n_classes': n_classes,
        })

    logger.info(f"Mock HDF5 written: {out_path}")
    return str(out_path)
