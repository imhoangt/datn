"""
src/datasets/base_dataset.py
Base PyTorch Dataset that reads from HDF5 files produced by pipeline_*.py.
Applies on-the-fly augmentation (H1b CutMix applied in training loop instead).

src/datasets/own_dataset.py   → thin wrappers calling CHARMDataset
src/datasets/xrf55_dataset.py
src/datasets/exposing_dataset.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class CHARMDataset(Dataset):
    """
    PyTorch Dataset backed by a fold HDF5 file.

    Returns: (X_amp [350,52,3,4], X_dfs [28,128,3], label)

    Augmentation (H1a subcarrier masking, H2 DFS masking) is done here.
    H1b CutMix-Time is done in the training loop (needs paired samples).

    Args:
        hdf5_path   : path to fold_XX.h5
        split       : 'train' | 'val' | 'test'
        augment     : if True, apply H1a + H2 augmentations
        global_epoch: shared int (set externally before each epoch) for CutMix threshold
        cfg_aug     : augmentation config dict
    """

    def __init__(
        self,
        hdf5_path   : str,
        split       : str = 'train',
        augment     : bool = False,
        cfg_aug     : Optional[dict] = None,
    ):
        super().__init__()
        assert split in ('train', 'val', 'test')
        self.hdf5_path    = hdf5_path
        self.split        = split
        self.augment      = augment
        self.cfg_aug      = cfg_aug or {}
        self.global_epoch = 0   # set externally per epoch

        # Load metadata only (data loaded lazily per __getitem__)
        with h5py.File(hdf5_path, 'r') as f:
            grp          = f[split]
            self.labels  = grp['labels'][:]              # [N] int8
            self.n       = len(self.labels)
            # Optional: cache small metadata
            meta         = grp.get('metadata', {})
            self.rec_ids = meta['recording_ids'][:] if 'recording_ids' in meta else None
            self.per_ids = meta['person_ids'][:]    if 'person_ids'    in meta else None

        self._file : Optional[h5py.File] = None   # opened lazily in worker

    # ------------------------------------------------------------------
    def _open(self):
        """Open HDF5 in worker process (not in main process to avoid fork issues)."""
        if self._file is None or not self._file.id.valid:
            try:
                self._file = h5py.File(self.hdf5_path, 'r', swmr=True)
            except Exception:
                # Files written without libver='latest' don't support SWMR
                self._file = h5py.File(self.hdf5_path, 'r')

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        self._open()
        grp   = self._file[self.split]
        X_amp = grp['X_amp'][idx]   # [350, 52, 3, 4] float32
        X_dfs = grp['X_dfs'][idx]   # [28, 128, 3]    float32
        label = int(self.labels[idx])

        # On-the-fly augmentation (H1a + H2 only; H1b done in training loop)
        if self.augment:
            X_amp, X_dfs = self._augment(X_amp, X_dfs)

        return (
            torch.from_numpy(X_amp.astype(np.float32)),
            torch.from_numpy(X_dfs.astype(np.float32)),
            label,
        )

    def _augment(
        self,
        X_amp: np.ndarray,
        X_dfs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """H1a subcarrier masking + H2 DFS masking."""
        cfg = self.cfg_aug

        # H1a: Subcarrier masking
        p_mask   = cfg.get('subcarrier_mask_prob', 0.4)
        mask_min = cfg.get('subcarrier_mask_min', 3)
        mask_max = cfg.get('subcarrier_mask_max', 9)
        if np.random.rand() < p_mask:
            n_mask  = int(np.random.randint(mask_min, mask_max + 1))
            f_start = int(np.random.randint(0, X_amp.shape[1] - n_mask))
            X_amp = X_amp.copy()
            X_amp[:, f_start:f_start + n_mask, :, :] = 0.0

        # H2a: Doppler masking
        if np.random.rand() < cfg.get('doppler_mask_prob', 0.3):
            n_m = int(np.random.randint(2, 7))
            v_s = int(np.random.randint(0, X_dfs.shape[1] - n_m))
            X_dfs = X_dfs.copy()
            X_dfs[:, v_s:v_s + n_m, :] = 0.0

        # H2b: Time masking
        if np.random.rand() < cfg.get('time_mask_prob', 0.3):
            t_m = int(np.random.randint(1, 4))
            t_s = int(np.random.randint(0, X_dfs.shape[0] - t_m))
            X_dfs = X_dfs.copy()
            X_dfs[t_s:t_s + t_m, :, :] = 0.0

        return X_amp, X_dfs

    def __del__(self):
        if self._file is not None and self._file.id.valid:
            self._file.close()


def build_loaders(
    hdf5_path   : str,
    batch_size  : int = 16,
    num_workers : int = 4,
    cfg_aug     : Optional[dict] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple:
    """
    Build train/val/test DataLoaders from a fold HDF5.

    WeightedRandomSampler on train for class imbalance.
    persistent_workers=False so global_epoch is visible to workers.
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_ds = CHARMDataset(hdf5_path, 'train', augment=True,  cfg_aug=cfg_aug)
    val_ds   = CHARMDataset(hdf5_path, 'val',   augment=False)
    test_ds  = CHARMDataset(hdf5_path, 'test',  augment=False)

    # WeightedRandomSampler for train
    sampler = None
    if class_weights is not None and len(train_ds) > 0:
        labels    = torch.tensor(train_ds.labels.astype(np.int64))
        sample_w  = class_weights[labels]
        sampler   = WeightedRandomSampler(
            sample_w.numpy(), len(sample_w), replacement=True
        )

    train_loader = DataLoader(
        train_ds,
        batch_size        = batch_size,
        sampler           = sampler,
        shuffle           = sampler is None,
        num_workers       = num_workers,
        pin_memory        = True,
        drop_last         = True,
        persistent_workers= False,  # False so global_epoch visible
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    return train_loader, val_loader, test_loader, train_ds
