"""
src/datasets/base_dataset.py
Base PyTorch Dataset that reads from HDF5 files produced by pipeline_*.py.

All augmentation (E-stage + H1a/H2) is done offline during preprocessing and
stored in HDF5. No on-the-fly augmentation is applied here.

src/datasets/own_dataset.py   → thin wrappers calling CHARMDataset
src/datasets/xrf55_dataset.py
src/datasets/exposing_dataset.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class CHARMDataset(Dataset):
    """
    PyTorch Dataset backed by a fold HDF5 file.

    Returns: (X_amp [350,52,3,4], X_dfs [28,128,3], label)

    All augmentations (E-stage + H1a/H2) are stored in HDF5 at preprocessing time.

    Args:
        hdf5_path: path to fold_XX.h5
        split    : 'train' | 'val' | 'test'
    """

    def __init__(
        self,
        hdf5_path   : str,
        split       : str = 'train',
    ):
        super().__init__()
        assert split in ('train', 'val', 'test')
        self.hdf5_path = hdf5_path
        self.split     = split

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

        return (
            torch.from_numpy(X_amp.astype(np.float32)),
            torch.from_numpy(X_dfs.astype(np.float32)),
            label,
        )

    def __del__(self):
        if self._file is not None and self._file.id.valid:
            self._file.close()


def build_loaders(
    hdf5_path    : str,
    batch_size   : int = 16,
    num_workers  : int = 4,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple:
    """Build train/val/test DataLoaders from a fold HDF5.

    WeightedRandomSampler on train for class imbalance.
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_ds = CHARMDataset(hdf5_path, 'train')
    val_ds   = CHARMDataset(hdf5_path, 'val')
    test_ds  = CHARMDataset(hdf5_path, 'test')

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
        persistent_workers= False,
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
