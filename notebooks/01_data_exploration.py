# ---
# jupyter:
#   jupytext:
#     cell_type: code
# ---
"""
01_data_exploration.ipynb
Khám phá dataset: thống kê phân phối class, packet rates, kiểm tra .pcap.
"""

import sys; sys.path.insert(0, '..')
import numpy as np, h5py, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

CLASS_NAMES = ['walk','run','stand_up','sit_down','bend','fall','lying_still','empty']

# ── Cell 1: Kiểm tra HDF5 đã có chưa ─────────────────────────────────────────
def check_hdf5(hdf5_path: str):
    with h5py.File(hdf5_path, 'r') as f:
        print(f"=== {hdf5_path} ===")
        print(f"Dataset: {f.attrs.get('dataset')}, Protocol: {f.attrs.get('protocol')}")
        print(f"Pipeline version: {f.attrs.get('pipeline_version')}")
        for split in ['train', 'val', 'test']:
            if split in f:
                n = len(f[split]['labels'])
                dist = Counter(f[split]['labels'][:].tolist())
                print(f"\n  [{split}] N={n}")
                for c in range(8):
                    print(f"    {CLASS_NAMES[c]:15s}: {dist.get(c,0):5d}")

# ── Cell 2: Nếu chưa có HDF5, tạo mock để test ───────────────────────────────
def make_mock_if_needed(hdf5_dir: str):
    p = Path(hdf5_dir)
    p.mkdir(parents=True, exist_ok=True)
    if not (p / 'fold_00.h5').exists():
        print("No HDF5 found — generating mock data for testing...")
        from src.utils.mock_generator import generate_mock_dataset_hdf5
        for fold_id in range(2):
            generate_mock_dataset_hdf5(
                str(p / f'fold_{fold_id:02d}.h5'),
                n_persons=8, n_recs_per=4, fold_id=fold_id
            )
        print("Mock data created.")

# ── Cell 3: Class distribution plot ──────────────────────────────────────────
def plot_class_distribution(hdf5_path: str, out_path: str):
    with h5py.File(hdf5_path, 'r') as f:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, split in zip(axes, ['train', 'val', 'test']):
            if split not in f:
                continue
            labels = f[split]['labels'][:]
            counts = [int((labels == c).sum()) for c in range(8)]
            bars   = ax.bar(CLASS_NAMES, counts, color='steelblue', edgecolor='black', lw=0.5)
            for bar, cnt in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        str(cnt), ha='center', fontsize=8)
            ax.set_title(f'{split.capitalize()} Split')
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
            ax.set_ylabel('Windows')
            ax.grid(True, axis='y', alpha=0.3)
        plt.suptitle('Class Distribution per Split', fontsize=13, y=1.02)
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    hdf5_dir = '../data/processed/own'
    make_mock_if_needed(hdf5_dir)
    hdf5_path = str(Path(hdf5_dir) / 'fold_00.h5')
    check_hdf5(hdf5_path)
    plot_class_distribution(hdf5_path, '../outputs/figures/fold00_class_distribution.png')
