"""
02_preprocessing_verify.ipynb
Verify DFS spectrograms và Amplitude patterns — bắt buộc trước khi train.
"""
import sys; sys.path.insert(0, '..')
import numpy as np, h5py, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

CLASS_NAMES = ['walk','run','stand_up','sit_down','bend','fall','lying_still','empty']

def verify_dfs_spectrograms(hdf5_path: str, out_dir: str):
    """Plot one DFS spectrogram per class from train split."""
    from src.utils.visualization import plot_dfs_spectrograms
    examples = {}
    with h5py.File(hdf5_path, 'r') as f:
        labels = f['train']['labels'][:]
        X_dfs  = f['train']['X_dfs']
        for c in range(8):
            idxs = np.where(labels == c)[0]
            if len(idxs) > 0:
                examples[c] = X_dfs[idxs[0]]   # [28,128,3]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plot_dfs_spectrograms(examples, f'{out_dir}/dfs_verification.png')
    print("DFS verification plot saved.")
    print("\nExpected signatures:")
    for i, (cls, sig) in enumerate(zip(CLASS_NAMES, [
        'Oscillating 1–3Hz','Strong 3–8Hz','Directional spike','Directional spike',
        'Moderate 0.5–2Hz','Broadband <0.5s','Near-dark 0.2–0.3Hz','Dark (noise only)'
    ])):
        print(f"  {cls:15s}: {sig}")

def verify_amp_patterns(hdf5_path: str, out_dir: str):
    """Plot mean amplitude envelope per class."""
    with h5py.File(hdf5_path, 'r') as f:
        labels = f['train']['labels'][:]
        X_amp  = f['train']['X_amp']

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        axes = axes.flatten()
        for c in range(8):
            idxs = np.where(labels == c)[0][:10]
            if len(idxs) == 0: continue
            samples = X_amp[list(idxs)]               # [N,350,52,3,4]
            env = np.abs(samples).mean(axis=(2,3,4))  # [N,350]
            mean_env = env.mean(axis=0)
            std_env  = env.std(axis=0)
            t = np.arange(350) / 350.0
            axes[c].plot(t, mean_env, linewidth=1.5)
            axes[c].fill_between(t, mean_env-std_env, mean_env+std_env, alpha=0.2)
            axes[c].set_title(CLASS_NAMES[c])
            axes[c].set_xlabel('Time (s)')
            axes[c].set_ylabel('Amplitude')
            axes[c].grid(True, alpha=0.3)

    plt.suptitle('Mean Amplitude Envelope per Class', fontsize=13)
    plt.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{out_dir}/amplitude_verification.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Amplitude verification plot saved.")

if __name__ == '__main__':
    hdf5_dir = '../data/processed/own'
    from pathlib import Path as P
    if not (P(hdf5_dir) / 'fold_00.h5').exists():
        from src.utils.mock_generator import generate_mock_dataset_hdf5
        P(hdf5_dir).mkdir(parents=True, exist_ok=True)
        generate_mock_dataset_hdf5(f'{hdf5_dir}/fold_00.h5')

    hdf5 = f'{hdf5_dir}/fold_00.h5'
    verify_dfs_spectrograms(hdf5, '../outputs/figures')
    verify_amp_patterns(hdf5, '../outputs/figures')
