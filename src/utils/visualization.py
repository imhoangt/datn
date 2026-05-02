"""
src/utils/visualization.py
Visualization: DFS spectrograms, confusion matrix, training curves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES = ['walk', 'run', 'stand_up', 'sit_down', 'bend', 'fall', 'lying_still', 'empty']


def plot_dfs_spectrograms(
    X_dfs_examples : Dict[int, np.ndarray],
    out_path       : str,
    class_names    : List[str] = CLASS_NAMES,
    fs_out         : float = 350.0,
    n_fft          : int   = 128,
) -> None:
    """
    Plot DFS spectrogram for each class (one example each).

    X_dfs_examples: {label_int: X_dfs [28, 128, 3]}
    """
    n_cls = len(class_names)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    freq_bins = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0/fs_out))

    for i, (label, X_dfs) in enumerate(sorted(X_dfs_examples.items())):
        if i >= n_cls:
            break
        ax = axes[i]
        # Average over receivers
        S_avg = X_dfs.mean(axis=-1)   # [28, 128]
        im = ax.imshow(
            S_avg.T,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            extent=[0, 1.0, freq_bins[0], freq_bins[-1]],
            vmin=np.percentile(S_avg, 5),
            vmax=np.percentile(S_avg, 95),
        )
        ax.set_title(class_names[label], fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Doppler (Hz)')
        ax.axhline(0, color='white', linewidth=0.5, linestyle='--')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused axes
    for j in range(len(X_dfs_examples), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('DFS Spectrograms by Class (Verification)', fontsize=14, y=1.01)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"DFS spectrograms saved: {out_path}")


def plot_confusion_matrix(
    cm          : np.ndarray,
    out_path    : str,
    class_names : List[str] = CLASS_NAMES,
    title       : str = 'Confusion Matrix',
    normalize   : bool = True,
) -> None:
    """Plot and save confusion matrix."""
    import seaborn as sns

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)
        fmt      = '.2f'
        vmax     = 1.0
    else:
        cm_plot  = cm
        fmt      = 'd'
        vmax     = None

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_plot,
        annot=True, fmt=fmt,
        xticklabels=class_names, yticklabels=class_names,
        cmap='Blues', vmin=0, vmax=vmax, ax=ax,
        linewidths=0.5, linecolor='lightgray',
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {out_path}")


def plot_training_curves(
    csv_path : str,
    out_path : str,
    fold_id  : int,
) -> None:
    """Plot loss and F1 curves from training CSV log."""
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"Could not read CSV: {e}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Losses (train)
    train_df = df[df['phase'] == 'train']
    if 'total' in train_df.columns:
        axes[0].plot(train_df['epoch'], train_df['total'], label='Total')
        axes[0].plot(train_df['epoch'], train_df.get('L_main', 0), label='L_main', linestyle='--')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Training Loss (fold {fold_id})')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Validation F1
    val_df = df[df['phase'] == 'val']
    if 'macro_f1' in val_df.columns:
        axes[1].plot(val_df['epoch'], val_df['macro_f1'], color='orange')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Macro F1')
        axes[1].set_title(f'Validation Macro F1 (fold {fold_id})')
        axes[1].grid(True, alpha=0.3)

    # Learning rate
    if 'lr' in train_df.columns:
        axes[2].semilogy(train_df['epoch'], train_df['lr'], color='green')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('LR (log scale)')
        axes[2].set_title(f'Learning Rate Schedule (fold {fold_id})')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Training curves saved: {out_path}")


def plot_loso_summary(
    summary_json: str,
    out_path    : str,
    class_names : List[str] = CLASS_NAMES,
) -> None:
    """Bar chart of per-fold macro F1 and overall mean±std."""
    import json
    with open(summary_json) as f:
        summary = json.load(f)

    fold_ids = sorted(summary['per_fold'].keys(), key=lambda x: int(x))
    f1s      = [summary['per_fold'][k]['macro_f1'] for k in fold_ids]
    mean_f1  = summary['mean_macro_f1']
    std_f1   = summary['std_macro_f1']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fold_ids))
    bars = ax.bar(x, f1s, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(mean_f1, color='red', linestyle='--', linewidth=2,
               label=f'Mean={mean_f1:.4f} ±{std_f1:.4f}')
    ax.fill_between([-0.5, len(fold_ids)-0.5],
                    mean_f1 - std_f1, mean_f1 + std_f1,
                    alpha=0.15, color='red')
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {k}' for k in fold_ids])
    ax.set_ylim(0, min(1.1, max(f1s) + 0.1))
    ax.set_ylabel('Macro F1')
    ax.set_title('LOSO Cross-Validation Results')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"LOSO summary plot saved: {out_path}")
