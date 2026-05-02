"""
src/evaluation/metrics.py   — per-fold and per-class metrics
src/evaluation/benchmark.py — 3-dataset comparison table
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ['walk', 'run', 'stand_up', 'sit_down', 'bend', 'fall', 'lying_still', 'empty']


# ============================================================================
# metrics.py
# ============================================================================

def compute_fold_metrics(
    probs : np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Dict:
    """
    Compute all metrics for one fold.

    Returns dict with:
        macro_f1, per_class_f1, fall_recall, empty_f1,
        confusion_matrix, classification_report
    """
    preds = probs.argmax(-1)
    n_cl  = probs.shape[-1]

    # Overall macro F1
    macro_f1 = float(f1_score(labels, preds, average='macro', zero_division=0))

    # Per-class P / R / F1
    prec, rec, f1, supp = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0,
        labels=list(range(n_cl)),
    )
    per_class = {
        class_names[c]: {
            'precision': float(prec[c]),
            'recall'   : float(rec[c]),
            'f1'       : float(f1[c]),
            'support'  : int(supp[c]),
        }
        for c in range(min(n_cl, len(class_names)))
    }

    # Special metrics
    fall_idx = class_names.index('fall') if 'fall' in class_names else 5
    fall_recall = float(rec[fall_idx]) if fall_idx < len(rec) else 0.0

    empty_idx = class_names.index('empty') if 'empty' in class_names else 7
    empty_f1  = float(f1[empty_idx]) if empty_idx < len(f1) else 0.0

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(n_cl))).tolist()

    # ECE (Expected Calibration Error)
    ece = _compute_ece(probs, labels)

    return {
        'macro_f1'              : macro_f1,
        'fall_recall'           : fall_recall,
        'empty_f1'              : empty_f1,
        'per_class_f1'          : {k: v['f1'] for k, v in per_class.items()},
        'per_class_metrics'     : per_class,
        'confusion_matrix'      : cm,
        'ece'                   : ece,
        'n_samples'             : int(len(labels)),
    }


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (10-bin ECE)."""
    preds    = probs.argmax(-1)
    confs    = probs.max(-1)
    correct  = (preds == labels).astype(float)
    bins     = np.linspace(0, 1, n_bins + 1)
    ece      = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() > 0:
            avg_conf = confs[mask].mean()
            avg_acc  = correct[mask].mean()
            ece     += mask.sum() / len(labels) * abs(avg_conf - avg_acc)
    return float(ece)


def aggregate_recording_preds(
    window_probs  : np.ndarray,
    recording_ids : np.ndarray,
    ground_truth  : Dict[str, int],
) -> tuple:
    """
    Mean-pool window probabilities per recording.
    Recording-level F1 typically 2–5% higher than window-level.
    """
    from collections import defaultdict
    rec_probs = defaultdict(list)
    for i, rid in enumerate(recording_ids):
        rec_probs[rid].append(window_probs[i])

    rec_ids    = list(rec_probs.keys())
    rec_labels = np.array([ground_truth[rid] for rid in rec_ids])
    rec_means  = np.stack([np.stack(v).mean(0) for v in rec_probs.values()])
    return rec_labels, rec_means


# ============================================================================
# benchmark.py
# ============================================================================

def build_comparison_table(
    result_files: Dict[str, str],
    out_path    : Optional[str] = None,
    class_names : List[str] = CLASS_NAMES,
) -> str:
    """
    Build 3-dataset comparison table from LOSO summary JSON files.

    result_files: {'own': path_to_json, 'xrf55': ..., 'exposing': ...}
    Returns: formatted table as string (also printed and optionally saved).
    """
    rows = []
    for ds_name, json_path in result_files.items():
        if not Path(json_path).exists():
            logger.warning(f"Result not found: {json_path}")
            continue
        with open(json_path) as f:
            summary = json.load(f)

        row = {
            'Dataset'       : ds_name,
            'Macro F1 (mean)': f"{summary['mean_macro_f1']*100:.2f}%",
            'Macro F1 (std)' : f"±{summary['std_macro_f1']*100:.2f}%",
            'Fall Recall'   : f"{summary['mean_fall_recall']*100:.2f}%",
        }

        # Per-class F1 from best fold (highest macro F1)
        best_fold = max(
            summary['per_fold'].items(),
            key=lambda x: x[1]['macro_f1'],
        )
        pf1 = best_fold[1].get('per_class_f1', {})
        for cls in class_names:
            row[cls] = f"{pf1.get(cls, 0)*100:.1f}%" if cls in pf1 else '-'

        rows.append(row)

    if not rows:
        return "No results available."

    # Format table
    cols     = list(rows[0].keys())
    col_w    = {c: max(len(c), max(len(str(r.get(c,'-'))) for r in rows)) for c in cols}
    sep      = '+' + '+'.join('-' * (col_w[c] + 2) for c in cols) + '+'
    header   = '|' + '|'.join(f" {c:<{col_w[c]}} " for c in cols) + '|'

    lines = [sep, header, sep]
    for r in rows:
        line = '|' + '|'.join(f" {str(r.get(c,'-')):<{col_w[c]}} " for c in cols) + '|'
        lines.append(line)
    lines.append(sep)

    table_str = '\n'.join(lines)
    print('\n=== 3-Dataset LOSO Comparison ===')
    print(table_str)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(table_str)
        logger.info(f"Comparison table saved: {out_path}")

    return table_str
