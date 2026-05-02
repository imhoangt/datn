"""
03_training_analysis.ipynb
Phân tích kết quả training: loss curves, F1 per epoch, confusion matrix.
"""
import sys; sys.path.insert(0, '..')
import numpy as np, matplotlib
matplotlib.use('Agg')
from pathlib import Path

CLASS_NAMES = ['walk','run','stand_up','sit_down','bend','fall','lying_still','empty']

def analyze_training(dataset: str = 'own', fold_id: int = 0, out_dir: str = '../outputs'):
    from src.utils.visualization import plot_training_curves, plot_confusion_matrix
    log_dir = Path(out_dir) / 'logs'
    fig_dir = Path(out_dir) / 'figures'

    csv_files = list(log_dir.glob(f'{dataset}_fold{fold_id:02d}*.csv'))
    if csv_files:
        plot_training_curves(str(csv_files[0]),
                             str(fig_dir / f'{dataset}_fold{fold_id:02d}_curves.png'),
                             fold_id)
        print(f"Training curves plotted.")
    else:
        print(f"No CSV log found for {dataset} fold {fold_id}.")

    # Confusion matrix from eval summary
    import json
    eval_json = Path(out_dir) / f'{dataset}_eval_summary.json'
    if eval_json.exists():
        with open(eval_json) as f:
            summary = json.load(f)
        fold_key = str(fold_id)
        if fold_key in summary.get('per_fold', {}):
            cm = np.array(summary['per_fold'][fold_key]['confusion_matrix'])
            plot_confusion_matrix(
                cm,
                str(fig_dir / f'{dataset}_fold{fold_id:02d}_confusion.png'),
                title=f'{dataset} Fold {fold_id} — Macro F1={summary["per_fold"][fold_key]["macro_f1"]:.4f}',
            )
            metrics = summary['per_fold'][fold_key]
            print(f"\n=== Fold {fold_id} Results ===")
            print(f"Macro F1 : {metrics['macro_f1']:.4f}")
            print(f"Fall Recall: {metrics.get('fall_recall',0):.4f}")
            print(f"Empty F1   : {metrics.get('empty_f1',0):.4f}")
            print(f"ECE        : {metrics.get('ece',0):.4f}")
            print("\nPer-class F1:")
            for cls, f1 in metrics.get('per_class_f1', {}).items():
                bar = '█' * int(f1 * 20)
                print(f"  {cls:15s}: {f1:.4f} {bar}")
    else:
        print(f"No eval summary found at {eval_json}. Run evaluate.py first.")

if __name__ == '__main__':
    analyze_training('own', 0)
