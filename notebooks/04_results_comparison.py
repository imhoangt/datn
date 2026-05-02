"""
04_results_comparison.ipynb
So sánh kết quả LOSO trên 3 datasets. Bảng kết quả cho báo cáo đồ án.
"""
import sys; sys.path.insert(0, '..')
import json, numpy as np
from pathlib import Path

CLASS_NAMES = ['walk','run','stand_up','sit_down','bend','fall','lying_still','empty']


def compare_3_datasets(out_dir: str = '../outputs'):
    from src.evaluation.metrics import build_comparison_table
    result_files = {}
    for ds in ['own', 'xrf55', 'exposing']:
        p = Path(out_dir) / f'{ds}_loso_summary.json'
        if p.exists():
            result_files[ds] = str(p)
        else:
            print(f"[Missing] {p} — run train.py + evaluate.py for {ds} first")

    if result_files:
        build_comparison_table(
            result_files,
            out_path=str(Path(out_dir) / 'figures' / '3dataset_comparison.txt'),
        )

def print_latex_table(out_dir: str = '../outputs'):
    """Generate LaTeX table for thesis report."""
    rows = []
    for ds in ['own', 'xrf55', 'exposing']:
        p = Path(out_dir) / f'{ds}_loso_summary.json'
        if not p.exists():
            continue
        with open(p) as f:
            s = json.load(f)
        row = [
            ds.upper(),
            f"{s['mean_macro_f1']*100:.2f}",
            f"{s['std_macro_f1']*100:.2f}",
            f"{s['mean_fall_recall']*100:.2f}",
        ]
        # Best fold per-class
        best_fold = max(s['per_fold'].items(), key=lambda x: x[1]['macro_f1'])
        pf1 = best_fold[1].get('per_class_f1', {})
        for cls in CLASS_NAMES:
            row.append(f"{pf1.get(cls, 0)*100:.1f}")
        rows.append(row)

    if not rows:
        print("No results to display. Run training first.")
        return

    header = ['Dataset', 'MF1 (%)', '±Std', 'Fall R'] + CLASS_NAMES
    print('\n=== LaTeX Table ===')
    print('\\begin{tabular}{l' + 'c'*(len(header)-1) + '}')
    print('\\hline')
    print(' & '.join(header) + ' \\\\')
    print('\\hline')
    for row in rows:
        print(' & '.join(row) + ' \\\\')
    print('\\hline')
    print('\\end{tabular}')

if __name__ == '__main__':
    compare_3_datasets()
    print_latex_table()
