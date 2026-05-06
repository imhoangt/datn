"""
scripts/evaluate.py
Load best checkpoint and evaluate on test set.

Usage:
    python scripts/evaluate.py --dataset own --fold 0
    python scripts/evaluate.py --dataset own --fold all --plot
"""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['own', 'xrf55', 'exposing'], required=True)
    parser.add_argument('--fold',    default='all')
    parser.add_argument('--output',  default='outputs/')
    parser.add_argument('--plot',    action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s — %(message)s')
    logger = logging.getLogger(__name__)

    from scripts.train import load_config, set_seed
    from src.models.charm_net import build_model
    from src.datasets.base_dataset import build_loaders, CHARMDataset
    from src.training.loso_runner import load_checkpoint
    from src.training.losses import evaluate_with_ema
    from src.evaluation.metrics import compute_fold_metrics, CLASS_NAMES
    from src.models.heads import TemperatureScaler
    from torch.utils.data import DataLoader

    try:
        from torch_ema import ExponentialMovingAverage
    except ImportError:
        logger.error("torch-ema not installed: pip install torch-ema")
        sys.exit(1)

    cfg    = load_config(args.dataset)
    set_seed(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processed_dir = Path(cfg['dataset']['processed_dir'])
    out_dir       = Path(args.output)
    ckpt_dir      = out_dir / 'checkpoints'
    fig_dir       = out_dir / 'figures'
    class_names   = cfg.get('dataset', {}).get('class_names', CLASS_NAMES)

    n_folds = cfg.get('evaluation', {}).get('n_folds', 8)
    folds   = list(range(n_folds)) if args.fold == 'all' else [int(args.fold)]

    all_results = {}
    for fold_id in folds:
        hdf5 = processed_dir / f"fold_{fold_id:02d}.h5"
        ckpt = ckpt_dir / f"fold_{fold_id:02d}_best.pt"
        if not hdf5.exists():
            logger.warning(f"HDF5 missing: {hdf5}")
            continue
        if not ckpt.exists():
            logger.warning(f"Checkpoint missing: {ckpt}")
            continue

        _, _, test_loader, _ = build_loaders(
            str(hdf5), batch_size=32, num_workers=2,
        )

        model = build_model(cfg.get('model', {}), device)
        ema   = ExponentialMovingAverage(model.parameters(), decay=0.999)
        load_checkpoint(str(ckpt), model, ema, device=device)

        # Calibrate temperature scaler on val set (consistent with loso_runner.py)
        val_ds     = CHARMDataset(str(hdf5), 'val')
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
        ts = TemperatureScaler()
        ts.calibrate(model, val_loader, device)
        logger.info(f"  Fold {fold_id} temperature: {ts.temperature.item():.4f}")

        _, probs, labels = evaluate_with_ema(model, ema, test_loader, device, ts)
        metrics = compute_fold_metrics(probs, labels, class_names=class_names)
        all_results[fold_id] = metrics

        logger.info(
            f"Fold {fold_id}: macro_f1={metrics['macro_f1']:.4f} "
            f"fall_recall={metrics.get('fall_recall', 0):.4f}"
        )

        if args.plot:
            from src.utils.visualization import plot_confusion_matrix, plot_loso_summary
            cm = np.array(metrics['confusion_matrix'])
            plot_confusion_matrix(
                cm, str(fig_dir / f"{args.dataset}_fold{fold_id:02d}_cm.png"),
                class_names=class_names,
                title=f'{args.dataset} Fold {fold_id} Confusion Matrix',
            )

    # Summary
    if all_results:
        f1s = [r['macro_f1'] for r in all_results.values()]
        frs = [r.get('fall_recall', 0) for r in all_results.values()]
        summary = {
            'mean_macro_f1'    : float(np.mean(f1s)),
            'std_macro_f1'     : float(np.std(f1s)),
            'mean_fall_recall' : float(np.mean(frs)),
            'per_fold'         : {str(k): v for k, v in all_results.items()},
        }
        out_json = out_dir / f"{args.dataset}_eval_summary.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(summary, f, indent=2, default=float)

        print(f"\n[{args.dataset}] Macro F1: {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
        print(f"[{args.dataset}] Fall Recall: {np.mean(frs)*100:.2f}%")

        if args.plot:
            from src.utils.visualization import plot_loso_summary
            plot_loso_summary(
                str(out_json),
                str(fig_dir / f"{args.dataset}_loso_summary.png"),
            )


if __name__ == '__main__':
    main()
