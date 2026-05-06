"""
scripts/train.py
Main training script. Run with:
    python scripts/train.py --dataset own    --fold 0
    python scripts/train.py --dataset xrf55  --fold all
    python scripts/train.py --dataset exposing
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_config(dataset: str) -> dict:
    root    = Path(__file__).parent.parent
    with open(root / 'configs' / 'base.yaml') as f:
        base = yaml.safe_load(f)
    ds_path = root / 'configs' / f'dataset_{dataset}.yaml'
    if ds_path.exists():
        with open(ds_path) as f:
            ds_cfg = yaml.safe_load(f)
        # Deep merge dataset config over base
        def _merge(base, override):
            for k, v in override.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    _merge(base[k], v)
                else:
                    base[k] = v
        _merge(base, ds_cfg)
    return base


def main():
    parser = argparse.ArgumentParser(description='CHARM-Net v7.2 Training')
    parser.add_argument('--dataset', choices=['own', 'xrf55', 'exposing'], required=True)
    parser.add_argument('--fold',    default='all',
                        help='Fold id (0-7) or "all" for full LOSO')
    parser.add_argument('--output',  default='outputs/',
                        help='Output directory')
    parser.add_argument('--resume',  type=int, default=None,
                        help='Resume from fold id')
    parser.add_argument('--device',  default='auto',
                        help='cuda / cpu / auto')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(name)s — %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                Path(args.output) / f'train_{args.dataset}.log', mode='a'
            ),
        ],
    )
    logger = logging.getLogger(__name__)

    # Config
    cfg  = load_config(args.dataset)
    seed = cfg.get('seed', 42)
    set_seed(seed)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # HDF5 dir
    processed_dir = Path(cfg['dataset']['processed_dir'])
    if not processed_dir.exists() or not any(processed_dir.glob('fold_*.h5')):
        logger.error(
            f"No HDF5 files found in {processed_dir}.\n"
            f"Run preprocessing first:\n"
            f"  python scripts/preprocess_{args.dataset}.py"
        )
        sys.exit(1)

    # LOSO run
    from src.training.loso_runner import run_loso

    n_folds = cfg.get('evaluation', {}).get('n_folds', 8)

    summary = run_loso(
        cfg          = cfg,
        dataset_name = args.dataset,
        hdf5_dir     = str(processed_dir),
        out_dir      = args.output,
        device       = device,
        n_folds      = n_folds if args.fold == 'all' else int(args.fold) + 1,
        resume_fold  = int(args.fold) if args.fold != 'all' else args.resume,
    )

    if summary:
        if 'mean_macro_f1' in summary:
            logger.info(
                f"\n[FINAL] {args.dataset} "
                f"Macro F1 = {summary['mean_macro_f1']*100:.2f}% "
                f"± {summary['std_macro_f1']*100:.2f}%"
            )
        else:
            logger.info(
                f"\n[FINAL] {args.dataset} fold {args.fold} "
                f"Macro F1 = {summary['macro_f1']*100:.2f}%"
            )


if __name__ == '__main__':
    main()
