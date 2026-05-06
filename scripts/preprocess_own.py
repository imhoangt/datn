"""
scripts/preprocess_own.py — Run full v7.2 pipeline on own-collected data.
"""
import sys, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from scripts.train import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--jobs',  type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s — %(message)s')

    cfg = load_config('own')
    from src.preprocessing.pipeline_own import preprocess_own_dataset

    for fold_id in args.folds:
        print(f'\n=== Preprocessing fold {fold_id} ===')
        out = preprocess_own_dataset(cfg, fold_id, n_jobs=args.jobs)
        if out:
            print(f'  → {out}')
        else:
            print(f'  FAILED fold {fold_id}')


if __name__ == '__main__':
    main()
