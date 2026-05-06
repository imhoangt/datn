"""scripts/preprocess_exposing.py"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s — %(message)s')
    from scripts.train import load_config
    from src.preprocessing.pipeline_exposing import preprocess_exposing_dataset
    cfg = load_config('exposing')

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    args = p.parse_args()

    for fold_id in args.folds:
        print(f'\n=== Exposing CSI Preprocessing fold {fold_id} ===')
        out = preprocess_exposing_dataset(cfg, fold_id)
        print(f'  → {out}' if out else f'  FAILED fold {fold_id}')

if __name__ == '__main__':
    main()
