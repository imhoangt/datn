"""scripts/preprocess_xrf55.py"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s — %(message)s')
    from scripts.train import load_config
    from src.preprocessing.pipeline_xrf55 import preprocess_xrf55_dataset
    cfg = load_config('xrf55')

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--folds', nargs='+', type=int, default=list(range(8)))
    args = p.parse_args()

    for fold_id in args.folds:
        print(f'\n=== XRF55 Preprocessing fold {fold_id} ===')
        out = preprocess_xrf55_dataset(cfg, fold_id)
        print(f'  → {out}' if out else f'  FAILED fold {fold_id}')

if __name__ == '__main__':
    main()
