"""
src/training/loso_runner.py  — Full LOSO training loop (all folds)
src/utils/checkpoint.py      — Save / load / resume checkpoints
src/utils/logger.py          — CSV + TensorBoard logging
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# checkpoint.py
# ============================================================================

def save_checkpoint(
    model        : nn.Module,
    ema,
    optimizer    : torch.optim.Optimizer,
    scheduler    : Optional[object],
    scaler,
    global_epoch : int,
    phase        : str,
    best_val_f1  : float,
    fold_id      : int,
    out_dir      : str,
) -> str:
    """Save training checkpoint. Returns path."""
    path = Path(out_dir) / f"fold_{fold_id:02d}_best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'global_epoch'         : global_epoch,
        'fold_id'              : fold_id,
        'model_state_dict'     : model.state_dict(),
        'ema_state_dict'       : ema.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict() if scheduler else None,
        'scaler_state_dict'    : scaler.state_dict(),
        'best_val_f1'          : best_val_f1,
        'current_phase'        : phase,
        'ema_decay'            : ema.decay,
    }, str(path))
    return str(path)


def load_checkpoint(
    path         : str,
    model        : nn.Module,
    ema,
    optimizer    : Optional[torch.optim.Optimizer] = None,
    scheduler    : Optional[object] = None,
    scaler       = None,
    device       : torch.device = torch.device('cpu'),
) -> Dict:
    """Load checkpoint. Returns metadata dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    ema.load_state_dict(ckpt['ema_state_dict'])
    ema.decay = ckpt.get('ema_decay', 0.999)
    if optimizer and ckpt.get('optimizer_state_dict'):
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and ckpt.get('scaler_state_dict'):
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    logger.info(
        f"Loaded checkpoint: fold={ckpt['fold_id']}, epoch={ckpt['global_epoch']}, "
        f"val_f1={ckpt['best_val_f1']:.4f}, phase={ckpt['current_phase']}"
    )
    return ckpt


# ============================================================================
# logger.py
# ============================================================================

class TrainLogger:
    """
    Logs metrics to CSV and optionally TensorBoard.
    """

    def __init__(
        self,
        log_dir        : str,
        fold_id        : int,
        dataset_name   : str,
        use_tensorboard: bool = True,
    ):
        self.log_dir     = Path(log_dir)
        self.fold_id     = fold_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        self.csv_path = self.log_dir / f"{dataset_name}_fold{fold_id:02d}_log.csv"
        self._csv_file   = open(self.csv_path, 'w', newline='')
        self._csv_writer = None   # written on first log

        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / 'tensorboard' / f"{dataset_name}_fold{fold_id:02d}"
                self.writer = SummaryWriter(str(tb_dir))
            except ImportError:
                logger.warning("TensorBoard not available")

    def log(self, epoch: int, phase: str, metrics: Dict):
        """Log one epoch of metrics."""
        row = {'epoch': epoch, 'phase': phase, **metrics}

        # CSV
        if self._csv_writer is None:
            fieldnames        = list(row.keys())
            self._csv_writer  = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()
        try:
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        except ValueError:
            pass   # new keys added mid-run

        # TensorBoard
        if self.writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"{phase}/{k}", v, epoch)

    def close(self):
        self._csv_file.close()
        if self.writer:
            self.writer.close()


# ============================================================================
# loso_runner.py
# ============================================================================

def run_loso(
    cfg         : dict,
    dataset_name: str,
    hdf5_dir    : str,
    out_dir     : str,
    device      : torch.device,
    n_folds     : int = 8,
    resume_fold : Optional[int] = None,
) -> Dict:
    """
    Run full LOSO training across n_folds.

    For each fold:
        Phase 1: warmup (5 ep, LR 1e-5→1e-4)
        Phase 2: main   (50 ep, cosine 1e-4→1e-6)
        Phase 3: finetune (20 ep, cosine 1e-5→1e-7, EMA 0.9999)
        Post: temperature calibration

    Returns: dict with per-fold and mean results.
    """
    from torch_ema import ExponentialMovingAverage
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from src.models.charm_net import build_model
    from src.datasets.base_dataset import build_loaders
    from src.training.losses import train_one_epoch, evaluate_with_ema
    from src.models.heads import TemperatureScaler
    from src.evaluation.metrics import compute_fold_metrics

    cfg_tr = cfg.get('training', {})
    cfg_m  = cfg.get('model', {})

    class_weights = torch.tensor(
        cfg_tr.get('class_weights', [1.,1.,1.,1.,1.,2.,1.5,3.]),
        dtype=torch.float32,
    )

    all_fold_results = {}

    for fold_id in range(n_folds):
        if resume_fold is not None and fold_id < resume_fold:
            continue

        logger.info(f"\n{'='*60}\nFold {fold_id}/{n_folds-1} — {dataset_name}\n{'='*60}")

        hdf5_path = Path(hdf5_dir) / f"fold_{fold_id:02d}.h5"
        if not hdf5_path.exists():
            logger.warning(f"HDF5 not found: {hdf5_path}, skipping fold")
            continue

        # DataLoaders
        train_loader, val_loader, test_loader, train_ds = build_loaders(
            str(hdf5_path),
            batch_size   = cfg_tr.get('batch_size', 16),
            num_workers  = 4,
            cfg_aug      = cfg_tr,
            class_weights= class_weights,
        )

        # Model + optimizer + EMA + scaler
        model    = build_model(cfg_m, device)
        optimizer = AdamW(model.parameters(),
                          lr=cfg_tr.get('main_lr_start', 1e-4),
                          weight_decay=cfg_tr.get('weight_decay', 1e-4),
                          betas=tuple(cfg_tr.get('betas', [0.9, 0.999])))
        ema      = ExponentialMovingAverage(model.parameters(),
                                            decay=cfg_tr.get('ema_decay_main', 0.999))
        try:
            scaler = torch.amp.GradScaler(device.type)
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler()
        log      = TrainLogger(out_dir + '/logs', fold_id, dataset_name,
                               cfg.get('logging', {}).get('use_tensorboard', True))

        best_val_f1  = 0.0
        global_epoch = 0
        ckpt_dir     = out_dir + '/checkpoints'

        # ── Phase 1: Warmup ─────────────────────────────────────────────
        lr_start = cfg_tr.get('warmup_lr_start', 1e-5)
        lr_end   = cfg_tr.get('warmup_lr_end', 1e-4)
        warmup_ep = cfg_tr.get('warmup_epochs', 5)
        for ep in range(warmup_ep):
            global_epoch += 1
            lr = lr_start + (lr_end - lr_start) * ep / max(warmup_ep - 1, 1)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            train_ds.global_epoch = global_epoch
            train_m = train_one_epoch(model, train_loader, optimizer, scaler,
                                      ema, global_epoch, class_weights, cfg_tr,
                                      device, aug_level='no_aug')
            val_f1, _, _ = evaluate_with_ema(model, ema, val_loader, device)
            log.log(global_epoch, 'train', {**train_m, 'lr': lr})
            log.log(global_epoch, 'val',   {'macro_f1': val_f1})
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_checkpoint(model, ema, optimizer, None, scaler,
                                global_epoch, 'warmup', best_val_f1, fold_id, ckpt_dir)
            logger.info(f"  [Warmup ep{ep+1}] LR={lr:.2e} val_f1={val_f1:.4f}")

        # ── Phase 2: Main cosine ─────────────────────────────────────────
        for pg in optimizer.param_groups:
            pg['lr'] = cfg_tr.get('main_lr_start', 1e-4)
        sched2 = CosineAnnealingLR(optimizer,
                                   T_max=cfg_tr.get('main_epochs', 50),
                                   eta_min=cfg_tr.get('main_lr_end', 1e-6))
        for ep in range(cfg_tr.get('main_epochs', 50)):
            global_epoch += 1
            train_ds.global_epoch = global_epoch
            train_m = train_one_epoch(model, train_loader, optimizer, scaler,
                                      ema, global_epoch, class_weights, cfg_tr,
                                      device, aug_level='full')
            sched2.step()
            val_f1, _, _ = evaluate_with_ema(model, ema, val_loader, device)
            log.log(global_epoch, 'train', {**train_m, 'lr': optimizer.param_groups[0]['lr']})
            log.log(global_epoch, 'val',   {'macro_f1': val_f1})
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_checkpoint(model, ema, optimizer, sched2, scaler,
                                global_epoch, 'main', best_val_f1, fold_id, ckpt_dir)
            if (ep + 1) % 10 == 0:
                logger.info(f"  [Main ep{ep+1}] val_f1={val_f1:.4f} best={best_val_f1:.4f}")

        # ── Phase 3: Fine-tune ───────────────────────────────────────────
        for pg in optimizer.param_groups:
            pg['lr'] = cfg_tr.get('finetune_lr_start', 1e-5)
        ema.decay = cfg_tr.get('ema_decay_finetune', 0.9999)
        sched3    = CosineAnnealingLR(optimizer,
                                      T_max=cfg_tr.get('finetune_epochs', 20),
                                      eta_min=cfg_tr.get('finetune_lr_end', 1e-7))
        for ep in range(cfg_tr.get('finetune_epochs', 20)):
            global_epoch += 1
            train_ds.global_epoch = global_epoch
            train_m = train_one_epoch(model, train_loader, optimizer, scaler,
                                      ema, global_epoch, class_weights, cfg_tr,
                                      device, aug_level='no_aug')
            sched3.step()
            val_f1, _, _ = evaluate_with_ema(model, ema, val_loader, device)
            log.log(global_epoch, 'train', {**train_m, 'lr': optimizer.param_groups[0]['lr']})
            log.log(global_epoch, 'val',   {'macro_f1': val_f1})
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_checkpoint(model, ema, optimizer, sched3, scaler,
                                global_epoch, 'finetune', best_val_f1, fold_id, ckpt_dir)

        # ── Load best, calibrate, test ───────────────────────────────────
        best_ckpt = str(Path(ckpt_dir) / f"fold_{fold_id:02d}_best.pt")
        if Path(best_ckpt).exists():
            load_checkpoint(best_ckpt, model, ema, device=device)

        ts  = TemperatureScaler()
        ts.calibrate(model, val_loader, device)

        _, test_probs, test_labels = evaluate_with_ema(model, ema, test_loader, device)
        fold_metrics = compute_fold_metrics(test_probs, test_labels)
        fold_metrics['best_val_f1']  = best_val_f1
        fold_metrics['temperature']  = ts.temperature.item()
        all_fold_results[fold_id]    = fold_metrics
        log.log(global_epoch, 'test', fold_metrics)
        log.close()

        logger.info(
            f"  Fold {fold_id} TEST: macro_f1={fold_metrics['macro_f1']:.4f} "
            f"fall_recall={fold_metrics.get('fall_recall', 0):.4f}"
        )

    # Summary
    if all_fold_results:
        f1s  = [r['macro_f1']   for r in all_fold_results.values()]
        frs  = [r.get('fall_recall', 0) for r in all_fold_results.values()]
        summary = {
            'mean_macro_f1' : float(np.mean(f1s)),
            'std_macro_f1'  : float(np.std(f1s)),
            'mean_fall_recall': float(np.mean(frs)),
            'per_fold'      : all_fold_results,
        }
        summary_path = Path(out_dir) / f"{dataset_name}_loso_summary.json"
        with open(summary_path, 'w') as fp:
            json.dump(summary, fp, indent=2, default=lambda x: float(x))
        logger.info(
            f"\n[{dataset_name}] LOSO result: "
            f"macro_f1 = {summary['mean_macro_f1']:.4f} ± {summary['std_macro_f1']:.4f}"
        )
        return summary

    return {}
