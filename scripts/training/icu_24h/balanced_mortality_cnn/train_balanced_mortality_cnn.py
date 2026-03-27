#!/usr/bin/env python3
"""Isolated training entry: CNNScratch on balanced_mortality + manifest (binary mortality).

Does not use create_dataloaders, Trainer, or shared train_loop.
Run from project root:
  python scripts/training/icu_24h/balanced_mortality_cnn/train_balanced_mortality_cnn.py
  python .../train_balanced_mortality_cnn.py --config configs/experiments/balanced_mortality_cnn.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent


def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "src" / "models").is_dir() and (p / "configs").is_dir():
            return p
    raise RuntimeError(f"Could not locate MA-thesis project root from {start}")


_PROJECT_ROOT = _find_project_root(_HERE)
# Ensure this repo's `src` wins over any other `src` on PYTHONPATH (e.g. other checkouts).
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

from balanced_mortality_dataset import (
    BalancedMortalityDataset,
    collate_balanced_mortality,
    load_manifest,
)
from balanced_mortality_loop import (
    build_optimizer,
    build_scheduler,
    evaluate,
    train_one_epoch,
)
from balanced_mortality_splits import stratified_subject_split

from src.data.augmentation import create_augmentation_transform
from src.models import CNNScratch
from src.utils.config_loader import load_config
from src.utils.device import get_device, set_seed


def _resolve(p: str, root: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _find_best_threshold_for_f1(probs: np.ndarray, labels: np.ndarray) -> float:
    """Find threshold in [0.05, 0.95] maximizing F1 on validation set."""
    if probs.size == 0 or labels.size == 0:
        return 0.5
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.05, 0.951, 0.01):
        preds = (probs >= thr).astype(np.int64)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/balanced_mortality_cnn.yaml",
        help="Path to experiment YAML (relative to project root or absolute).",
    )
    args = parser.parse_args()

    cfg_path = _resolve(args.config, _PROJECT_ROOT)
    config = load_config(model_config_path=cfg_path, base_dir=_PROJECT_ROOT)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device_cfg = config.get("device", {})
    device = get_device(device_cfg.get("device"))

    data_cfg = config.get("data", {})
    data_dir = _resolve(data_cfg["data_dir"], _PROJECT_ROOT)
    manifest_path = _resolve(data_cfg["manifest_path"], _PROJECT_ROOT)

    prefix = config.get("logging", {}).get("log_prefix", "[balanced_mortality_cnn]")

    df = load_manifest(manifest_path)
    val_split = float(config.get("validation", {}).get("val_split", 0.1))
    test_split = float(config.get("test_split", 0.1))
    idx_train, idx_val, idx_test = stratified_subject_split(
        df, val_split=val_split, test_split=test_split, random_state=seed
    )

    train_tf = create_augmentation_transform(config)
    train_ds = BalancedMortalityDataset(data_dir, df.iloc[idx_train], transform=train_tf)
    val_ds = BalancedMortalityDataset(data_dir, df.iloc[idx_val], transform=None)
    test_ds = BalancedMortalityDataset(data_dir, df.iloc[idx_test], transform=None)

    tcfg = config.get("training", {})
    bs = int(tcfg.get("batch_size", 64))
    nw = int(tcfg.get("num_workers", 4))
    pin = bool(tcfg.get("pin_memory", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        collate_fn=collate_balanced_mortality,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        collate_fn=collate_balanced_mortality,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        collate_fn=collate_balanced_mortality,
    )

    model = CNNScratch(config)
    model.to(device)

    optimizer = build_optimizer(model, config)
    scheduler, sched_kind = build_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss()

    ckpt_cfg = config.get("checkpoint", {})
    save_dir = _resolve(ckpt_cfg.get("save_dir", "outputs/checkpoints/balanced_mortality_cnn"), _PROJECT_ROOT)
    save_dir.mkdir(parents=True, exist_ok=True)
    job_id = os.getenv("SLURM_JOB_ID", "local")
    best_name = f"CNNScratch_balanced_mortality_best_{job_id}.pt"

    num_epochs = int(tcfg.get("num_epochs", 50))
    es_cfg = config.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", True))
    patience = int(es_cfg.get("patience", 7))
    monitor_mode = es_cfg.get("mode", "max")
    best_metric = float("-inf") if monitor_mode == "max" else float("inf")
    stale = 0
    best_state = None

    print(f"{prefix} device={device} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"{prefix} checkpoints -> {save_dir / best_name}")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_auc = val_metrics["auc"]

        if sched_kind == "plateau" and scheduler is not None:
            scheduler.step(val_auc)

        print(
            f"{prefix} epoch {epoch}/{num_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_auc={val_auc:.4f}"
        )

        improved = val_auc > best_metric if monitor_mode == "max" else val_auc < best_metric
        if improved:
            best_metric = val_auc
            stale = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_cfg.get("save_best", True):
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_auc": val_auc,
                        "config_path": str(cfg_path),
                    },
                    save_dir / best_name,
                )
        else:
            stale += 1

        if es_enabled and stale >= patience:
            print(f"{prefix} early stopping at epoch {epoch} (best val_auc={best_metric:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Calibrate decision threshold on validation set (keeps AUC unchanged, improves class metrics)
    val_for_threshold = evaluate(model, val_loader, criterion, device, threshold=0.5, return_arrays=True)
    best_threshold = _find_best_threshold_for_f1(
        probs=val_for_threshold.get("probs", np.array([])),
        labels=val_for_threshold.get("labels", np.array([])),
    )

    test_metrics = evaluate(model, test_loader, criterion, device, threshold=best_threshold)
    # Keep the concise one-liner for quick scans
    print(
        f"{prefix} TEST loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['accuracy']:.4f} auc={test_metrics['auc']:.4f}"
    )

    # Print a parser-friendly summary block (mirrors parse_training_results.py patterns)
    # This enables: python scripts/analysis/parse_training_results.py --log <this slurm out>
    print("=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Best Validation AUC: {best_metric:.4f}" if best_state is not None else "Best Validation AUC: <not available>")
    print(f"Best Validation Threshold (max F1): {best_threshold:.2f}")
    print(f"Loss:      {test_metrics['loss']:.4f}")
    print(f"Accuracy:  {test_metrics.get('accuracy', 0.0):.4f}")
    print(f"Precision: {test_metrics.get('precision', 0.0):.4f}")
    print(f"Recall:    {test_metrics.get('recall', 0.0):.4f}")
    print(f"F1:        {test_metrics.get('f1', 0.0):.4f}")
    print(f"AUC:       {test_metrics.get('auc', 0.0):.4f}")
    print(f"TP:        {int(test_metrics.get('tp', 0.0))}")
    print(f"FP:        {int(test_metrics.get('fp', 0.0))}")
    print(f"TN:        {int(test_metrics.get('tn', 0.0))}")
    print(f"FN:        {int(test_metrics.get('fn', 0.0))}")
    print("=" * 60)
    print(f"✅ Checkpoints: {save_dir / best_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
