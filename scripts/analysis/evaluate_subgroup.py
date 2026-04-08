#!/usr/bin/env python3
"""Post-hoc subgroup evaluation for already-trained models.

Loads a saved checkpoint, rebuilds the model and test dataloader from the
embedded config, runs inference on the test set, and reports:

  - Overall metrics  (MAE, RMSE, R², Median AE, percentile errors)
  - Subgroup metrics (LOS ≤ 10 days)
  - Per-bin metrics  (0-3 d / 3-10 d / >10 d)

Usage
-----
# By checkpoint path
python scripts/analysis/evaluate_subgroup.py \\
    --checkpoint outputs/checkpoints/LSTM1D_best_3291177.pt

# By SLURM job ID (auto-locates checkpoint in outputs/checkpoints/)
python scripts/analysis/evaluate_subgroup.py --job 3291177

# JSON output
python scripts/analysis/evaluate_subgroup.py --job 3291177 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so src.* imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    MultiTaskECGModel,
    CNNScratch,
    HybridCNNLSTM,
    DeepECG_SL,
    LSTM1D,
)
from src.models.lstm import LSTM1D_Unidirectional, LSTM1D_Bidirectional
from src.data.ecg import create_dataloaders
from src.training import setup_icustays_mapper
from src.training.losses import get_multi_task_loss, get_loss
from src.training.train_loop import MultiTaskLoss


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "LSTM1D": LSTM1D_Unidirectional,
    "LSTM1D_Unidirectional": LSTM1D_Unidirectional,
    "LSTM1D_Bidirectional": LSTM1D_Bidirectional,
    "HybridCNNLSTM": HybridCNNLSTM,
    "CNNScratch": CNNScratch,
    "DeepECG_SL": DeepECG_SL,
}


def _build_model(config: dict) -> torch.nn.Module:
    model_type = config.get("model", {}).get("type", "")
    cls = _MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Supported: {list(_MODEL_REGISTRY)}"
        )
    return cls(config)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _regression_metrics(
    predictions: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    ae = np.abs(predictions - labels)
    mse = float(((predictions - labels) ** 2).mean())
    ss_res = ((labels - predictions) ** 2).sum()
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
        "median_ae": float(np.median(ae)),
        "p25_error": float(np.percentile(ae, 25)),
        "p50_error": float(np.percentile(ae, 50)),
        "p75_error": float(np.percentile(ae, 75)),
        "p90_error": float(np.percentile(ae, 90)),
        "n": int(len(labels)),
    }


def _print_metrics(title: str, m: Dict[str, float]) -> None:
    n = m.get("n", "?")
    print(f"\n{title} (N={n:,}):" if isinstance(n, int) else f"\n{title}:")
    print(f"  MAE:       {m['mae']:.4f} days")
    print(f"  RMSE:      {m['rmse']:.4f} days")
    print(f"  R²:        {m['r2']:.4f}")
    print(f"  Median AE: {m['median_ae']:.4f} days")
    print(f"  P25 error: {m['p25_error']:.4f} days")
    print(f"  P50 error: {m['p50_error']:.4f} days")
    print(f"  P75 error: {m['p75_error']:.4f} days")
    print(f"  P90 error: {m['p90_error']:.4f} days")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    is_multi_task: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predictions, labels) arrays (stay-level mean aggregation)."""
    model.eval()
    stay_preds: Dict[str, List[float]] = {}
    stay_labels: Dict[str, float] = {}

    with torch.no_grad():
        for batch in test_loader:
            signals = batch["signal"].to(device)
            labels = batch["label"].to(device)
            meta = batch["meta"]

            valid = labels >= 0
            if not valid.any():
                continue
            signals = signals[valid]
            labels = labels[valid]
            meta = [meta[i] for i in range(len(meta)) if valid[i]]

            # Optional auxiliary features
            def _feat(key):
                if key in batch and batch[key] is not None:
                    return batch[key].to(device)[valid]
                return None

            demo = _feat("demographic_features")
            diag = _feat("diagnosis_features")
            icu = _feat("icu_unit_features")

            outputs = model(
                signals,
                demographic_features=demo,
                diagnosis_features=diag,
                icu_unit_features=icu,
            )

            if is_multi_task and isinstance(outputs, dict):
                los_preds = outputs["los"]
            elif is_multi_task and isinstance(outputs, tuple):
                los_preds = outputs[0]
            else:
                los_preds = outputs

            los_preds = los_preds.squeeze(-1) if los_preds.dim() > 1 else los_preds

            for i in range(len(labels)):
                sid = meta[i].get("stay_id")
                if sid is None:
                    continue
                stay_preds.setdefault(sid, []).append(los_preds[i].cpu().item())
                stay_labels[sid] = labels[i].item()

    if not stay_preds:
        raise RuntimeError("No predictions collected — check test dataloader.")

    preds_np = np.array([np.mean(stay_preds[s]) for s in stay_preds])
    labels_np = np.array([stay_labels[s] for s in stay_preds])
    return preds_np, labels_np


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_checkpoint(job_id: int) -> Path:
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    candidates = sorted(ckpt_dir.glob(f"*_best_{job_id}.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching '*_best_{job_id}.pt' in {ckpt_dir}"
        )
    if len(candidates) > 1:
        print(f"[warn] Multiple checkpoints found for job {job_id}, using: {candidates[0].name}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc subgroup evaluation for a saved checkpoint."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to .pt checkpoint file")
    group.add_argument("--job", type=int, help="SLURM job ID (auto-finds checkpoint)")
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    parser.add_argument(
        "--device", type=str, default=None, help="Device override (cpu / cuda)"
    )
    args = parser.parse_args()

    # Locate checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _find_checkpoint(args.job)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config: dict = ckpt.get("config", {})
    job_id = ckpt.get("job_id", "unknown")
    epoch = ckpt.get("epoch", "?")
    model_type = config.get("model", {}).get("type", "unknown")

    print(f"  Job ID:     {job_id}")
    print(f"  Model type: {model_type}")
    print(f"  Epoch:      {epoch}")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device:     {device}")

    # Build model
    base_model = _build_model(config)
    multi_task_config = config.get("multi_task", {})
    is_multi_task = multi_task_config.get("enabled", False)

    if is_multi_task:
        model = MultiTaskECGModel(base_model, config)
    else:
        model = base_model

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build test dataloader (same split as training)
    icu_mapper = setup_icustays_mapper(config)
    _, _, test_loader = create_dataloaders(
        config=config,
        labels=None,
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,
    )

    # Run inference
    print("\nRunning inference on test set...")
    predictions, labels = _run_inference(model, test_loader, device, is_multi_task)
    print(f"  Test stays: {len(labels):,}")
    print(f"  LOS range:  {labels.min():.2f} – {labels.max():.2f} days (mean {labels.mean():.2f})")

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    overall = _regression_metrics(predictions, labels)

    mask_leq10 = labels <= 10.0
    mask_3_10 = (labels > 3.0) & (labels <= 10.0)
    mask_leq3 = labels <= 3.0
    mask_gt10 = labels > 10.0

    results = {
        "job_id": job_id,
        "model_type": model_type,
        "epoch": epoch,
        "checkpoint": str(ckpt_path),
        "overall": overall,
        "subgroup_leq10": _regression_metrics(predictions[mask_leq10], labels[mask_leq10]) if mask_leq10.any() else {},
        "subgroup_leq3": _regression_metrics(predictions[mask_leq3], labels[mask_leq3]) if mask_leq3.any() else {},
        "subgroup_3_10": _regression_metrics(predictions[mask_3_10], labels[mask_3_10]) if mask_3_10.any() else {},
        "subgroup_gt10": _regression_metrics(predictions[mask_gt10], labels[mask_gt10]) if mask_gt10.any() else {},
    }

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # -----------------------------------------------------------------------
    # Pretty print
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"📊 SUBGROUP EVALUATION — {model_type}  (Job {job_id}, epoch {epoch})")
    print("=" * 70)

    _print_metrics("🔹 Overall (all stays)", overall)

    if mask_leq10.any():
        _print_metrics(
            f"🔹 LOS ≤ 10 days [{mask_leq10.sum():,}/{len(labels):,} stays]",
            results["subgroup_leq10"],
        )
    if mask_leq3.any():
        _print_metrics(
            f"  └─ LOS ≤ 3 days [{mask_leq3.sum():,} stays]",
            results["subgroup_leq3"],
        )
    if mask_3_10.any():
        _print_metrics(
            f"  └─ LOS 3–10 days [{mask_3_10.sum():,} stays]",
            results["subgroup_3_10"],
        )
    if mask_gt10.any():
        _print_metrics(
            f"🔹 LOS > 10 days [{mask_gt10.sum():,} stays — outliers]",
            results["subgroup_gt10"],
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
