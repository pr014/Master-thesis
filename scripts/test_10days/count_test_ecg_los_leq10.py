#!/usr/bin/env python3
"""Count ECG windows in the **test split** whose matched ICU stay has true LOS in [0, max_days].

Uses the same dataloader path as training / ``evaluate_subgroup.py`` (config from a checkpoint).

Example
-------
  cd <repo-root>
  python scripts/test_10days/count_test_ecg_los_leq10.py --job 3119696
  python scripts/test_10days/count_test_ecg_los_leq10.py --checkpoint outputs/checkpoints/CNNScratch_best_3119696.pt --max-days 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.data.ecg import create_dataloaders
from src.training import setup_icustays_mapper


def _find_checkpoint(job_id: int) -> Path:
    ckpt_dir = REPO_ROOT / "outputs" / "checkpoints"
    candidates = sorted(ckpt_dir.glob(f"*_best_{job_id}.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matching '*_best_{job_id}.pt' in {ckpt_dir}")
    return candidates[0]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Count test ECG windows with matched stay LOS ≤ max_days (same loader as training)."
    )
    p.add_argument("--job", type=int, default=None, help="SLURM job id (finds *_best_<job>.pt)")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint (overrides --job)")
    p.add_argument(
        "--max-days",
        type=float,
        default=10.0,
        help="Upper bound on true stay LOS in days (inclusive). Default: 10.",
    )
    p.add_argument("--json", action="store_true", help="Print one JSON object only.")
    args = p.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).expanduser().resolve()
    elif args.job is not None:
        ckpt_path = _find_checkpoint(int(args.job))
    else:
        p.error("Provide --job or --checkpoint")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config: Dict[str, Any] = ckpt.get("config", {})
    if not config:
        raise SystemExit("Checkpoint has no 'config'")

    icu_mapper = setup_icustays_mapper(config)
    _, _, test_loader = create_dataloaders(
        config=config,
        labels=None,
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,
    )
    if test_loader is None:
        raise SystemExit("No test loader (test_split may be 0 in config).")

    max_d = float(args.max_days)
    n_valid = 0
    n_leq = 0
    stays_valid: Set[Any] = set()
    stays_leq: Set[Any] = set()

    for batch in test_loader:
        labels = batch["label"].cpu().numpy()
        meta = batch["meta"]
        for i in range(len(labels)):
            los = float(labels[i])
            if los < 0.0:
                continue
            n_valid += 1
            m = meta[i] if i < len(meta) else {}
            sid = m.get("stay_id")
            if sid is not None:
                stays_valid.add(sid)
            if 0.0 <= los <= max_d:
                n_leq += 1
                if sid is not None:
                    stays_leq.add(sid)

    ds = test_loader.dataset
    n_dataset = len(ds)

    out: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "test_dataset_len": int(n_dataset),
        "max_true_los_days_inclusive": max_d,
        "n_ecg_windows_valid_label": int(n_valid),
        "n_ecg_windows_true_stay_los_leq_max": int(n_leq),
        "frac_windows_leq_max": float(n_leq / n_valid) if n_valid else None,
        "n_distinct_stay_id_among_valid_windows": len(stays_valid),
        "n_distinct_stay_id_among_leq_max_windows": len(stays_leq),
        "note": (
            "Each test-loader row is one ECG window/record. Label = true LOS (days) of the ICU stay "
            "matched by map_ecg_to_stay (same as training). Windows with label<0 are excluded from counts."
        ),
    }

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Test dataset __len__ (windows in split): {n_dataset:,}")
    print(f"ECG windows with valid match (label >= 0): {n_valid:,}")
    print(
        f"ECG windows whose matched stay has LOS in [0, {max_d:g}] days: "
        f"{n_leq:,} ({100.0 * n_leq / n_valid:.2f}% of valid windows)" if n_valid else "n/a"
    )
    print(f"Distinct stay_id among valid windows: {len(stays_valid):,}")
    print(f"Distinct stay_id among LOS≤{max_d:g}d windows: {len(stays_leq):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
