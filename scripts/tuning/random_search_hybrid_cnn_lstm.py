#!/usr/bin/env python3
"""Run random-search tuning for the Hybrid CNN-LSTM.

This script samples trial configurations, writes small override YAML files,
submits one SLURM job per trial via the existing Hybrid sbatch wrapper, and
records a manifest linking trial IDs to sampled hyperparameters and job IDs.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import save_config


DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/model/hybrid_cnn_lstm/hybrid_cnn_lstm.yaml"
DEFAULT_SBATCH_SCRIPT = PROJECT_ROOT / (
    "scripts/cluster/icu_24h/hybrid_cnn_lstm/train_hybrid_cnn_lstm_24h.sbatch"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/tuning/hybrid_cnn_lstm"

TRIAL_FIELDNAMES = [
    "sweep_id",
    "trial_id",
    "job_id",
    "status",
    "submit_time_utc",
    "model_config_path",
    "experiment_config_path",
    "training_seed",
    "optimizer_type",
    "optimizer_lr",
    "optimizer_weight_decay",
    "training_batch_size",
    "training_dropout_rate",
    "model_lstm_dropout",
    "model_hidden_dim",
    "model_hidden_dim_layer2",
    "scheduler_factor",
    "scheduler_patience",
    "gradient_clip_norm",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random-search tuning for the Hybrid CNN-LSTM."
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=20,
        help="Number of random-search trials to sample and submit.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Optional sweep identifier. Defaults to a timestamp-based name.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=str(DEFAULT_MODEL_CONFIG),
        help="Base Hybrid model config to override per trial.",
    )
    parser.add_argument(
        "--sbatch-script",
        type=str,
        default=str(DEFAULT_SBATCH_SCRIPT),
        help="Existing sbatch wrapper used to submit each trial.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for generated trial configs and manifests.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=42,
        help="Seed for random-search sampling.",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Optional fixed training seed written into each override config.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index for trial numbering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create trial configs and manifest without submitting SLURM jobs.",
    )
    return parser.parse_args()


def _default_sweep_id() -> str:
    return datetime.utcnow().strftime("random_search_%Y%m%d_%H%M%S")


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _round_sig(value: float, digits: int = 2) -> float:
    """Round a float to significant digits for cleaner sampled values."""
    if value == 0:
        return 0.0
    return round(value, digits - int(math.floor(math.log10(abs(value)))) - 1)


def _sample_trial(rng: random.Random, training_seed: int | None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    optimizer_type = rng.choice(["Adam", "AdamW"])

    # Keep the log-scale behavior but round to readable, reproducible values.
    lr = _round_sig(_log_uniform(rng, 1e-4, 8e-4), digits=2)
    weight_decay = _round_sig(_log_uniform(rng, 1e-6, 5e-3), digits=2)
    batch_size = rng.choice([32, 64, 128])
    dropout_rate = round(rng.uniform(0.1, 0.5), 3)
    lstm_dropout = round(rng.uniform(0.1, 0.4), 3)
    hidden_dim = rng.choice([64, 128, 256])
    hidden_dim_layer2 = rng.choice([64, 128, 256])
    scheduler_factor = rng.choice([0.1, 0.2, 0.5])
    scheduler_patience = rng.choice([3, 5, 7, 10])
    gradient_clip_norm = rng.choice([0.5, 1.0, 2.0])

    override = {
        "model": {
            "hidden_dim": hidden_dim,
            "hidden_dim_layer2": hidden_dim_layer2,
            "lstm_dropout": lstm_dropout,
        },
        "training": {
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "gradient_clip_norm": gradient_clip_norm,
            "optimizer": {
                "type": optimizer_type,
                "lr": lr,
                "weight_decay": weight_decay,
            },
            "scheduler": {
                "factor": scheduler_factor,
                "patience": scheduler_patience,
            },
        },
    }
    if training_seed is not None:
        override["seed"] = training_seed

    flat = {
        "training_seed": training_seed if training_seed is not None else "",
        "optimizer_type": optimizer_type,
        "optimizer_lr": f"{lr:.8g}",
        "optimizer_weight_decay": f"{weight_decay:.8g}",
        "training_batch_size": batch_size,
        "training_dropout_rate": dropout_rate,
        "model_lstm_dropout": lstm_dropout,
        "model_hidden_dim": hidden_dim,
        "model_hidden_dim_layer2": hidden_dim_layer2,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "gradient_clip_norm": gradient_clip_norm,
    }
    return override, flat


def _submit_trial(
    sbatch_script: Path,
    model_config_path: Path,
    experiment_config_path: Path,
    trial_id: str,
    sweep_id: str,
) -> tuple[str, str]:
    export_args = (
        f"ALL,"
        f"MODEL_CONFIG_PATH={model_config_path.resolve()},"
        f"EXPERIMENT_CONFIG_PATH={experiment_config_path.resolve()},"
        f"TRIAL_ID={trial_id},"
        f"SWEEP_ID={sweep_id}"
    )
    cmd = [
        "sbatch",
        "--export",
        export_args,
        str(sbatch_script.resolve()),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        error_text = stderr or stdout or f"sbatch failed with exit code {completed.returncode}"
        raise RuntimeError(error_text)

    match = re.search(r"Submitted batch job\s+(\d+)", stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from sbatch output: {stdout}")

    return match.group(1), stdout


def _write_manifest_header(manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDNAMES)
        writer.writeheader()


def _append_manifest_row(manifest_path: Path, row: Dict[str, Any]) -> None:
    with open(manifest_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDNAMES)
        writer.writerow(row)


def main() -> None:
    args = _parse_args()

    model_config_path = Path(args.model_config)
    sbatch_script = Path(args.sbatch_script)
    output_root = Path(args.output_root)
    sweep_id = args.sweep_id or _default_sweep_id()

    sweep_dir = output_root / sweep_id
    trial_config_dir = sweep_dir / "trial_configs"
    manifest_path = sweep_dir / "trials.csv"
    trial_config_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest_header(manifest_path)

    rng = random.Random(args.sampler_seed)

    print(f"Sweep ID: {sweep_id}")
    print(f"Base model config: {model_config_path}")
    print(f"SBATCH script: {sbatch_script}")
    print(f"Manifest: {manifest_path}")
    print(f"Dry run: {args.dry_run}")
    print("")

    for trial_number in range(args.start_index, args.start_index + args.num_trials):
        trial_id = f"trial_{trial_number:03d}"
        override, flat = _sample_trial(rng, args.training_seed)
        config_path = trial_config_dir / f"{trial_id}.yaml"
        save_config(override, config_path)

        row = {
            "sweep_id": sweep_id,
            "trial_id": trial_id,
            "job_id": "",
            "status": "dry_run" if args.dry_run else "pending_submission",
            "submit_time_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "model_config_path": str(model_config_path.resolve()),
            "experiment_config_path": str(config_path.resolve()),
            **flat,
        }

        print(f"[{trial_id}] override config -> {config_path}")
        if args.dry_run:
            _append_manifest_row(manifest_path, row)
            continue

        try:
            job_id, sbatch_output = _submit_trial(
                sbatch_script=sbatch_script,
                model_config_path=model_config_path,
                experiment_config_path=config_path,
                trial_id=trial_id,
                sweep_id=sweep_id,
            )
            row["job_id"] = job_id
            row["status"] = "submitted"
            print(f"[{trial_id}] submitted as job {job_id}")
            print(f"[{trial_id}] sbatch: {sbatch_output}")
        except Exception as exc:  # noqa: BLE001
            row["status"] = "submission_failed"
            print(f"[{trial_id}] submission failed: {exc}")

        _append_manifest_row(manifest_path, row)

    print("")
    print("Random-search setup complete.")
    print(f"Trials manifest: {manifest_path}")
    print(
        "After jobs finish, aggregate metrics with "
        "scripts/analysis/parse_training_results.py --jobs-csv <manifest> --out-csv <results.csv>"
    )


if __name__ == "__main__":
    main()
