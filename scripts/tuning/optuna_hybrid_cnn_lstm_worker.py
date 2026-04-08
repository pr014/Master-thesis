#!/usr/bin/env python3
"""Optuna worker for Hybrid CNN-LSTM hyperparameter search (SLURM-friendly).

Merges configs/tuning/hybrid_cnn_lstm/optuna_base_p1_tabular.yaml (default) with sampled
hyperparameters, writes one YAML per trial, runs train_hybrid_cnn_lstm.run_training,
minimizes validation LOS MAE.

Environment (typical):
  OPTUNA_STORAGE    - RDB URL, e.g. postgresql://... or sqlite:///abs/path/optuna.db
  OPTUNA_STUDY_NAME - study name (shared across workers)
  N_TRIALS_PER_JOB  - trials to run in this process (default: 1)

Optional smoke test (no GPU training; validates YAML + Optuna + CSV export):
  OPTUNA_HYBRID_SMOKE_OBJECTIVE=1

Per tuning run, use a unique OPTUNA_STUDY_NAME. Artifacts land under
outputs/tuning/hybrid_cnn_lstm/<YYYY-MM-DD>/<study-name>/:
  - trial_configs/    merged YAML per trial
  - exports/          optuna_trials.csv when --export-csv-after is enabled

Default base YAML: optuna_base_p1_tabular.yaml (demographics + ICU unit one-hot + SOFA). Override with
  --base-tuning-yaml .../optuna_base_p1_no_tabular.yaml for ECG-only runs.

Search space (P1 default, P2 wide — new study name required for P2):
  --search-space p1   Original knobs only (optimizer, dropout, scheduler, Huber delta, aug strengths).
  --search-space p2   P1 superset: CNN/LSTM shape presets, pooling, layers, epochs, early stopping,
                      scheduler min_lr, Adam betas, LOS loss type (mse/mae/huber), multi-task weights,
                      optional amplitude scaling. Dataset stays P1; preprocessing not tuned.
  Env: OPTUNA_HYBRID_SEARCH_SPACE=p1|p2 if CLI omitted.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
from optuna import pruners

from src.utils.config_loader import load_yaml, merge_configs, save_config

_TRAIN_PATH = PROJECT_ROOT / "scripts/training/icu_24h/hybrid_cnn_lstm/train_hybrid_cnn_lstm_24h.py"
_spec = importlib.util.spec_from_file_location("train_hybrid_cnn_lstm_24h", _TRAIN_PATH)
assert _spec and _spec.loader
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)
run_training = _train_mod.run_training


DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/model/hybrid_cnn_lstm/hybrid_cnn_lstm.yaml"
DEFAULT_BASE_TUNING = PROJECT_ROOT / "configs/tuning/hybrid_cnn_lstm/optuna_base_p1_tabular.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/tuning/hybrid_cnn_lstm"


def _study_exports_dir(study_dir: Path) -> Path:
    """Per-study exports (CSV, etc.). Keeps trial YAMLs under trial_configs/."""
    return study_dir / "exports"


def _normalize_output_date(value: Optional[str]) -> str:
    """Return YYYY-MM-DD for artifact subfolder."""
    raw = (value or os.getenv("OPTUNA_OUTPUT_DATE") or "").strip()
    if not raw:
        return date.today().isoformat()
    raw = raw.replace("/", "-")
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def _build_trial_override_p1(trial: optuna.Trial) -> Dict[str, Any]:
    opt_type = trial.suggest_categorical("optimizer_type", ["Adam", "AdamW"])
    lr = trial.suggest_float("lr", 3e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    grad_clip = trial.suggest_categorical("gradient_clip_norm", [0.5, 1.0, 2.0, 5.0])
    dropout_rate = trial.suggest_float("dropout_rate", 0.15, 0.55)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.15, 0.55)
    sched_factor = trial.suggest_categorical("scheduler_factor", [0.1, 0.2, 0.5])
    sched_patience = trial.suggest_int("scheduler_patience", 3, 10)
    loss_delta = trial.suggest_float("loss_delta", 0.5, 2.0)

    noise_std = trial.suggest_float("aug_noise_std", 0.01, 0.06)
    lead_dropout_prob = trial.suggest_float("aug_lead_dropout_prob", 0.05, 0.25)
    bw_freq_min = trial.suggest_float("aug_bw_freq_min", 0.1, 0.2)
    bw_freq_max = trial.suggest_float("aug_bw_freq_max", 0.35, 0.5)
    bw_amp_min = trial.suggest_float("aug_bw_amp_min", 0.02, 0.05)
    bw_amp_max = trial.suggest_float("aug_bw_amp_max", 0.06, 0.12)

    return {
        "model": {
            "lstm_dropout": lstm_dropout,
        },
        "training": {
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "gradient_clip_norm": grad_clip,
            "optimizer": {
                "type": opt_type,
                "lr": lr,
                "weight_decay": weight_decay,
            },
            "scheduler": {
                "factor": sched_factor,
                "patience": sched_patience,
            },
            "loss": {
                "delta": loss_delta,
            },
        },
        "data": {
            "augmentation": {
                "noise_std": noise_std,
                "lead_dropout_prob": lead_dropout_prob,
                "bw_freq_min": bw_freq_min,
                "bw_freq_max": max(bw_freq_max, bw_freq_min + 0.05),
                "bw_amp_min": bw_amp_min,
                "bw_amp_max": max(bw_amp_max, bw_amp_min + 0.01),
            },
        },
    }


_CNN_WIDTH_PRESETS: Dict[str, tuple[int, int, int]] = {
    "32_64_128": (32, 64, 128),
    "48_96_192": (48, 96, 192),
    "64_128_256": (64, 128, 256),
    "24_48_96": (24, 48, 96),
}

_LSTM_HIDDEN_PRESETS: Dict[str, tuple[int, int]] = {
    "64_64": (64, 64),
    "96_96": (96, 96),
    "128_128": (128, 128),
    "192_192": (192, 192),
    "128_96": (128, 96),
    "128_64": (128, 64),
    "192_128": (192, 128),
}


def _build_trial_override_p2(trial: optuna.Trial) -> Dict[str, Any]:
    """Wide search: P1-style knobs plus architecture, schedule tail, loss family, multi-task, amp aug."""
    opt_type = trial.suggest_categorical("optimizer_type", ["Adam", "AdamW"])
    lr = trial.suggest_float("lr", 3e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    betas_tag = trial.suggest_categorical("optimizer_betas", ["b09_0999", "b095_0999"])
    betas = [0.9, 0.999] if betas_tag == "b09_0999" else [0.95, 0.999]

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    grad_clip = trial.suggest_categorical("gradient_clip_norm", [0.5, 1.0, 2.0, 5.0])
    dropout_rate = trial.suggest_float("dropout_rate", 0.15, 0.55)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.15, 0.55)

    loss_type = trial.suggest_categorical("loss_type", ["mse", "mae", "huber"])
    loss_delta = trial.suggest_float("loss_delta", 0.5, 2.0)

    sched_factor = trial.suggest_categorical("scheduler_factor", [0.1, 0.2, 0.5])
    sched_patience = trial.suggest_int("scheduler_patience", 3, 10)
    min_lr = trial.suggest_float("scheduler_min_lr", 1e-7, 1e-5, log=True)

    num_epochs = trial.suggest_categorical("num_epochs", [40, 50, 60, 80])
    es_patience = trial.suggest_int("early_stopping_patience", 7, 20)

    num_layers = trial.suggest_categorical("num_layers", [1, 2])
    pooling = trial.suggest_categorical("pooling", ["mean", "last", "max"])

    lstm_key = trial.suggest_categorical(
        "lstm_hidden_preset",
        list(_LSTM_HIDDEN_PRESETS.keys()),
    )
    h1, h2 = _LSTM_HIDDEN_PRESETS[lstm_key]

    cnn_key = trial.suggest_categorical(
        "cnn_width_preset",
        list(_CNN_WIDTH_PRESETS.keys()),
    )
    c1, c2, c3 = _CNN_WIDTH_PRESETS[cnn_key]

    noise_std = trial.suggest_float("aug_noise_std", 0.01, 0.06)
    lead_dropout_prob = trial.suggest_float("aug_lead_dropout_prob", 0.05, 0.25)
    bw_freq_min = trial.suggest_float("aug_bw_freq_min", 0.1, 0.2)
    bw_freq_max = trial.suggest_float("aug_bw_freq_max", 0.35, 0.5)
    bw_amp_min = trial.suggest_float("aug_bw_amp_min", 0.02, 0.05)
    bw_amp_max = trial.suggest_float("aug_bw_amp_max", 0.06, 0.12)
    bw_freq_max = max(bw_freq_max, bw_freq_min + 0.05)
    bw_amp_max = max(bw_amp_max, bw_amp_min + 0.01)

    amp_on = trial.suggest_categorical("aug_amplitude_scaling", [False, True])
    scale_min = trial.suggest_float("aug_scale_min", 0.82, 0.94)
    scale_max = trial.suggest_float("aug_scale_max", 1.06, 1.22)
    scale_max_adj = max(scale_max, scale_min + 0.05)

    aug_block: Dict[str, Any] = {
        "noise_std": noise_std,
        "lead_dropout_prob": lead_dropout_prob,
        "bw_freq_min": bw_freq_min,
        "bw_freq_max": bw_freq_max,
        "bw_amp_min": bw_amp_min,
        "bw_amp_max": bw_amp_max,
        "amplitude_scaling": amp_on,
    }
    if amp_on:
        aug_block["scale_min"] = scale_min
        aug_block["scale_max"] = scale_max_adj

    los_w = trial.suggest_float("los_loss_weight", 0.5, 2.0, log=True)
    mort_w = trial.suggest_float("mortality_loss_weight", 0.25, 2.0, log=True)

    loss_section: Dict[str, Any] = {"type": loss_type}
    if loss_type == "huber":
        loss_section["delta"] = loss_delta

    return {
        "model": {
            "lstm_dropout": lstm_dropout,
            "num_layers": num_layers,
            "pooling": pooling,
            "hidden_dim": h1,
            "hidden_dim_layer2": h2,
            "conv1_out": c1,
            "conv2_out": c2,
            "conv3_out": c3,
        },
        "training": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "gradient_clip_norm": grad_clip,
            "optimizer": {
                "type": opt_type,
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": betas,
            },
            "scheduler": {
                "factor": sched_factor,
                "patience": sched_patience,
                "min_lr": min_lr,
            },
            "loss": loss_section,
        },
        "early_stopping": {
            "patience": es_patience,
        },
        "multi_task": {
            "los_loss_weight": los_w,
            "mortality_loss_weight": mort_w,
        },
        "data": {
            "augmentation": aug_block,
        },
    }


def _resolve_search_space(cli_value: Optional[str]) -> str:
    raw = (cli_value or "").strip().lower()
    if not raw:
        raw = os.getenv("OPTUNA_HYBRID_SEARCH_SPACE", "").strip().lower()
    return "p2" if raw == "p2" else "p1"


def _trial_builder_for_space(space: str) -> Callable[[optuna.Trial], Dict[str, Any]]:
    if space == "p2":
        return _build_trial_override_p2
    return _build_trial_override_p1


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna worker for Hybrid CNN-LSTM")
    parser.add_argument("--storage", type=str, default=os.getenv("OPTUNA_STORAGE", ""))
    parser.add_argument("--study-name", type=str, default=os.getenv("OPTUNA_STUDY_NAME", "hybrid_cnn_lstm_default"))
    parser.add_argument("--n-trials", type=int, default=int(os.getenv("N_TRIALS_PER_JOB", "1")))
    parser.add_argument("--model-config", type=str, default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument(
        "--base-tuning-yaml",
        type=str,
        default=str(DEFAULT_BASE_TUNING),
        help="Merged before trial overrides. Default: optuna_base_p1_tabular (demographics + ICU unit + SOFA). "
        "Use optuna_base_p1_no_tabular.yaml for ECG-only ablations.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root: <output-root>/<YYYY-MM-DD>/<study-name>/trial_configs/, exports/",
    )
    parser.add_argument(
        "--output-date",
        type=str,
        default=None,
        help="Date folder (YYYY-MM-DD or YYYYMMDD). Default: today or OPTUNA_OUTPUT_DATE.",
    )
    parser.add_argument(
        "--export-csv-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After optimize, write exports/optuna_trials.csv (default: on; --no-export-csv-after to skip).",
    )
    parser.add_argument(
        "--search-space",
        type=str,
        default=None,
        choices=["p1", "p2"],
        help="p1=narrow (default). p2=wide (+ arch, epochs, ES, min_lr, betas, loss type, multi-task, amp scaling). "
        "If omitted, use env OPTUNA_HYBRID_SEARCH_SPACE.",
    )
    args = parser.parse_args()

    if not args.storage.strip():
        raise SystemExit(
            "Set OPTUNA_STORAGE or pass --storage (e.g. postgresql://user:pass@host/db or sqlite:///abs/path.db)"
        )

    model_config_path = Path(args.model_config).resolve()
    base_tuning_path = Path(args.base_tuning_yaml).resolve()
    output_root = Path(args.output_root)
    run_date = _normalize_output_date(args.output_date)
    study_dir = output_root / run_date / args.study_name
    trial_config_dir = study_dir / "trial_configs"
    trial_config_dir.mkdir(parents=True, exist_ok=True)
    _study_exports_dir(study_dir).mkdir(parents=True, exist_ok=True)
    search_space = _resolve_search_space(args.search_space)
    trial_builder = _trial_builder_for_space(search_space)
    print(f"Output date folder: {run_date}")
    print(f"search_space={search_space}")
    print(f"Study artifacts: {study_dir}")
    print(f"  trial_configs -> {trial_config_dir}")
    print(f"  exports       -> {_study_exports_dir(study_dir)}")

    base_fragment = load_yaml(base_tuning_path)

    def objective(trial: optuna.Trial) -> float:
        override = trial_builder(trial)
        merged = merge_configs(copy.deepcopy(base_fragment), override)
        trial_tag = f"trial_{trial.number:04d}"
        trial_yaml = trial_config_dir / f"{trial_tag}.yaml"
        save_config(merged, trial_yaml)
        trial.set_user_attr("search_space", search_space)

        slurm = os.getenv("SLURM_JOB_ID", "local")
        os.environ["OPTUNA_TRIAL_JOB_ID"] = f"{slurm}_{trial.number}"

        if os.getenv("OPTUNA_HYBRID_SMOKE_OBJECTIVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            trial.set_user_attr("best_val_loss", 1.0)
            trial.set_user_attr("best_val_r2", 0.0)
            trial.set_user_attr("experiment_config_path", str(trial_yaml))
            trial.set_user_attr("job_id", os.environ.get("OPTUNA_TRIAL_JOB_ID", ""))
            trial.set_user_attr("test_los_mae", 2.0)
            trial.set_user_attr("test_los_r2", 0.0)
            return 2.5

        metrics = run_training(
            model_config_path=model_config_path,
            experiment_config_path=trial_yaml,
            trial_id=trial_tag,
            sweep_id=args.study_name,
            print_parameter_debug=False,
        )
        best_mae = float(metrics["best_val_mae"])
        trial.set_user_attr("best_val_loss", float(metrics["best_val_loss"]))
        trial.set_user_attr("best_val_r2", float(metrics["best_val_r2"]))
        trial.set_user_attr("experiment_config_path", str(trial_yaml))
        trial.set_user_attr("job_id", metrics.get("job_id", ""))
        if metrics.get("test_los_mae") is not None:
            trial.set_user_attr("test_los_mae", float(metrics["test_los_mae"]))
        if metrics.get("test_los_r2") is not None:
            trial.set_user_attr("test_los_r2", float(metrics["test_los_r2"]))
        return best_mae

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage.strip(),
        direction="minimize",
        load_if_exists=True,
        pruner=pruners.NopPruner(),
    )
    study.optimize(objective, n_trials=args.n_trials)

    if args.export_csv_after:
        _export_study_csv(study, _study_exports_dir(study_dir) / "optuna_trials.csv")

    try:
        print(f"Study finished. Best value: {study.best_value}, params: {study.best_params}")
    except ValueError:
        print("Study finished (no completed trials with a value yet).")


def _export_study_csv(study: optuna.Study, out_path: Path) -> None:
    import pandas as pd

    df = study.trials_dataframe()
    # Optuna returns a (0, 0) frame when the study has no trials yet; to_csv becomes a bare newline.
    if df.shape[1] == 0:
        df = pd.DataFrame(
            columns=[
                "number",
                "value",
                "datetime_start",
                "datetime_complete",
                "duration",
                "state",
            ]
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
