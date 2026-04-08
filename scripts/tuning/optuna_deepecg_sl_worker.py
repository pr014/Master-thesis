#!/usr/bin/env python3
"""Optuna worker for DeepECG-SL hyperparameter search (SLURM-friendly).

Merges a tuning base YAML with sampled hyperparameters, runs train_deepecg_sl.run_training,
minimizes validation LOS MAE.

Default base: configs/tuning/deepecg_sl/optuna_base_p1_tabular.yaml (demographics + ICU unit + SOFA).
Override with --base-tuning-yaml .../optuna_base_p1_no_tabular.yaml for ECG-only runs.

Search space (tuned for WCR + tabular + multi-task; wide P2 for high-compute final runs):
  --search-space p1   Huber-only LOS loss, backbone/head LR, aug, scheduler, batch up to 64.
  --search-space p2   + loss type (mse/mae/huber), Huber delta (sampled always; applied if huber),
                      three Adam beta presets (incl. beta2=0.98), min_lr, long epoch grid, wide ES.
                      WCR architecture fixed. Amplitude aug stays off in base YAML.

Environment:
  OPTUNA_STORAGE, OPTUNA_STUDY_NAME, N_TRIALS_PER_JOB
  OPTUNA_OUTPUT_DATE — artifact date folder
  OPTUNA_DEEPECG_SEARCH_SPACE=p1|p2 if --search-space omitted
  OPTUNA_DEEPECG_SMOKE_OBJECTIVE=1 — skip GPU training (YAML + Optuna smoke test)

Per tuning run, use a unique OPTUNA_STUDY_NAME when changing search space or base data setup.
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

_TRAIN_PATH = PROJECT_ROOT / "scripts/training/icu_24h/deepecg_sl/train_deepecg_sl_24h.py"
_spec = importlib.util.spec_from_file_location("train_deepecg_sl_24h", _TRAIN_PATH)
assert _spec and _spec.loader
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)
run_training = _train_mod.run_training


DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/model/deepecg_sl/deepecg_sl.yaml"
DEFAULT_BASE_TUNING = PROJECT_ROOT / "configs/tuning/deepecg_sl/optuna_base_p1_tabular.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/tuning/deepecg_sl"


def _study_exports_dir(study_dir: Path) -> Path:
    """Per-study exports (CSV, etc.). Keeps trial YAMLs under trial_configs/."""
    return study_dir / "exports"


def _normalize_output_date(value: Optional[str]) -> str:
    """Return ``YYYY-MM-DD`` for artifact subfolder."""
    raw = (value or os.getenv("OPTUNA_OUTPUT_DATE") or "").strip()
    if not raw:
        return date.today().isoformat()
    raw = raw.replace("/", "-")
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def _build_trial_override_p1(trial: optuna.Trial) -> Dict[str, Any]:
    opt_type = trial.suggest_categorical("optimizer_type", ["Adam", "AdamW"])
    backbone_lr = trial.suggest_float("backbone_lr", 5e-7, 1.5e-4, log=True)
    head_lr = trial.suggest_float("head_lr", 5e-6, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    grad_clip = trial.suggest_categorical("gradient_clip_norm", [0.5, 1.0, 2.0, 5.0])
    dropout_rate = trial.suggest_float("dropout_rate", 0.03, 0.45)
    sched_factor = trial.suggest_categorical("scheduler_factor", [0.1, 0.2, 0.25, 0.5])
    sched_patience = trial.suggest_int("scheduler_patience", 2, 14)
    loss_delta = trial.suggest_float("loss_delta", 0.25, 3.5)

    noise_std = trial.suggest_float("aug_noise_std", 0.005, 0.07)
    lead_dropout_prob = trial.suggest_float("aug_lead_dropout_prob", 0.03, 0.28)
    bw_freq_min = trial.suggest_float("aug_bw_freq_min", 0.08, 0.22)
    bw_freq_max = trial.suggest_float("aug_bw_freq_max", 0.32, 0.52)
    bw_amp_min = trial.suggest_float("aug_bw_amp_min", 0.015, 0.055)
    bw_amp_max = trial.suggest_float("aug_bw_amp_max", 0.05, 0.14)

    return {
        "training": {
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "gradient_clip_norm": grad_clip,
            "optimizer": {
                "type": opt_type,
                "lr": head_lr,
                "backbone_lr": backbone_lr,
                "head_lr": head_lr,
                "weight_decay": weight_decay,
            },
            "scheduler": {
                "factor": sched_factor,
                "patience": sched_patience,
            },
            "loss": {
                "type": "huber",
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


def _build_trial_override_p2(trial: optuna.Trial) -> Dict[str, Any]:
    """Wide search for WCR DeepECG-SL: full training budget, loss family, multi-task balance."""
    opt_type = trial.suggest_categorical("optimizer_type", ["Adam", "AdamW"])
    backbone_lr = trial.suggest_float("backbone_lr", 5e-7, 1.5e-4, log=True)
    head_lr = trial.suggest_float("head_lr", 5e-6, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    betas_tag = trial.suggest_categorical(
        "optimizer_betas",
        ["b09_0999", "b095_0999", "b09_098"],
    )
    if betas_tag == "b09_0999":
        betas = [0.9, 0.999]
    elif betas_tag == "b095_0999":
        betas = [0.95, 0.999]
    else:
        betas = [0.9, 0.98]

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    grad_clip = trial.suggest_categorical("gradient_clip_norm", [0.5, 1.0, 2.0, 5.0])
    dropout_rate = trial.suggest_float("dropout_rate", 0.03, 0.5)

    loss_type = trial.suggest_categorical("loss_type", ["mse", "mae", "huber"])
    # Always sample (stable for TPE / parallel workers); only used if loss_type == huber.
    loss_delta = trial.suggest_float("loss_delta", 0.25, 3.5)

    sched_factor = trial.suggest_categorical("scheduler_factor", [0.1, 0.2, 0.25, 0.5])
    sched_patience = trial.suggest_int("scheduler_patience", 2, 14)
    min_lr = trial.suggest_float("scheduler_min_lr", 1e-8, 5e-5, log=True)

    num_epochs = trial.suggest_categorical("num_epochs", [30, 40, 50, 60, 80, 100])
    es_patience = trial.suggest_int("early_stopping_patience", 8, 28)

    los_w = trial.suggest_float("los_loss_weight", 0.3, 4.0, log=True)
    mort_w = trial.suggest_float("mortality_loss_weight", 0.15, 4.0, log=True)

    noise_std = trial.suggest_float("aug_noise_std", 0.005, 0.07)
    lead_dropout_prob = trial.suggest_float("aug_lead_dropout_prob", 0.03, 0.28)
    bw_freq_min = trial.suggest_float("aug_bw_freq_min", 0.08, 0.22)
    bw_freq_max = trial.suggest_float("aug_bw_freq_max", 0.32, 0.52)
    bw_amp_min = trial.suggest_float("aug_bw_amp_min", 0.015, 0.055)
    bw_amp_max = trial.suggest_float("aug_bw_amp_max", 0.05, 0.14)
    bw_freq_max = max(bw_freq_max, bw_freq_min + 0.05)
    bw_amp_max = max(bw_amp_max, bw_amp_min + 0.01)

    loss_section: Dict[str, Any] = {"type": loss_type}
    if loss_type == "huber":
        loss_section["delta"] = loss_delta

    return {
        "training": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "gradient_clip_norm": grad_clip,
            "optimizer": {
                "type": opt_type,
                "lr": head_lr,
                "backbone_lr": backbone_lr,
                "head_lr": head_lr,
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
            "augmentation": {
                "noise_std": noise_std,
                "lead_dropout_prob": lead_dropout_prob,
                "bw_freq_min": bw_freq_min,
                "bw_freq_max": bw_freq_max,
                "bw_amp_min": bw_amp_min,
                "bw_amp_max": bw_amp_max,
            },
        },
    }


def _resolve_search_space(cli_value: Optional[str]) -> str:
    raw = (cli_value or "").strip().lower()
    if not raw:
        raw = os.getenv("OPTUNA_DEEPECG_SEARCH_SPACE", "p2").strip().lower()
    return "p2" if raw == "p2" else "p1"


def _trial_builder_for_space(space: str) -> Callable[[optuna.Trial], Dict[str, Any]]:
    if space == "p2":
        return _build_trial_override_p2
    return _build_trial_override_p1


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna worker for DeepECG-SL")
    parser.add_argument("--storage", type=str, default=os.getenv("OPTUNA_STORAGE", ""))
    parser.add_argument("--study-name", type=str, default=os.getenv("OPTUNA_STUDY_NAME", "deepecg_sl_default"))
    parser.add_argument("--n-trials", type=int, default=int(os.getenv("N_TRIALS_PER_JOB", "1")))
    parser.add_argument("--model-config", type=str, default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument(
        "--base-tuning-yaml",
        type=str,
        default=str(DEFAULT_BASE_TUNING),
        help="Merged before trial overrides. Default: optuna_base_p1_tabular.yaml.",
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
        help="p1=narrow Huber-only LOS loss. p2=+ loss type, multi-task weights, betas, min_lr, epochs, ES. "
        "If omitted, env OPTUNA_DEEPECG_SEARCH_SPACE (default p2).",
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

        if os.getenv("OPTUNA_DEEPECG_SMOKE_OBJECTIVE", "").strip().lower() in ("1", "true", "yes"):
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
