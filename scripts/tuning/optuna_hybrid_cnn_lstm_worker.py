#!/usr/bin/env python3
"""Optuna worker for Hybrid CNN-LSTM hyperparameter search (SLURM-friendly).

Merges configs/tuning/hybrid_cnn_lstm/optuna_base.yaml with sampled hyperparameters,
writes one YAML per trial, runs train_hybrid_cnn_lstm.run_training, and minimizes
**best validation total loss** ``min_t val_loss(t)`` — the same multi-task combined
loss (LOS + mortality) as in training when loss weights are fixed to 1:1.

MedianPruner receives per-epoch ``val_loss`` from the trainer (aligned with this objective).

Environment (typical):
  OPTUNA_STORAGE    - RDB URL, e.g. postgresql://... or sqlite:///abs/path/optuna.db
  OPTUNA_STUDY_NAME - study name (shared across workers)
  N_TRIALS_PER_JOB  - trials to run in this process (default: 1)

Optional smoke test (no GPU training; validates YAML + Optuna + CSV export):
  OPTUNA_HYBRID_SMOKE_OBJECTIVE=1

Study: ``TPESampler`` + ``MedianPruner`` — see ``scripts/tuning/hybrid_cnn_lstm_study_config.py``.

Per tuning run, use a unique OPTUNA_STUDY_NAME. Artifacts land under
outputs/tuning/hybrid_cnn_lstm/<YYYY-MM-DD>/<study-name>/:
  - trial_configs/    merged YAML per trial
  - exports/          optuna_trials.csv, optuna_best_trial.csv (when --export-csv-after)

Search space (hybrid CNN-LSTM):
  Multi-task training (LOS + mortality) with fixed loss weights 1/1 in optuna_base.
  Fixed via optuna_base.yaml + model YAML: num_epochs=200, Huber(delta=1) LOS loss,
  batch_size=64, CNN 32/64/128, num_layers=2, pooling=mean, los/mortality weights=1,
  early_stopping.patience=15, demographics + ICU unit + EHR window CSV from optuna_base, AG6
  numeric strengths, AdamW + betas [0.9,0.999], gradient_clip_norm=2.0,
  ReduceLROnPlateau (factor=0.5, patience=6, min_lr=1e-6).
  Tuned per trial: lr, weight_decay, dropout_rate, lstm_dropout, LSTM hidden
  preset among (64,64), (128,128), (256,256).
  Dataset stays P1; preprocessing is not tuned.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
_Tuning_DIR = PROJECT_ROOT / "scripts" / "tuning"
if str(_Tuning_DIR) not in sys.path:
    sys.path.insert(0, str(_Tuning_DIR))
import hybrid_cnn_lstm_study_config  # TPESampler + MedianPruner (shared with launch precreate)

import optuna

from src.utils.config_loader import load_yaml, merge_configs, save_config

_TRAIN_PATH = PROJECT_ROOT / "scripts/training/icu_24h/hybrid_cnn_lstm/train_hybrid_cnn_lstm_24h.py"
_spec = importlib.util.spec_from_file_location("train_hybrid_cnn_lstm_24h", _TRAIN_PATH)
assert _spec and _spec.loader
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)
run_training = _train_mod.run_training


DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/model/hybrid_cnn_lstm/hybrid_cnn_lstm.yaml"
DEFAULT_BASE_TUNING = PROJECT_ROOT / "configs/tuning/hybrid_cnn_lstm/optuna_base.yaml"
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


# Symmetric LSTM hidden sizes (layer1, layer2); tuned via suggest_categorical keys.
_LSTM_HIDDEN_PRESETS: Dict[str, tuple[int, int]] = {
    "64_64": (64, 64),
    "128_128": (128, 128),
    "256_256": (256, 256),
}


def _max_val_mortality_auc(history: Dict[str, Any]) -> float:
    """Max validation mortality AUC over epochs (auxiliary logging for analysis)."""
    series = history.get("val_mortality_auc") or []
    best: Optional[float] = None
    for x in series:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN
            continue
        if best is None or v > best:
            best = v
    return 0.5 if best is None else best


def _set_mortality_user_attrs_from_history(trial: optuna.Trial, history: Dict[str, Any]) -> None:
    """Log test and validation mortality metrics for Optuna CSV (multi-task)."""
    float_keys = [
        ("test_mortality_auc", "test_mortality_auc"),
        ("test_mortality_acc", "test_mortality_acc"),
        ("test_mortality_f1", "test_mortality_f1"),
        ("test_mortality_precision", "test_mortality_precision"),
        ("test_mortality_recall", "test_mortality_recall"),
        ("test_mortality_threshold", "test_mortality_threshold"),
        ("test_mortality_auc_leq10", "test_mortality_auc_leq10"),
    ]
    for attr, hkey in float_keys:
        v = history.get(hkey)
        if v is not None:
            trial.set_user_attr(attr, float(v))
    n_stays = history.get("test_mortality_n_stays_leq10")
    if n_stays is not None:
        trial.set_user_attr("test_mortality_n_stays_leq10", int(n_stays))

    val_los_mae: List[Any] = history.get("val_los_mae") or []
    val_mort_acc: List[Any] = history.get("val_mortality_acc") or []
    val_mort_auc: List[Any] = history.get("val_mortality_auc") or []
    if val_los_mae and val_mort_acc and len(val_mort_acc) == len(val_los_mae):
        best_i = min(range(len(val_los_mae)), key=lambda i: float(val_los_mae[i]))
        trial.set_user_attr("val_mortality_acc_at_best_los_mae", float(val_mort_acc[best_i]))
    if val_los_mae and val_mort_auc and len(val_mort_auc) == len(val_los_mae):
        best_i = min(range(len(val_los_mae)), key=lambda i: float(val_los_mae[i]))
        trial.set_user_attr("val_mortality_auc_at_best_los_mae", float(val_mort_auc[best_i]))


def _build_trial_override(trial: optuna.Trial) -> Dict[str, Any]:
    """Build the single Hybrid CNN-LSTM Optuna search-space override.

    Fixed architecture, batch, Huber loss, pooling, layer count, CNN widths, and
    multi-task loss weights live in optuna_base.yaml; this dict only supplies tuned keys.
    """
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    dropout_rate = trial.suggest_float("dropout_rate", 0.25, 0.5)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.2, 0.5)

    lstm_key = trial.suggest_categorical(
        "lstm_hidden_preset",
        list(_LSTM_HIDDEN_PRESETS.keys()),
    )
    h1, h2 = _LSTM_HIDDEN_PRESETS[lstm_key]

    return {
        "model": {
            "lstm_dropout": lstm_dropout,
            "hidden_dim": h1,
            "hidden_dim_layer2": h2,
        },
        "training": {
            "dropout_rate": dropout_rate,
            "optimizer": {
                "lr": lr,
                "weight_decay": weight_decay,
            },
        },
    }


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
        help="Merged before trial overrides. Default: optuna_base.yaml "
        "(demographics + ICU unit + EHR window table from that YAML + fixed AG6 / AdamW / scheduler / grad clip).",
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
        help="After optimize, write exports/optuna_trials.csv and optuna_best_trial.csv "
        "(default: on; --no-export-csv-after to skip).",
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
    print(f"Output date folder: {run_date}")
    print(f"Study artifacts: {study_dir}")
    print(f"  trial_configs -> {trial_config_dir}")
    print(f"  exports       -> {_study_exports_dir(study_dir)}")
    print(f"Optuna study: {hybrid_cnn_lstm_study_config.study_config_summary()}")

    base_fragment = load_yaml(base_tuning_path)

    def objective(trial: optuna.Trial) -> float:
        override = _build_trial_override(trial)
        # merge_configs(base, override): nested dicts merge; trial keys win only where sampled.
        # Training then load_config(model_yaml, trial_yaml): experiment wins over model —
        # optuna_base therefore overrides hybrid_cnn_lstm.yaml for keys defined here.
        merged = merge_configs(copy.deepcopy(base_fragment), override)
        trial_tag = f"trial_{trial.number:04d}"
        trial_yaml = trial_config_dir / f"{trial_tag}.yaml"
        save_config(merged, trial_yaml)
        trial.set_user_attr("tuning_setup", "hybrid_cnn_lstm_optuna")

        slurm = os.getenv("SLURM_JOB_ID", "local")
        os.environ["OPTUNA_TRIAL_JOB_ID"] = f"{slurm}_{trial.number}"

        if os.getenv("OPTUNA_HYBRID_SMOKE_OBJECTIVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            trial.set_user_attr("best_val_loss", 1.0)
            trial.set_user_attr("best_val_mae", 2.5)
            trial.set_user_attr("best_val_r2", 0.0)
            trial.set_user_attr("aux_val_mortality_auc_max_epochs", 0.65)
            trial.set_user_attr("experiment_config_path", str(trial_yaml))
            trial.set_user_attr("job_id", os.environ.get("OPTUNA_TRIAL_JOB_ID", ""))
            trial.set_user_attr("test_los_mae", 2.0)
            trial.set_user_attr("test_los_r2", 0.0)
            trial.set_user_attr("test_mortality_auc", 0.5)
            trial.set_user_attr("test_mortality_f1", 0.0)
            trial.set_user_attr("val_mortality_auc_at_best_los_mae", 0.5)
            return 1.0

        metrics = run_training(
            model_config_path=model_config_path,
            experiment_config_path=trial_yaml,
            trial_id=trial_tag,
            sweep_id=args.study_name,
            print_parameter_debug=False,
            optuna_trial=trial,
        )
        best_loss = float(metrics["best_val_loss"])
        best_mae = float(metrics["best_val_mae"])
        hist = metrics.get("history") or {}
        max_val_mort_auc = (
            _max_val_mortality_auc(hist) if isinstance(hist, dict) else 0.5
        )

        trial.set_user_attr("best_val_loss", best_loss)
        trial.set_user_attr("best_val_mae", best_mae)
        trial.set_user_attr("best_val_r2", float(metrics["best_val_r2"]))
        trial.set_user_attr("aux_val_mortality_auc_max_epochs", max_val_mort_auc)
        trial.set_user_attr("experiment_config_path", str(trial_yaml))
        trial.set_user_attr("job_id", metrics.get("job_id", ""))
        if metrics.get("test_los_mae") is not None:
            trial.set_user_attr("test_los_mae", float(metrics["test_los_mae"]))
        if metrics.get("test_los_r2") is not None:
            trial.set_user_attr("test_los_r2", float(metrics["test_los_r2"]))
        if isinstance(hist, dict):
            _set_mortality_user_attrs_from_history(trial, hist)
        return best_loss

    study = hybrid_cnn_lstm_study_config.create_hybrid_study(
        study_name=args.study_name,
        storage=args.storage.strip(),
    )
    study.optimize(objective, n_trials=args.n_trials)

    if args.export_csv_after:
        _export_study_csv(study, _study_exports_dir(study_dir) / "optuna_trials.csv")
        _export_best_trial_csv(study, _study_exports_dir(study_dir) / "optuna_best_trial.csv")

    try:
        print(f"Study finished. Best value (min best val_loss): {study.best_value}, params: {study.best_params}")
    except (ValueError, RuntimeError) as exc:
        print(f"Study finished (could not read best trial: {exc}).")


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


def _export_best_trial_csv(study: optuna.Study, out_path: Path) -> None:
    import pandas as pd

    try:
        trial = study.best_trial
    except ValueError:
        df = pd.DataFrame(columns=["number", "value", "state"])
    else:
        row: Dict[str, Any] = {
            "number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
        }
        for key, value in trial.params.items():
            row[f"param_{key}"] = value
        for key, value in trial.user_attrs.items():
            row[f"user_attr_{key}"] = value
        df = pd.DataFrame([row])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
