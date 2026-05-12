#!/usr/bin/env python3
"""Build a compact CSV: tuned hyperparameters + key metrics from Optuna trials export.

Reads ``exports/optuna_trials.csv`` (from ``study.trials_dataframe()``) and writes a
table with readable column names. PRUNED trials keep params and ``objective_best_val_loss``
but test metrics are empty (training stopped early).

Example:
  python scripts/tuning/build_hybrid_optuna_combinations_metrics_csv.py \\
    --in-csv outputs/tuning/hybrid_cnn_lstm/2026-05-08/hybrid_cnn_lstm_hpo_tab_20260508/exports/optuna_trials.csv \\
    --out-csv outputs/tuning/hybrid_cnn_lstm/2026-05-08/combinations_metrics_hybrid_cnn_lstm_hpo_tab_20260508.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Source columns from optuna.trials_dataframe() (hybrid worker)
_SOURCE_COLS = [
    "number",
    "state",
    "duration",
    "datetime_start",
    "datetime_complete",
    "params_lr",
    "params_weight_decay",
    "params_dropout_rate",
    "params_lstm_dropout",
    "params_lstm_hidden_preset",
    "value",
    "user_attrs_best_val_loss",
    "user_attrs_best_val_mae",
    "user_attrs_best_val_r2",
    "user_attrs_test_los_mae",
    "user_attrs_test_los_r2",
    "user_attrs_test_mortality_auc",
    "user_attrs_test_mortality_f1",
    "user_attrs_test_mortality_acc",
    "user_attrs_aux_val_mortality_auc_max_epochs",
    "user_attrs_val_mortality_auc_at_best_los_mae",
    "user_attrs_val_mortality_acc_at_best_los_mae",
    "user_attrs_experiment_config_path",
    "user_attrs_job_id",
]

_RENAME = {
    "number": "trial_number",
    "state": "state",
    "duration": "duration",
    "datetime_start": "datetime_start",
    "datetime_complete": "datetime_complete",
    "params_lr": "lr",
    "params_weight_decay": "weight_decay",
    "params_dropout_rate": "dropout_rate",
    "params_lstm_dropout": "lstm_dropout",
    "params_lstm_hidden_preset": "lstm_hidden_preset",
    "value": "objective_best_val_loss",
    "user_attrs_best_val_loss": "best_val_loss",
    "user_attrs_best_val_mae": "best_val_mae_days",
    "user_attrs_best_val_r2": "best_val_r2",
    "user_attrs_test_los_mae": "test_los_mae_days",
    "user_attrs_test_los_r2": "test_los_r2",
    "user_attrs_test_mortality_auc": "test_mortality_auc",
    "user_attrs_test_mortality_f1": "test_mortality_f1",
    "user_attrs_test_mortality_acc": "test_mortality_acc",
    "user_attrs_aux_val_mortality_auc_max_epochs": "aux_max_val_mortality_auc",
    "user_attrs_val_mortality_auc_at_best_los_mae": "val_mortality_auc_at_best_los_mae",
    "user_attrs_val_mortality_acc_at_best_los_mae": "val_mortality_acc_at_best_los_mae",
    "user_attrs_experiment_config_path": "trial_config_path",
    "user_attrs_job_id": "slurm_or_trial_job_id",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Optuna trials -> compact combinations+metrics CSV")
    parser.add_argument(
        "--in-csv",
        type=str,
        default="outputs/tuning/hybrid_cnn_lstm/2026-05-08/hybrid_cnn_lstm_hpo_tab_20260508/exports/optuna_trials.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="outputs/tuning/hybrid_cnn_lstm/2026-05-08/combinations_metrics_hybrid_cnn_lstm_hpo_tab_20260508.csv",
    )
    args = parser.parse_args()

    in_path = Path(args.in_csv).resolve()
    out_path = Path(args.out_csv).resolve()
    df = pd.read_csv(in_path)

    missing = [c for c in _SOURCE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Input CSV missing columns: {missing}")

    sub = df[_SOURCE_COLS].copy()
    sub = sub.rename(columns=_RENAME)
    sub.insert(
        1,
        "trial_config_basename",
        sub["trial_config_path"].map(lambda p: Path(str(p)).name if pd.notna(p) and str(p) else ""),
    )

    def _format_job_id(x: object) -> str:
        if x is None or pd.isna(x):
            return ""
        try:
            xf = float(x)
            if xf == int(xf):
                return str(int(xf))
        except (TypeError, ValueError):
            pass
        return str(x)

    sub["slurm_or_trial_job_id"] = sub["slurm_or_trial_job_id"].map(_format_job_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
