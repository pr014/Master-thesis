#!/usr/bin/env python3
"""Export an Optuna study to CSV (params, value, user_attrs, state).

Example:
  python scripts/tuning/export_optuna_deepecg_sl_study.py \\
    --storage sqlite:////path/to/optuna.db \\
    --study-name deepecg_sl_default \\
    --out-csv outputs/tuning/deepecg_sl/2026-04-02/deepecg_sl_default/exports/optuna_trials.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Optuna study to CSV")
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage.strip())
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(study.trials)} trials)")


if __name__ == "__main__":
    main()
