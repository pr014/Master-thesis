#!/usr/bin/env python3
"""
Builds ``EHR_feature_data_small.csv`` by selecting a safe subset of columns from
the already generated ``EHR_feature_data.csv``.

This keeps all values exactly aligned with the existing full EHR export while
producing a smaller table with only simple bedside features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IN_CSV = PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "EHR_feature_data.csv"
DEFAULT_OUT_CSV = PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "EHR_feature_data_small.csv"

METADATA_COLUMNS = [
    "base_path",
    "study_id",
    "subject_id",
    "ecg_time",
    "stay_id",
    "stay_intime",
    "hadm_id",
    "t_cut",
    "ehr_window_hours",
]

FEATURE_COLUMNS = [
    "respiratory_rate",
    "oxygen_saturation",
    "sbp",
    "dbp",
    "temperature",
    "weight",
]

FINAL_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create EHR_feature_data_small.csv as a column subset of EHR_feature_data.csv."
    )
    ap.add_argument("--in-csv", type=Path, default=DEFAULT_IN_CSV)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return ap.parse_args()


def build_small_ehr_table(in_csv: Path, out_csv: Path) -> pd.DataFrame:
    if not in_csv.exists():
        raise FileNotFoundError(f"Pfad fehlt: {in_csv}")

    df = pd.read_csv(in_csv, low_memory=False)
    missing = [col for col in FINAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"EHR_feature_data.csv fehlt Spalten: {missing}")

    if df["base_path"].astype(str).duplicated().any():
        raise ValueError("EHR_feature_data.csv hat doppelte base_path-Eintraege; Small-Export waere nicht eindeutig.")

    out = df.loc[:, FINAL_COLUMNS].copy()
    out["base_path"] = out["base_path"].astype(str)

    if "stay_id" in out.columns:
        out["stay_id"] = pd.to_numeric(out["stay_id"], errors="coerce").astype("Int64")
    if "study_id" in out.columns:
        out["study_id"] = pd.to_numeric(out["study_id"], errors="coerce").astype("Int64")
    if "subject_id" in out.columns:
        out["subject_id"] = pd.to_numeric(out["subject_id"], errors="coerce").astype("Int64")
    if "hadm_id" in out.columns:
        out["hadm_id"] = pd.to_numeric(out["hadm_id"], errors="coerce").astype("Int64")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def print_summary(out: pd.DataFrame, out_csv: Path) -> None:
    print(f"Datei: {out_csv}")
    print(f"Zeilen: {len(out):,}")
    print(f"Spalten: {len(out.columns)}")
    print("Feature-Spalten:")
    for col in FEATURE_COLUMNS:
        available = int(out[col].notna().sum())
        print(f"  {col}: {available:,} / {len(out):,} ({100.0 * available / len(out):.2f} %)")


def main() -> None:
    args = parse_args()
    try:
        out = build_small_ehr_table(args.in_csv, args.out_csv)
    except Exception as exc:  # pragma: no cover - CLI error path
        sys.exit(str(exc))

    print_summary(out, args.out_csv)


if __name__ == "__main__":
    main()
