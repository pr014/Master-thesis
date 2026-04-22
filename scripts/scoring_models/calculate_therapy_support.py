"""
Therapie-Flags pro ECG (MIMIC-IV): Beatmung, NIV/BiPAP/CPAP, Katecholamine,
Vasopressin/Phenylephrin, Dialyse/CRRT.

Zeitlogik wie SOFA-pro-ECG: charttime bzw. Infusionsüberlappung nur bis ecg_time, pro stay_id.

Schreibt data/labeling/labels_csv/icu_therapy_support.csv (eine Zeile pro base_path).

Usage:
    python scripts/scoring_models/calculate_therapy_support.py
    python scripts/scoring_models/calculate_therapy_support.py --skip-path-validation
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scoring_models import config  # noqa: E402
from src.scoring_models.sofa.data_loader import build_therapy_support_export  # noqa: E402
from src.scoring_models.sofa.ecg_stay_mapping import build_ecg_stay_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = config.get_project_root()
    labels = root / "data" / "labeling" / "labels_csv"
    p = argparse.ArgumentParser(
        description="Therapie-Flags pro ECG (≤ ecg_time, leakage-frei)"
    )
    p.add_argument(
        "--timestamps-csv",
        type=Path,
        default=labels / "timestamps_mapping_24h_P1.csv",
        help="Timestamp-Mapping (base_path, subject_id, ecg-Zeit)",
    )
    p.add_argument(
        "--icustays-csv",
        type=Path,
        default=labels / "icustays.csv",
        help="icustays für ICUStayMapper",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=labels / "icu_therapy_support.csv",
        help="Ausgabe: eine Zeile pro ECG",
    )
    p.add_argument(
        "--skip-path-validation",
        action="store_true",
        help="MIMIC-Pfadprüfung überspringen",
    )
    return p.parse_args()


def print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    avail = df["therapy_labels_available"].fillna(False)
    n_av = int(avail.sum())
    print("\n" + "=" * 70)
    print("THERAPIE-FLAGS PRO ECG")
    print("=" * 70)
    print(f"  Zeilen gesamt:                    {n:,}")
    print(f"  therapy_labels_available=True:    {n_av:,}")
    print(f"  therapy_stay_matched (stay_id):   {int(df['stay_id'].notna().sum()):,}")
    if n_av > 0:
        sub = df.loc[avail]
        print(f"  mech_vent==1:                     {int((sub['mech_vent'] == 1).sum()):,}")
        print(f"  niv_hfnc==1:                      {int((sub['niv_hfnc'] == 1).sum()):,}")
        print(f"  vaso_any==1:                      {int((sub['vaso_any'] == 1).sum()):,}")
        print(f"  vaso_non_catechol_any==1:         {int((sub['vaso_non_catechol_any'] == 1).sum()):,}")
        print(f"  rrt==1:                           {int((sub['rrt'] == 1).sum()):,}")
    print("=" * 70 + "\n")


def main() -> int:
    args = parse_args()
    print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if not args.timestamps_csv.is_file():
        print(f"❌ Nicht gefunden: {args.timestamps_csv}")
        return 1
    if not args.icustays_csv.is_file():
        print(f"❌ Nicht gefunden: {args.icustays_csv}")
        return 1

    if not args.skip_path_validation:
        if not config.validate_paths():
            print("\n❌ MIMIC-IV Pfade ungültig (oder --skip-path-validation).")
            return 1
    else:
        print("(Pfadvalidierung übersprungen)")

    print("ECG → stay_id …")
    ecg_mapping = build_ecg_stay_mapping(args.timestamps_csv, args.icustays_csv)
    print(f"  {len(ecg_mapping):,} ECG-Zeilen, {ecg_mapping['stay_id'].notna().sum():,} mit stay_id\n")

    print("MIMIC chartevents + inputevents (Therapie) …")
    try:
        export_df = build_therapy_support_export(ecg_mapping)
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback

        traceback.print_exc()
        return 1

    cols = [
        "base_path",
        "subject_id",
        "ecg_time",
        "stay_id",
        "hadm_id",
        "therapy_stay_matched",
        "therapy_labels_available",
        "mech_vent",
        "niv_hfnc",
        "vaso_any",
        "vaso_non_catechol_any",
        "rrt",
    ]
    if "study_id" in export_df.columns:
        cols.insert(2, "study_id")
    cols = [c for c in cols if c in export_df.columns]
    out = export_df[cols].copy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"✓ Gespeichert: {args.output_csv} ({len(out):,} Zeilen)")

    print_summary(out)
    print(f"Ende: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
