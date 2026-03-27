"""
SOFA Score Berechnung für die 24h-ECG-Kohorte (MIMIC-IV).

Pro ECG: Nur Messwerte mit charttime/starttime ≤ ecg_time (und ≥ ICU-intime).
Leakage-frei für Late Fusion mit ECG zum Aufnahmezeitpunkt.

Schreibt data/labeling/labels_csv/sofa_scores.csv (eine Zeile pro base_path/ECG).

Usage:
    python scripts/scoring_models/calculate_sofa.py
    python scripts/scoring_models/calculate_sofa.py --skip-path-validation
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Projekt-Root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scoring_models import config, utils  # noqa: E402
from src.scoring_models.sofa import calculator, itemid_mappings  # noqa: E402
from src.scoring_models.sofa.data_loader import load_sofa_data_per_ecg  # noqa: E402
from src.scoring_models.sofa.ecg_stay_mapping import (  # noqa: E402
    build_ecg_stay_mapping,
)

# Klinische Inputs für Mindestabdeckung (siehe config.MIN_COMPONENTS_FOR_VALID_SCORE)
SOFA_INPUT_COLUMNS = [
    "pao2_fio2_ratio",
    "platelets",
    "bilirubin",
    "map",
    "gcs",
    "creatinine",
    "urine_output_24h",
]

SOFA_SCORE_COLUMNS = [
    "sofa_respiration",
    "sofa_coagulation",
    "sofa_liver",
    "sofa_cardiovascular",
    "sofa_cns",
    "sofa_renal",
    "sofa_total",
]


def _count_available_inputs(row: pd.Series, columns: list[str]) -> int:
    n = 0
    for c in columns:
        if c not in row.index:
            continue
        v = row[c]
        if pd.notna(v):
            n += 1
    return n


def apply_sofa_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Set sofa_available; wenn zu wenig Messungen: SOFA-Spalten auf NaN."""
    out = df.copy()
    min_c = config.MIN_COMPONENTS_FOR_VALID_SCORE
    counts = out.apply(
        lambda r: _count_available_inputs(r, SOFA_INPUT_COLUMNS), axis=1
    )
    out["sofa_available"] = counts >= min_c
    mask = ~out["sofa_available"]
    for c in SOFA_SCORE_COLUMNS:
        if c in out.columns:
            out.loc[mask, c] = np.nan
    return out


def merge_ecg_mapping_with_scores(
    ecg_mapping: pd.DataFrame,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Eine Zeile pro ECG aus ecg_mapping; SOFA per left join auf base_path."""
    base = ecg_mapping.copy()
    if len(scored) == 0:
        for c in SOFA_SCORE_COLUMNS:
            base[c] = np.nan
        base["sofa_available"] = False
        base["hadm_id"] = np.nan
        return base

    scored_u = scored.drop_duplicates(subset=["base_path"], keep="first")
    extra = ["hadm_id"] + SOFA_SCORE_COLUMNS + ["sofa_available"]
    extra = [c for c in extra if c in scored_u.columns]
    right = scored_u[["base_path"] + extra].copy()
    merged = base.merge(right, on="base_path", how="left")
    if "sofa_available" in merged.columns:
        merged["sofa_available"] = merged["sofa_available"].fillna(False)
    else:
        merged["sofa_available"] = False
    for c in SOFA_SCORE_COLUMNS:
        if c not in merged.columns:
            merged[c] = np.nan
    return merged


def run_completeness_check_per_ecg(
    ecg_mapping: pd.DataFrame,
    export_df: pd.DataFrame,
) -> tuple[int, int, int, list[str]]:
    """(n_no_stay, n_no_sofa_row, n_invalid_sofa, sample_no_stay)."""
    by_path = export_df.drop_duplicates(subset=["base_path"]).set_index("base_path")

    n_no_stay = 0
    n_no_sofa_row = 0
    n_invalid = 0
    sample_no_stay: list[str] = []

    for _, row in ecg_mapping.iterrows():
        bp = str(row["base_path"])
        sid = row["stay_id"]
        if sid is None or (isinstance(sid, float) and np.isnan(sid)):
            n_no_stay += 1
            if len(sample_no_stay) < 20:
                sample_no_stay.append(bp)
            continue

        if bp not in by_path.index:
            n_no_sofa_row += 1
            continue

        r = by_path.loc[bp]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        avail = bool(r.get("sofa_available", False))
        total = r.get("sofa_total")
        if not avail or pd.isna(total):
            n_invalid += 1

    return n_no_stay, n_no_sofa_row, n_invalid, sample_no_stay


def print_completeness_report(
    n_no_stay: int,
    n_no_sofa_row: int,
    n_invalid: int,
    sample_no_stay: list[str],
    n_ecg: int,
    n_valid_sofa: int,
) -> None:
    print("\n" + "=" * 70)
    print("BERICHT: SOFA pro ECG (Messwerte ≤ ecg_time)")
    print("=" * 70)
    print(f"  ECG-Records gesamt:                 {n_ecg:,}")
    print(f"  Mit gültigem SOFA (sofa_available): {n_valid_sofa:,}")
    print(f"  (1) Kein ICU-Stay (Mapper None):    {n_no_stay:,}")
    print(f"  (2) Keine SOFA-Zeile (kein Merge):   {n_no_sofa_row:,}")
    print(f"  (3) SOFA-Zeile, aber ungültig:       {n_invalid:,}")
    if sample_no_stay and n_no_stay > 0:
        print("  Beispiel base_path ohne Stay (max 20):")
        for p in sample_no_stay:
            print(f"    - {p}")
    print("=" * 70 + "\n")


def write_report_file(
    path: Path,
    n_ecg: int,
    n_valid: int,
    n_no_stay: int,
    n_no_row: int,
    n_invalid: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "SOFA-Berechnung pro ECG (Leakage-frei: charttime ≤ ecg_time)",
        f"Erstellt: {datetime.now().isoformat()}",
        "",
        f"ECG-Records gesamt:                 {n_ecg:,}",
        f"Mit gültigem SOFA (sofa_available): {n_valid:,}",
        f"Ohne ICU-Stay (Mapper None):        {n_no_stay:,}",
        f"Mit Stay, aber keine SOFA-Zeile:    {n_no_row:,}",
        f"Mit Stay, SOFA ungültig (zu wenig Komponenten): {n_invalid:,}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Bericht gespeichert: {path}")


def parse_args() -> argparse.Namespace:
    root = config.get_project_root()
    labels = root / "data" / "labeling" / "labels_csv"
    p = argparse.ArgumentParser(
        description="SOFA pro ECG (Messwerte bis ecg_time, ohne Leakage)"
    )
    p.add_argument(
        "--timestamps-csv",
        type=Path,
        default=labels / "timestamps_mapping_24h_P1.csv",
        help="Timestamp-Mapping CSV",
    )
    p.add_argument(
        "--icustays-csv",
        type=Path,
        default=labels / "icustays.csv",
        help="icustays für ICUStayMapper (labels_csv)",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=labels / "sofa_scores.csv",
        help="Ausgabe sofa_scores.csv (eine Zeile pro ECG/base_path)",
    )
    p.add_argument(
        "--report-path",
        type=Path,
        default=root / "outputs" / "sofa_per_ecg_report.txt",
        help="Textbericht mit Zählungen",
    )
    p.add_argument(
        "--skip-path-validation",
        action="store_true",
        help="MIMIC-Pfadprüfung überspringen (nur für Entwicklung)",
    )
    p.add_argument(
        "--skip-calculator-tests",
        action="store_true",
        help="Unit-Tests der SOFA-Logik überspringen",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Bei fehlgeschlagenen Checks nach Bestätigung fragen (sonst Exit)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  SOFA PRO ECG (charttime ≤ ecg_time, leakage-frei)".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("Schritt 1: ECG-Records → stay_id (ICUStayMapper)")
    print("-" * 70)
    if not args.timestamps_csv.is_file():
        print(f"❌ FEHLER: {args.timestamps_csv} nicht gefunden.")
        return 1
    if not args.icustays_csv.is_file():
        print(f"❌ FEHLER: {args.icustays_csv} nicht gefunden.")
        return 1

    ecg_mapping = build_ecg_stay_mapping(args.timestamps_csv, args.icustays_csv)
    n_with_stay = ecg_mapping["stay_id"].notna().sum()
    print(f"  ✓ {len(ecg_mapping):,} ECG-Zeilen, {n_with_stay:,} mit stay_id")

    print("\nSchritt 2: MIMIC-Pfade")
    print("-" * 70)
    if not args.skip_path_validation:
        if not config.validate_paths():
            print("\n❌ MIMIC-IV Pfade ungültig (oder --skip-path-validation nutzen).")
            return 1
    else:
        print("  (Pfadvalidierung übersprungen)")

    if not args.skip_calculator_tests:
        print("\n🔍 Validiere itemid Mappings...")
        itemid_mappings.validate_itemids()
        print("\n🧪 SOFA Calculator Tests...")
        ok = calculator.validate_sofa_calculation()
        if not ok:
            msg = "SOFA Calculator-Tests fehlgeschlagen."
            if args.interactive:
                response = input(f"{msg} Trotzdem fortfahren? (j/n): ")
                if response.lower() != "j":
                    return 1
            else:
                print(f"❌ {msg} (--interactive für manuelle Bestätigung)")
                return 1

    print("\nSchritt 3: MIMIC laden & SOFA pro ECG berechnen")
    print("-" * 70)
    try:
        sofa_data = load_sofa_data_per_ecg(ecg_mapping)
    except Exception as e:
        print(f"\n❌ FEHLER beim Laden: {e}")
        import traceback

        traceback.print_exc()
        return 1

    if len(sofa_data) == 0:
        print("⚠️  Keine ECG-Zeilen nach ICU-Merge – alle SOFA-Werte leer.")

    scored = (
        calculator.calculate_sofa_batch(sofa_data) if len(sofa_data) else pd.DataFrame()
    )
    if len(scored):
        scored = apply_sofa_availability(scored)

    final_df = merge_ecg_mapping_with_scores(ecg_mapping, scored)

    out_cols = [
        "base_path",
        "subject_id",
        "ecg_time",
        "stay_id",
        "hadm_id",
    ] + SOFA_SCORE_COLUMNS + ["sofa_available"]
    if "study_id" in final_df.columns:
        out_cols.insert(4, "study_id")
    out_cols = [c for c in out_cols if c in final_df.columns]
    export_df = final_df[out_cols].copy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Gespeichert: {args.output_csv} ({len(export_df):,} Zeilen)")

    n_valid = int(export_df["sofa_available"].sum()) if "sofa_available" in export_df.columns else 0
    n1, n2, n3, sample = run_completeness_check_per_ecg(ecg_mapping, export_df)
    print_completeness_report(n1, n2, n3, sample, len(ecg_mapping), n_valid)
    write_report_file(
        args.report_path,
        len(ecg_mapping),
        n_valid,
        n1,
        n2,
        n3,
    )

    if len(scored):
        print("Schritt 4: Datenqualität (nur Zeilen mit berechnetem Roh-SOFA)")
        print("-" * 70)
        utils.validate_data(sofa_data, SOFA_INPUT_COLUMNS[:6], "SOFA")
        comp_cols = SOFA_SCORE_COLUMNS[:-1]
        stats = utils.calculate_score_statistics(
            scored.dropna(subset=["sofa_total"]),
            score_column="sofa_total",
            component_columns=comp_cols,
        )
        if stats:
            utils.print_statistics(stats, "SOFA")

    if n2 > 0:
        print(
            "⚠️  Hinweis: Einige ECGs mit Stay hatten keine SOFA-Zeile "
            "(sollte bei vollem Merge selten sein)."
        )

    print(f"Ende: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n\nAbbruch (Ctrl+C)")
        raise SystemExit(130) from None
