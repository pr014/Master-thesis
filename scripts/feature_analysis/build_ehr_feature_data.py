#!/usr/bin/env python3
"""
Builds ``EHR_feature_data.csv`` for ECG samples with a strict leakage-safe rule:

  stay_intime <= event_time <= min(ecg_time, stay_intime + icu_window_hours)

The output intentionally contains only:
- ECG/stay join metadata
- the curated variables agreed for the multimodal table

Aggregation:
- chartevents / labevents: median within the window
- outputevents (urine): sum within the window
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x: Iterable, **kwargs):
        return x


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labeling import load_icustays  # noqa: E402
from src.data.labeling.icu_los_labels import ICUStayMapper  # noqa: E402
from src.scoring_models import config as sm_config  # noqa: E402
from src.scoring_models.sofa import itemid_mappings as im  # noqa: E402


CHUNK_SIZE = sm_config.CHUNK_SIZE
DEFAULT_ICU_WINDOW_HOURS = 6.0

CHART_SPECS: list[tuple[str, tuple[int, ...]]] = [
    ("respiratory_rate", (220210, 224688, 224689, 224690)),
    ("oxygen_saturation", (220277, 220227)),
    ("map", tuple(int(x) for x in im.MAP_ITEMIDS)),
    ("sbp", tuple(int(x) for x in im.SBP_ITEMIDS)),
    ("dbp", tuple(int(x) for x in im.DBP_ITEMIDS)),
    ("temperature", (223762, 223761)),
    ("gcs_eye", tuple(int(x) for x in im.GCS_EYE_ITEMIDS)),
    ("gcs_motor", tuple(int(x) for x in im.GCS_MOTOR_ITEMIDS)),
]

LAB_SPECS: list[tuple[str, tuple[int, ...]]] = [
    ("platelets", tuple(int(x) for x in im.PLATELETS_ITEMIDS)),
    ("creatinine", tuple(int(x) for x in im.CREATININE_ITEMIDS)),
]

OUTPUT_COL = "urine_output"


def _itemid_to_concept(specs: list[tuple[str, tuple[int, ...]]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for concept, itemids in specs:
        for itemid in itemids:
            mapping[int(itemid)] = concept
    return mapping


def _feature_shell(base_paths: pd.Series, feature_names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({"base_path": base_paths.astype(str).unique()})
    for feature in feature_names:
        out[feature] = np.nan
    return out


def build_ecg_frame(
    timestamps_csv: Path,
    icustays_csv: Path,
    icu_window_hours: float,
    max_ecg: Optional[int],
) -> pd.DataFrame:
    ts = pd.read_csv(timestamps_csv, nrows=max_ecg)
    required_cols = {"base_path", "base_date", "base_time", "subject_id"}
    missing_cols = required_cols - set(ts.columns)
    if missing_cols:
        raise ValueError(f"timestamps CSV fehlt: {missing_cols}")

    ts["ecg_time"] = pd.to_datetime(ts["base_date"] + " " + ts["base_time"])
    ts["subject_id"] = ts["subject_id"].astype(int)
    if "study_id" in ts.columns:
        ts["study_id"] = ts["study_id"].astype(int)
    else:
        ts["study_id"] = pd.Series(pd.NA, index=ts.index, dtype="Int64")

    icu = load_icustays(str(icustays_csv))
    if "hadm_id" not in icu.columns:
        raise ValueError("icustays.csv braucht hadm_id für den sicheren ECG↔Stay-Join")
    mapper = ICUStayMapper(icu, mortality_mapping=None)

    stay_meta = icu[["stay_id", "intime", "hadm_id"]].drop_duplicates(subset=["stay_id"]).copy()
    stay_meta["stay_id"] = stay_meta["stay_id"].astype(int)
    stay_meta["intime"] = pd.to_datetime(stay_meta["intime"])

    rows = []
    for _, row in ts.iterrows():
        subject_id = int(row["subject_id"])
        ecg_time = row["ecg_time"]
        stay_id = mapper.map_ecg_to_stay(subject_id, ecg_time)
        rows.append(
            {
                "base_path": str(row["base_path"]),
                "study_id": row["study_id"],
                "subject_id": subject_id,
                "ecg_time": ecg_time,
                "stay_id": stay_id,
            }
        )

    out = pd.DataFrame(rows)
    out = out.merge(stay_meta.rename(columns={"intime": "stay_intime"}), on="stay_id", how="left")
    out["t_cut"] = pd.NaT
    has_stay = out["stay_intime"].notna()
    delta = pd.Timedelta(hours=icu_window_hours)
    out.loc[has_stay, "t_cut"] = np.minimum(out.loc[has_stay, "ecg_time"], out.loc[has_stay, "stay_intime"] + delta)
    out["ehr_window_hours"] = (out["t_cut"] - out["stay_intime"]).dt.total_seconds() / 3600.0
    return out


def scan_chartevents_median(ecg: pd.DataFrame, chart_path: Path, chunk_size: int) -> pd.DataFrame:
    base_paths = ecg["base_path"]
    features = [name for name, _ in CHART_SPECS]
    out = _feature_shell(base_paths, features)

    sub = ecg[ecg["stay_id"].notna() & ecg["stay_intime"].notna() & ecg["t_cut"].notna()].copy()
    if sub.empty:
        return out

    sub["stay_id"] = sub["stay_id"].astype(int)
    stay_set = set(sub["stay_id"].unique())
    keys = sub[["base_path", "stay_id", "stay_intime", "t_cut"]].copy()
    itemid_to_concept = _itemid_to_concept(CHART_SPECS)
    chart_ids = set(itemid_to_concept.keys())

    values: DefaultDict[tuple[str, str], list[float]] = defaultdict(list)

    for chunk in tqdm(
        pd.read_csv(
            chart_path,
            usecols=["stay_id", "itemid", "charttime", "valuenum"],
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc="chartevents",
    ):
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "stay_id", "itemid", "valuenum"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stay_set) & chunk["itemid"].isin(chart_ids)]
        if chunk.empty:
            continue

        merged = chunk.merge(keys, on="stay_id", how="inner")
        merged = merged[(merged["charttime"] >= merged["stay_intime"]) & (merged["charttime"] <= merged["t_cut"])]
        if merged.empty:
            continue

        merged["concept"] = merged["itemid"].map(itemid_to_concept)
        merged["valuenum"] = pd.to_numeric(merged["valuenum"], errors="coerce")
        merged = merged.dropna(subset=["concept", "valuenum"])
        if merged.empty:
            continue

        for (base_path, concept), grp in merged.groupby(["base_path", "concept"], sort=False):
            values[(str(base_path), str(concept))].extend(grp["valuenum"].astype(float).tolist())

    if not values:
        return out

    rows = []
    for (base_path, concept), observed in values.items():
        rows.append({"base_path": base_path, "concept": concept, "value": float(np.median(np.asarray(observed)))})
    long_df = pd.DataFrame(rows)
    pivot = long_df.pivot(index="base_path", columns="concept", values="value")
    wide = out.set_index("base_path")
    for concept in pivot.columns:
        wide.loc[pivot.index, concept] = pivot[concept]
    return wide.reset_index()


def scan_labevents_median(ecg: pd.DataFrame, lab_path: Path, chunk_size: int) -> pd.DataFrame:
    base_paths = ecg["base_path"]
    features = [name for name, _ in LAB_SPECS]
    out = _feature_shell(base_paths, features)

    sub = ecg[ecg["hadm_id"].notna() & ecg["stay_intime"].notna() & ecg["t_cut"].notna()].copy()
    if sub.empty:
        return out

    sub["hadm_id"] = sub["hadm_id"].astype(int)
    sub["subject_id"] = sub["subject_id"].astype(int)
    hadm_set = set(sub["hadm_id"].unique())
    subject_set = set(sub["subject_id"].unique())
    keys = sub[["base_path", "subject_id", "hadm_id", "stay_intime", "t_cut"]].copy()
    itemid_to_concept = _itemid_to_concept(LAB_SPECS)
    lab_ids = set(itemid_to_concept.keys())

    values: DefaultDict[tuple[str, str], list[float]] = defaultdict(list)

    for chunk in tqdm(
        pd.read_csv(
            lab_path,
            usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc="labevents",
    ):
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "subject_id", "hadm_id", "itemid", "valuenum"])
        chunk["subject_id"] = chunk["subject_id"].astype(int)
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[
            chunk["subject_id"].isin(subject_set)
            & chunk["hadm_id"].isin(hadm_set)
            & chunk["itemid"].isin(lab_ids)
        ]
        if chunk.empty:
            continue

        merged = chunk.merge(keys, on=["subject_id", "hadm_id"], how="inner")
        merged = merged[(merged["charttime"] >= merged["stay_intime"]) & (merged["charttime"] <= merged["t_cut"])]
        if merged.empty:
            continue

        merged["concept"] = merged["itemid"].map(itemid_to_concept)
        merged["valuenum"] = pd.to_numeric(merged["valuenum"], errors="coerce")
        merged = merged.dropna(subset=["concept", "valuenum"])
        if merged.empty:
            continue

        for (base_path, concept), grp in merged.groupby(["base_path", "concept"], sort=False):
            values[(str(base_path), str(concept))].extend(grp["valuenum"].astype(float).tolist())

    if not values:
        return out

    rows = []
    for (base_path, concept), observed in values.items():
        rows.append({"base_path": base_path, "concept": concept, "value": float(np.median(np.asarray(observed)))})
    long_df = pd.DataFrame(rows)
    pivot = long_df.pivot(index="base_path", columns="concept", values="value")
    wide = out.set_index("base_path")
    for concept in pivot.columns:
        wide.loc[pivot.index, concept] = pivot[concept]
    return wide.reset_index()


def scan_outputevents_urine_sum(ecg: pd.DataFrame, output_path: Path, chunk_size: int) -> pd.DataFrame:
    out = _feature_shell(ecg["base_path"], [OUTPUT_COL])

    sub = ecg[ecg["stay_id"].notna() & ecg["stay_intime"].notna() & ecg["t_cut"].notna()].copy()
    if sub.empty:
        return out

    sub["stay_id"] = sub["stay_id"].astype(int)
    stay_set = set(sub["stay_id"].unique())
    keys = sub[["base_path", "subject_id", "hadm_id", "stay_id", "stay_intime", "t_cut"]].copy()
    urine_ids = set(int(x) for x in im.get_all_outputevents_itemids())
    partials: list[pd.DataFrame] = []

    for chunk in tqdm(
        pd.read_csv(
            output_path,
            usecols=["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "value"],
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc="outputevents",
    ):
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "subject_id", "hadm_id", "stay_id", "itemid", "value"])
        chunk["subject_id"] = chunk["subject_id"].astype(int)
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stay_set) & chunk["itemid"].isin(urine_ids)]
        if chunk.empty:
            continue

        merged = chunk.merge(keys, on=["subject_id", "hadm_id", "stay_id"], how="inner")
        merged = merged[(merged["charttime"] >= merged["stay_intime"]) & (merged["charttime"] <= merged["t_cut"])]
        if merged.empty:
            continue

        merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
        merged = merged.dropna(subset=["value"])
        if merged.empty:
            continue

        partials.append(merged.groupby("base_path", as_index=False).agg(**{OUTPUT_COL: ("value", "sum")}))

    if not partials:
        return out

    big = pd.concat(partials, ignore_index=True)
    agg = big.groupby("base_path", as_index=False).agg(**{OUTPUT_COL: (OUTPUT_COL, "sum")})
    wide = out.set_index("base_path")
    agg = agg.set_index("base_path")
    wide.loc[agg.index, OUTPUT_COL] = agg[OUTPUT_COL]
    return wide.reset_index()


def add_empty_feature_columns(ecg: pd.DataFrame) -> pd.DataFrame:
    out = ecg.copy()
    for feature, _ in CHART_SPECS:
        out[feature] = np.nan
    for feature, _ in LAB_SPECS:
        out[feature] = np.nan
    out[OUTPUT_COL] = np.nan
    return out


def final_column_order(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "base_path",
        "study_id",
        "subject_id",
        "ecg_time",
        "stay_id",
        "stay_intime",
        "hadm_id",
        "t_cut",
        "ehr_window_hours",
        "respiratory_rate",
        "oxygen_saturation",
        "map",
        "sbp",
        "dbp",
        "temperature",
        "gcs_eye",
        "gcs_motor",
        "platelets",
        "creatinine",
        OUTPUT_COL,
    ]
    return df[columns]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the curated ECG-aligned EHR feature table.")
    ap.add_argument(
        "--timestamps-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "timestamps_mapping_24h_P1.csv",
    )
    ap.add_argument(
        "--icustays-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "icustays.csv",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "EHR_feature_data.csv",
    )
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--max-ecg", type=int, default=None)
    ap.add_argument("--icu-window-hours", type=float, default=DEFAULT_ICU_WINDOW_HOURS)
    ap.add_argument(
        "--skip-mimic-tables",
        action="store_true",
        help="Do not read MIMIC event tables; feature columns stay empty.",
    )
    args = ap.parse_args()

    chart_path = Path(sm_config.MIMIC_IV_PATHS["chartevents"])
    lab_path = Path(sm_config.MIMIC_IV_PATHS["labevents"])
    output_path = Path(sm_config.MIMIC_IV_PATHS["outputevents"])

    required = [args.timestamps_csv, args.icustays_csv]
    if not args.skip_mimic_tables:
        required.extend([chart_path, lab_path, output_path])
    for path in required:
        if not path.exists():
            sys.exit(f"Pfad fehlt: {path}")

    print("Baue ECG↔Stay Frame…")
    ecg = build_ecg_frame(
        args.timestamps_csv,
        args.icustays_csv,
        args.icu_window_hours,
        args.max_ecg,
    )
    ecg["base_path"] = ecg["base_path"].astype(str)

    if args.skip_mimic_tables:
        print("Überspringe MIMIC-Tabellen (--skip-mimic-tables); Feature-Spalten bleiben leer.")
        ecg = add_empty_feature_columns(ecg)
    else:
        print("Scan chartevents (Median)…")
        wide_chart = scan_chartevents_median(ecg, chart_path, args.chunk_size)
        wide_chart["base_path"] = wide_chart["base_path"].astype(str)
        ecg = ecg.merge(wide_chart, on="base_path", how="left")

        print("Scan labevents (Median)…")
        wide_lab = scan_labevents_median(ecg, lab_path, args.chunk_size)
        wide_lab["base_path"] = wide_lab["base_path"].astype(str)
        ecg = ecg.merge(wide_lab, on="base_path", how="left")

        print("Scan outputevents (Urin-Summe)…")
        wide_output = scan_outputevents_urine_sum(ecg, output_path, args.chunk_size)
        wide_output["base_path"] = wide_output["base_path"].astype(str)
        ecg = ecg.merge(wide_output, on="base_path", how="left")

    ecg = final_column_order(ecg)
    ecg["stay_id"] = pd.to_numeric(ecg["stay_id"], errors="coerce").astype("Int64")
    ecg["hadm_id"] = pd.to_numeric(ecg["hadm_id"], errors="coerce").astype("Int64")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    ecg.to_csv(args.out_csv, index=False)

    print(f"Geschrieben: {args.out_csv} ({len(ecg)} Zeilen)")
    for feature in [name for name, _ in CHART_SPECS] + [name for name, _ in LAB_SPECS] + [OUTPUT_COL]:
        availability = 100.0 * float(ecg[feature].notna().mean())
        print(f"  {feature}: {availability:.2f} % belegt")


if __name__ == "__main__":
    main()
