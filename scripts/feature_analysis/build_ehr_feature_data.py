#!/usr/bin/env python3
"""
Baut EHR_feature_data.csv: nur ICU-/MIMIC-EHR (chartevents, labevents, outputevents, inputevents) pro ECG
im Fenster [stay_intime, t_cut] mit t_cut = min(ecg_time, stay_intime + 24h) (Leakage-sicher).

Enthalten sind ausschließlich Join-/Zeitfenster-Metadaten (base_path, study_id, subject_id, ecg_time, stay_id,
hadm_id, stay_intime, t_cut, …) plus aggregierte Messwerte. **Keine** Demographics (Alter/Geschlecht) und
**keine** Outcome-Labels (Mortalität) — die kommen in euren anderen Label-CSVs.

Hinweis: Ein vollständiger MIMIC-Scan (ohne --skip-mimic-tables) liest chartevents/labevents zeilenweise
und kann je nach Hardware sehr lange dauern. Mit ``--skip-mimic-tables`` bleiben alle MIMIC-Spalten leer.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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

HOURS_24 = pd.Timedelta(hours=24)
CHUNK_SIZE = sm_config.CHUNK_SIZE
DEFAULT_WEIGHT_KG = float(sm_config.DEFAULT_PATIENT_WEIGHT_KG)

# Chart-Konzepte: (Name, itemids, unbenutzt) — drittes Feld historisch für FiO2-Normierung
CHART_SPECS: List[Tuple[str, Tuple[int, ...], bool]] = [
    ("heart_rate", (220045,), False),
    ("respiratory_rate", (220210, 224688, 224689, 224690), False),
    ("oxygen_saturation", (220277, 220227), False),
    ("map", tuple(im.MAP_ITEMIDS), False),
    ("sbp", tuple(im.SBP_ITEMIDS), False),
    ("dbp", tuple(im.DBP_ITEMIDS), False),
    ("temperature", (223762, 223761), False),
]

MECH_IDS = set(im.MECHANICAL_VENTILATION_ITEMIDS)
NIV_IDS = set(im.NIV_HFNC_CHART_ITEMIDS)
RRT_IDS = set(im.DIALYSIS_RRT_CHART_ITEMIDS)

LAB_SPECS: List[Tuple[str, Tuple[int, ...], str]] = [
    ("platelets", tuple(im.PLATELETS_ITEMIDS), "min"),
    ("creatinine", tuple(im.CREATININE_ITEMIDS), "max"),
]

VASO_ITEMID_TO_COL = {}
for name, ids in im.VASOPRESSOR_MAPPING.items():
    for i in ids:
        VASO_ITEMID_TO_COL[int(i)] = name
for i in im.NON_CATECHOLAMINE_PRESSOR_INPUT_ITEMIDS:
    key = int(i)
    if key in (222315,):
        VASO_ITEMID_TO_COL[key] = "vasopressin"
    else:
        VASO_ITEMID_TO_COL[key] = "phenylephrine"

VASO_DRUG_NAMES = sorted(set(VASO_ITEMID_TO_COL.values()))


def _itemid_to_chart_concept() -> Dict[int, str]:
    m: Dict[int, str] = {}
    for key, itemids, _ in CHART_SPECS:
        for i in itemids:
            m[int(i)] = key
    return m


def _all_chart_itemids() -> Set[int]:
    s = set(MECH_IDS) | set(NIV_IDS) | set(RRT_IDS)
    for _, itemids, _ in CHART_SPECS:
        s.update(int(x) for x in itemids)
    return s


def _all_lab_itemids() -> Set[int]:
    s: Set[int] = set()
    for _, itemids, _ in LAB_SPECS:
        s.update(int(x) for x in itemids)
    return s


def build_ecg_frame(
    timestamps_csv: Path,
    icustays_csv: Path,
    max_ecg: Optional[int],
) -> pd.DataFrame:
    ts = pd.read_csv(timestamps_csv, nrows=max_ecg)
    need = {"base_path", "base_date", "base_time", "subject_id"}
    miss = need - set(ts.columns)
    if miss:
        raise ValueError(f"timestamps CSV fehlt: {miss}")
    ts["ecg_time"] = pd.to_datetime(ts["base_date"] + " " + ts["base_time"])
    ts["subject_id"] = ts["subject_id"].astype(int)
    if "study_id" in ts.columns:
        ts["study_id"] = ts["study_id"].astype(int)
    else:
        ts["study_id"] = pd.NA
        ts["study_id"] = ts["study_id"].astype("Int64")

    icu = load_icustays(str(icustays_csv))
    if "hadm_id" not in icu.columns:
        raise ValueError("icustays.csv braucht hadm_id für Labevents-Join")
    mapper = ICUStayMapper(icu, mortality_mapping=None)

    stay_intime = icu[["stay_id", "intime", "hadm_id"]].drop_duplicates(subset=["stay_id"]).copy()
    stay_intime["stay_id"] = stay_intime["stay_id"].astype(int)
    stay_intime["intime"] = pd.to_datetime(stay_intime["intime"])

    rows = []
    for _, row in ts.iterrows():
        sid = int(row["subject_id"])
        ecg_t = row["ecg_time"]
        stay_id = mapper.map_ecg_to_stay(sid, ecg_t)
        rows.append(
            {
                "base_path": str(row["base_path"]),
                "study_id": row["study_id"],
                "subject_id": sid,
                "ecg_time": ecg_t,
                "stay_id": stay_id,
            }
        )
    out = pd.DataFrame(rows)
    out = out.merge(stay_intime.rename(columns={"intime": "stay_intime"}), on="stay_id", how="left")
    out["t_cut"] = pd.NaT
    ok = out["stay_intime"].notna()
    out.loc[ok, "t_cut"] = np.minimum(out.loc[ok, "ecg_time"], out.loc[ok, "stay_intime"] + HOURS_24)
    out["ehr_window_hours"] = (out["t_cut"] - out["stay_intime"]).dt.total_seconds() / 3600.0
    return out


def _combine_numeric_partials(partials: List[pd.DataFrame], group_cols: List[str]) -> pd.DataFrame:
    if not partials:
        return pd.DataFrame()
    big = pd.concat(partials, ignore_index=True)
    g = big.groupby(group_cols, as_index=False).agg(
        vmin=("vmin", "min"),
        vmax=("vmax", "max"),
        vsum=("vsum", "sum"),
        vcnt=("vcnt", "sum"),
    )
    g["vmean"] = np.where(g["vcnt"] > 0, g["vsum"] / g["vcnt"], np.nan)
    g = g.loc[:, ~g.columns.duplicated()].copy()
    return g


def _combine_last_partials(partials: List[pd.DataFrame]) -> pd.DataFrame:
    if not partials:
        return pd.DataFrame()
    big = pd.concat(partials, ignore_index=True).sort_values("charttime")
    tail = big.groupby(["base_path", "concept"], as_index=False, sort=False).tail(1)
    return tail[["base_path", "concept", "valuenum"]].rename(columns={"valuenum": "vlast"})


def scan_chartevents(ecg: pd.DataFrame, chart_path: Path, chunk_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (numeric_stats wide, flags wide) for base_path."""
    item_map = _itemid_to_chart_concept()
    chart_ids = _all_chart_itemids()
    sub = ecg[ecg["stay_id"].notna() & ecg["stay_intime"].notna() & ecg["t_cut"].notna()].copy()
    if sub.empty:
        all_bp = ecg["base_path"].astype(str).tolist()
        flags = pd.DataFrame(
            {
                "base_path": all_bp,
                "mech_vent_during_window": 0,
                "niv_hfnc_during_window": 0,
                "rrt_during_window": 0,
            }
        )
        return pd.DataFrame({"base_path": all_bp}), flags
    sub["stay_id"] = sub["stay_id"].astype(int)
    stay_set = set(sub["stay_id"].unique())
    ecg_keys = sub[
        ["base_path", "stay_id", "stay_intime", "t_cut", "ecg_time"]
    ].copy()

    partial_num: List[pd.DataFrame] = []
    partial_last: List[pd.DataFrame] = []
    flags_mech: Dict[str, bool] = defaultdict(bool)
    flags_niv: Dict[str, bool] = defaultdict(bool)
    flags_rrt: Dict[str, bool] = defaultdict(bool)

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
        chunk = chunk.dropna(subset=["charttime", "stay_id", "itemid"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stay_set) & chunk["itemid"].isin(chart_ids)]
        if chunk.empty:
            continue

        m = chunk.merge(ecg_keys, on="stay_id", how="inner")
        m = m[(m["charttime"] >= m["stay_intime"]) & (m["charttime"] <= m["t_cut"])]
        if m.empty:
            continue

        for bp in m.loc[m["itemid"].isin(MECH_IDS), "base_path"].unique():
            flags_mech[str(bp)] = True
        for bp in m.loc[m["itemid"].isin(NIV_IDS), "base_path"].unique():
            flags_niv[str(bp)] = True
        for bp in m.loc[m["itemid"].isin(RRT_IDS), "base_path"].unique():
            flags_rrt[str(bp)] = True

        m = m[m["itemid"].isin(item_map.keys())].copy()
        if m.empty:
            continue
        m["concept"] = m["itemid"].map(item_map)
        m = m.dropna(subset=["concept", "valuenum"])
        m["valuenum"] = pd.to_numeric(m["valuenum"], errors="coerce")
        m = m.dropna(subset=["valuenum"])

        g = (
            m.groupby(["base_path", "concept"], as_index=False)
            .agg(
                vmin=("valuenum", "min"),
                vmax=("valuenum", "max"),
                vsum=("valuenum", "sum"),
                vcnt=("valuenum", "count"),
            )
        )
        partial_num.append(g)

        m_sort = m.sort_values("charttime")
        idx = m_sort.groupby(["base_path", "concept"])["charttime"].idxmax()
        last_df = m_sort.loc[idx, ["base_path", "concept", "charttime", "valuenum"]]
        partial_last.append(last_df)

    num_long = _combine_numeric_partials(partial_num, ["base_path", "concept"])
    last_long = _combine_last_partials(partial_last)
    if not num_long.empty:
        num_long = num_long.drop_duplicates(subset=["base_path", "concept"], keep="first")
    if not last_long.empty:
        last_long = last_long.drop_duplicates(subset=["base_path", "concept"], keep="first")
    if not num_long.empty and not last_long.empty:
        last_slim = last_long[["base_path", "concept", "vlast"]]
        long_df = num_long.merge(last_slim, on=["base_path", "concept"], how="outer")
    elif not num_long.empty:
        long_df = num_long.copy()
        long_df["vlast"] = np.nan
    elif not last_long.empty:
        long_df = last_long.copy()
        long_df["vmin"] = long_df["vlast"]
        long_df["vmax"] = long_df["vlast"]
        long_df["vmean"] = long_df["vlast"]
        long_df["vcnt"] = 1
    else:
        long_df = pd.DataFrame()

    wide_num = pd.DataFrame({"base_path": ecg["base_path"].astype(str).unique()})
    if not long_df.empty:
        for concept in sorted(long_df["concept"].dropna().unique()):
            # vsum is only needed to compute vmean; keeping it causes repeated merge(..., vsum) → MergeError.
            subc = long_df[long_df["concept"] == concept].drop(
                columns=["concept", "charttime", "vsum"], errors="ignore"
            )
            subc = subc.drop_duplicates(subset=["base_path"], keep="first")
            subc = subc.rename(
                columns={
                    "vmin": f"{concept}_min",
                    "vmax": f"{concept}_max",
                    "vmean": f"{concept}_mean",
                    "vcnt": f"{concept}_n",
                    "vlast": f"{concept}_last",
                }
            )
            wide_num = wide_num.merge(subc, on="base_path", how="left")

    all_bp = ecg["base_path"].astype(str).tolist()
    flags = pd.DataFrame(
        {
            "base_path": all_bp,
            "mech_vent_during_window": [int(flags_mech.get(bp, False)) for bp in all_bp],
            "niv_hfnc_during_window": [int(flags_niv.get(bp, False)) for bp in all_bp],
            "rrt_during_window": [int(flags_rrt.get(bp, False)) for bp in all_bp],
        }
    )
    return wide_num, flags


def _itemid_to_lab_concept() -> Dict[int, str]:
    m: Dict[int, str] = {}
    for key, itemids, _ in LAB_SPECS:
        for i in itemids:
            m[int(i)] = key
    return m


def scan_labevents(ecg: pd.DataFrame, lab_path: Path, chunk_size: int) -> pd.DataFrame:
    lab_map = _itemid_to_lab_concept()
    lab_ids = _all_lab_itemids()
    sub = ecg[ecg["hadm_id"].notna() & ecg["stay_intime"].notna() & ecg["t_cut"].notna()].copy()
    if sub.empty:
        return pd.DataFrame({"base_path": ecg["base_path"].astype(str).unique()})
    sub["hadm_id"] = sub["hadm_id"].astype(int)
    sub["subject_id"] = sub["subject_id"].astype(int)
    hadm_set = set(sub["hadm_id"].unique())
    subj_set = set(sub["subject_id"].unique())
    keys = sub[
        ["base_path", "subject_id", "hadm_id", "stay_intime", "t_cut"]
    ].drop_duplicates()

    partials: Dict[str, List[pd.DataFrame]] = defaultdict(list)

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
        chunk = chunk.dropna(subset=["charttime", "hadm_id", "itemid", "valuenum"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["subject_id"] = chunk["subject_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[
            chunk["hadm_id"].isin(hadm_set)
            & chunk["subject_id"].isin(subj_set)
            & chunk["itemid"].isin(lab_ids)
        ]
        if chunk.empty:
            continue
        m = chunk.merge(keys, on=["subject_id", "hadm_id"], how="inner")
        m = m[(m["charttime"] >= m["stay_intime"]) & (m["charttime"] <= m["t_cut"])]
        if m.empty:
            continue
        m["concept"] = m["itemid"].map(lab_map)
        m = m.dropna(subset=["concept"])
        m["valuenum"] = pd.to_numeric(m["valuenum"], errors="coerce")
        m = m.dropna(subset=["valuenum"])
        for concept, _, how in LAB_SPECS:
            mc = m[m["concept"] == concept]
            if mc.empty:
                continue
            if how == "min":
                g = mc.groupby("base_path", as_index=False).agg(v=("valuenum", "min"), n=("valuenum", "count"))
            else:
                g = mc.groupby("base_path", as_index=False).agg(v=("valuenum", "max"), n=("valuenum", "count"))
            g = g.rename(columns={"v": f"{concept}_worst", "n": f"{concept}_n"})
            partials[concept].append(g)

    wide = pd.DataFrame({"base_path": ecg["base_path"].astype(str).unique()})
    for concept, _, how in LAB_SPECS:
        if not partials[concept]:
            continue
        big = pd.concat(partials[concept], ignore_index=True)
        if how == "min":
            agg = big.groupby("base_path", as_index=False).agg(
                **{f"{concept}_worst": (f"{concept}_worst", "min"), f"{concept}_n": (f"{concept}_n", "sum")}
            )
        else:
            agg = big.groupby("base_path", as_index=False).agg(
                **{f"{concept}_worst": (f"{concept}_worst", "max"), f"{concept}_n": (f"{concept}_n", "sum")}
            )
        wide = wide.merge(agg, on="base_path", how="left")
    return wide


def scan_outputevents_urine(ecg: pd.DataFrame, out_path: Path, chunk_size: int) -> pd.DataFrame:
    urine_ids = set(im.get_all_outputevents_itemids())
    sub = ecg[ecg["stay_id"].notna() & ecg["stay_intime"].notna() & ecg["t_cut"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["stay_id"] = sub["stay_id"].astype(int)
    stay_set = set(sub["stay_id"].unique())
    keys = sub[["base_path", "subject_id", "hadm_id", "stay_id", "stay_intime", "t_cut"]].copy()
    partials: List[pd.DataFrame] = []

    for chunk in tqdm(
        pd.read_csv(
            out_path,
            usecols=["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "value"],
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc="outputevents",
    ):
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "stay_id", "itemid"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stay_set) & chunk["itemid"].isin(urine_ids)]
        if chunk.empty:
            continue
        m = chunk.merge(keys, on=["subject_id", "hadm_id", "stay_id"], how="inner")
        m = m[(m["charttime"] >= m["stay_intime"]) & (m["charttime"] <= m["t_cut"])]
        if m.empty:
            continue
        m["value"] = pd.to_numeric(m["value"], errors="coerce")
        m = m.dropna(subset=["value"])
        g = m.groupby("base_path", as_index=False).agg(urine_ml_sum=("value", "sum"))
        partials.append(g)

    if not partials:
        return pd.DataFrame({"base_path": ecg["base_path"].unique(), "urine_ml_sum": np.nan})
    big = pd.concat(partials, ignore_index=True)
    agg = big.groupby("base_path", as_index=False).agg(urine_ml_sum=("urine_ml_sum", "sum"))
    return agg


def scan_inputevents_vasopressors(ecg: pd.DataFrame, inp_path: Path, chunk_size: int) -> pd.DataFrame:
    vaso_ids = set(VASO_ITEMID_TO_COL.keys())
    sub = ecg[ecg["stay_id"].notna() & ecg["stay_intime"].notna() & ecg["t_cut"].notna()].copy()
    drugs = VASO_DRUG_NAMES
    empty_shell = pd.DataFrame({"base_path": ecg["base_path"].astype(str).unique()})
    for d in drugs:
        empty_shell[f"vaso_{d}_max"] = np.nan
    if sub.empty:
        return empty_shell
    sub["stay_id"] = sub["stay_id"].astype(int)
    stay_set = set(sub["stay_id"].unique())
    keys = sub[
        ["base_path", "subject_id", "hadm_id", "stay_id", "stay_intime", "t_cut"]
    ].copy()

    partials: List[pd.DataFrame] = []

    for chunk in tqdm(
        pd.read_csv(
            inp_path,
            usecols=["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime", "rate", "amount"],
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc="inputevents (vaso)",
    ):
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk["endtime"] = pd.to_datetime(chunk["endtime"], errors="coerce")
        chunk = chunk.dropna(subset=["starttime", "stay_id", "itemid"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stay_set) & chunk["itemid"].isin(vaso_ids)]
        if chunk.empty:
            continue
        m = chunk.merge(keys, on=["subject_id", "hadm_id", "stay_id"], how="inner")
        end_eff = m["endtime"].fillna(m["t_cut"])
        m = m[(m["starttime"] <= m["t_cut"]) & (end_eff >= m["stay_intime"])]
        if m.empty:
            continue
        m["drug"] = m["itemid"].map(VASO_ITEMID_TO_COL)
        m = m.dropna(subset=["drug"])
        m["rate"] = pd.to_numeric(m["rate"], errors="coerce").fillna(0.0)
        m["amount"] = pd.to_numeric(m["amount"], errors="coerce").fillna(0.0)
        m["signal"] = np.where(m["rate"] > 0, m["rate"], m["amount"])
        g = m.groupby(["base_path", "drug"], as_index=False).agg(vmax=("signal", "max"))
        partials.append(g)

    if not partials:
        return empty_shell

    big = pd.concat(partials, ignore_index=True)
    agg = big.groupby(["base_path", "drug"], as_index=False).agg(vmax=("vmax", "max"))
    wide = agg.pivot(index="base_path", columns="drug", values="vmax")
    wide = wide.rename(columns={c: f"vaso_{c}_max" for c in wide.columns})
    wide = wide.reset_index()
    out = pd.DataFrame({"base_path": ecg["base_path"].astype(str).unique()}).merge(wide, on="base_path", how="left")
    for d in drugs:
        col = f"vaso_{d}_max"
        if col not in out.columns:
            out[col] = np.nan
    return out


def add_empty_mimic_columns(ecg: pd.DataFrame) -> pd.DataFrame:
    """Alle MIMIC-abgeleiteten Spalten als fehlend (NaN bzw. 0 für Flags), gleiches Schema wie voller Lauf."""
    out = ecg.copy()
    for key, _, _ in CHART_SPECS:
        for s in ("min", "max", "mean", "n", "last"):
            out[f"{key}_{s}"] = np.nan
    out["mech_vent_during_window"] = 0
    out["niv_hfnc_during_window"] = 0
    out["rrt_during_window"] = 0
    for concept, _, _ in LAB_SPECS:
        out[f"{concept}_worst"] = np.nan
        out[f"{concept}_n"] = np.nan
    out["urine_ml_sum"] = np.nan
    out["vasopressor_during_window"] = 0
    return out


def normalize_vasopressors_by_weight(out: pd.DataFrame, ecg: pd.DataFrame) -> pd.DataFrame:
    """Skaliert Katecholamin-Raten grob auf µg/kg/min (wie SOFA-Pfad); Non-Katecholamine unverändert.

    Ohne Gewichtsspalte in ``ecg`` (kein ``weight_kg`` mehr in der Ausgabe-Tabelle) wird durchgehend
    ``DEFAULT_WEIGHT_KG`` aus ``src/scoring_models/config.py`` verwendet.
    """
    wcol = "weight_kg_mean"
    if wcol not in ecg.columns:
        ecg = ecg.assign(weight_kg_mean=np.nan)
    m = ecg[["base_path", wcol]].copy()
    df = out.merge(m, on="base_path", how="left")
    w = df[wcol].where(df[wcol] > 0, DEFAULT_WEIGHT_KG)
    catechol = ("dopamine", "norepinephrine", "epinephrine", "dobutamine")
    for name in catechol:
        c = f"vaso_{name}_max"
        if c in df.columns:
            df[c] = df[c] / w
    return df.drop(columns=[wcol], errors="ignore")


def collapse_vasopressor_max_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Ersetzt ``vaso_<drug>_max``-Spalten durch ``vasopressor_during_window`` (0/1).

    1 = mindestens ein Vasopressor im Fenster mit Rate/Amount > 0 (nach Gewichts-Normierung
    für Katecholamine, siehe ``normalize_vasopressors_by_weight``).
    """
    n = len(df)
    bp = df["base_path"].astype(str).to_numpy()
    cols = [f"vaso_{d}_max" for d in VASO_DRUG_NAMES if f"vaso_{d}_max" in df.columns]
    if not cols:
        return pd.DataFrame({"base_path": bp, "vasopressor_during_window": np.zeros(n, dtype=np.int64)})
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    binary = ((sub > 0).any(axis=1)).astype(np.int64).to_numpy()
    return pd.DataFrame({"base_path": bp, "vasopressor_during_window": binary})


def main() -> None:
    ap = argparse.ArgumentParser(description="EHR_feature_data.csv erzeugen")
    ap.add_argument(
        "--timestamps-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "timestamps_mapping_24h_P1.csv",
    )
    ap.add_argument("--icustays-csv", type=Path, default=PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "icustays.csv")
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "EHR_feature_data.csv",
    )
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--max-ecg", type=int, default=None)
    ap.add_argument(
        "--skip-mimic-tables",
        action="store_true",
        help="Kein Lesen von chartevents/labevents/outputevents/inputevents; MIMIC-Spalten = NaN/0.",
    )
    args = ap.parse_args()

    chart_path = Path(sm_config.MIMIC_IV_PATHS["chartevents"])
    lab_path = Path(sm_config.MIMIC_IV_PATHS["labevents"])
    out_path_m = Path(sm_config.MIMIC_IV_PATHS["outputevents"])
    inp_path = Path(sm_config.MIMIC_IV_PATHS["inputevents"])

    required = [args.timestamps_csv, args.icustays_csv]
    if not args.skip_mimic_tables:
        required += [chart_path, lab_path, out_path_m, inp_path]
    for p in required:
        if not p.exists():
            sys.exit(f"Pfad fehlt: {p}")

    print("Baue ECG↔Stay Frame…")
    ecg = build_ecg_frame(
        args.timestamps_csv,
        args.icustays_csv,
        args.max_ecg,
    )
    ecg["base_path"] = ecg["base_path"].astype(str)

    if args.skip_mimic_tables:
        print("Überspringe MIMIC-Tabellen (--skip-mimic-tables); EHR-Messwerte = NaN/0.")
        ecg = add_empty_mimic_columns(ecg)
    else:
        print("Scan chartevents…")
        wide_chart, flags = scan_chartevents(ecg, chart_path, args.chunk_size)
        wide_chart["base_path"] = wide_chart["base_path"].astype(str)
        flags["base_path"] = flags["base_path"].astype(str)
        ecg = ecg.merge(wide_chart, on="base_path", how="left")
        ecg = ecg.merge(flags, on="base_path", how="left")

        print("Scan labevents…")
        wide_lab = scan_labevents(ecg, lab_path, args.chunk_size)
        wide_lab["base_path"] = wide_lab["base_path"].astype(str)
        ecg = ecg.merge(wide_lab, on="base_path", how="left")

        print("Scan outputevents…")
        wide_out = scan_outputevents_urine(ecg, out_path_m, args.chunk_size)
        if not wide_out.empty:
            wide_out["base_path"] = wide_out["base_path"].astype(str)
        ecg = ecg.merge(wide_out, on="base_path", how="left")

        print("Scan inputevents (Vasopressoren)…")
        wide_vaso = scan_inputevents_vasopressors(ecg, inp_path, args.chunk_size)
        wide_vaso["base_path"] = wide_vaso["base_path"].astype(str)
        wide_vaso = normalize_vasopressors_by_weight(wide_vaso, ecg)
        wide_vaso = collapse_vasopressor_max_to_binary(wide_vaso)
        ecg = ecg.merge(wide_vaso, on="base_path", how="left")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if "stay_id" in ecg.columns:
        ecg["stay_id"] = pd.to_numeric(ecg["stay_id"], errors="coerce").astype("Int64")
    if "hadm_id" in ecg.columns:
        ecg["hadm_id"] = pd.to_numeric(ecg["hadm_id"], errors="coerce").astype("Int64")
    ecg.to_csv(args.out_csv, index=False)
    n = len(ecg)
    na_hr = float(ecg["heart_rate_n"].isna().mean()) if "heart_rate_n" in ecg.columns else 1.0
    print(f"Geschrieben: {args.out_csv} ({n} Zeilen)")
    print(f"  Fehlende Vitale (heart_rate_n): {100 * na_hr:.1f} %")
    if args.skip_mimic_tables:
        print(
            "  → Vitale/Labs/Urin/Vasopressoren sind absichtlich leer (--skip-mimic-tables). "
            "Ohne dieses Flag neu bauen, um echte MIMIC-Werte zu füllen (Laufzeit kann sehr lang sein)."
        )
    elif na_hr > 0.99:
        print(
            "  Warnung: Fast keine HR-Messungen — prüfen, ob chartevents-Pfad stimmt und stay_id-Matches existieren."
        )


if __name__ == "__main__":
    main()
