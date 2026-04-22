#!/usr/bin/env python3
"""
Breite EHR-Coverage-Discovery für die ECG-Kohorte (icu_ecgs_24h / P1).

Zeitregel (leakage-sicher, wie Thesis-Skript):
  stay_intime <= event_time <= min(ecg_time, stay_intime + --icu-window-hours)

Quellen: chartevents, labevents, outputevents, inputevents (alle Starts im Fenster +
Vasopressoren), optional wide Lab/Output; **standardmäßig** alle nutzbaren Spalten aus
``patients.csv`` / ``admissions.csv`` (``--no-static-patient-tables`` zum Abschalten);
**standardmäßig** Zusatz-MIMIC-Tabellen, falls unter ``--mimic-iv-dir`` vorhanden:
``diagnoses_icd`` (hadm-Ebene, ohne Tageszeitstempel), ``prescriptions``, ``procedures_icd``,
``microbiologyevents`` (zeitlich im Fenster; ``--no-mimic-extras`` zum Abschalten).

Ausgabe: CSV (+ optional Markdown) mit Anteil der ECG-Zeilen mit Treffer — zur Auswahl
weiterer Variablen, ohne neue Modell-Pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

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

from scripts.feature_analysis.build_ehr_feature_data import (  # noqa: E402
    CHART_SPECS,
    LAB_SPECS,
    MECH_IDS,
    NIV_IDS,
    RRT_IDS,
    VASO_ITEMID_TO_COL,
)
from src.data.labeling import load_icustays  # noqa: E402
from src.scoring_models import config as sm_config  # noqa: E402
from src.scoring_models.sofa import itemid_mappings as im  # noqa: E402
from src.scoring_models.sofa.ecg_stay_mapping import build_ecg_stay_mapping  # noqa: E402

CHUNK_DEFAULT = int(sm_config.CHUNK_SIZE)


def build_ecg_frame(
    timestamps_csv: Path,
    icustays_csv: Path,
    icu_window_hours: float,
    max_ecg: int | None = None,
) -> pd.DataFrame:
    icu = load_icustays(str(icustays_csv))
    stay_intime = icu[["stay_id", "intime"]].drop_duplicates(subset=["stay_id"]).copy()
    stay_intime["stay_id"] = stay_intime["stay_id"].astype(int)
    stay_intime["intime"] = pd.to_datetime(stay_intime["intime"])

    hadm = None
    if "hadm_id" in icu.columns:
        hadm = icu[["stay_id", "hadm_id"]].drop_duplicates(subset=["stay_id"]).copy()
        hadm = hadm.dropna(subset=["hadm_id"])
        hadm["stay_id"] = hadm["stay_id"].astype(int)
        hadm["hadm_id"] = hadm["hadm_id"].astype(int)

    mapping = build_ecg_stay_mapping(timestamps_csv, icustays_csv, nrows=max_ecg)
    mapping["ecg_id"] = np.arange(len(mapping), dtype=np.int64)
    mapping["ecg_time"] = pd.to_datetime(mapping["ecg_time"])
    mapping = mapping.merge(stay_intime, on="stay_id", how="left")
    if hadm is not None:
        hadm = hadm.drop_duplicates(subset=["stay_id"], keep="first")
        mapping = mapping.merge(hadm, on="stay_id", how="left")
    else:
        mapping["hadm_id"] = np.nan

    delta = pd.Timedelta(hours=icu_window_hours)
    mapping["t_window_end"] = np.minimum(mapping["ecg_time"], mapping["intime"] + delta)
    return mapping


def _itemid_to_hit_keys(
    chart_rows: Sequence[Tuple[str, Tuple[int, ...], str]],
    therapy: Sequence[Tuple[str, Set[int], str]],
) -> Dict[int, List[str]]:
    m: DefaultDict[int, List[str]] = defaultdict(list)
    for key, itemids, _ in chart_rows:
        for i in itemids:
            m[int(i)].append(key)
    for key, idset, _ in therapy:
        for i in idset:
            m[int(i)].append(key)
    return dict(m)


def _scan_chartevents_combined(
    chart_path: Path,
    ecg: pd.DataFrame,
    hit_keys: List[str],
    itemid_to_keys: Dict[int, List[str]],
    all_itemids: Set[int],
    chunk_size: int,
    keys_for_early_exit: List[str],
    tqdm_desc: str = "chartevents",
) -> Dict[str, np.ndarray]:
    n = len(ecg)
    hits: Dict[str, np.ndarray] = {k: np.zeros(n, dtype=bool) for k in hit_keys}
    sub = ecg[ecg["stay_id"].notna() & ecg["intime"].notna()].copy()
    if sub.empty:
        return hits
    sub["stay_id"] = sub["stay_id"].astype(int)
    stay_set = set(sub["stay_id"].unique())

    for chunk in tqdm(
        pd.read_csv(
            chart_path,
            usecols=["stay_id", "itemid", "charttime"],
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc=tqdm_desc,
    ):
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "stay_id", "itemid"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stay_set) & chunk["itemid"].isin(all_itemids)]
        if chunk.empty:
            continue
        m = chunk.merge(
            sub[["ecg_id", "stay_id", "intime", "t_window_end"]],
            on="stay_id",
            how="inner",
        )
        m = m[(m["charttime"] >= m["intime"]) & (m["charttime"] <= m["t_window_end"])]
        if m.empty:
            continue
        for iid, grp in m.groupby("itemid", sort=False):
            keys = itemid_to_keys.get(int(iid), [])
            if not keys:
                continue
            eids = grp["ecg_id"].astype(int).to_numpy()
            for vk in keys:
                hits[vk][eids] = True
        if keys_for_early_exit and all(hits[k].all() for k in keys_for_early_exit):
            break
    return hits


def _chart_specs_with_extras() -> List[Tuple[str, Tuple[int, ...], str]]:
    """CHART_SPECS + FiO₂ + Gewicht + GCS (SOFA-relevante Chart-Größen)."""
    labels = {
        "heart_rate": "Herzfrequenz (HR)",
        "respiratory_rate": "Atemfrequenz (RR)",
        "oxygen_saturation": "Sauerstoffsättigung (SpO₂)",
        "map": "Mittlerer arterieller Druck (MAP)",
        "sbp": "Systolischer Blutdruck (SBP)",
        "dbp": "Diastolischer Blutdruck (DBP)",
        "temperature": "Körpertemperatur",
    }
    out: List[Tuple[str, Tuple[int, ...], str]] = []
    for name, itemids, _ in CHART_SPECS:
        out.append((name, tuple(int(x) for x in itemids), labels.get(name, name)))
    out.append(("fio2", tuple(int(x) for x in im.FIO2_ITEMIDS), "Inspiratorische O₂-Fraktion (FiO₂, Chart)"))
    out.append(("weight", tuple(int(x) for x in im.WEIGHT_ITEMIDS), "Körpergewicht (Chart)"))
    out.append(("gcs_total", tuple(int(x) for x in im.GCS_TOTAL_ITEMIDS), "GCS Total (Chart)"))
    out.append(("gcs_eye", tuple(int(x) for x in im.GCS_EYE_ITEMIDS), "GCS Augenöffnung (Chart)"))
    out.append(("gcs_verbal", tuple(int(x) for x in im.GCS_VERBAL_ITEMIDS), "GCS verbale Antwort (Chart)"))
    out.append(("gcs_motor", tuple(int(x) for x in im.GCS_MOTOR_ITEMIDS), "GCS motorische Antwort (Chart)"))
    return out


def _lab_specs_extended() -> List[Tuple[str, Tuple[int, ...], str]]:
    """LAB_SPECS + PaO₂ + Bilirubin (explizit)."""
    rows: List[Tuple[str, Tuple[int, ...], str]] = []
    for name, itemids, _ in LAB_SPECS:
        de = {"platelets": "Thrombozyten (Lab)", "creatinine": "Kreatinin (Lab)"}.get(name, name)
        rows.append((name, tuple(int(x) for x in itemids), de))
    rows.append(("pao2", tuple(int(x) for x in im.PAO2_ITEMIDS), "PaO₂ / PO₂ (Blutgas, Lab)"))
    rows.append(("bilirubin", tuple(int(x) for x in im.BILIRUBIN_ITEMIDS), "Bilirubin total (Lab)"))
    return rows


def _therapy_specs() -> List[Tuple[str, Set[int], str]]:
    return [
        ("mech_vent_chart", MECH_IDS, "Mechanische Beatmung (dokumentiert im Chart)"),
        ("niv_hfnc_chart", NIV_IDS, "NIV / HFNC (dokumentiert im Chart)"),
        ("rrt_chart", RRT_IDS, "Dialyse / RRT (dokumentiert im Chart)"),
    ]


def _input_drug_catalog() -> Tuple[Dict[int, str], List[str]]:
    """itemid → logischer Medikamentenname (wie build_ehr_feature_data)."""
    drug_keys = sorted(set(VASO_ITEMID_TO_COL.values()))
    return VASO_ITEMID_TO_COL, drug_keys


def scan_labevents_catalog(
    lab_path: Path,
    ecg: pd.DataFrame,
    lab_rows: Sequence[Tuple[str, Tuple[int, ...], str]],
    chunk_size: int,
    hits: Dict[str, np.ndarray],
) -> None:
    lab_map: Dict[int, str] = {}
    lab_ids: Set[int] = set()
    for concept, itemids, _ in lab_rows:
        for i in itemids:
            lab_ids.add(int(i))
            lab_map[int(i)] = concept

    sub = ecg[ecg["hadm_id"].notna() & ecg["intime"].notna()].copy()
    if sub.empty:
        return
    sub["hadm_id"] = sub["hadm_id"].astype(int)
    sub["subject_id"] = sub["subject_id"].astype(int)
    hadm_set = set(sub["hadm_id"].unique())
    subj_set = set(sub["subject_id"].unique())
    keys = sub[["ecg_id", "subject_id", "hadm_id", "intime", "t_window_end"]].copy()

    for chunk in tqdm(
        pd.read_csv(
            lab_path,
            usecols=["subject_id", "hadm_id", "itemid", "charttime"],
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc="labevents (Katalog)",
    ):
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "hadm_id", "itemid", "subject_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["subject_id"] = chunk["subject_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(hadm_set) & chunk["subject_id"].isin(subj_set) & chunk["itemid"].isin(lab_ids)]
        if chunk.empty:
            continue
        m = chunk.merge(keys, on=["subject_id", "hadm_id"], how="inner")
        m = m[(m["charttime"] >= m["intime"]) & (m["charttime"] <= m["t_window_end"])]
        if m.empty:
            continue
        for concept in m["itemid"].map(lab_map).dropna().unique():
            eids = m[m["itemid"].map(lab_map) == concept]["ecg_id"].astype(int).to_numpy()
            hits[str(concept)][eids] = True


def scan_labevents_wide(
    lab_path: Path,
    ecg: pd.DataFrame,
    allowed_itemids: Set[int],
    itemid_labels: Dict[int, str],
    chunk_size: int,
) -> Dict[int, np.ndarray]:
    """Pro Lab-itemid ein Bool-Vektor (nur itemids in allowed_itemids)."""
    n = len(ecg)
    hits: Dict[int, np.ndarray] = {int(i): np.zeros(n, dtype=bool) for i in allowed_itemids}

    sub = ecg[ecg["hadm_id"].notna() & ecg["intime"].notna()].copy()
    if sub.empty:
        return hits
    sub["hadm_id"] = sub["hadm_id"].astype(int)
    sub["subject_id"] = sub["subject_id"].astype(int)
    hadm_set = set(sub["hadm_id"].unique())
    subj_set = set(sub["subject_id"].unique())
    keys = sub[["ecg_id", "subject_id", "hadm_id", "intime", "t_window_end"]].copy()

    for chunk in tqdm(
        pd.read_csv(
            lab_path,
            usecols=["subject_id", "hadm_id", "itemid", "charttime"],
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc="labevents (wide d_labitems)",
    ):
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "hadm_id", "itemid", "subject_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["subject_id"] = chunk["subject_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(hadm_set) & chunk["subject_id"].isin(subj_set) & chunk["itemid"].isin(allowed_itemids)]
        if chunk.empty:
            continue
        m = chunk.merge(keys, on=["subject_id", "hadm_id"], how="inner")
        m = m[(m["charttime"] >= m["intime"]) & (m["charttime"] <= m["t_window_end"])]
        if m.empty:
            continue
        for iid, grp in m.groupby("itemid", sort=False):
            ii = int(iid)
            if ii not in hits:
                continue
            hits[ii][grp["ecg_id"].astype(int).to_numpy()] = True
    return hits


def scan_outputevents_wide(
    out_path: Path,
    ecg: pd.DataFrame,
    allowed_itemids: Set[int],
    chunk_size: int,
) -> Dict[int, np.ndarray]:
    """Pro Output-itemid ein Bool-Vektor (stay-basiert, gleiches Zeitfenster)."""
    n = len(ecg)
    hits: Dict[int, np.ndarray] = {int(i): np.zeros(n, dtype=bool) for i in allowed_itemids}

    sub = ecg[ecg["stay_id"].notna() & ecg["intime"].notna()].copy()
    if sub.empty:
        return hits
    sub["stay_id"] = sub["stay_id"].astype(int)
    stay_set = set(sub["stay_id"].unique())

    for chunk in tqdm(
        pd.read_csv(out_path, usecols=["stay_id", "itemid", "charttime"], chunksize=chunk_size, low_memory=False),
        desc="outputevents (wide d_items)",
    ):
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "stay_id", "itemid"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stay_set) & chunk["itemid"].isin(allowed_itemids)]
        if chunk.empty:
            continue
        m = chunk.merge(sub[["ecg_id", "stay_id", "intime", "t_window_end"]], on="stay_id", how="inner")
        m = m[(m["charttime"] >= m["intime"]) & (m["charttime"] <= m["t_window_end"])]
        if m.empty:
            continue
        for iid, grp in m.groupby("itemid", sort=False):
            ii = int(iid)
            if ii not in hits:
                continue
            hits[ii][grp["ecg_id"].astype(int).to_numpy()] = True
    return hits


def scan_outputevents_catalog(
    out_path: Path,
    ecg: pd.DataFrame,
    output_rows: Sequence[Tuple[str, Set[int], str]],
    chunk_size: int,
    hits: Dict[str, np.ndarray],
) -> None:
    itemid_to_keys: DefaultDict[int, List[str]] = defaultdict(list)
    all_ids: Set[int] = set()
    for key, idset, _ in output_rows:
        for i in idset:
            all_ids.add(int(i))
            itemid_to_keys[int(i)].append(key)

    sub = ecg[ecg["stay_id"].notna() & ecg["intime"].notna()].copy()
    if sub.empty:
        return
    sub["stay_id"] = sub["stay_id"].astype(int)
    stay_set = set(sub["stay_id"].unique())

    for chunk in tqdm(
        pd.read_csv(out_path, usecols=["stay_id", "itemid", "charttime"], chunksize=chunk_size, low_memory=False),
        desc="outputevents (Katalog)",
    ):
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "stay_id", "itemid"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stay_set) & chunk["itemid"].isin(all_ids)]
        if chunk.empty:
            continue
        m = chunk.merge(sub[["ecg_id", "stay_id", "intime", "t_window_end"]], on="stay_id", how="inner")
        m = m[(m["charttime"] >= m["intime"]) & (m["charttime"] <= m["t_window_end"])]
        if m.empty:
            continue
        for iid, grp in m.groupby("itemid", sort=False):
            for vk in itemid_to_keys.get(int(iid), []):
                hits[vk][grp["ecg_id"].astype(int).to_numpy()] = True


def scan_inputevents_combined(
    inp_path: Path,
    ecg: pd.DataFrame,
    itemid_to_drug: Dict[int, str],
    vaso_itemids: Set[int],
    chunk_size: int,
    hits: Dict[str, np.ndarray],
) -> None:
    """Alle inputevents-Starts im ICU-Fenster + Vasopressor-Substanz-Hits."""
    sub = ecg[ecg["stay_id"].notna() & ecg["intime"].notna()].copy()
    if sub.empty:
        return
    sub["stay_id"] = sub["stay_id"].astype(int)
    stay_set = set(sub["stay_id"].unique())

    for chunk in tqdm(
        pd.read_csv(
            inp_path,
            usecols=["stay_id", "itemid", "starttime"],
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc="inputevents (alle + Vasopressoren)",
    ):
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk = chunk.dropna(subset=["starttime", "stay_id", "itemid"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stay_set)]
        if chunk.empty:
            continue
        m = chunk.merge(sub[["ecg_id", "stay_id", "intime", "t_window_end"]], on="stay_id", how="inner")
        m = m[(m["starttime"] >= m["intime"]) & (m["starttime"] <= m["t_window_end"])]
        if m.empty:
            continue
        for row in m.itertuples(index=False):
            eid = int(row.ecg_id)
            hits["any_inputevent_start_in_window"][eid] = True
            drug = itemid_to_drug.get(int(row.itemid))
            if drug:
                hits[drug][eid] = True
                hits["any_vasopressor_iv"][eid] = True


def scan_diagnoses_icd_any_for_hadm(path: Path, ecg: pd.DataFrame, chunk_size: int, hits: np.ndarray) -> None:
    """Mind. eine ICD-Zeile zu hadm (MIMIC: keine charttime pro Diagnose)."""
    sub = ecg[ecg["hadm_id"].notna()].copy()
    if sub.empty:
        return
    sub["hadm_id"] = sub["hadm_id"].astype(int)
    hadm_set = set(sub["hadm_id"].unique())
    hadm_to_ecg_ids: DefaultDict[int, List[int]] = defaultdict(list)
    for _, r in sub.iterrows():
        hadm_to_ecg_ids[int(r["hadm_id"])].append(int(r["ecg_id"]))

    for chunk in tqdm(
        pd.read_csv(path, usecols=["hadm_id"], chunksize=chunk_size, low_memory=False),
        desc="diagnoses_icd (hadm)",
    ):
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(hadm_set)]
        if chunk.empty:
            continue
        for h in chunk["hadm_id"].unique():
            for eid in hadm_to_ecg_ids.get(int(h), []):
                hits[eid] = True


def scan_prescriptions_in_window(path: Path, ecg: pd.DataFrame, chunk_size: int, hits: np.ndarray) -> None:
    sub = ecg[ecg["hadm_id"].notna() & ecg["intime"].notna()].copy()
    if sub.empty:
        return
    sub["hadm_id"] = sub["hadm_id"].astype(int)
    sub["subject_id"] = sub["subject_id"].astype(int)
    hadm_set = set(sub["hadm_id"].unique())
    subj_set = set(sub["subject_id"].unique())
    keys = sub[["ecg_id", "subject_id", "hadm_id", "intime", "t_window_end"]].copy()

    for chunk in tqdm(
        pd.read_csv(
            path,
            usecols=["subject_id", "hadm_id", "starttime"],
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc="prescriptions (Fenster)",
    ):
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk = chunk.dropna(subset=["starttime", "hadm_id", "subject_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["subject_id"] = chunk["subject_id"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(hadm_set) & chunk["subject_id"].isin(subj_set)]
        if chunk.empty:
            continue
        m = chunk.merge(keys, on=["subject_id", "hadm_id"], how="inner")
        m = m[(m["starttime"] >= m["intime"]) & (m["starttime"] <= m["t_window_end"])]
        if m.empty:
            continue
        hits[m["ecg_id"].astype(int).to_numpy()] = True


def scan_procedures_icd_in_window(path: Path, ecg: pd.DataFrame, chunk_size: int, hits: np.ndarray) -> None:
    sub = ecg[ecg["hadm_id"].notna() & ecg["intime"].notna()].copy()
    if sub.empty:
        return
    sub["hadm_id"] = sub["hadm_id"].astype(int)
    sub["subject_id"] = sub["subject_id"].astype(int)
    hadm_set = set(sub["hadm_id"].unique())
    subj_set = set(sub["subject_id"].unique())
    keys = sub[["ecg_id", "subject_id", "hadm_id", "intime", "t_window_end"]].copy()

    for chunk in tqdm(
        pd.read_csv(
            path,
            usecols=["subject_id", "hadm_id", "chartdate"],
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc="procedures_icd (Fenster)",
    ):
        chunk["chartdate"] = pd.to_datetime(chunk["chartdate"], errors="coerce")
        chunk = chunk.dropna(subset=["chartdate", "hadm_id", "subject_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["subject_id"] = chunk["subject_id"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(hadm_set) & chunk["subject_id"].isin(subj_set)]
        if chunk.empty:
            continue
        m = chunk.merge(keys, on=["subject_id", "hadm_id"], how="inner")
        m = m[(m["chartdate"] >= m["intime"]) & (m["chartdate"] <= m["t_window_end"])]
        if m.empty:
            continue
        hits[m["ecg_id"].astype(int).to_numpy()] = True


def scan_microbiologyevents_in_window(path: Path, ecg: pd.DataFrame, chunk_size: int, hits: np.ndarray) -> None:
    sub = ecg[ecg["hadm_id"].notna() & ecg["intime"].notna()].copy()
    if sub.empty:
        return
    sub["hadm_id"] = sub["hadm_id"].astype(int)
    sub["subject_id"] = sub["subject_id"].astype(int)
    hadm_set = set(sub["hadm_id"].unique())
    subj_set = set(sub["subject_id"].unique())
    keys = sub[["ecg_id", "subject_id", "hadm_id", "intime", "t_window_end"]].copy()

    for chunk in tqdm(
        pd.read_csv(
            path,
            usecols=["subject_id", "hadm_id", "chartdate"],
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc="microbiologyevents (Fenster)",
    ):
        chunk["chartdate"] = pd.to_datetime(chunk["chartdate"], errors="coerce")
        chunk = chunk.dropna(subset=["chartdate", "hadm_id", "subject_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["subject_id"] = chunk["subject_id"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(hadm_set) & chunk["subject_id"].isin(subj_set)]
        if chunk.empty:
            continue
        m = chunk.merge(keys, on=["subject_id", "hadm_id"], how="inner")
        m = m[(m["chartdate"] >= m["intime"]) & (m["chartdate"] <= m["t_window_end"])]
        if m.empty:
            continue
        hits[m["ecg_id"].astype(int).to_numpy()] = True


def load_d_labitems_itemids(d_labitems_csv: Path, max_ids: int | None) -> Tuple[Set[int], Dict[int, str]]:
    dl = pd.read_csv(d_labitems_csv, usecols=["itemid", "label"], low_memory=False)
    dl = dl.dropna(subset=["itemid"])
    dl["itemid"] = dl["itemid"].astype(int)
    dl["label"] = dl["label"].astype(str)
    if max_ids is not None and len(dl) > max_ids:
        dl = dl.head(max_ids)
    ids = set(dl["itemid"].tolist())
    labels = dict(zip(dl["itemid"], dl["label"]))
    return ids, labels


def mimic_table_csv(mimic_dir: Path, table: str) -> Path:
    """PhysioNet-Layout: mimic-iv/<table>.csv/<table>.csv"""
    return mimic_dir / f"{table}.csv" / f"{table}.csv"


def load_d_items_output_itemids(d_items_csv: Path, max_ids: int | None) -> Tuple[Set[int], Dict[int, str]]:
    di = pd.read_csv(d_items_csv, usecols=["itemid", "label", "linksto"], low_memory=False)
    di = di[di["linksto"].astype(str).str.lower() == "outputevents"].dropna(subset=["itemid"])
    di["itemid"] = di["itemid"].astype(int)
    di["label"] = di["label"].astype(str)
    if max_ids is not None and len(di) > max_ids:
        di = di.head(max_ids)
    ids = set(di["itemid"].tolist())
    labels = dict(zip(di["itemid"], di["label"]))
    return ids, labels


def static_patient_and_admissions_coverage(
    ecg: pd.DataFrame,
    patients_csv: Path | None,
    admissions_csv: Path | None,
) -> List[Dict[str, object]]:
    """Eine Zeile pro Spalte in patients/admissions (Join über subject_id bzw. hadm_id)."""
    rows: List[Dict[str, object]] = []
    n = len(ecg)
    if n == 0:
        return rows

    def add_row(source: str, field: str, label_de: str, mask: pd.Series) -> None:
        c = int(mask.sum())
        rows.append(
            {
                "variable_key": field,
                "source": source,
                "label_de": label_de,
                "itemids": "",
                "n_ecg": n,
                "n_positive": c,
                "pct_ecg": round(100.0 * c / n, 4),
            }
        )

    add_row("cohorte", "stay_id_matched", "ICU-Stay zum ECG-Zeitpunkt gematcht", ecg["stay_id"].notna())
    if "hadm_id" in ecg.columns:
        add_row("cohorte", "hadm_id_matched", "Hospitalisierung (hadm) für Stay vorhanden", ecg["hadm_id"].notna())

    patient_skip = {"subject_id", "dod"}
    admission_skip = {"hadm_id"}

    if patients_csv and patients_csv.exists() and "subject_id" in ecg.columns:
        p = pd.read_csv(patients_csv, low_memory=False)
        if "subject_id" in p.columns:
            base = ecg[["ecg_id", "subject_id"]].dropna(subset=["subject_id"]).copy()
            base["subject_id"] = base["subject_id"].astype(int)
            pm = p.drop_duplicates(subset=["subject_id"], keep="first").copy()
            pm["subject_id"] = pm["subject_id"].astype(int)
            merged = base.merge(pm, on="subject_id", how="left")
            for col in sorted(pm.columns, key=str):
                if col in patient_skip:
                    continue
                if col not in merged.columns:
                    continue
                hit_ecg = set(merged.loc[merged[col].notna(), "ecg_id"].astype(int))
                add_row("patients.csv", col, f"patients.{col} (non-null)", ecg["ecg_id"].isin(hit_ecg))

    if admissions_csv and admissions_csv.exists() and "hadm_id" in ecg.columns:
        a = pd.read_csv(admissions_csv, low_memory=False)
        if "hadm_id" in a.columns:
            base = ecg[["ecg_id", "hadm_id"]].dropna(subset=["hadm_id"]).copy()
            base["hadm_id"] = base["hadm_id"].astype(int)
            am = a.drop_duplicates(subset=["hadm_id"], keep="first").copy()
            am["hadm_id"] = am["hadm_id"].astype(int)
            merged = base.merge(am, on="hadm_id", how="left")
            for col in sorted(am.columns, key=str):
                if col in admission_skip:
                    continue
                if col not in merged.columns:
                    continue
                hit_ecg = set(merged.loc[merged[col].notna(), "ecg_id"].astype(int))
                add_row("admissions.csv", col, f"admissions.{col} (non-null)", ecg["ecg_id"].isin(hit_ecg))

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Breite EHR-Coverage-Discovery (zeitlich/leakage-sicher) für die ECG-Kohorte."
    )
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
        "--mimic-iv-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeling" / "mimic-iv",
    )
    ap.add_argument("--icu-window-hours", type=float, default=6.0, help="ICU-Fenster ab intime (Default 6, z. B. 24)")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_DEFAULT)
    ap.add_argument("--max-ecg", type=int, default=None)
    ap.add_argument(
        "--wide-labevents",
        action="store_true",
        help="Zusätzlich alle itemids aus d_labitems.csv scannen (sehr langsam, große CSV).",
    )
    ap.add_argument(
        "--max-wide-lab-itemids",
        type=int,
        default=None,
        help="Optional: nur erste N Zeilen aus d_labitems (Debug).",
    )
    ap.add_argument(
        "--wide-outputevents",
        action="store_true",
        help="Zusätzlich alle outputevents-ItemIDs aus d_items (linksto=outputevents) scannen.",
    )
    ap.add_argument(
        "--max-wide-output-itemids",
        type=int,
        default=None,
        help="Optional: nur erste N Output-ItemIDs aus d_items.",
    )
    ap.add_argument(
        "--no-static-patient-tables",
        action="store_true",
        help="Keine patients/admissions-Spalten (sonst: alle Spalten automatisch, dod aus patients ausgelassen).",
    )
    ap.add_argument(
        "--no-mimic-extras",
        action="store_true",
        help="Keine diagnoses_icd / prescriptions / procedures_icd / microbiologyevents (sonst falls Dateien existieren).",
    )
    ap.add_argument(
        "--patients-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "patients.csv",
    )
    ap.add_argument(
        "--admissions-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "admissions.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "feature_analysis",
    )
    ap.add_argument("--write-md", action="store_true", help="Kurze Markdown-Tabelle schreiben.")
    args = ap.parse_args()

    chart_path = args.mimic_iv_dir / "chartevents.csv" / "chartevents.csv"
    lab_path = args.mimic_iv_dir / "labevents.csv" / "labevents.csv"
    out_path = args.mimic_iv_dir / "outputevents.csv" / "outputevents.csv"
    inp_path = args.mimic_iv_dir / "inputevents.csv" / "inputevents.csv"
    d_lab_path = args.mimic_iv_dir / "d_labitems.csv" / "d_labitems.csv"
    d_items_path = args.mimic_iv_dir / "d_items.csv" / "d_items.csv"

    for p in (args.timestamps_csv, args.icustays_csv, chart_path, lab_path, out_path, inp_path):
        if not p.exists():
            sys.exit(f"Pfad fehlt: {p}")
    if args.wide_labevents and not d_lab_path.exists():
        sys.exit(f"--wide-labevents: fehlt {d_lab_path}")
    if args.wide_outputevents and not d_items_path.exists():
        sys.exit(f"--wide-outputevents: fehlt {d_items_path}")

    chart_specs = _chart_specs_with_extras()
    therapy_specs = _therapy_specs()
    itemid_to_keys = _itemid_to_hit_keys(chart_specs, therapy_specs)
    all_chart_itemids: Set[int] = set(itemid_to_keys.keys())

    ecg = build_ecg_frame(
        args.timestamps_csv,
        args.icustays_csv,
        args.icu_window_hours,
        max_ecg=args.max_ecg,
    )
    n = len(ecg)
    if "ecg_id" not in ecg.columns:
        ecg["ecg_id"] = np.arange(n, dtype=np.int64)
    n_matched = int((ecg["stay_id"].notna() & ecg["intime"].notna()).sum())
    n_hadm = int(ecg["hadm_id"].notna().sum()) if "hadm_id" in ecg.columns else 0

    chart_hit_keys = [k for k, _, _ in chart_specs] + [k for k, _, _ in therapy_specs]
    chart_hits = _scan_chartevents_combined(
        chart_path,
        ecg,
        chart_hit_keys,
        itemid_to_keys,
        all_chart_itemids,
        args.chunk_size,
        keys_for_early_exit=[],
    )

    lab_rows = _lab_specs_extended()
    lab_keys = [k for k, _, _ in lab_rows]
    lab_hits = {k: np.zeros(n, dtype=bool) for k in lab_keys}
    scan_labevents_catalog(lab_path, ecg, lab_rows, args.chunk_size, lab_hits)

    urine_ids = set(im.get_all_outputevents_itemids())
    out_catalog: List[Tuple[str, Set[int], str]] = [
        ("urine_output", urine_ids, "Urin-Output (summierte ItemIDs, dokumentiert)"),
    ]
    out_hits = {k: np.zeros(n, dtype=bool) for k, _, _ in out_catalog}
    scan_outputevents_catalog(out_path, ecg, out_catalog, args.chunk_size, out_hits)

    itemid_to_drug, drug_keys = _input_drug_catalog()
    vaso_ids = set(itemid_to_drug.keys())
    in_hits = {d: np.zeros(n, dtype=bool) for d in drug_keys}
    in_hits["any_vasopressor_iv"] = np.zeros(n, dtype=bool)
    in_hits["any_inputevent_start_in_window"] = np.zeros(n, dtype=bool)
    scan_inputevents_combined(inp_path, ecg, itemid_to_drug, vaso_ids, args.chunk_size, in_hits)

    label_chart = {k: lab for k, _, lab in chart_specs}
    label_therapy = {k: lab for k, _, lab in therapy_specs}
    label_lab = {k: lab for k, _, lab in lab_rows}

    result_rows: List[Dict[str, object]] = []

    def append_hits(source: str, key: str, vec: np.ndarray, label_de: str, itemid_note: str = "") -> None:
        c = int(vec.sum())
        result_rows.append(
            {
                "variable_key": key,
                "source": source,
                "label_de": label_de,
                "itemids": itemid_note,
                "n_ecg": n,
                "n_positive": c,
                "pct_ecg": round(100.0 * float(c) / n, 4),
            }
        )

    def itemids_for_chart_key(k: str) -> str:
        for kk, ids, _ in chart_specs:
            if kk == k:
                return ",".join(str(x) for x in ids[:16]) + ("..." if len(ids) > 16 else "")
        for kk, idset, _ in therapy_specs:
            if kk == k:
                sl = sorted(idset)
                return ",".join(str(x) for x in sl[:16]) + ("..." if len(sl) > 16 else "")
        return ""

    for k in chart_hit_keys:
        lab = label_chart.get(k) or label_therapy.get(k) or k
        append_hits("chartevents", k, chart_hits[k], lab, itemids_for_chart_key(k))

    def itemids_for_lab_key(k: str) -> str:
        for kk, ids, _ in lab_rows:
            if kk == k:
                return ",".join(str(x) for x in ids)
        return ""

    for k in lab_keys:
        append_hits("labevents", k, lab_hits[k], label_lab[k], itemids_for_lab_key(k))

    for k, _, lab in out_catalog:
        append_hits("outputevents", k, out_hits[k], lab, "see urine itemid list in itemid_mappings")

    for d in drug_keys:
        append_hits(
            "inputevents",
            f"vaso_{d}",
            in_hits[d],
            f"Vasopressor/Infusion: {d} (inputevents, Start im Fenster)",
            ",".join(str(i) for i, nm in itemid_to_drug.items() if nm == d),
        )
    append_hits(
        "inputevents",
        "any_vasopressor_iv",
        in_hits["any_vasopressor_iv"],
        "Beliebiger Vasopressor aus inputevents (Kategorie-Mapping) im Fenster",
        "",
    )
    append_hits(
        "inputevents",
        "any_inputevent_start_in_window",
        in_hits["any_inputevent_start_in_window"],
        "Beliebiger inputevents-Start im ICU-Fenster (alle Medikamente/Infusionen)",
        "",
    )

    mimic_extra_status: Dict[str, str | bool] = {}
    if not args.no_mimic_extras:
        md = args.mimic_iv_dir
        d_icd = mimic_table_csv(md, "diagnoses_icd")
        if d_icd.exists():
            dx = np.zeros(n, dtype=bool)
            scan_diagnoses_icd_any_for_hadm(d_icd, ecg, args.chunk_size, dx)
            append_hits(
                "diagnoses_icd",
                "any_diagnosis_row_for_hadm",
                dx,
                "Mind. eine ICD-Zeile zu hadm (hadm-Ebene, kein Intra-Stay-Zeitstempel)",
                "",
            )
            mimic_extra_status["diagnoses_icd"] = True
        else:
            mimic_extra_status["diagnoses_icd"] = "skipped_missing_file"

        rx_path = mimic_table_csv(md, "prescriptions")
        if rx_path.exists():
            rx = np.zeros(n, dtype=bool)
            scan_prescriptions_in_window(rx_path, ecg, args.chunk_size, rx)
            append_hits(
                "prescriptions",
                "any_prescription_start_in_window",
                rx,
                "Mind. eine Verordnung mit starttime im ICU-Fenster",
                "",
            )
            mimic_extra_status["prescriptions"] = True
        else:
            mimic_extra_status["prescriptions"] = "skipped_missing_file"

        proc_path = mimic_table_csv(md, "procedures_icd")
        if proc_path.exists():
            pr = np.zeros(n, dtype=bool)
            scan_procedures_icd_in_window(proc_path, ecg, args.chunk_size, pr)
            append_hits(
                "procedures_icd",
                "any_procedure_icd_chartdate_in_window",
                pr,
                "Mind. eine procedures_icd-Zeile mit chartdate im ICU-Fenster",
                "",
            )
            mimic_extra_status["procedures_icd"] = True
        else:
            mimic_extra_status["procedures_icd"] = "skipped_missing_file"

        micro_path = mimic_table_csv(md, "microbiologyevents")
        if micro_path.exists():
            mb = np.zeros(n, dtype=bool)
            scan_microbiologyevents_in_window(micro_path, ecg, args.chunk_size, mb)
            append_hits(
                "microbiologyevents",
                "any_microbiology_chartdate_in_window",
                mb,
                "Mind. eine Mikrobiologie-Zeile mit chartdate im ICU-Fenster",
                "",
            )
            mimic_extra_status["microbiologyevents"] = True
        else:
            mimic_extra_status["microbiologyevents"] = "skipped_missing_file"
    else:
        mimic_extra_status["disabled"] = True

    if not args.no_static_patient_tables:
        static_rows = static_patient_and_admissions_coverage(
            ecg,
            args.patients_csv if args.patients_csv.exists() else None,
            args.admissions_csv if args.admissions_csv.exists() else None,
        )
        result_rows.extend(static_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    htag = str(int(args.icu_window_hours)) if args.icu_window_hours == int(args.icu_window_hours) else str(args.icu_window_hours).replace(".", "p")
    base = args.out_dir / f"ehr_wide_discovery_first{htag}h_curated"
    curated_csv = base.with_suffix(".csv")

    wide_lab_rows: List[Dict[str, object]] = []
    if args.wide_labevents:
        lab_id_set, lab_labels = load_d_labitems_itemids(d_lab_path, args.max_wide_lab_itemids)
        wide_hits = scan_labevents_wide(lab_path, ecg, lab_id_set, lab_labels, args.chunk_size)
        for iid, vec in wide_hits.items():
            c = int(vec.sum())
            if c == 0:
                continue
            wide_lab_rows.append(
                {
                    "variable_key": f"lab_itemid_{iid}",
                    "source": "labevents",
                    "label_de": lab_labels.get(int(iid), str(iid)),
                    "itemids": str(iid),
                    "n_ecg": n,
                    "n_positive": c,
                    "pct_ecg": round(100.0 * float(c) / n, 4),
                }
            )
        wide_lab_rows.sort(key=lambda r: float(r["pct_ecg"]), reverse=True)
        pd.DataFrame(wide_lab_rows).to_csv(base.parent / f"{base.name}_wide_labevents.csv", index=False)

    if args.wide_outputevents:
        oid_set, o_labels = load_d_items_output_itemids(d_items_path, args.max_wide_output_itemids)
        wide_out_hits = scan_outputevents_wide(out_path, ecg, oid_set, args.chunk_size)
        wide_out_rows: List[Dict[str, object]] = []
        for iid, vec in wide_out_hits.items():
            c = int(vec.sum())
            if c == 0:
                continue
            wide_out_rows.append(
                {
                    "variable_key": f"output_itemid_{iid}",
                    "source": "outputevents",
                    "label_de": o_labels.get(int(iid), str(iid)),
                    "itemids": str(iid),
                    "n_ecg": n,
                    "n_positive": c,
                    "pct_ecg": round(100.0 * float(c) / n, 4),
                }
            )
        wide_out_rows.sort(key=lambda r: float(r["pct_ecg"]), reverse=True)
        pd.DataFrame(wide_out_rows).to_csv(base.parent / f"{base.name}_wide_outputevents.csv", index=False)

    out_df = pd.DataFrame(result_rows)
    out_df = out_df.sort_values(["source", "pct_ecg"], ascending=[True, False])
    out_df.to_csv(curated_csv, index=False)

    temporal_rule = (
        f"stay_intime <= event_time <= min(ecg_time, stay_intime + {args.icu_window_hours:g} h); "
        "kein Look-ahead nach ECG."
    )
    meta = {
        "cohort_timestamps_csv": str(args.timestamps_csv),
        "icustays_csv": str(args.icustays_csv),
        "icu_window_hours": args.icu_window_hours,
        "temporal_rule": temporal_rule,
        "n_ecg_rows": n,
        "n_ecg_icu_stay_matched": n_matched,
        "n_ecg_with_hadm": n_hadm,
        "curated_csv": str(curated_csv),
        "wide_labevents_csv": str(base.parent / f"{base.name}_wide_labevents.csv") if args.wide_labevents else None,
        "wide_outputevents_csv": str(base.parent / f"{base.name}_wide_outputevents.csv")
        if args.wide_outputevents
        else None,
        "static_patient_tables": not args.no_static_patient_tables,
        "mimic_extras": mimic_extra_status,
    }
    meta_path = base.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.write_md:
        lines = [
            "# EHR Wide Discovery (kuratiert)",
            "",
            temporal_rule,
            "",
            f"n_ecg={n}, ICU-match={n_matched}, hadm={n_hadm}",
            "",
            "| Variable | Quelle | Beschreibung | % ECGs |",
            "|----------|--------|--------------|--------|",
        ]
        for _, r in out_df.iterrows():
            desc = str(r["label_de"]).replace("|", "\\|")
            lines.append(
                f"| `{r['variable_key']}` | {r['source']} | {desc} | {float(r['pct_ecg']):.2f} |"
            )
        (base.with_suffix(".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"ECG-Zeilen: {n:,}, ICU-match: {n_matched:,}, mit hadm: {n_hadm:,}")
    print(f"Zeitregel: {temporal_rule}")
    print(f"CSV: {curated_csv.resolve()}")
    print(f"Meta: {meta_path.resolve()}")
    if args.wide_labevents:
        print(f"Wide labs: {(base.parent / f'{base.name}_wide_labevents.csv').resolve()}")
    if args.wide_outputevents:
        print(f"Wide output: {(base.parent / f'{base.name}_wide_outputevents.csv').resolve()}")
    print()
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()