"""
Map 24h ECG records (timestamp CSV) to ICU stay_id via ICUStayMapper.

Used by scripts/scoring_models/calculate_sofa.py to restrict SOFA to ECG cohort.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **kwargs):
        return it

from ...data.labeling import ICUStayMapper, load_icustays as load_labeling_icustays


def build_ecg_stay_mapping(
    timestamps_csv: Path | str,
    icustays_csv: Path | str,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Build one row per ECG record with optional stay_id.

    Args:
        timestamps_csv: e.g. timestamps_mapping_24h_P1.csv with base_path, base_date,
            base_time, subject_id, study_id.
        icustays_csv: project icustays under labels_csv (same source as training mapper).
        nrows: Optional limit on rows read from timestamps_csv (first nrows records).

    Returns:
        DataFrame with base_path, subject_id, study_id (if present), ecg_time, stay_id.
    """
    timestamps_csv = Path(timestamps_csv)
    icustays_csv = Path(icustays_csv)

    ts = pd.read_csv(timestamps_csv, nrows=nrows)
    required = {"base_path", "base_date", "base_time", "subject_id"}
    missing = required - set(ts.columns)
    if missing:
        raise ValueError(f"timestamps CSV missing columns: {missing}")

    icu_df = load_labeling_icustays(str(icustays_csv))
    mapper = ICUStayMapper(icu_df)

    records = []
    for _, row in tqdm(ts.iterrows(), total=len(ts), desc="ECG→Stay"):
        subject_id = int(row["subject_id"])
        ecg_time = pd.to_datetime(f"{row['base_date']} {row['base_time']}")
        stay_id = mapper.map_ecg_to_stay(subject_id, ecg_time)
        rec = {
            "base_path": row["base_path"],
            "subject_id": subject_id,
            "ecg_time": ecg_time,
            "stay_id": stay_id,
        }
        if "study_id" in ts.columns:
            rec["study_id"] = row["study_id"]
        records.append(rec)

    return pd.DataFrame(records)


def unique_stay_ids_from_mapping(ecg_mapping: pd.DataFrame) -> List[int]:
    """Sorted unique stay_ids with a successful ICU match (excludes None)."""
    valid = ecg_mapping[ecg_mapping["stay_id"].notna()]
    if len(valid) == 0:
        return []
    return sorted(valid["stay_id"].astype(int).unique().tolist())
