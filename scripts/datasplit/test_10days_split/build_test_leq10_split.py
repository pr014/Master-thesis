#!/usr/bin/env python3
"""
Build a frozen **test-split-only** cohort: preprocessed ECG windows (.npy) whose mapped
ICU stay has true LOS ≤ ``max_los_days`` (default 10).

Uses the **same** record list, cohort filters (SOFA / ICU therapy), timestamp mapping,
and **temporal (_stratified_) subject split** as ``create_dataloaders`` in
``src/data/ecg/dataloader_factory.py``. Only subjects in the **test** segment are
considered, so no train/val leakage.

Output layout mirrors ``data_dir`` (e.g. P1): relative paths under
``data/icu_ecgs_24h/test_10days_split/``.

Example (from repo root)
------------------------
  python scripts/datasplit/test_10days_split/build_test_leq10_split.py \\
      --config configs/model/deepecg_sl/deepecg_sl.yaml

  python scripts/datasplit/test_10days_split/build_test_leq10_split.py \\
      --config configs/model/CNN/cnn_scratch.yaml \\
      --max-los-days 10 --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.data.ecg.dataloader_factory import (  # noqa: E402
    create_temporal_stratified_split,
    filter_records_to_icu_therapy_labeled,
    filter_records_to_valid_sofa,
    get_subject_los_values,
    get_subject_timestamps,
)
from src.data.ecg.ecg_dataset import construct_ecg_time, extract_subject_id_from_path  # noqa: E402
from src.data.ecg.ecg_loader import build_npy_index  # noqa: E402
from src.data.ecg.timestamp_mapping import get_timestamp_mapping_path, load_timestamp_mapping  # noqa: E402
from src.training.training_utils import setup_icustays_mapper  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402


def _resolve_config_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _record_stay_los(
    record: Dict[str, Any],
    icu_mapper: Any,
    timestamp_mapping: Optional[Dict[str, Dict[str, Optional[str]]]],
    data_dir: Path,
) -> Tuple[Optional[int], Optional[float]]:
    """Match one record to stay LOS; mirrors ``get_los_values_for_records`` + stay_id."""
    base_path = record.get("base_path", "")
    if not base_path:
        return None, None
    try:
        subject_id = extract_subject_id_from_path(base_path)
    except ValueError:
        return None, None

    ecg_time = None
    if timestamp_mapping is not None:
        try:
            base_path_obj = Path(base_path)
            if base_path_obj.is_absolute():
                rel_path = base_path_obj.relative_to(data_dir)
            else:
                rel_path = Path(base_path)
            rel_path_str = str(rel_path).replace("\\", "/")
            if rel_path_str in timestamp_mapping:
                ts_info = timestamp_mapping[rel_path_str]
                ecg_time = construct_ecg_time(ts_info.get("base_date"), ts_info.get("base_time"))
        except (ValueError, KeyError):
            pass

    if ecg_time is None and icu_mapper is not None:
        subject_stays = icu_mapper.icustays_df[icu_mapper.icustays_df["subject_id"] == subject_id]
        if len(subject_stays) > 0:
            ecg_time = pd.to_datetime(subject_stays.iloc[0]["intime"])

    if ecg_time is None:
        return None, None

    stay_id = icu_mapper.map_ecg_to_stay(subject_id, ecg_time)
    if stay_id is None:
        return None, None
    los_days = icu_mapper.get_los(stay_id)
    if los_days is None:
        return None, None
    return int(stay_id), float(los_days)


def _build_test_records_same_as_dataloader(
    config: Dict[str, Any],
) -> Tuple[List[Dict], List[int], Any, Dict[str, Dict[str, Optional[str]]]]:
    """Return test records, test subject ids, icu_mapper, timestamp_mapping."""
    data_config = config.get("data", {})
    data_dir = Path(data_config["data_dir"])
    split_strategy = data_config.get("split_strategy", "temporal_stratified")
    if split_strategy not in ("temporal", "temporal_stratified"):
        raise ValueError(f"Unsupported split_strategy: {split_strategy!r}")

    val_split = float(config.get("validation", {}).get("val_split", 0.2))
    test_split = float(config.get("test_split", 0.0))
    if test_split <= 0:
        raise ValueError("config test_split must be > 0 to define a test cohort")

    records = build_npy_index(data_dir=str(data_dir))
    records = filter_records_to_valid_sofa(records, data_config, str(data_dir))
    records = filter_records_to_icu_therapy_labeled(records, data_config, str(data_dir))

    mapping_path = get_timestamp_mapping_path(str(data_dir))
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Timestamp mapping missing: {mapping_path}\n"
            "Create it before running this script (same as training)."
        )
    timestamp_mapping = load_timestamp_mapping(str(mapping_path))

    icu_mapper = setup_icustays_mapper(config)

    records_with_subject: List[Tuple[int, Dict[str, Any]]] = []
    for record in records:
        try:
            sid = extract_subject_id_from_path(record["base_path"])
        except ValueError:
            continue
        records_with_subject.append((sid, record))

    subject_to_records: Dict[int, List[Dict[str, Any]]] = {}
    for sid, rec in records_with_subject:
        subject_to_records.setdefault(sid, []).append(rec)

    unique_subjects = list(subject_to_records.keys())
    stratify = split_strategy == "temporal_stratified"
    subject_timestamps = get_subject_timestamps(
        subjects=unique_subjects,
        subject_to_records=subject_to_records,
        icu_mapper=icu_mapper,
        timestamp_mapping=timestamp_mapping,
        data_dir=str(data_dir),
    )
    if not subject_timestamps:
        raise ValueError("No subject timestamps; cannot temporal-split.")

    subject_los = None
    if stratify:
        subject_los = get_subject_los_values(unique_subjects, subject_to_records, icu_mapper)

    train_subjects, val_subjects, test_subjects = create_temporal_stratified_split(
        subjects=unique_subjects,
        subject_timestamps=subject_timestamps,
        subject_los=subject_los,
        test_size=test_split,
        val_size=val_split,
        stratify=stratify,
        n_bins=10,
        random_state=config.get("seed", 42),
    )

    test_records = []
    for sid in test_subjects:
        test_records.extend(subject_to_records[sid])

    print(
        f"Split: train={len(train_subjects)} val={len(val_subjects)} test={len(test_subjects)} subjects; "
        f"test records={len(test_records):,}"
    )
    return test_records, test_subjects, icu_mapper, timestamp_mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/deepecg_sl/deepecg_sl.yaml",
        help="Training YAML (same data_dir / filters / test_split / seed as training).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/icu_ecgs_24h/test_10days_split",
        help="Dataset root; mirrors relative paths under data_dir.",
    )
    parser.add_argument(
        "--max-los-days",
        type=float,
        default=10.0,
        help="Keep only windows whose mapped ICU stay LOS is in [0, max_los_days].",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and write manifests only; do not copy files.",
    )
    args = parser.parse_args()

    cfg_path = _resolve_config_path(args.config)
    config = load_config(model_config_path=cfg_path, base_dir=REPO_ROOT)
    data_dir = Path(config["data"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    test_records, test_subject_ids, icu_mapper, timestamp_mapping = _build_test_records_same_as_dataloader(
        config
    )

    rows: List[Dict[str, Any]] = []
    copy_plan: List[Tuple[Path, Path, Dict[str, Any]]] = []

    max_los = float(args.max_los_days)
    skipped_no_match = 0
    skipped_los = 0

    for rec in test_records:
        stay_id, los = _record_stay_los(rec, icu_mapper, timestamp_mapping, data_dir)
        if stay_id is None or los is None:
            skipped_no_match += 1
            continue
        if los < 0 or los > max_los:
            skipped_los += 1
            continue

        base_path = Path(rec["base_path"])
        try:
            rel = base_path.relative_to(data_dir)
        except ValueError:
            rel = Path(rec["base_path"])

        src_npy = Path(rec.get("npy_path", str(base_path) + ".npy"))
        if not src_npy.exists():
            skipped_no_match += 1
            continue

        subject_id = extract_subject_id_from_path(str(base_path))
        rows.append(
            {
                "rel_path": str(rel).replace("\\", "/"),
                "subject_id": subject_id,
                "stay_id": stay_id,
                "los_days": los,
                "src_npy": str(src_npy),
            }
        )
        dest_npy = out_dir / rel.with_suffix(".npy")
        copy_plan.append((src_npy, dest_npy, rows[-1]))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(rows).sort_values(["subject_id", "stay_id", "rel_path"]).to_csv(
        manifest_path, index=False
    )

    meta = {
        "config": str(cfg_path),
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "seed": config.get("seed"),
        "val_split": config.get("validation", {}).get("val_split"),
        "test_split": config.get("test_split"),
        "split_strategy": config.get("data", {}).get("split_strategy"),
        "max_los_days": max_los,
        "n_test_subjects": len(test_subject_ids),
        "n_test_records_total": len(test_records),
        "n_manifest_rows": len(rows),
        "skipped_no_match_or_missing_file": skipped_no_match,
        "skipped_los_outside_0_max": skipped_los,
        "dry_run": bool(args.dry_run),
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Manifest: {manifest_path} ({len(rows):,} rows)")
    print(
        f"Skipped (no stay LOS / missing npy): {skipped_no_match:,}; "
        f"skipped (LOS not in [0, {max_los}]): {skipped_los:,}"
    )

    if args.dry_run:
        print("Dry-run: no files copied.")
        return 0

    copied = 0
    for src, dest, _ in copy_plan:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    print(f"Copied {copied:,} .npy files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
