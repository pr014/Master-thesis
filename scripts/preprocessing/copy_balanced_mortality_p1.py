"""Build a balanced (50/50) mortality cohort from P1 .npy ECGs via file COPY.

Copies selected ``.npy`` files into ``data/icu_ecgs_24h/balanced_mortality/`` using
``shutil.copy2`` (source files under P1 are left unchanged).

Label (ever died after ECG, consistent with prior plan):
  - y=1: ``dod`` parseable and ``ecg_time < dod``
  - y=0: ``dod`` missing / not parseable

One ECG per ``subject_id`` (earliest qualifying ``ecg_time``). Patients with any
qualifying y=1 row are treated as positive; otherwise negative if only y=0 rows.

Requires ``pandas``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--labels-csv",
        type=str,
        default="data/labeling/labels_csv/records_w_diag_icd10.csv",
        help="Path to records_w_diag_icd10.csv (repo-relative or absolute).",
    )
    p.add_argument(
        "--p1-dir",
        type=str,
        default="data/icu_ecgs_24h/P1",
        help="Root of P1 preprocessed .npy tree.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="data/icu_ecgs_24h/balanced_mortality",
        help="Destination root; structure mirrors P1 relative paths.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for negative sampling.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute cohort and print stats but do not copy files or write manifest.",
    )
    return p.parse_args()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def _build_study_id_to_p1_relpath(p1_root: Path) -> Tuple[Dict[int, str], List[str]]:
    """Map study_id -> relative POSIX path from P1 root (without .npy)."""
    npy_files = sorted(p1_root.rglob("*.npy"))
    mapping: Dict[int, str] = {}
    warnings: List[str] = []
    for npy in npy_files:
        try:
            rel = npy.relative_to(p1_root)
        except ValueError:
            continue
        stem = npy.stem
        if not stem.isdigit():
            warnings.append(f"Skip non-numeric stem: {npy}")
            continue
        study_id = int(stem)
        parent_name = npy.parent.name
        if parent_name != f"s{study_id}":
            warnings.append(
                f"study_id folder mismatch for {npy} (parent={parent_name}, expected=s{study_id})"
            )
        rel_no_ext = rel.with_suffix("").as_posix()
        if study_id in mapping:
            prev = mapping[study_id]
            if rel_no_ext != prev:
                warnings.append(
                    f"Duplicate study_id {study_id}: keeping lexicographically first path "
                    f"({prev!r} vs {rel_no_ext!r})"
                )
            if rel_no_ext < prev:
                mapping[study_id] = rel_no_ext
        else:
            mapping[study_id] = rel_no_ext
    return mapping, warnings


def _row_label_and_valid(
    ecg_time: pd.Timestamp, dod_ts: Optional[pd.Timestamp]
) -> Optional[int]:
    """Return 0, 1, or None if row should be dropped."""
    if dod_ts is None or pd.isna(dod_ts):
        return 0
    if pd.isna(ecg_time):
        return None
    if ecg_time < dod_ts:
        return 1
    return None


def main() -> None:
    args = _parse_args()
    labels_path = _resolve(args.labels_csv)
    p1_root = _resolve(args.p1_dir)
    out_root = _resolve(args.output_dir)

    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")
    if not p1_root.is_dir():
        raise FileNotFoundError(f"P1 directory not found: {p1_root}")

    study_to_rel, p1_warnings = _build_study_id_to_p1_relpath(p1_root)
    for w in p1_warnings:
        print(f"WARN: {w}")
    print(f"P1: {len(study_to_rel):,} study_id keys under {p1_root}")

    usecols = [
        "subject_id",
        "study_id",
        "ecg_time",
        "dod",
        "gender",
        "age",
    ]
    df = pd.read_csv(labels_path, usecols=lambda c: c in usecols, low_memory=False)
    df["ecg_time"] = pd.to_datetime(df["ecg_time"], errors="coerce")
    df["dod"] = pd.to_datetime(df["dod"], errors="coerce")
    df["subject_id"] = df["subject_id"].astype(int)
    df["study_id"] = df["study_id"].astype(int)

    df["in_p1"] = df["study_id"].map(lambda sid: sid in study_to_rel)
    df_p1 = df[df["in_p1"]].copy()

    rows_dropped_no_time = df_p1["ecg_time"].isna().sum()
    df_p1 = df_p1.dropna(subset=["ecg_time"])

    df_p1["y_row"] = [
        _row_label_and_valid(t, d) for t, d in zip(df_p1["ecg_time"], df_p1["dod"])
    ]
    dropped_after_death = (df_p1["y_row"].isna() & df_p1["dod"].notna()).sum()
    df_p1 = df_p1.dropna(subset=["y_row"])
    df_p1["y_row"] = df_p1["y_row"].astype(int)

    # One row per subject: positives win; earliest qualifying ecg_time per side.
    positive_pick: List[Dict[str, Any]] = []
    negative_pick: List[Dict[str, Any]] = []

    for sid, grp in df_p1.groupby("subject_id", sort=True):
        gpos = grp[grp["y_row"] == 1]
        if len(gpos) > 0:
            row = gpos.loc[gpos["ecg_time"].idxmin()]
            positive_pick.append(
                {
                    "subject_id": int(sid),
                    "study_id": int(row["study_id"]),
                    "y": 1,
                    "ecg_time": row["ecg_time"],
                    "dod": row["dod"],
                    "gender": row.get("gender"),
                    "age": row.get("age"),
                }
            )
            continue
        gneg = grp[grp["y_row"] == 0]
        if len(gneg) == 0:
            continue
        row = gneg.loc[gneg["ecg_time"].idxmin()]
        negative_pick.append(
            {
                "subject_id": int(sid),
                "study_id": int(row["study_id"]),
                "y": 0,
                "ecg_time": row["ecg_time"],
                "dod": row["dod"],
                "gender": row.get("gender"),
                "age": row.get("age"),
            }
        )

    n_pos = len(positive_pick)
    neg_df = pd.DataFrame(negative_pick)
    if neg_df.empty and n_pos > 0:
        raise RuntimeError("No negative patients available after filtering.")

    n_need = n_pos
    if len(neg_df) < n_need:
        raise RuntimeError(
            f"Not enough negative patients: need {n_need}, have {len(neg_df)}. "
            "Relax filters or use a larger pool."
        )

    neg_sample = neg_df.sample(n=n_need, random_state=args.seed).sort_values(
        "subject_id"
    )

    pos_df = pd.DataFrame(positive_pick).sort_values("subject_id")
    cohort = pd.concat([pos_df, neg_sample], ignore_index=True)
    cohort["p1_rel_path"] = cohort["study_id"].map(lambda s: study_to_rel[int(s)])

    meta = {
        "seed": args.seed,
        "labels_csv": str(labels_path),
        "p1_dir": str(p1_root),
        "output_dir": str(out_root),
        "n_p1_study_ids": len(study_to_rel),
        "rows_total_in_csv": int(len(df)),
        "rows_in_p1_study_match": int(df["in_p1"].sum()),
        "rows_dropped_missing_ecg_time": int(rows_dropped_no_time),
        "rows_dropped_ecg_not_before_dod": int(dropped_after_death),
        "n_positive_patients": n_pos,
        "n_negative_patients_sampled": n_need,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(meta, indent=2, default=str))

    if args.dry_run:
        print("Dry run: skipping copy and manifest.")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []

    for _, row in cohort.iterrows():
        rel = str(row["p1_rel_path"])
        src_npy = p1_root / f"{rel}.npy"
        if not src_npy.is_file():
            raise FileNotFoundError(f"Expected P1 file missing: {src_npy}")
        dst_npy = out_root / f"{rel}.npy"
        dst_npy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_npy, dst_npy)
        manifest_rows.append(
            {
                "subject_id": row["subject_id"],
                "study_id": row["study_id"],
                "y": row["y"],
                "p1_rel_path": rel,
                "dest_relpath": f"{rel}.npy",
                "ecg_time": row["ecg_time"],
                "dod": row["dod"],
                "gender": row["gender"],
                "age": row["age"],
            }
        )

    manifest_path = out_root / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    meta_path = out_root / "cohort_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(manifest_rows)} copies under {out_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
