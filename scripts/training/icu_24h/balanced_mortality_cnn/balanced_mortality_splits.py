"""Patient-disjoint stratified train/val/test splits for balanced mortality manifest rows."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_subject_split(
    df: pd.DataFrame,
    val_split: float,
    test_split: float,
    random_state: int,
    subject_col: str = "subject_id",
    label_col: str = "y",
) -> Tuple[List[int], List[int], List[int]]:
    """Split row indices so each subject appears in exactly one split; stratify by label.

    Expects one row per subject (as in balanced_mortality manifest). If duplicates exist,
    drops duplicate subjects keeping the first row (deterministic sort by subject_col).
    """
    if subject_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"DataFrame must contain {subject_col!r} and {label_col!r}")

    work = df[[subject_col, label_col]].copy()
    work["_idx"] = np.arange(len(df))
    work = work.sort_values(subject_col).drop_duplicates(subset=[subject_col], keep="first")

    indices = work["_idx"].to_numpy()
    y = work[label_col].to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError("Stratified split requires at least two classes in the cohort.")

    if not (0.0 < test_split < 1.0):
        raise ValueError("test_split must be in (0, 1)")
    if not (0.0 <= val_split < 1.0 - test_split):
        raise ValueError("val_split invalid for given test_split")

    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=test_split,
        stratify=y,
        random_state=random_state,
    )

    y_tv = df.loc[idx_train_val, label_col].to_numpy()
    val_size = val_split / (1.0 - test_split)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_size,
        stratify=y_tv,
        random_state=random_state,
    )

    return idx_train.tolist(), idx_val.tolist(), idx_test.tolist()
