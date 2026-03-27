"""Torch Dataset: load .npy ECGs from balanced_mortality using manifest.csv rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BalancedMortalityDataset(Dataset):
    """One manifest row -> one sample; signal shape (12, 5000) for CNNScratch."""

    def __init__(
        self,
        data_dir: Path,
        rows: pd.DataFrame,
        transform: Optional[Any] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows.iloc[idx]
        rel = str(row["dest_relpath"]).replace("\\", "/")
        npy_path = self.data_dir / rel
        if not npy_path.is_file():
            raise FileNotFoundError(f"Missing npy: {npy_path}")

        x = np.load(npy_path)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.ndim == 2:
            if x.shape[0] < x.shape[1]:
                x = x.T
        x = x.astype(np.float32, copy=False)

        if torch.isnan(torch.from_numpy(x)).any():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        signal = torch.from_numpy(x).float()
        if signal.dim() == 2 and signal.shape[0] > signal.shape[1]:
            signal = signal.transpose(0, 1)

        y = int(row["y"])
        subject_id = int(row["subject_id"])

        if self.transform is not None:
            if hasattr(self.transform, "train"):
                self.transform.train()
            signal = self.transform(signal)

        return {
            "signal": signal,
            "label": torch.tensor(y, dtype=torch.long),
            "subject_id": subject_id,
        }


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load manifest.csv; require dest_relpath, y, subject_id."""
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found: {p}")
    df = pd.read_csv(p)
    required = {"dest_relpath", "y", "subject_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return df


def collate_balanced_mortality(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    signals = torch.stack([b["signal"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"signal": signals, "label": labels}
