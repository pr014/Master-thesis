"""ECG data loading utilities for MIMIC-IV-ECG (demo dataset).

Based on PhysioNet Usage Notes:
- ECG timestamps (base_date, base_time) are from the machine's internal clock
- May not be synchronized with other MIMIC-IV databases (Clinical, Waveform)
- Some ECGs collected outside ED/ICU may not overlap with Clinical Database timestamps
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass
class ECGRecord:
    base_path: str  # path without extension to the record (PhysioNet-style)


def build_demo_index(data_dir: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Discover demo ECG records in a directory.

    Looks for PhysioNet-style pairs of .hea/.dat files and returns a list of
    metadata dicts with a `base_path` key (path without extension).
    """

    root = Path(data_dir) if data_dir is not None else Path("data/raw/demo/ecg/mimic-iv-ecg-demo")
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    # Find all .hea header files that have a corresponding .dat
    header_files = sorted(root.rglob("*.hea"))
    records: List[Dict[str, str]] = []
    for hea_path in header_files:
        base = hea_path.with_suffix("")
        dat_path = base.with_suffix(".dat")
        if dat_path.exists():
            records.append({"base_path": str(base)})
            if limit is not None and len(records) >= limit:
                break

    if not records:
        raise RuntimeError(f"No demo ECG records found in {root}")

    return records


class ECGDemoDataset:
    def __init__(
        self,
        records: List[Dict[str, str]],
        preprocess: Optional[Callable] = None,
        window_seconds: Optional[float] = None,
    ) -> None:
        self.records = records
        self.preprocess = preprocess
        self.window_seconds = window_seconds

    def __len__(self) -> int:  # noqa: D401
        return len(self.records)

    def _load_record(self, base_path: str):
        try:
            import wfdb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "wfdb package is required to read PhysioNet-format ECG files"
            ) from e

        record = wfdb.rdrecord(base_path)
        # p_signal shape: (T, C) as float64; fs: float
        x = record.p_signal
        fs = float(record.fs)
        
        # Extract timestamp metadata (base_date and base_time from WFDB format)
        # Note: These timestamps may not be synchronized with other MIMIC-IV databases
        # See PhysioNet Usage Notes for limitations
        base_date = getattr(record, 'base_date', None)
        base_time = getattr(record, 'base_time', None)
        
        return x, fs, base_date, base_time

    def __getitem__(self, idx: int):
        item = self.records[idx]
        base_path = item["base_path"]

        x, fs, base_date, base_time = self._load_record(base_path)

        # Apply fixed-length window from start if requested
        if self.window_seconds is not None:
            window_samples = int(self.window_seconds * fs)
            end = min(window_samples, x.shape[0])
            x = x[0:end]

        # Apply preprocessing per-lead if provided
        if self.preprocess is not None:
            try:
                # Prefer preprocess(x, fs) if supported
                x = self.preprocess(x, fs)  # type: ignore[misc]
            except TypeError:
                x = self.preprocess(x)

        return {
            "signal": x,
            "meta": {
                "base_path": base_path,
                "fs": fs,
                "num_samples": x.shape[0],
                "num_leads": x.shape[1] if x.ndim == 2 else 1,
                "base_date": base_date,  # WFDB format date (may not be synchronized with other MIMIC-IV databases)
                "base_time": base_time,  # WFDB format time (may not be synchronized with other MIMIC-IV databases)
            },
        }


def build_npy_index(data_dir: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Discover preprocessed .npy ECG files in a directory.
    
    Looks for .npy files in structure: data_dir/p<patient_id>/p<stay_id>/s<study_id>/<study_id>.npy
    Returns a list of metadata dicts with a `base_path` key (path to .npy file without extension).
    """
    root = Path(data_dir) if data_dir is not None else Path("data/processed/ecg")
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")
    
    # Find all .npy files
    npy_files = sorted(root.rglob("*.npy"))
    records: List[Dict[str, str]] = []
    
    for npy_path in npy_files:
        # base_path is path without .npy extension (for compatibility)
        base_path = str(npy_path.with_suffix(""))
        records.append({"base_path": base_path, "npy_path": str(npy_path)})
        if limit is not None and len(records) >= limit:
            break
    
    if not records:
        raise RuntimeError(f"No .npy ECG records found in {root}")
    
    return records


class ECGNPYDataset:
    """Dataset for loading preprocessed .npy ECG files."""
    
    def __init__(
        self,
        records: List[Dict[str, str]],
        preprocess: Optional[Callable] = None,
        window_seconds: Optional[float] = None,
        timestamp_mapping: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
        data_dir: Optional[str] = None,
    ) -> None:
        """Initialize ECGNPYDataset.
        
        Args:
            records: List of record dictionaries with 'base_path' and 'npy_path'.
            preprocess: Optional preprocessing function.
            window_seconds: Optional window length in seconds.
            timestamp_mapping: Optional mapping from relative base_path to timestamps.
                             Format: {base_path: {"base_date": str, "base_time": str}}
            data_dir: Optional data directory root (for computing relative paths).
        """
        self.records = records
        self.preprocess = preprocess
        self.window_seconds = window_seconds
        self.timestamp_mapping = timestamp_mapping or {}
        self.data_dir = Path(data_dir) if data_dir else None
        # Assume preprocessed data is at 500 Hz and already windowed
        self.fs = 500.0
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int):
        import numpy as np
        
        item = self.records[idx]
        npy_path = item.get("npy_path", item["base_path"] + ".npy")
        
        # Load .npy file
        x = np.load(npy_path)
        
        # Ensure shape is (T, C) or (C, T)
        if x.ndim == 1:
            # Single lead, reshape to (T, 1)
            x = x.reshape(-1, 1)
        elif x.ndim == 2:
            # Check if (C, T) or (T, C)
            if x.shape[0] < x.shape[1]:
                # Likely (C, T), transpose to (T, C)
                x = x.T
        
        # Apply window if needed (data should already be windowed, but just in case)
        if self.window_seconds is not None:
            window_samples = int(self.window_seconds * self.fs)
            if x.shape[0] > window_samples:
                x = x[:window_samples]
            elif x.shape[0] < window_samples:
                # Pad with zeros
                pad_length = window_samples - x.shape[0]
                x = np.pad(x, ((0, pad_length), (0, 0)), mode='constant')
        
        # Apply preprocessing if provided
        if self.preprocess is not None:
            try:
                x = self.preprocess(x, self.fs)
            except TypeError:
                x = self.preprocess(x)
        
        # Try to get timestamps from mapping
        base_date = None
        base_time = None
        
        if self.timestamp_mapping:
            base_path_obj = Path(item["base_path"])
            
            # Try to get relative path from data_dir
            rel_path_str = None
            if self.data_dir:
                try:
                    rel_path = base_path_obj.relative_to(self.data_dir)
                    rel_path_str = str(rel_path).replace("\\", "/")
                except ValueError:
                    # If not relative to data_dir, try just the name
                    rel_path_str = base_path_obj.name
            
            # Also try with full base_path as key (for compatibility)
            base_path_str = str(base_path_obj).replace("\\", "/")
            
            # Look up in mapping
            if rel_path_str and rel_path_str in self.timestamp_mapping:
                timestamp_info = self.timestamp_mapping[rel_path_str]
                base_date = timestamp_info.get("base_date")
                base_time = timestamp_info.get("base_time")
            elif base_path_str in self.timestamp_mapping:
                timestamp_info = self.timestamp_mapping[base_path_str]
                base_date = timestamp_info.get("base_date")
                base_time = timestamp_info.get("base_time")
        
        return {
            "signal": x,
            "meta": {
                "base_path": item["base_path"],
                "npy_path": npy_path,
                "fs": self.fs,
                "num_samples": x.shape[0],
                "num_leads": x.shape[1] if x.ndim == 2 else 1,
                "base_date": base_date,  # From timestamp mapping if available
                "base_time": base_time,  # From timestamp mapping if available
            },
        }
