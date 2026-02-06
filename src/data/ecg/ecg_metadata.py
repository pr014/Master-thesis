"""Utilities for extracting and working with ECG metadata from MIMIC-IV-ECG.

Based on PhysioNet Usage Notes:
- ECG timestamps (base_date, base_time) are from the machine's internal clock
- May not be synchronized with other MIMIC-IV databases (Clinical, Waveform)
- Some ECGs collected outside ED/ICU may not overlap with Clinical Database timestamps
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def extract_timestamp_from_record(base_path: str) -> Dict[str, Optional[str]]:
    """Extract base_date and base_time from a WFDB record.

    Args:
        base_path: Path to ECG record (without .hea/.dat extension)

    Returns:
        Dictionary with keys: 'study', 'date', 'time'
        - study: Record name (study ID)
        - date: base_date as string (YYYY-MM-DD) or None
        - time: base_time as string (HH:MM:SS) or None
    """
    try:
        import wfdb  # type: ignore
    except ImportError as e:
        raise RuntimeError("wfdb is required. Install with: pip install wfdb") from e

    record = wfdb.rdrecord(base_path)
    study = getattr(record, 'record_name', Path(base_path).name)
    base_date = getattr(record, 'base_date', None)
    base_time = getattr(record, 'base_time', None)

    # Convert to strings if available
    date_str = None
    time_str = None

    if base_date is not None:
        if isinstance(base_date, datetime):
            date_str = base_date.strftime('%Y-%m-%d')
        elif hasattr(base_date, 'isoformat'):
            date_str = str(base_date)
        else:
            date_str = str(base_date)

    if base_time is not None:
        if isinstance(base_time, datetime):
            time_str = base_time.strftime('%H:%M:%S')
        elif hasattr(base_time, 'isoformat'):
            time_str = str(base_time)
        else:
            time_str = str(base_time)

    return {
        'study': study,
        'date': date_str,
        'time': time_str,
    }


def extract_timestamps_from_directory(
    data_dir: str,
    limit: Optional[int] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Extract timestamps from all ECG records in a directory.

    Args:
        data_dir: Directory containing ECG records (.hea/.dat pairs)
        limit: Optional limit on number of records to process
        output_csv: Optional path to save results as CSV

    Returns:
        DataFrame with columns: 'study', 'date', 'time', 'datetime'
    """
    from .ecg_loader import build_demo_index

    records = build_demo_index(data_dir=data_dir, limit=limit)

    date_times: Dict[str, List] = {
        'study': [],
        'date': [],
        'time': [],
    }

    for record in records:
        base_path = record['base_path']
        try:
            timestamp_info = extract_timestamp_from_record(base_path)
            date_times['study'].append(timestamp_info['study'])
            date_times['date'].append(timestamp_info['date'])
            date_times['time'].append(timestamp_info['time'])
        except Exception as e:
            print(f"Warning: Failed to extract timestamp from {base_path}: {e}")
            date_times['study'].append(Path(base_path).name)
            date_times['date'].append(None)
            date_times['time'].append(None)

    df = pd.DataFrame(date_times)

    # Create combined datetime column if both date and time are available
    if 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = df.apply(
            lambda row: f"{row['date']}T{row['time']}" if row['date'] and row['time'] else None,
            axis=1
        )

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved timestamps to {output_path}")

    return df


def get_wfdb_record_metadata(base_path: str) -> Dict:
    """Get full metadata from a WFDB record.

    Args:
        base_path: Path to ECG record (without .hea/.dat extension)

    Returns:
        Dictionary with record metadata including:
        - record_name, fs, n_sig, sig_len, sig_name
        - base_date, base_time (WFDB timestamps)
        - units, comments, etc.
    """
    try:
        import wfdb  # type: ignore
    except ImportError as e:
        raise RuntimeError("wfdb is required. Install with: pip install wfdb") from e

    record = wfdb.rdrecord(base_path)

    metadata = {
        'record_name': getattr(record, 'record_name', None),
        'fs': getattr(record, 'fs', None),
        'n_sig': getattr(record, 'n_sig', None),
        'sig_len': getattr(record, 'sig_len', None),
        'sig_name': getattr(record, 'sig_name', None),
        'base_date': getattr(record, 'base_date', None),
        'base_time': getattr(record, 'base_time', None),
        'units': getattr(record, 'units', None),
        'comments': getattr(record, 'comments', None),
    }

    return metadata

