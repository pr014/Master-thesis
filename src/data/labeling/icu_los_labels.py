"""ICU Length of Stay (LOS) label mapping utilities.

Provides functions to:
1. Load icustays.csv
2. Map ECG records to ICU stays via time matching
3. Convert LOS (days) to bin classes (0-9)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np


def load_icustays(icustays_path: str) -> pd.DataFrame:
    """Load icustays.csv into a DataFrame.
    
    Args:
        icustays_path: Path to icustays.csv file.
    
    Returns:
        DataFrame with columns: subject_id, hadm_id, stay_id, intime, outtime, los
        - subject_id: int
        - hadm_id: int (hospital admission ID, for merging with admissions.csv)
        - stay_id: int
        - intime: datetime
        - outtime: datetime
        - los: float (length of stay in days)
    """
    icustays_path = Path(icustays_path)
    if not icustays_path.exists():
        raise FileNotFoundError(f"ICU stays file not found: {icustays_path}")
    
    # Load CSV
    df = pd.read_csv(icustays_path)
    
    # Required columns
    required_cols = {'subject_id', 'stay_id', 'intime', 'outtime', 'los'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in icustays.csv: {missing}")
    
    # hadm_id is optional but recommended for mortality analysis
    has_hadm_id = 'hadm_id' in df.columns
    
    # Convert types
    df['subject_id'] = df['subject_id'].astype(int)
    df['stay_id'] = df['stay_id'].astype(int)
    if has_hadm_id:
        df['hadm_id'] = df['hadm_id'].astype(int)
    df['intime'] = pd.to_datetime(df['intime'])
    df['outtime'] = pd.to_datetime(df['outtime'])
    df['los'] = df['los'].astype(float)
    
    # Select columns (include hadm_id if available)
    if has_hadm_id:
        df = df[['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los']].copy()
    else:
        df = df[['subject_id', 'stay_id', 'intime', 'outtime', 'los']].copy()
    
    # Sort by subject_id and intime for efficient lookup
    df = df.sort_values(['subject_id', 'intime']).reset_index(drop=True)
    
    return df


class ICUStayMapper:
    """Efficient mapper for ECG to ICU stay matching."""
    
    def __init__(
        self,
        icustays_df: pd.DataFrame,
        mortality_mapping: Optional[Dict[int, int]] = None,
    ):
        """Initialize mapper with ICU stays DataFrame.
        
        Args:
            icustays_df: DataFrame from load_icustays().
            mortality_mapping: Optional dictionary mapping stay_id -> mortality_label (0/1).
                             If provided, enables get_mortality_label() method.
        """
        self.icustays_df = icustays_df.copy()
        
        # Build lookup structure: grouped by subject_id
        # Each group is sorted by intime
        self._lookup: Dict[int, pd.DataFrame] = {}
        for subject_id, group in self.icustays_df.groupby('subject_id'):
            self._lookup[subject_id] = group.reset_index(drop=True)
        
        # Build stay_id -> los mapping for quick lookup
        self._stay_to_los: Dict[int, float] = dict(
            zip(self.icustays_df['stay_id'], self.icustays_df['los'])
        )
        
        # Store mortality mapping if provided
        self._mortality_mapping = mortality_mapping
    
    def map_ecg_to_stay(
        self,
        subject_id: int,
        ecg_time: datetime,
    ) -> Optional[int]:
        """Map ECG record to ICU stay via time matching.
        
        For ECG with (subject_id, ecg_time), find the ICU stay row with same
        subject_id where intime <= ecg_time <= outtime.
        
        If multiple matches (rare), pick the one with smallest |ecg_time - intime|.
        If no match, return None.
        
        Args:
            subject_id: Patient subject ID.
            ecg_time: ECG timestamp (datetime).
        
        Returns:
            stay_id if match found, None otherwise.
        """
        # Check if subject_id exists
        if subject_id not in self._lookup:
            return None
        
        # Get all stays for this subject
        subject_stays = self._lookup[subject_id]
        
        # Find matching stays: intime <= ecg_time <= outtime
        matches = subject_stays[
            (subject_stays['intime'] <= ecg_time) & 
            (ecg_time <= subject_stays['outtime'])
        ]
        
        if len(matches) == 0:
            return None
        
        if len(matches) == 1:
            return int(matches.iloc[0]['stay_id'])
        
        # Multiple matches: pick closest to intime
        matches = matches.copy()
        matches['time_diff'] = (matches['intime'] - ecg_time).abs()
        best_match = matches.loc[matches['time_diff'].idxmin()]
        return int(best_match['stay_id'])
    
    def get_los(self, stay_id: int) -> Optional[float]:
        """Get LOS (days) for a given stay_id.
        
        Args:
            stay_id: ICU stay ID.
        
        Returns:
            LOS in days, or None if stay_id not found.
        """
        return self._stay_to_los.get(stay_id)
    
    def get_mortality_label(self, stay_id: int) -> Optional[int]:
        """Get mortality label (0/1) for a given stay_id.
        
        Args:
            stay_id: ICU stay ID.
        
        Returns:
            Mortality label: 0 = survived, 1 = died, or None if:
            - stay_id not found, or
            - mortality_mapping was not provided during initialization.
        """
        if self._mortality_mapping is None:
            return None
        return self._mortality_mapping.get(stay_id)


def los_to_bin(los_days: float, binning_strategy: str = "intervals", max_days: int = 9) -> int:
    """Convert LOS (days) to bin class.
    
    Supports two binning strategies:
    1. "intervals": [0,1), [1,2), [2,3), ..., [9, +inf) => 10 classes (old)
    2. "exact_days": 1 day [0,1), 2 days [1,2), ..., max_days days [max_days-1, max_days), >= max_days+1 [max_days, +inf) => max_days+1 classes (new)
    
    Args:
        los_days: Length of stay in days (float).
        binning_strategy: "intervals" or "exact_days" (default: "intervals")
        max_days: Maximum number of exact day classes for "exact_days" strategy (default: 9)
    
    Returns:
        Class index (range depends on strategy).
    """
    if los_days < 0:
        return 0  # Invalid, default to bin 0
    
    if binning_strategy == "intervals":
        # Old strategy: [0,1), [1,2), ..., [9, +inf) => 10 classes
        if los_days >= 9:
            return 9  # [9, +inf)
        return int(np.floor(los_days))  # [0,1), [1,2), ..., [8,9)
    
    elif binning_strategy == "exact_days":
        # New strategy: exactly 1 day, 2 days, ..., max_days days, >= max_days+1
        # Class boundaries: [0, 1), [1, 2), [2, 3), ..., [max_days, +inf)
        # Class 0 = 1 day (0-24h), Class 1 = 2 days (24-48h), etc.
        if los_days < 0:
            return 0  # Invalid, default to bin 0
        
        # Floor to get the day interval
        day_interval = int(np.floor(los_days))
        
        if day_interval > max_days:
            return max_days  # >= max_days+1 days -> last class
        else:
            return day_interval  # Class 0 = [0,1) = 1 day, Class 1 = [1,2) = 2 days, etc.
    
    else:
        raise ValueError(f"Unknown binning_strategy: {binning_strategy}. Must be 'intervals' or 'exact_days'")


def load_mortality_mapping(
    admissions_path: str,
    icustays_df: pd.DataFrame,
) -> Dict[int, int]:
    """Create mapping from stay_id to mortality_label (0/1).
    
    Logic: died_in_icu = (deathtime is not None and pd.notna(deathtime) 
                          and intime <= deathtime <= outtime)
    
    Args:
        admissions_path: Path to admissions.csv file.
        icustays_df: DataFrame from load_icustays() (must contain hadm_id).
    
    Returns:
        Dictionary mapping stay_id -> mortality_label (0 = survived, 1 = died).
    """
    admissions_path = Path(admissions_path)
    if not admissions_path.exists():
        raise FileNotFoundError(f"Admissions file not found: {admissions_path}")
    
    # Check if hadm_id is available in icustays_df
    if 'hadm_id' not in icustays_df.columns:
        raise ValueError(
            "icustays_df must contain 'hadm_id' column for mortality mapping. "
            "Please reload icustays_df using load_icustays() with a CSV that includes hadm_id."
        )
    
    # Load admissions.csv
    admissions_df = pd.read_csv(admissions_path)
    
    # Check required columns
    if 'hadm_id' not in admissions_df.columns:
        raise ValueError("admissions.csv must contain 'hadm_id' column")
    if 'deathtime' not in admissions_df.columns:
        raise ValueError("admissions.csv must contain 'deathtime' column")
    
    # Prepare admissions data
    admissions_with_deathtime = admissions_df[['hadm_id', 'deathtime']].copy()
    admissions_with_deathtime['deathtime'] = pd.to_datetime(
        admissions_with_deathtime['deathtime'], errors='coerce'
    )
    
    # Merge with icustays to get stay_id -> hadm_id -> deathtime
    icustays_with_death = icustays_df.merge(
        admissions_with_deathtime,
        on='hadm_id',
        how='left'
    )
    
    # Calculate mortality label: died_in_icu = (deathtime notna) and (intime <= deathtime <= outtime)
    icustays_with_death['died_in_icu'] = (
        icustays_with_death['deathtime'].notna() &
        (icustays_with_death['deathtime'] >= icustays_with_death['intime']) &
        (icustays_with_death['deathtime'] <= icustays_with_death['outtime'])
    )
    
    # Create mapping: stay_id -> mortality_label (0 or 1)
    mortality_mapping = dict(
        zip(
            icustays_with_death['stay_id'],
            icustays_with_death['died_in_icu'].astype(int)
        )
    )
    
    return mortality_mapping


def map_ecg_to_stay(
    subject_id: int,
    ecg_time: datetime,
    mapper: ICUStayMapper,
) -> Optional[int]:
    """Convenience function to map ECG to stay.
    
    Args:
        subject_id: Patient subject ID.
        ecg_time: ECG timestamp (datetime).
        mapper: ICUStayMapper instance.
    
    Returns:
        stay_id if match found, None otherwise.
    """
    return mapper.map_ecg_to_stay(subject_id, ecg_time)


def get_num_classes_from_config(config: Dict[str, Any]) -> Optional[int]:
    """Get number of classes from config.
    
    For regression tasks, returns None (no classes).
    For classification tasks, returns the number of bins.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Number of classes based on binning strategy, or None for regression.
    """
    data_config = config.get("data", {})
    
    # Check task type - regression doesn't use classes
    task_type = data_config.get("task_type", "regression")
    if task_type == "regression":
        return None  # No classes for regression
    
    # Classification: compute number of classes
    los_binning_config = data_config.get("los_binning", {})
    strategy = los_binning_config.get("strategy", "intervals")
    max_days = los_binning_config.get("max_days", 9)
    
    if strategy == "intervals":
        return 10  # [0,1), [1,2), ..., [9, +inf)
    elif strategy == "exact_days":
        return max_days + 1  # 1 day, 2 days, ..., max_days days, >= max_days+1
    else:
        raise ValueError(f"Unknown binning_strategy: {strategy}")


def is_regression_task(config: Dict[str, Any]) -> bool:
    """Check if the task is a regression task.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if task_type is "regression", False otherwise.
    """
    data_config = config.get("data", {})
    task_type = data_config.get("task_type", "regression")
    return task_type == "regression"


def get_class_labels_from_config(config: Dict[str, Any]) -> List[str]:
    """Get class labels from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of class labels (e.g., ["1 day", "2 days", ...] or ["[0,1)", "[1,2)", ...])
    """
    data_config = config.get("data", {})
    los_binning_config = data_config.get("los_binning", {})
    strategy = los_binning_config.get("strategy", "intervals")
    max_days = los_binning_config.get("max_days", 9)
    
    if strategy == "intervals":
        labels = [f"[{i},{i+1})" for i in range(9)]
        labels.append("[9, +inf)")
        return labels
    elif strategy == "exact_days":
        labels = [f"{i} day" if i == 1 else f"{i} days" for i in range(1, max_days + 1)]
        labels.append(f">={max_days+1} days")
        return labels
    else:
        raise ValueError(f"Unknown binning_strategy: {strategy}")

