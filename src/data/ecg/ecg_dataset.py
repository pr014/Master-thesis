"""PyTorch Dataset wrapper for ECG data."""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from .ecg_loader import ECGDemoDataset, build_demo_index, ECGNPYDataset


def extract_subject_id_from_path(base_path: str) -> int:
    """Extract subject_id from ECG file path (longest p{ID} segment).
    
    Args:
        base_path: Path to ECG record.
    
    Returns:
        subject_id as int.
    """
    path_parts = Path(base_path).parts
    subject_ids = []
    for part in path_parts:
        if part.startswith('p') and len(part) > 1 and part[1:].isdigit():
            subject_ids.append(part[1:])
    
    if not subject_ids:
        raise ValueError(f"Could not extract subject_id from path: {base_path}")
    
    return int(max(subject_ids, key=len))


def construct_ecg_time(base_date: Optional[str], base_time: Optional[str]) -> Optional[datetime]:
    """Construct datetime from WFDB base_date and base_time.
    
    Args:
        base_date: Date string (YYYY-MM-DD) or None.
        base_time: Time string (HH:MM:SS) or None.
    
    Returns:
        datetime object or None if date/time not available.
    """
    if base_date is None or base_time is None:
        return None
    
    try:
        # Try parsing as combined string
        dt_str = f"{base_date} {base_time}"
        return pd.to_datetime(dt_str)
    except Exception:
        return None


class ECGDataset(Dataset):
    """PyTorch Dataset for ECG signals with LOS regression labels.
    
    Wraps ECGDemoDataset and converts outputs to PyTorch tensors.
    Handles label generation via ICU stay matching.
    
    For regression: labels are continuous LOS values in days (float).
    """
    
    def __init__(
        self,
        records: List[Dict[str, str]],
        labels: Optional[Dict[str, float]] = None,
        preprocess: Optional[Callable] = None,
        window_seconds: Optional[float] = None,
        transform: Optional[Callable] = None,
        icu_mapper: Optional[Any] = None,
        mortality_labels: Optional[Dict[str, int]] = None,
        timestamp_mapping: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
        data_dir: Optional[str] = None,
        demographic_features_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize ECG Dataset.
        
        Args:
            records: List of record dictionaries with 'base_path' key.
            labels: Optional dictionary mapping base_path to label (float for regression).
                    If None and icu_mapper provided, labels will be generated.
            preprocess: Optional preprocessing function (signal, fs) -> signal.
            window_seconds: Optional window length in seconds.
            transform: Optional PyTorch transform to apply.
            icu_mapper: Optional ICUStayMapper for automatic label generation.
            mortality_labels: Optional dictionary mapping base_path to mortality_label (0/1).
                             If None and icu_mapper has mortality_mapping, labels will be generated.
            demographic_features_config: Optional config dict for demographic features (Age & Sex).
                                        Should contain 'enabled', 'records_csv_path', etc.
            config: Optional full configuration dictionary. Used to read task_type settings.
        """
        # Use ECGNPYDataset if records contain 'npy_path', otherwise use ECGDemoDataset
        if records and "npy_path" in records[0]:
            self.ecg_dataset = ECGNPYDataset(
                records=records,
                preprocess=preprocess,
                window_seconds=window_seconds,
                timestamp_mapping=timestamp_mapping,
                data_dir=data_dir,
            )
        else:
            self.ecg_dataset = ECGDemoDataset(
                records=records,
                preprocess=preprocess,
                window_seconds=window_seconds,
            )
        self.labels = labels if labels is not None else {}
        self.mortality_labels = mortality_labels if mortality_labels is not None else {}
        # If caller did not provide per-ECG mortality labels, generate them from ICUStayMapper
        self._should_generate_mortality = mortality_labels is None
        self.transform = transform
        self.icu_mapper = icu_mapper
        
        # Demographic features configuration
        self.demographic_features_config = demographic_features_config or {}
        self.demographic_features_enabled = self.demographic_features_config.get("enabled", False)
        self.demographic_mapping = {}  # base_path -> (age, gender)
        self.age_normalization_stats = {}  # For zscore: {'mean': float, 'std': float}
        
        # Load demographic features if enabled
        if self.demographic_features_enabled:
            self._load_demographic_features()
        
        # Diagnosis features configuration
        data_config = config.get("data", {}) if config else {}
        self.diagnosis_features_config = data_config.get("diagnosis_features", {})
        self.diagnosis_features_enabled = self.diagnosis_features_config.get("enabled", False)
        self.diagnosis_mapping = {}  # record_name -> binary_vector (List[int])
        self.diagnosis_list = self.diagnosis_features_config.get("diagnosis_list", [])
        self.diagnosis_dim = len(self.diagnosis_list) if self.diagnosis_list else 0
        
        # Load diagnosis features if enabled
        if self.diagnosis_features_enabled:
            self._load_diagnosis_features()
        
        # ICU unit features configuration
        data_config = config.get("data", {}) if config else {}
        self.icu_unit_features_config = data_config.get("icu_unit_features", {})
        self.icu_unit_features_enabled = self.icu_unit_features_config.get("enabled", False)
        self.icu_unit_mapping = {}  # stay_id -> icu_unit_name (str)
        self.icu_unit_list = self.icu_unit_features_config.get("icu_unit_list", [])
        self.icu_unit_encoding = self.icu_unit_features_config.get("encoding", "onehot")
        self.icu_unit_dim = len(self.icu_unit_list) if self.icu_unit_encoding == "onehot" and self.icu_unit_list else 0
        
        # Load ICU unit features if enabled
        if self.icu_unit_features_enabled:
            self._load_icu_unit_features()
        
        # Task type: "regression" (default) or "classification" (backward compatibility)
        self.task_type = data_config.get("task_type", "regression")
        
        # Statistics
        self.matched_count = 0
        self.unmatched_count = 0
        self.los_values = []  # Store LOS values for statistics (regression)
        
        # Pre-compute labels if icu_mapper provided and labels not given
        if icu_mapper is not None and not self.labels:
            self._generate_labels()
        
        # Filter out unmatched samples (label < 0) BEFORE training
        # This ensures unmatched samples never enter training/validation/testing
        self._filter_unmatched()
        
        # Log statistics
        self._log_statistics()
    
    def _filter_unmatched(self) -> None:
        """Filter out records with unmatched labels (label < 0).
        
        This ensures unmatched samples are excluded deterministically before
        training, not filtered during training loop.
        """
        if not self.labels:
            return
        
        # Get indices of matched records
        matched_indices = []
        for idx in range(len(self.ecg_dataset)):
            base_path = self.ecg_dataset.records[idx]["base_path"]
            label = self.labels.get(base_path, -1.0)
            if label >= 0:  # Only keep matched samples (valid LOS >= 0)
                matched_indices.append(idx)
        
        # Filter records and update dataset
        original_count = len(self.ecg_dataset.records)
        self.ecg_dataset.records = [self.ecg_dataset.records[i] for i in matched_indices]
        self.filtered_count = original_count - len(matched_indices)
    
    def __len__(self) -> int:
        """Return dataset size (only matched samples)."""
        return len(self.ecg_dataset)
    
    def _generate_labels(self) -> None:
        """Generate labels by matching ECGs to ICU stays.
        
        For regression: stores LOS in days (float) directly.
        Also generates mortality labels if mortality_mapping is available.
        """
        if self.icu_mapper is None:
            return
        
        for idx in range(len(self.ecg_dataset)):
            item = self.ecg_dataset[idx]
            base_path = item["meta"]["base_path"]
            base_date = item["meta"].get("base_date")
            base_time = item["meta"].get("base_time")
            
            # Extract subject_id from path
            try:
                subject_id = extract_subject_id_from_path(base_path)
            except ValueError:
                self.unmatched_count += 1
                continue
            
            # Construct ecg_time
            ecg_time = construct_ecg_time(base_date, base_time)
            
            # Fallback for .npy files without timestamps: use first ICU stay's intime
            if ecg_time is None:
                # Try to get first ICU stay for this subject and use its intime
                try:
                    import pandas as pd
                    subject_stays = self.icu_mapper.icustays_df[
                        self.icu_mapper.icustays_df['subject_id'] == subject_id
                    ]
                    if len(subject_stays) > 0:
                        # Use intime of first stay as fallback
                        first_stay = subject_stays.iloc[0]
                        ecg_time = pd.to_datetime(first_stay['intime'])
                    else:
                        self.unmatched_count += 1
                        continue
                except Exception:
                    self.unmatched_count += 1
                    continue
            
            # Map to stay
            stay_id = self.icu_mapper.map_ecg_to_stay(subject_id, ecg_time)
            if stay_id is None:
                self.unmatched_count += 1
                continue
            
            # Get LOS in days (continuous value for regression)
            los_days = self.icu_mapper.get_los(stay_id)
            if los_days is None:
                self.unmatched_count += 1
                continue
            
            # Store LOS directly as float (regression)
            self.labels[base_path] = float(los_days)
            self.los_values.append(los_days)
            self.matched_count += 1
            
            # Get mortality label if available (per ECG) – only if not provided by caller
            if self._should_generate_mortality and base_path not in self.mortality_labels:
                mortality_label = self.icu_mapper.get_mortality_label(stay_id)
                if mortality_label is not None:
                    self.mortality_labels[base_path] = int(mortality_label)
    
    def _load_demographic_features(self) -> None:
        """Load demographic features (Age & Sex) from CSV file.
        
        Creates a mapping from base_path to (age, gender) for fast lookup.
        """
        csv_path = self.demographic_features_config.get("records_csv_path")
        if not csv_path:
            print("Warning: demographic_features.enabled is True but records_csv_path not specified.")
            self.demographic_features_enabled = False
            return
        
        # Resolve path (relative to project root or absolute)
        csv_path = Path(csv_path)
        if not csv_path.is_absolute():
            # Try to find project root
            project_root = Path(__file__).parent.parent.parent.parent
            csv_path = project_root / csv_path
        
        if not csv_path.exists():
            print(f"Warning: Demographic features CSV not found at {csv_path}. Disabling demographic features.")
            self.demographic_features_enabled = False
            return
        
        try:
            # Load CSV (only needed columns to save memory)
            df = pd.read_csv(csv_path, usecols=['file_name', 'subject_id', 'gender', 'age'])
            print(f"Loaded demographic features from {csv_path}: {len(df):,} records")
        except Exception as e:
            print(f"Warning: Failed to load demographic features CSV: {e}. Disabling demographic features.")
            self.demographic_features_enabled = False
            return
        
        # Create mapping: base_path -> (age, gender)
        # Matching strategy: Try to match via file_name (normalized) or subject_id
        self.demographic_mapping = {}
        unmatched_demo = 0
        
        for _, row in df.iterrows():
            file_name = str(row['file_name']).strip()
            subject_id = int(row['subject_id']) if pd.notna(row['subject_id']) else None
            gender = str(row['gender']).strip().upper() if pd.notna(row['gender']) else None
            age = float(row['age']) if pd.notna(row['age']) else None
            
            # Normalize file_name: extract last part (study_id/record_name)
            # CSV format: "mimic-iv-ecg-.../files/p1000/p10000032/s40689238/40689238"
            # We want: "40689238" (the record name = study_id, which is unique per ECG)
            file_name_parts = file_name.replace('\\', '/').split('/')
            record_name = file_name_parts[-1] if file_name_parts else None
            
            # Store by record_name (study_id) - this is unique per ECG, so no fallback needed
            if record_name:
                self.demographic_mapping[record_name] = (age, gender, subject_id)
        
        print(f"Created demographic mapping: {len(self.demographic_mapping):,} entries")
        
        # Calculate normalization stats from available ages (for zscore)
        ages = [age for age, _, _ in self.demographic_mapping.values() if age is not None]
        if ages:
            self.age_normalization_stats = {
                'mean': float(np.mean(ages)),
                'std': float(np.std(ages)),
                'min': float(np.min(ages)),
                'max': float(np.max(ages)),
                'median': float(np.median(ages))
            }
            print(f"Age statistics: min={self.age_normalization_stats['min']:.1f}, "
                  f"max={self.age_normalization_stats['max']:.1f}, "
                  f"mean={self.age_normalization_stats['mean']:.1f}, "
                  f"median={self.age_normalization_stats['median']:.1f}")
    
    def _get_demographic_features(self, base_path: str) -> Optional[torch.Tensor]:
        """Extract and normalize demographic features for a given base_path.
        
        Args:
            base_path: ECG record base path.
        
        Returns:
            Tensor of shape (2,) for binary encoding [age, sex] or (3,) for onehot [age, sex_0, sex_1],
            or None if not found and skip strategy.
        """
        if not self.demographic_features_enabled:
            return None
        
        # Extract record name from base_path (study_id, unique per ECG)
        # base_path format: "data/icu_ecgs_24h/P1/p10000032/40689238"
        base_path_obj = Path(base_path)
        record_name = base_path_obj.name  # "40689238" (study_id, unique per ECG)
        
        # Match by record_name (study_id) - this is unique, so direct lookup
        age, gender, subject_id = None, None, None
        if record_name in self.demographic_mapping:
            age, gender, subject_id = self.demographic_mapping[record_name]
        
        # Handle missing values
        missing_age_strategy = self.demographic_features_config.get("missing_age_strategy", "median")
        missing_sex_strategy = self.demographic_features_config.get("missing_sex_strategy", "default")
        
        if age is None:
            if missing_age_strategy == "median" and self.age_normalization_stats:
                age = self.age_normalization_stats['median']
            elif missing_age_strategy == "zero":
                age = 0.0
            elif missing_age_strategy == "skip":
                return None
            else:
                age = 0.0  # Default fallback
        
        if gender is None:
            if missing_sex_strategy == "default":
                gender = "M"  # Default to most common
            elif missing_sex_strategy == "skip":
                return None
            else:
                gender = "M"  # Default fallback
        
        # Normalize age
        age_normalization = self.demographic_features_config.get("age_normalization", "minmax")
        if age_normalization == "minmax":
            age_min = self.demographic_features_config.get("age_min")
            age_max = self.demographic_features_config.get("age_max")
            if age_min is None or age_max is None:
                # Use stats from data
                if self.age_normalization_stats:
                    age_min = self.age_normalization_stats.get('min', 0)
                    age_max = self.age_normalization_stats.get('max', 120)
                else:
                    age_min = 0
                    age_max = 120
            if age_max > age_min:
                age_normalized = (age - age_min) / (age_max - age_min)
            else:
                age_normalized = 0.0
        elif age_normalization == "zscore":
            if self.age_normalization_stats:
                mean = self.age_normalization_stats['mean']
                std = self.age_normalization_stats['std']
                if std > 0:
                    age_normalized = (age - mean) / std
                else:
                    age_normalized = 0.0
            else:
                age_normalized = age / 100.0  # Fallback normalization
        else:  # "none"
            age_normalized = age
        
        # Encode sex
        sex_encoding = self.demographic_features_config.get("sex_encoding", "binary")
        if sex_encoding == "binary":
            sex_encoded = 0.0 if gender == "M" else 1.0
            return torch.tensor([age_normalized, sex_encoded], dtype=torch.float32)
        else:  # "onehot"
            if gender == "M":
                sex_encoded = torch.tensor([1.0, 0.0], dtype=torch.float32)
            else:
                sex_encoded = torch.tensor([0.0, 1.0], dtype=torch.float32)
            return torch.cat([torch.tensor([age_normalized], dtype=torch.float32), sex_encoded])
    
    def _load_diagnosis_features(self) -> None:
        """Load diagnosis features (ICD-10 codes) from CSV file.
        
        Creates a mapping from record_name to binary_vector for fast lookup.
        Binary vector indicates presence (1) or absence (0) of each diagnosis.
        
        IMPORTANT: Automatically filters diagnoses where admittime < ecg_time 
        to prevent data leakage when enabled=True.
        """
        csv_path = self.diagnosis_features_config.get("records_csv_path")
        if not csv_path:
            print("Warning: diagnosis_features.enabled is True but records_csv_path not specified.")
            self.diagnosis_features_enabled = False
            return
        
        if not self.diagnosis_list:
            print("Warning: diagnosis_features.enabled is True but diagnosis_list is empty.")
            self.diagnosis_features_enabled = False
            return
        
        # Resolve path (relative to project root or absolute)
        csv_path = Path(csv_path)
        if not csv_path.is_absolute():
            # Try to find project root
            project_root = Path(__file__).parent.parent.parent.parent
            csv_path = project_root / csv_path
        
        if not csv_path.exists():
            print(f"Warning: Diagnosis features CSV not found at {csv_path}. Disabling diagnosis features.")
            self.diagnosis_features_enabled = False
            return
        
        # Get diagnosis columns to search
        diagnosis_columns = self.diagnosis_features_config.get("diagnosis_columns", ["all_diag_all"])
        if not isinstance(diagnosis_columns, list):
            diagnosis_columns = [diagnosis_columns]
        
        # Load CSV with diagnosis columns + time/ID columns for filtering
        try:
            # Load all needed columns (including time/ID for filtering)
            needed_cols = ['file_name', 'ecg_time', 'subject_id', 'hosp_hadm_id'] + diagnosis_columns
            # Filter to only columns that exist
            df_all = pd.read_csv(csv_path, nrows=0)  # Read header only
            available_cols = [col for col in needed_cols if col in df_all.columns]
            
            if not any(col in available_cols for col in diagnosis_columns):
                print(f"Warning: None of the diagnosis columns {diagnosis_columns} found in CSV. Disabling diagnosis features.")
                self.diagnosis_features_enabled = False
                return
            
            # Check if we have the required columns for time filtering
            has_ecg_time = 'ecg_time' in available_cols
            has_subject_id = 'subject_id' in available_cols
            has_hadm_id = 'hosp_hadm_id' in available_cols
            
            if not has_ecg_time or not has_subject_id or not has_hadm_id:
                print(f"Warning: Missing required columns for time filtering (ecg_time, subject_id, hosp_hadm_id). Disabling diagnosis features.")
                print(f"  Available columns: {available_cols}")
                self.diagnosis_features_enabled = False
                return
            
            df = pd.read_csv(csv_path, usecols=available_cols, low_memory=False)
            print(f"Loaded diagnosis features from {csv_path}: {len(df):,} records")
            
            # TIME-BASED FILTERING: Match to ICU stays and filter admittime < ecg_time
            print("Applying time-based filtering: admittime < ecg_time")
            
            # Load icustays.csv to get stay_id -> hadm_id mapping
            icustays_path = csv_path.parent / "icustays.csv"
            if not icustays_path.exists():
                print(f"Warning: icustays.csv not found at {icustays_path}. Cannot apply time filtering. Disabling diagnosis features.")
                self.diagnosis_features_enabled = False
                return
            
            # Load icustays with required columns
            icustays_header = pd.read_csv(icustays_path, nrows=0)
            icustays_cols = ['stay_id', 'subject_id', 'intime', 'outtime']
            if 'hadm_id' in icustays_header.columns:
                icustays_cols.append('hadm_id')
            
            icustays_df = pd.read_csv(icustays_path, usecols=icustays_cols, low_memory=False)
            icustays_df['intime'] = pd.to_datetime(icustays_df['intime'], utc=True, errors='coerce')
            icustays_df['outtime'] = pd.to_datetime(icustays_df['outtime'], utc=True, errors='coerce')
            icustays_df = icustays_df.dropna(subset=['intime', 'outtime'])
            print(f"Loaded ICU stays data: {len(icustays_df):,} records")
            
            # Convert ecg_time to datetime
            df['ecg_time'] = pd.to_datetime(df['ecg_time'], utc=True, errors='coerce')
            
            # Drop rows with invalid timestamps
            before_time_clean = len(df)
            df = df.dropna(subset=['ecg_time']).copy()
            after_time_clean = len(df)
            if before_time_clean != after_time_clean:
                print(f"Removed {before_time_clean - after_time_clean:,} records with invalid ecg_time")
            
            if len(df) == 0:
                print("Warning: No records with valid ecg_time. Disabling diagnosis features.")
                self.diagnosis_features_enabled = False
                return
            
            # Match ECGs to ICU stays: subject_id + ecg_time within intime/outtime
            print("Matching ECGs to ICU stays...")
            df_with_stays = df.merge(
                icustays_df[['stay_id', 'subject_id', 'intime', 'outtime'] + (['hadm_id'] if 'hadm_id' in icustays_df.columns else [])],
                on='subject_id',
                how='inner'
            )
            
            # Filter: ecg_time must be within intime/outtime window
            time_mask = (df_with_stays['ecg_time'] >= df_with_stays['intime']) & (df_with_stays['ecg_time'] <= df_with_stays['outtime'])
            df_matched = df_with_stays[time_mask].copy()
            
            # Handle multiple matches: if one ECG matches multiple stays, pick the closest to intime
            if len(df_matched) > 0:
                df_matched['time_diff'] = (df_matched['ecg_time'] - df_matched['intime']).dt.total_seconds()
                df_matched['rank'] = df_matched.groupby('file_name')['time_diff'].rank(method='min')
                df_matched = df_matched[df_matched['rank'] == 1].drop(columns=['time_diff', 'rank'])
            
            print(f"Matched {len(df_matched):,} ECGs to ICU stays (from {len(df):,} total)")
            
            if len(df_matched) == 0:
                print("Warning: No ECGs matched to ICU stays. Disabling diagnosis features.")
                self.diagnosis_features_enabled = False
                return
            
            # Get hadm_id from icustays (if not already present)
            if 'hadm_id' not in df_matched.columns:
                if 'hadm_id' in icustays_df.columns:
                    df_matched = df_matched.merge(
                        icustays_df[['stay_id', 'hadm_id']],
                        on='stay_id',
                        how='left'
                    )
                else:
                    print("Warning: hadm_id not available in icustays.csv. Cannot apply time filtering. Disabling diagnosis features.")
                    self.diagnosis_features_enabled = False
                    return
            
            # Check for missing hadm_id after merge
            if df_matched['hadm_id'].isna().any():
                missing_count = df_matched['hadm_id'].isna().sum()
                print(f"Warning: {missing_count:,} ECGs have missing hadm_id. Removing them.")
                df_matched = df_matched.dropna(subset=['hadm_id'])
            
            if len(df_matched) == 0:
                print("Warning: No ECGs with valid hadm_id. Disabling diagnosis features.")
                self.diagnosis_features_enabled = False
                return
            
            # Load admissions.csv to get admittime
            admissions_path = csv_path.parent / "admissions.csv"
            if not admissions_path.exists():
                print(f"Warning: admissions.csv not found at {admissions_path}. Cannot apply time filtering. Disabling diagnosis features.")
                self.diagnosis_features_enabled = False
                return
            
            admissions_df = pd.read_csv(
                admissions_path,
                usecols=['subject_id', 'hadm_id', 'admittime'],
                low_memory=False
            )
            admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'], utc=True, errors='coerce')
            admissions_df = admissions_df.dropna(subset=['admittime'])
            print(f"Loaded admissions data: {len(admissions_df):,} records")
            
            # Merge with admissions to get admittime
            df_merged = df_matched.merge(
                admissions_df,
                on=['subject_id', 'hadm_id'],
                how='inner'
            )
            
            # Filter: only keep records where admittime < ecg_time
            before_filter = len(df_merged)
            df_filtered = df_merged[df_merged['admittime'] < df_merged['ecg_time']].copy()
            after_filter = len(df_filtered)
            
            print(f"Time filtering: {before_filter:,} records -> {after_filter:,} records ({after_filter/before_filter*100:.1f}% kept)")
            
            if len(df_filtered) == 0:
                print("Warning: No records remain after time filtering. Disabling diagnosis features.")
                self.diagnosis_features_enabled = False
                return
            
            # Remove duplicate ECGs (keep first if any remain after filtering)
            duplicate_ecgs = df_filtered.groupby('file_name').size()
            if (duplicate_ecgs > 1).any():
                num_duplicates = (duplicate_ecgs > 1).sum()
                print(f"Warning: Found {num_duplicates:,} ECGs with multiple matches after filtering. Keeping first occurrence.")
                df_filtered = df_filtered.drop_duplicates(subset=['file_name'], keep='first')
            
            df = df_filtered
            
        except Exception as e:
            print(f"Warning: Failed to load diagnosis features CSV: {e}. Disabling diagnosis features.")
            import traceback
            traceback.print_exc()
            self.diagnosis_features_enabled = False
            return
        
        # Create mapping: record_name -> binary_vector
        self.diagnosis_mapping = {}
        import ast
        
        for _, row in df.iterrows():
            file_name = str(row['file_name']).strip()
            
            # Normalize file_name: extract last part (study_id/record_name)
            # CSV format: "mimic-iv-ecg-.../files/p1000/p10000032/s40689238/40689238"
            # We want: "40689238" (the record name = study_id, which is unique per ECG)
            file_name_parts = file_name.replace('\\', '/').split('/')
            record_name = file_name_parts[-1] if file_name_parts else None
            
            if not record_name:
                continue
            
            # Parse diagnoses from diagnosis columns (search in order)
            found_diagnoses = set()
            for col in diagnosis_columns:
                if col not in row or pd.isna(row[col]):
                    continue
                
                diag_str = str(row[col]).strip()
                if not diag_str or diag_str == '[]' or diag_str == 'nan':
                    continue
                
                diagnoses = []
                
                # Try to parse as Python list string first (most common format)
                diag_str_clean = diag_str.strip()
                if diag_str_clean.startswith('[') and diag_str_clean.endswith(']'):
                    try:
                        parsed_list = ast.literal_eval(diag_str_clean)
                        if isinstance(parsed_list, list):
                            diagnoses = [str(d).strip().strip("'\"") for d in parsed_list if d]
                    except:
                        pass
                
                # If not a list, try semicolon-separated
                if not diagnoses:
                    diagnoses = [d.strip().strip("'\"") for d in diag_str.split(';') if d.strip()]
                
                # Clean and add to found_diagnoses
                for diag in diagnoses:
                    diag = diag.strip().strip("'\"[]")
                    if diag and diag != 'nan' and diag != '':
                        # Extract ICD-10 code (first part before space, if any)
                        icd10_code = diag.split()[0] if ' ' in diag else diag
                        icd10_code = icd10_code.strip("'\"")
                        if icd10_code:
                            found_diagnoses.add(icd10_code)
            
            # Create binary vector: 1 if diagnosis in found_diagnoses, else 0
            binary_vector = [1 if diag_code in found_diagnoses else 0 for diag_code in self.diagnosis_list]
            self.diagnosis_mapping[record_name] = binary_vector
        
        print(f"Created diagnosis mapping: {len(self.diagnosis_mapping):,} entries")
        print(f"Diagnosis features dimension: {self.diagnosis_dim} (diagnoses: {', '.join(self.diagnosis_list)})")
    
    def _get_diagnosis_features(self, base_path: str) -> Optional[torch.Tensor]:
        """Extract diagnosis features for a given base_path.
        
        Args:
            base_path: ECG record base path.
        
        Returns:
            Tensor of shape (diagnosis_dim,) with binary values [0 or 1] for each diagnosis,
            or None if not found and skip strategy.
        """
        if not self.diagnosis_features_enabled:
            return None
        
        # Extract record name from base_path (study_id, unique per ECG)
        # base_path format: "data/icu_ecgs_24h/P1/p10000032/40689238"
        base_path_obj = Path(base_path)
        record_name = base_path_obj.name  # "40689238" (study_id, unique per ECG)
        
        # Match by record_name (study_id) - this is unique, so direct lookup
        binary_vector = None
        if record_name in self.diagnosis_mapping:
            binary_vector = self.diagnosis_mapping[record_name]
        
        # Handle missing values
        missing_strategy = self.diagnosis_features_config.get("missing_strategy", "zero")
        
        if binary_vector is None:
            if missing_strategy == "zero":
                binary_vector = [0] * self.diagnosis_dim
            elif missing_strategy == "skip":
                return None
            else:
                binary_vector = [0] * self.diagnosis_dim  # Default fallback
        
        # Convert to tensor
        return torch.tensor(binary_vector, dtype=torch.float32)
    
    def _load_icu_unit_features(self) -> None:
        """Load ICU unit features from icustays.csv.
        
        Creates a mapping from stay_id to ICU unit name (first_careunit).
        Uses icu_mapper.icustays_df if available, otherwise loads icustays.csv directly.
        """
        if self.icu_mapper is None:
            print("Warning: icu_mapper not available. Cannot load ICU unit features. Disabling ICU unit features.")
            self.icu_unit_features_enabled = False
            return
        
        try:
            # Get icustays DataFrame from icu_mapper
            icustays_df = self.icu_mapper.icustays_df
            
            # Check if first_careunit column exists
            if 'first_careunit' not in icustays_df.columns:
                print("Warning: first_careunit column not found in icustays.csv. Disabling ICU unit features.")
                self.icu_unit_features_enabled = False
                return
            
            # Create mapping: stay_id -> first_careunit
            self.icu_unit_mapping = {}
            for _, row in icustays_df.iterrows():
                stay_id = int(row['stay_id'])
                icu_unit = str(row['first_careunit']).strip() if pd.notna(row['first_careunit']) else None
                if icu_unit:
                    self.icu_unit_mapping[stay_id] = icu_unit
            
            # Calculate statistics
            total_stays = len(icustays_df)
            mapped_stays = len(self.icu_unit_mapping)
            unit_counts = {}
            for unit in self.icu_unit_mapping.values():
                unit_counts[unit] = unit_counts.get(unit, 0) + 1
            
            # Find most common unit for missing strategy
            if unit_counts:
                most_common_unit = max(unit_counts.items(), key=lambda x: x[1])[0]
            else:
                most_common_unit = None
            
            self._most_common_icu_unit = most_common_unit
            
            print(f"Loaded ICU unit features: {mapped_stays:,} stays mapped (from {total_stays:,} total)")
            if unit_counts:
                print(f"  Most common ICU unit: {most_common_unit} ({unit_counts[most_common_unit]:,} stays)")
            print(f"  ICU unit list: {len(self.icu_unit_list)} units")
            print(f"  Encoding: {self.icu_unit_encoding}")
            print(f"  Feature dimension: {self.icu_unit_dim}")
            
        except Exception as e:
            print(f"Warning: Failed to load ICU unit features: {e}. Disabling ICU unit features.")
            import traceback
            traceback.print_exc()
            self.icu_unit_features_enabled = False
            return
    
    def _get_icu_unit_features(self, stay_id: Optional[int]) -> Optional[torch.Tensor]:
        """Extract ICU unit features for a given stay_id.
        
        Args:
            stay_id: ICU stay ID, or None if not available.
        
        Returns:
            Tensor of shape (icu_unit_dim,) with one-hot encoded ICU unit features,
            or None if not found and skip strategy.
        """
        if not self.icu_unit_features_enabled:
            return None
        
        if stay_id is None:
            # Handle missing stay_id
            missing_strategy = self.icu_unit_features_config.get("missing_strategy", "most_common")
            if missing_strategy == "skip":
                return None
            elif missing_strategy == "most_common":
                # Use most common ICU unit
                icu_unit = getattr(self, '_most_common_icu_unit', None)
                if icu_unit is None:
                    return None
            elif missing_strategy == "zero":
                # Return all zeros
                return torch.zeros(self.icu_unit_dim, dtype=torch.float32)
            else:
                return None
        
        # Get ICU unit from mapping
        icu_unit = self.icu_unit_mapping.get(stay_id)
        
        # Handle missing ICU unit
        if icu_unit is None:
            missing_strategy = self.icu_unit_features_config.get("missing_strategy", "most_common")
            if missing_strategy == "skip":
                return None
            elif missing_strategy == "most_common":
                icu_unit = getattr(self, '_most_common_icu_unit', None)
                if icu_unit is None:
                    return None
            elif missing_strategy == "zero":
                return torch.zeros(self.icu_unit_dim, dtype=torch.float32)
            else:
                return None
        
        # Handle unknown ICU units (not in the list)
        unknown_strategy = self.icu_unit_features_config.get("unknown_strategy", "other")
        if icu_unit not in self.icu_unit_list:
            if unknown_strategy == "skip":
                return None
            elif unknown_strategy == "most_common":
                icu_unit = getattr(self, '_most_common_icu_unit', None)
                if icu_unit is None or icu_unit not in self.icu_unit_list:
                    return None
            elif unknown_strategy == "other":
                # Return all zeros (treat as "Other" category)
                # Note: If we want to add an "Other" category, we'd need to add it to icu_unit_list
                return torch.zeros(self.icu_unit_dim, dtype=torch.float32)
            else:
                return None
        
        # Create one-hot encoding
        if self.icu_unit_encoding == "onehot":
            onehot_vector = [1 if unit == icu_unit else 0 for unit in self.icu_unit_list]
            return torch.tensor(onehot_vector, dtype=torch.float32)
        elif self.icu_unit_encoding == "embedding":
            # For embedding, we return the index (will be handled by model)
            # But for now, we'll use one-hot as placeholder
            # The model should handle embedding conversion
            idx = self.icu_unit_list.index(icu_unit)
            return torch.tensor([idx], dtype=torch.long)
        else:
            # Unknown encoding, return None
            return None
    
    def _log_statistics(self) -> None:
        """Log dataset statistics after filtering."""
        total_before_filter = self.matched_count + self.unmatched_count
        total_after_filter = len(self.ecg_dataset)
        filtered_out = getattr(self, 'filtered_count', 0)
        
        if total_before_filter > 0:
            print(f"ECGDataset Statistics:")
            print(f"  Task type: {self.task_type}")
            print(f"  Total samples: {total_before_filter}")
            print(f"  Matched samples kept: {total_after_filter}")
            print(f"  Unmatched samples dropped: {filtered_out}")
            
            # LOS statistics for regression
            if self.los_values:
                los_array = np.array(self.los_values)
                print(f"  LOS statistics (days):")
                print(f"    Mean: {los_array.mean():.2f}")
                print(f"    Median: {np.median(los_array):.2f}")
                print(f"    Std: {los_array.std():.2f}")
                print(f"    Min: {los_array.min():.2f}")
                print(f"    Max: {los_array.max():.2f}")
            
            if self.demographic_features_enabled:
                print(f"  Demographic features: Enabled ({len(self.demographic_mapping):,} mappings)")
            if self.diagnosis_features_enabled:
                print(f"  Diagnosis features: Enabled ({len(self.diagnosis_mapping):,} mappings, {self.diagnosis_dim} diagnoses)")
            if self.icu_unit_features_enabled:
                print(f"  ICU unit features: Enabled ({len(self.icu_unit_mapping):,} mappings, {self.icu_unit_dim} units)")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset.
        
        Args:
            idx: Index of item.
        
        Returns:
            Dictionary with:
                - 'signal': ECG signal tensor (C, T) = (12, 5000)
                - 'label': Label tensor (scalar) - LOS in days (float32) for regression, or -1 if unmatched
                - 'mortality_label': Mortality label tensor (scalar) - 0 or 1, or -1 if not available
                - 'meta': Metadata dictionary (includes subject_id, stay_id, los_days if matched)
        """
        # Get data from ECGDemoDataset
        item = self.ecg_dataset[idx]
        signal = item["signal"]  # numpy array (T, C) or (C, T)
        meta = item["meta"].copy()
        base_path = meta["base_path"]
        
        # Convert to tensor
        signal = torch.from_numpy(signal).float()
        
        # Check for NaN/Inf in loaded data
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            # Replace NaN/Inf with zeros (fallback)
            signal = torch.where(torch.isnan(signal) | torch.isinf(signal), torch.tensor(0.0), signal)
        
        # Ensure shape (C, T) = (12, 5000) for CNN
        if signal.dim() == 2:
            if signal.shape[0] > signal.shape[1]:
                # (T, C) -> (C, T)
                signal = signal.transpose(0, 1)
        
        # Get label (LOS in days for regression)
        label = self.labels.get(base_path, -1.0)
        label = torch.tensor(label, dtype=torch.float32)  # float32 for regression
        
        # Get mortality label
        mortality_label = self.mortality_labels.get(base_path, -1)
        mortality_label = torch.tensor(mortality_label, dtype=torch.long)
        
        # Add metadata for debugging
        if label >= 0 and self.icu_mapper is not None:
            try:
                subject_id = extract_subject_id_from_path(base_path)
                ecg_time = construct_ecg_time(meta.get("base_date"), meta.get("base_time"))
                
                # Fallback for .npy files without timestamps: use first ICU stay's intime
                if ecg_time is None:
                    import pandas as pd
                    subject_stays = self.icu_mapper.icustays_df[
                        self.icu_mapper.icustays_df['subject_id'] == subject_id
                    ]
                    if len(subject_stays) > 0:
                        # Use intime of first stay as fallback
                        first_stay = subject_stays.iloc[0]
                        ecg_time = pd.to_datetime(first_stay['intime'])
                
                if ecg_time is not None:
                    stay_id = self.icu_mapper.map_ecg_to_stay(subject_id, ecg_time)
                    if stay_id is not None:
                        los_days = self.icu_mapper.get_los(stay_id)
                        meta["subject_id"] = subject_id
                        meta["stay_id"] = stay_id
                        meta["los_days"] = los_days
            except Exception:
                pass  # Skip if metadata extraction fails
        
        # Apply transform if provided
        # Note: Augmentation transforms check self.training internally
        if self.transform is not None:
            # Ensure transform is in training mode (for augmentation)
            if hasattr(self.transform, 'train'):
                self.transform.train()
            signal = self.transform(signal)
        
        # Remove None values from meta (PyTorch DataLoader can't collate None)
        meta_clean = {k: v for k, v in meta.items() if v is not None}
        
        # Get demographic features if enabled
        demographic_features = None
        if self.demographic_features_enabled:
            demographic_features = self._get_demographic_features(base_path)
        
        # Get diagnosis features if enabled
        diagnosis_features = None
        if self.diagnosis_features_enabled:
            diagnosis_features = self._get_diagnosis_features(base_path)
        
        # Get ICU unit features if enabled
        icu_unit_features = None
        if self.icu_unit_features_enabled:
            # Extract stay_id from meta (set earlier in this method if label >= 0)
            stay_id = meta_clean.get("stay_id")
            icu_unit_features = self._get_icu_unit_features(stay_id)
        
        result = {
            "signal": signal,  # (C, T) = (12, 5000)
            "label": label,  # scalar, LOS in days (float32) for regression, or -1
            "mortality_label": mortality_label,  # scalar, 0 or 1, or -1 if not available
            "meta": meta_clean,
        }
        
        # Add demographic features if available
        if demographic_features is not None:
            result["demographic_features"] = demographic_features
        
        # Add diagnosis features if available
        if diagnosis_features is not None:
            result["diagnosis_features"] = diagnosis_features
        
        # Add ICU unit features if available
        if icu_unit_features is not None:
            result["icu_unit_features"] = icu_unit_features
        
        return result

