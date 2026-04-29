"""PyTorch Dataset wrapper for ECG data."""

from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime
from pathlib import Path
import hashlib
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from .ecg_loader import ECGDemoDataset, build_demo_index, ECGNPYDataset
from ..labeling.icu_los_labels import los_to_bin


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


def _project_root() -> Path:
    """Repository root (parent of ``src``)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def tabular_join_path_key(base_path: str, data_dir: Optional[str]) -> str:
    """Normalize ECG path for joining tabular files (e.g. SOFA CSV) to ``data_dir``."""
    if not base_path:
        return ""
    raw = Path(str(base_path).replace("\\", "/"))
    data_dir_path: Optional[Path] = None
    if data_dir:
        dp = Path(str(data_dir).replace("\\", "/"))
        if not dp.is_absolute():
            dp = _project_root() / dp
        try:
            data_dir_path = dp.resolve()
        except OSError:
            data_dir_path = dp
    try:
        if data_dir_path is not None:
            if raw.is_absolute():
                resolved = raw.resolve()
            else:
                resolved = (data_dir_path / raw).resolve()
            rel = resolved.relative_to(data_dir_path)
            return rel.as_posix()
    except (ValueError, OSError):
        pass
    return raw.as_posix()


# --- EHR late-fusion features from EHR_feature_data.csv ---

DEFAULT_EHR_METADATA_COLUMNS = [
    "base_path",
    "study_id",
    "subject_id",
    "ecg_time",
    "stay_id",
    "stay_intime",
    "hadm_id",
    "t_cut",
]


def _resolve_ehr_csv_path(cfg: Dict[str, Any], project_root: Path) -> Path:
    csv_path = cfg.get("csv_path", "data/labeling/labels_csv/EHR_feature_data.csv")
    path = Path(csv_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def get_ehr_feature_columns(
    cfg: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> List[str]:
    """Resolve feature columns from the full EHR CSV, excluding metadata columns."""
    if not cfg.get("enabled", False):
        return []
    root = project_root or _project_root()
    path = _resolve_ehr_csv_path(cfg, root)
    if not path.exists():
        raise FileNotFoundError(f"EHR feature CSV not found: {path}")

    df = pd.read_csv(path, nrows=0)
    if "base_path" not in df.columns:
        raise ValueError("EHR CSV must contain base_path")

    include_columns = cfg.get("include_columns")
    if include_columns:
        columns = [str(col) for col in include_columns]
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"EHR CSV missing requested include_columns: {missing}")
        return columns

    excluded = set(DEFAULT_EHR_METADATA_COLUMNS)
    excluded.update(str(col) for col in cfg.get("exclude_columns", []))
    return [str(col) for col in df.columns if str(col) not in excluded]


def ehr_window_feature_dim(cfg: Dict[str, Any]) -> int:
    """Length of the late-fusion vector for ``data.ehr_window_features``."""
    return len(get_ehr_feature_columns(cfg))


def _ehr_safe_float(x: Any) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


def raw_ehr_vector_from_row(row: pd.Series, feature_columns: List[str]) -> np.ndarray:
    """Raw numeric vector (no z-score) from one full EHR CSV row."""
    parts = [_ehr_safe_float(row.get(col)) for col in feature_columns]
    return np.asarray(parts, dtype=np.float32)


def compute_ehr_zscore_stats(
    train_records: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    project_root: Path,
    data_dir: Optional[str],
) -> Tuple[List[float], List[float]]:
    """Column-wise mean/std on train rows present in the EHR CSV (train-only normalization)."""
    path = _resolve_ehr_csv_path(cfg, project_root)
    if not path.exists():
        raise FileNotFoundError(f"EHR feature CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    if "base_path" not in df.columns:
        raise ValueError("EHR CSV must contain base_path")
    feature_columns = get_ehr_feature_columns(cfg, project_root)

    train_keys = {
        tabular_join_path_key(str(r.get("base_path", "")), data_dir)
        for r in train_records
    }
    train_keys.discard("")

    mats: List[np.ndarray] = []
    for _, row in df.iterrows():
        bp = row.get("base_path")
        if pd.isna(bp):
            continue
        key = tabular_join_path_key(str(bp), data_dir)
        if key not in train_keys:
            continue
        mats.append(raw_ehr_vector_from_row(row, feature_columns))

    dim = len(feature_columns)
    if not mats:
        return [0.0] * dim, [1.0] * dim

    mat = np.stack(mats, axis=0)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float64).tolist(), std.astype(np.float64).tolist()


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
        regression_weights: Optional[Dict[int, float]] = None,
        los_binning: Optional[Dict[str, Any]] = None,
        ehr_window_norm: Optional[Tuple[List[float], List[float]]] = None,
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
            regression_weights: Optional dict mapping LOS bin_idx -> weight for sample weighting.
            los_binning: Optional config for binning (strategy, max_days) for weight lookup.
            ehr_window_norm: Optional ``(mean, std)`` for EHR window features (train-computed z-score).
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
        self._data_dir = data_dir
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
        
        # SOFA tabular features (late fusion)
        self.sofa_features_config = data_config.get("sofa_features", {})
        self.sofa_features_enabled = self.sofa_features_config.get("enabled", False)
        self.sofa_mapping: Dict[str, List[float]] = {}
        self.sofa_features_dim = 0
        self._sofa_norm_mean: Optional[List[float]] = None
        self._sofa_norm_std: Optional[List[float]] = None
        if self.sofa_features_enabled:
            self._load_sofa_features()
        
        # ICU therapy support (late fusion; binary columns from labeling CSV)
        self.icu_therapy_support_config = data_config.get(
            "icu_therapy_support_features", {}
        )
        self.icu_therapy_support_enabled = self.icu_therapy_support_config.get(
            "enabled", False
        )
        self.icu_therapy_mapping: Dict[str, List[float]] = {}
        self.icu_therapy_dim = 0
        if self.icu_therapy_support_enabled:
            self._load_icu_therapy_support_features()

        # EHR window tabular (vitals mean/last/[n], labs worst/n, urine sum/[rate])
        self.ehr_window_features_config = data_config.get("ehr_window_features", {})
        self.ehr_window_features_enabled = self.ehr_window_features_config.get(
            "enabled", False
        )
        self._ehr_raw_by_key: Dict[str, np.ndarray] = {}
        self._ehr_feature_columns: List[str] = []
        self.ehr_window_features_dim = 0
        self._ehr_norm_mean: Optional[np.ndarray] = None
        self._ehr_norm_std: Optional[np.ndarray] = None
        if self.ehr_window_features_enabled:
            self._load_ehr_window_features(ehr_window_norm)
        
        # Task type: "regression" (default) or "classification" (backward compatibility)
        self.task_type = data_config.get("task_type", "regression")
        
        # Regression sample weighting
        self.regression_weights = regression_weights
        los_binning = los_binning or {}
        self._binning_strategy = los_binning.get("strategy", "intervals")
        self._binning_max_days = los_binning.get("max_days", 9)
        self._binning_boundaries = los_binning.get("boundaries")

        aug_cfg = data_config.get("augmentation", {})
        mb_ts_cfg = aug_cfg.get("mortality_balance_time_shuffle", {})
        self._mortality_balance_ts_enabled = bool(mb_ts_cfg.get("enabled", False))
        self._mortality_balance_ts_module: Optional[torch.nn.Module] = None
        if self._mortality_balance_ts_enabled:
            from ..augmentation.ecg_augmentation import TimeSegmentShuffle

            self._mortality_balance_ts_module = TimeSegmentShuffle(
                num_segments=int(aug_cfg.get("time_segment_shuffle_num_segments", 5)),
                p=1.0,
            )
        
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
    
    def _load_sofa_features(self) -> None:
        """Load SOFA columns from CSV; keys match ``tabular_join_path_key`` with ``data_dir``."""
        csv_path = self.sofa_features_config.get("csv_path", "data/labeling/labels_csv/sofa_scores.csv")
        path = Path(csv_path)
        if not path.is_absolute():
            path = _project_root() / path
        if not path.exists():
            print(f"Warning: SOFA CSV not found at {path}. Disabling sofa_features.")
            self.sofa_features_enabled = False
            return
        columns = self.sofa_features_config.get("columns", ["sofa_total"])
        if isinstance(columns, str):
            columns = [columns]
        require_available = self.sofa_features_config.get("require_available", True)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Warning: Failed to load SOFA CSV: {e}. Disabling sofa_features.")
            self.sofa_features_enabled = False
            return
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(f"Warning: SOFA CSV missing columns {missing}. Disabling sofa_features.")
            self.sofa_features_enabled = False
            return
        if require_available and "sofa_available" in df.columns:
            sa = df["sofa_available"]
            if sa.dtype == object:
                df = df.loc[sa.astype(str).str.lower().isin(("true", "1", "yes"))].copy()
            else:
                df = df.loc[sa.astype(bool)].copy()
        self.sofa_mapping = {}
        all_rows: List[List[float]] = []
        for _, row in df.iterrows():
            bp = row.get("base_path")
            if pd.isna(bp):
                continue
            key = tabular_join_path_key(str(bp), self._data_dir)
            vec: List[float] = []
            skip = False
            for col in columns:
                v = row[col]
                if pd.isna(v):
                    vec.append(0.0)
                else:
                    try:
                        vec.append(float(v))
                    except (TypeError, ValueError):
                        skip = True
                        break
            if skip:
                continue
            self.sofa_mapping[key] = vec
            all_rows.append(vec)
        self.sofa_features_dim = len(columns)
        norm = self.sofa_features_config.get("normalize", "none")
        if norm == "zscore" and all_rows and self.sofa_features_dim > 0:
            arr = np.array(all_rows, dtype=np.float64)
            self._sofa_norm_mean = arr.mean(axis=0).tolist()
            std = arr.std(axis=0)
            self._sofa_norm_std = [s if s > 1e-8 else 1.0 for s in std.tolist()]
        else:
            self._sofa_norm_mean = None
            self._sofa_norm_std = None
        print(
            f"Loaded SOFA features from {path}: {len(self.sofa_mapping):,} rows, "
            f"dim={self.sofa_features_dim}, columns={columns}"
        )
    
    def _get_sofa_features(self, base_path: str) -> torch.Tensor:
        """SOFA feature vector for ``base_path`` (zeros if missing)."""
        if not self.sofa_features_enabled or self.sofa_features_dim == 0:
            return torch.zeros(0, dtype=torch.float32)
        key = tabular_join_path_key(base_path, self._data_dir)
        vec = self.sofa_mapping.get(key)
        if vec is None:
            vec = [0.0] * self.sofa_features_dim
        if self._sofa_norm_mean is not None and self._sofa_norm_std is not None:
            vec = [
                (vec[i] - self._sofa_norm_mean[i]) / self._sofa_norm_std[i]
                for i in range(self.sofa_features_dim)
            ]
        return torch.tensor(vec, dtype=torch.float32)
    
    def _load_icu_therapy_support_features(self) -> None:
        """Load therapy-support columns; keys match ``tabular_join_path_key`` with ``data_dir``."""
        csv_path = self.icu_therapy_support_config.get(
            "csv_path", "data/labeling/labels_csv/icu_therapy_support.csv"
        )
        path = Path(csv_path)
        if not path.is_absolute():
            path = _project_root() / path
        if not path.exists():
            print(
                f"Warning: ICU therapy support CSV not found at {path}. "
                "Disabling icu_therapy_support_features."
            )
            self.icu_therapy_support_enabled = False
            return
        columns = self.icu_therapy_support_config.get(
            "columns",
            [
                "mech_vent",
                "niv_hfnc",
                "vaso_any",
                "vaso_non_catechol_any",
                "rrt",
            ],
        )
        if isinstance(columns, str):
            columns = [columns]
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(
                f"Warning: Failed to load ICU therapy support CSV: {e}. "
                "Disabling icu_therapy_support_features."
            )
            self.icu_therapy_support_enabled = False
            return
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(
                f"Warning: ICU therapy support CSV missing columns {missing}. "
                "Disabling icu_therapy_support_features."
            )
            self.icu_therapy_support_enabled = False
            return
        if "therapy_labels_available" in df.columns:
            ta = df["therapy_labels_available"]
            if ta.dtype == object:
                df = df.loc[
                    ta.astype(str).str.lower().isin(("true", "1", "yes"))
                ].copy()
            else:
                df = df.loc[ta.astype(bool)].copy()
        self.icu_therapy_mapping = {}
        for _, row in df.iterrows():
            bp = row.get("base_path")
            if pd.isna(bp):
                continue
            key = tabular_join_path_key(str(bp), self._data_dir)
            vec: List[float] = []
            skip = False
            for col in columns:
                v = row[col]
                if pd.isna(v):
                    vec.append(0.0)
                else:
                    try:
                        vec.append(float(v))
                    except (TypeError, ValueError):
                        skip = True
                        break
            if skip:
                continue
            self.icu_therapy_mapping[key] = vec
        self.icu_therapy_dim = len(columns)
        print(
            f"Loaded ICU therapy support features from {path}: "
            f"{len(self.icu_therapy_mapping):,} rows, dim={self.icu_therapy_dim}, "
            f"columns={columns}"
        )
    
    def _get_icu_therapy_support_features(self, base_path: str) -> torch.Tensor:
        """Therapy-support vector for ``base_path`` (zeros if missing)."""
        if not self.icu_therapy_support_enabled or self.icu_therapy_dim == 0:
            return torch.zeros(0, dtype=torch.float32)
        key = tabular_join_path_key(base_path, self._data_dir)
        vec = self.icu_therapy_mapping.get(key)
        if vec is None:
            vec = [0.0] * self.icu_therapy_dim
        return torch.tensor(vec, dtype=torch.float32)

    def _load_ehr_window_features(
        self,
        ehr_window_norm: Optional[Tuple[List[float], List[float]]],
    ) -> None:
        """Load the full EHR_feature_data.csv as a late-fusion vector."""
        cfg = self.ehr_window_features_config
        path = _resolve_ehr_csv_path(cfg, _project_root())
        if not path.exists():
            print(f"Warning: EHR window CSV not found at {path}. Disabling ehr_window_features.")
            self.ehr_window_features_enabled = False
            self.ehr_window_features_dim = 0
            return

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"Warning: Failed to load EHR window CSV: {e}. Disabling ehr_window_features.")
            self.ehr_window_features_enabled = False
            self.ehr_window_features_dim = 0
            return

        try:
            self._ehr_feature_columns = get_ehr_feature_columns(cfg)
        except Exception as e:
            print(
                f"Warning: Could not resolve EHR feature columns: {e}. "
                "Disabling ehr_window_features."
            )
            self.ehr_window_features_enabled = False
            self.ehr_window_features_dim = 0
            return
        self.ehr_window_features_dim = len(self._ehr_feature_columns)
        if self.ehr_window_features_dim == 0:
            print("Warning: No EHR feature columns selected. Disabling ehr_window_features.")
            self.ehr_window_features_enabled = False
            return

        self._ehr_raw_by_key = {}
        for _, row in df.iterrows():
            bp = row.get("base_path")
            if pd.isna(bp):
                continue
            key = tabular_join_path_key(str(bp), self._data_dir)
            self._ehr_raw_by_key[key] = raw_ehr_vector_from_row(
                row, self._ehr_feature_columns
            )

        norm_mode = cfg.get("normalize", "zscore")
        if norm_mode == "zscore" and ehr_window_norm is not None:
            m, s = ehr_window_norm
            self._ehr_norm_mean = np.asarray(m, dtype=np.float32)
            self._ehr_norm_std = np.asarray(s, dtype=np.float32)
        else:
            self._ehr_norm_mean = None
            self._ehr_norm_std = None

        print(
            f"Loaded EHR window features from {path}: {len(self._ehr_raw_by_key):,} rows, "
            f"dim={self.ehr_window_features_dim}, normalize={norm_mode}, "
            f"columns={self._ehr_feature_columns}"
        )

    def _get_ehr_window_features(self, base_path: str) -> torch.Tensor:
        if not self.ehr_window_features_enabled or self.ehr_window_features_dim == 0:
            return torch.zeros(0, dtype=torch.float32)
        key = tabular_join_path_key(base_path, self._data_dir)
        raw = self._ehr_raw_by_key.get(key)
        if raw is None:
            raw = np.zeros(self.ehr_window_features_dim, dtype=np.float32)
        if self._ehr_norm_mean is not None and self._ehr_norm_std is not None:
            out = (raw.astype(np.float32) - self._ehr_norm_mean) / self._ehr_norm_std
        else:
            out = raw.astype(np.float32)
        return torch.from_numpy(out.astype(np.float32))

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
            if self.icu_unit_features_enabled:
                print(f"  ICU unit features: Enabled ({len(self.icu_unit_mapping):,} mappings, {self.icu_unit_dim} units)")
            if self.sofa_features_enabled:
                print(f"  SOFA features: Enabled ({len(self.sofa_mapping):,} mappings, dim={self.sofa_features_dim})")
            if self.icu_therapy_support_enabled:
                print(
                    f"  ICU therapy support features: Enabled "
                    f"({len(self.icu_therapy_mapping):,} mappings, dim={self.icu_therapy_dim})"
                )
            if self.ehr_window_features_enabled:
                print(
                    f"  EHR window features: Enabled "
                    f"({len(self._ehr_raw_by_key):,} mappings, dim={self.ehr_window_features_dim})"
                )
    
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

        rec = self.ecg_dataset.records[idx]
        if self._mortality_balance_ts_module is not None and rec.get("mortality_ts_child") is not None:
            if hasattr(self._mortality_balance_ts_module, "train"):
                self._mortality_balance_ts_module.train()
            child_id = int(rec["mortality_ts_child"])
            digest = hashlib.blake2b(
                f"{base_path}\0{child_id}".encode("utf-8"), digest_size=8
            ).digest()
            seed = int.from_bytes(digest, "little") % (2**32)
            rng_state = np.random.get_state()
            try:
                np.random.seed(seed)
                signal = self._mortality_balance_ts_module(signal)
            finally:
                np.random.set_state(rng_state)
        
        # Remove None values from meta (PyTorch DataLoader can't collate None)
        meta_clean = {k: v for k, v in meta.items() if v is not None}
        
        # Get demographic features if enabled
        demographic_features = None
        if self.demographic_features_enabled:
            demographic_features = self._get_demographic_features(base_path)
        
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
        
        # Add ICU unit features if available
        if icu_unit_features is not None:
            result["icu_unit_features"] = icu_unit_features
        
        if self.sofa_features_enabled and self.sofa_features_dim > 0:
            result["sofa_features"] = self._get_sofa_features(base_path)
        
        if self.icu_therapy_support_enabled and self.icu_therapy_dim > 0:
            result["icu_therapy_support_features"] = (
                self._get_icu_therapy_support_features(base_path)
            )

        if self.ehr_window_features_enabled and self.ehr_window_features_dim > 0:
            result["ehr_window_features"] = self._get_ehr_window_features(base_path)
        
        # Sample weight for regression (only when weighting is active)
        if self.regression_weights is not None and label >= 0:
            bin_idx = los_to_bin(
                float(label),
                binning_strategy=self._binning_strategy,
                max_days=self._binning_max_days,
                boundaries=self._binning_boundaries,
            )
            sample_weight = self.regression_weights.get(bin_idx, 1.0)
            result["sample_weight"] = torch.tensor(sample_weight, dtype=torch.float32)
        elif self.regression_weights is not None:
            result["sample_weight"] = torch.tensor(1.0, dtype=torch.float32)
        
        return result

