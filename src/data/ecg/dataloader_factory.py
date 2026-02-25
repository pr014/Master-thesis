"""Factory for creating PyTorch DataLoaders."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from .ecg_dataset import ECGDataset, build_demo_index, extract_subject_id_from_path, construct_ecg_time
from .ecg_loader import build_npy_index
from ..labeling.icu_los_labels import los_to_bin
from .timestamp_mapping import (
    load_timestamp_mapping,
    create_timestamp_mapping,
    auto_detect_original_path,
    get_timestamp_mapping_path,
)


def get_los_values_for_records(
    records: List[Dict[str, Any]],
    icu_mapper: Any,
    timestamp_mapping: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    data_dir: Optional[str] = None,
) -> List[float]:
    """Extract LOS values for records by matching ECGs to ICU stays.
    
    Mirrors ECGDataset._generate_labels logic to get LOS for each matched record.
    
    Args:
        records: List of record dicts with 'base_path' key.
        icu_mapper: ICUStayMapper instance.
        timestamp_mapping: Optional mapping from relative base_path to timestamps.
        data_dir: Data directory for resolving relative paths.
    
    Returns:
        List of LOS values in days (float) for matched records only.
    """
    los_values = []
    data_dir_path = Path(data_dir) if data_dir else None
    
    for record in records:
        base_path = record.get("base_path", "")
        if not base_path:
            continue
        try:
            subject_id = extract_subject_id_from_path(base_path)
        except ValueError:
            continue
        
        ecg_time = None
        if timestamp_mapping is not None and data_dir_path is not None:
            try:
                base_path_obj = Path(base_path)
                if base_path_obj.is_absolute():
                    rel_path = base_path_obj.relative_to(data_dir_path)
                else:
                    rel_path = Path(base_path)
                rel_path_str = str(rel_path).replace("\\", "/")
                if rel_path_str in timestamp_mapping:
                    ts_info = timestamp_mapping[rel_path_str]
                    ecg_time = construct_ecg_time(
                        ts_info.get("base_date"), ts_info.get("base_time")
                    )
            except (ValueError, KeyError):
                pass
        
        if ecg_time is None and icu_mapper is not None:
            subject_stays = icu_mapper.icustays_df[
                icu_mapper.icustays_df["subject_id"] == subject_id
            ]
            if len(subject_stays) > 0:
                ecg_time = pd.to_datetime(subject_stays.iloc[0]["intime"])
        
        if ecg_time is None:
            continue
        
        stay_id = icu_mapper.map_ecg_to_stay(subject_id, ecg_time)
        if stay_id is None:
            continue
        
        los_days = icu_mapper.get_los(stay_id)
        if los_days is not None:
            los_values.append(float(los_days))
    
    return los_values


def compute_regression_weights(
    los_values: List[float],
    config: Dict[str, Any],
) -> Optional[Dict[int, float]]:
    """Compute sample weights per LOS bin for imbalanced regression.
    
    Args:
        los_values: List of LOS values in days from training set.
        config: Full config with data.los_binning and training.loss.
    
    Returns:
        Dict mapping bin_idx -> weight, normalized so mean=1.0.
        Returns None if los_values is empty.
    """
    if not los_values:
        return None
    
    data_config = config.get("data", {})
    los_binning = data_config.get("los_binning", {})
    strategy = los_binning.get("strategy", "intervals")
    max_days = los_binning.get("max_days", 9)
    
    loss_config = config.get("training", {}).get("loss", {})
    method = loss_config.get("method", "balanced")
    
    bins = [
        los_to_bin(los, binning_strategy=strategy, max_days=max_days)
        for los in los_values
    ]
    n_bins = max(bins) + 1
    counts = [bins.count(i) for i in range(n_bins)]
    n_total = sum(counts)
    
    if n_total == 0:
        return None
    
    weights = []
    for c in counts:
        if c > 0:
            if method == "balanced":
                w = n_total / (n_bins * c)
            else:
                w = np.sqrt(n_total) / np.sqrt(c)
        else:
            w = 0.0
        weights.append(w)
    
    mean_weight = np.mean([w for w in weights if w > 0])
    if mean_weight <= 0:
        return None
    weights_normalized = [w / mean_weight if w > 0 else 0.0 for w in weights]
    
    return {i: weights_normalized[i] for i in range(len(weights_normalized))}


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function that preserves meta as a list of dicts.
    
    PyTorch's default_collate tries to collate dicts, which breaks meta.
    This function keeps meta as a list of dictionaries.
    """
    # Separate meta from other fields
    signals = torch.stack([item["signal"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    meta = [item["meta"] for item in batch]
    
    # Handle mortality_label if present
    result = {
        "signal": signals,
        "label": labels,
        "meta": meta,  # Keep as list, don't collate
    }
    
    # Add mortality_label if present in batch
    if "mortality_label" in batch[0]:
        mortality_labels = torch.stack([item["mortality_label"] for item in batch])
        result["mortality_label"] = mortality_labels
    
    # Add demographic_features if present in batch
    if "demographic_features" in batch[0]:
        # Check if any sample has demographic features
        demo_features_list = [item.get("demographic_features") for item in batch]
        if any(df is not None for df in demo_features_list):
            # Replace None with zeros (same shape as first non-None feature)
            first_non_none = next(df for df in demo_features_list if df is not None)
            demo_dim = first_non_none.shape[0]
            demo_features_filled = [
                df if df is not None else torch.zeros(demo_dim, dtype=torch.float32)
                for df in demo_features_list
            ]
            demographic_features = torch.stack(demo_features_filled)
            result["demographic_features"] = demographic_features
        else:
            result["demographic_features"] = None
    else:
        result["demographic_features"] = None
    
    # Add diagnosis_features if present in batch
    if "diagnosis_features" in batch[0]:
        # Check if any sample has diagnosis features
        diag_features_list = [item.get("diagnosis_features") for item in batch]
        if any(df is not None for df in diag_features_list):
            # Replace None with zeros (same shape as first non-None feature)
            first_non_none = next(df for df in diag_features_list if df is not None)
            diag_dim = first_non_none.shape[0]
            diag_features_filled = [
                df if df is not None else torch.zeros(diag_dim, dtype=torch.float32)
                for df in diag_features_list
            ]
            diagnosis_features = torch.stack(diag_features_filled)
            result["diagnosis_features"] = diagnosis_features
        else:
            result["diagnosis_features"] = None
    else:
        result["diagnosis_features"] = None
    
    # Add icu_unit_features if present in batch
    if "icu_unit_features" in batch[0]:
        icu_features_list = [item.get("icu_unit_features") for item in batch]
        if any(f is not None for f in icu_features_list):
            first_non_none = next(f for f in icu_features_list if f is not None)
            icu_dim = first_non_none.shape[0]
            icu_features_filled = [
                f if f is not None else torch.zeros(icu_dim, dtype=torch.float32)
                for f in icu_features_list
            ]
            icu_unit_features = torch.stack(icu_features_filled)
            result["icu_unit_features"] = icu_unit_features
        else:
            result["icu_unit_features"] = None
    else:
        result["icu_unit_features"] = None
    
    if "sample_weight" in batch[0]:
        result["sample_weight"] = torch.stack([item["sample_weight"] for item in batch])
    
    return result


def get_subject_timestamps(
    subjects: List[int],
    subject_to_records: Dict[int, List[Dict]],
    icu_mapper: Any,
    timestamp_mapping: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    data_dir: Optional[str] = None,
) -> Dict[int, pd.Timestamp]:
    """Get earliest timestamp for each subject (for temporal splitting).
    
    Tries to get intime from ICU stays first, falls back to ecg_time from timestamp_mapping.
    
    Args:
        subjects: List of subject IDs.
        subject_to_records: Mapping from subject_id to list of records.
        icu_mapper: ICUStayMapper instance.
        timestamp_mapping: Optional mapping from relative base_path to timestamps.
        data_dir: Optional data directory path for constructing relative paths.
    
    Returns:
        Dictionary mapping subject_id -> earliest timestamp (pd.Timestamp).
        Subjects without valid timestamps are excluded.
    """
    from .ecg_dataset import extract_subject_id_from_path, construct_ecg_time
    
    subject_timestamps = {}
    data_dir_path = Path(data_dir) if data_dir else None
    
    for subject_id in subjects:
        earliest_timestamp = None
        
        # First, try to get intime from ICU stays (most reliable)
        if icu_mapper is not None:
            subject_stays = icu_mapper.icustays_df[
                icu_mapper.icustays_df['subject_id'] == subject_id
            ]
            if len(subject_stays) > 0:
                # Get minimum intime across all stays
                intimes = pd.to_datetime(subject_stays['intime'], errors='coerce')
                intimes = intimes.dropna()
                if len(intimes) > 0:
                    earliest_timestamp = intimes.min()
        
        # Fallback: use earliest ecg_time from timestamp_mapping
        if earliest_timestamp is None and timestamp_mapping is not None and data_dir_path:
            subject_records = subject_to_records.get(subject_id, [])
            ecg_times = []
            
            for record in subject_records:
                base_path = record.get("base_path", "")
                if not base_path:
                    continue
                
                # Convert to relative path for lookup (same as in ecg_dataset)
                try:
                    base_path_obj = Path(base_path)
                    if base_path_obj.is_absolute():
                        rel_path = base_path_obj.relative_to(data_dir_path)
                    else:
                        rel_path = Path(base_path)
                    rel_path_str = str(rel_path).replace("\\", "/")
                    
                    if rel_path_str in timestamp_mapping:
                        timestamp_info = timestamp_mapping[rel_path_str]
                        base_date = timestamp_info.get("base_date")
                        base_time = timestamp_info.get("base_time")
                        ecg_time = construct_ecg_time(base_date, base_time)
                        if ecg_time is not None:
                            ecg_times.append(pd.Timestamp(ecg_time))
                except (ValueError, KeyError):
                    continue
            
            if ecg_times:
                earliest_timestamp = min(ecg_times)
        
        if earliest_timestamp is not None:
            subject_timestamps[subject_id] = earliest_timestamp
    
    return subject_timestamps


def get_subject_los_values(
    subjects: List[int],
    subject_to_records: Dict[int, List[Dict]],
    icu_mapper: Any,
) -> Dict[int, float]:
    """Get mean LOS value for each subject (for stratification).
    
    Args:
        subjects: List of subject IDs.
        subject_to_records: Mapping from subject_id to list of records.
        icu_mapper: ICUStayMapper instance.
    
    Returns:
        Dictionary mapping subject_id -> mean LOS in days.
    """
    from .ecg_dataset import extract_subject_id_from_path
    
    subject_los = {}
    for subject_id in subjects:
        # Get LOS values for all stays of this subject
        if icu_mapper is not None:
            subject_stays = icu_mapper.icustays_df[
                icu_mapper.icustays_df['subject_id'] == subject_id
            ]
            if len(subject_stays) > 0:
                # Use mean LOS across all stays
                mean_los = subject_stays['los'].mean()
                subject_los[subject_id] = float(mean_los)
            else:
                subject_los[subject_id] = -1.0  # No stays found
        else:
            subject_los[subject_id] = -1.0  # No mapper
    
    return subject_los


def create_temporal_stratified_split(
    subjects: List[int],
    subject_timestamps: Dict[int, pd.Timestamp],
    subject_los: Optional[Dict[int, float]] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    stratify: bool = False,
    n_bins: int = 10,
    random_state: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Create temporal split with optional LOS stratification within segments.
    
    Sorts subjects by timestamp and splits temporally:
    - Train: earliest 70% (or 1 - val_size - test_size)
    - Val: next 15% (or val_size)
    - Test: latest 15% (or test_size)
    
    If stratify=True, applies LOS stratification within each temporal segment.
    
    Args:
        subjects: List of subject IDs.
        subject_timestamps: Dictionary mapping subject_id -> earliest timestamp.
        subject_los: Optional dictionary mapping subject_id -> LOS in days (for stratification).
        test_size: Fraction for test set (e.g., 0.15).
        val_size: Fraction for validation set (e.g., 0.15).
        stratify: If True, stratify by LOS within each temporal segment.
        n_bins: Number of quantile bins for stratification (if stratify=True).
        random_state: Random seed for stratification.
    
    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects).
    """
    # Filter subjects with valid timestamps
    valid_subjects = [s for s in subjects if s in subject_timestamps]
    
    if len(valid_subjects) == 0:
        print("Warning: No subjects with valid timestamps. Returning empty splits.")
        return [], [], []
    
    # Sort subjects by timestamp (ascending: oldest first)
    sorted_subjects = sorted(valid_subjects, key=lambda s: subject_timestamps[s])
    
    # Calculate split indices
    n_total = len(sorted_subjects)
    train_size = 1.0 - val_size - test_size
    
    train_end = int(n_total * train_size)
    val_end = int(n_total * (train_size + val_size))
    
    # Temporal split (no stratification)
    if not stratify or subject_los is None:
        train_subjects = sorted_subjects[:train_end]
        val_subjects = sorted_subjects[train_end:val_end]
        test_subjects = sorted_subjects[val_end:]
        
        # Log temporal split statistics
        train_timestamps = [subject_timestamps[s] for s in train_subjects]
        val_timestamps = [subject_timestamps[s] for s in val_subjects]
        test_timestamps = [subject_timestamps[s] for s in test_subjects]
        
        print(f"Temporal split (no stratification):")
        print(f"  Train: {len(train_subjects)} subjects, time range: {min(train_timestamps)} to {max(train_timestamps)}")
        print(f"  Val:   {len(val_subjects)} subjects, time range: {min(val_timestamps) if val_timestamps else 'N/A'} to {max(val_timestamps) if val_timestamps else 'N/A'}")
        print(f"  Test:  {len(test_subjects)} subjects, time range: {min(test_timestamps) if test_timestamps else 'N/A'} to {max(test_timestamps) if test_timestamps else 'N/A'}")
        
        # Log LOS statistics if available
        if subject_los:
            train_los = [subject_los.get(s, -1) for s in train_subjects if subject_los.get(s, -1) >= 0]
            val_los = [subject_los.get(s, -1) for s in val_subjects if subject_los.get(s, -1) >= 0]
            test_los = [subject_los.get(s, -1) for s in test_subjects if subject_los.get(s, -1) >= 0]
            
            if train_los:
                print(f"  Train LOS: mean={np.mean(train_los):.2f}, median={np.median(train_los):.2f}")
            if val_los:
                print(f"  Val LOS:   mean={np.mean(val_los):.2f}, median={np.median(val_los):.2f}")
            if test_los:
                print(f"  Test LOS:  mean={np.mean(test_los):.2f}, median={np.median(test_los):.2f}")
        
        return train_subjects, val_subjects, test_subjects
    
    # Temporal split WITH LOS stratification within segments
    # Split into temporal segments first
    train_segment = sorted_subjects[:train_end]
    val_segment = sorted_subjects[train_end:val_end]
    test_segment = sorted_subjects[val_end:]
    
    # Apply stratification within each segment
    def stratify_segment(segment_subjects: List[int], segment_name: str) -> List[int]:
        """Stratify a temporal segment by LOS."""
        valid_segment = [s for s in segment_subjects if subject_los.get(s, -1) >= 0]
        
        if len(valid_segment) < n_bins:
            # Not enough subjects for stratification, return as-is
            return valid_segment
        
        los_values = np.array([subject_los[s] for s in valid_segment])
        
        # Create quantile-based bins
        try:
            bins = pd.qcut(los_values, q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            bins = pd.cut(los_values, bins=n_bins, labels=False, duplicates='drop')
        
        bins = np.array(bins)
        
        # Handle NaN bins
        if np.isnan(bins).any():
            most_common_bin = np.bincount(bins[~np.isnan(bins)].astype(int)).argmax()
            bins = np.where(np.isnan(bins), most_common_bin, bins).astype(int)
        
        # Group by bin while maintaining temporal order
        # For stratification, we want to balance LOS distribution but keep temporal order
        # within each bin. We'll interleave subjects from different bins.
        from collections import defaultdict
        bin_to_subjects = defaultdict(list)
        for idx, subject in enumerate(valid_segment):
            bin_to_subjects[bins[idx]].append(subject)
        
        # Interleave subjects from different bins to balance LOS distribution
        # while maintaining approximate temporal order
        stratified_subjects = []
        max_bin_size = max(len(bin_to_subjects[bin_idx]) for bin_idx in bin_to_subjects.keys()) if bin_to_subjects else 0
        
        for i in range(max_bin_size):
            for bin_idx in sorted(bin_to_subjects.keys()):
                if i < len(bin_to_subjects[bin_idx]):
                    stratified_subjects.append(bin_to_subjects[bin_idx][i])
        
        return stratified_subjects
    
    train_subjects = stratify_segment(train_segment, "train")
    val_subjects = stratify_segment(val_segment, "val")
    test_subjects = stratify_segment(test_segment, "test")
    
    # Log temporal stratified split statistics
    train_timestamps = [subject_timestamps[s] for s in train_subjects]
    val_timestamps = [subject_timestamps[s] for s in val_subjects]
    test_timestamps = [subject_timestamps[s] for s in test_subjects]
    
    print(f"Temporal stratified split (n_bins={n_bins}):")
    print(f"  Train: {len(train_subjects)} subjects, time range: {min(train_timestamps)} to {max(train_timestamps)}")
    print(f"  Val:   {len(val_subjects)} subjects, time range: {min(val_timestamps) if val_timestamps else 'N/A'} to {max(val_timestamps) if val_timestamps else 'N/A'}")
    print(f"  Test:  {len(test_subjects)} subjects, time range: {min(test_timestamps) if test_timestamps else 'N/A'} to {max(test_timestamps) if test_timestamps else 'N/A'}")
    
    # Log LOS statistics
    train_los = [subject_los.get(s, -1) for s in train_subjects if subject_los.get(s, -1) >= 0]
    val_los = [subject_los.get(s, -1) for s in val_subjects if subject_los.get(s, -1) >= 0]
    test_los = [subject_los.get(s, -1) for s in test_subjects if subject_los.get(s, -1) >= 0]
    
    if train_los:
        print(f"  Train LOS: mean={np.mean(train_los):.2f}, median={np.median(train_los):.2f}")
    if val_los:
        print(f"  Val LOS:   mean={np.mean(val_los):.2f}, median={np.median(val_los):.2f}")
    if test_los:
        print(f"  Test LOS:  mean={np.mean(test_los):.2f}, median={np.median(test_los):.2f}")
    
    return train_subjects, val_subjects, test_subjects


def create_dataloaders(
    config: Dict[str, Any],
    labels: Optional[Dict[str, float]] = None,
    preprocess: Optional[Any] = None,
    transform: Optional[Any] = None,
    icu_mapper: Optional[Any] = None,
    mortality_labels: Optional[Dict[str, int]] = None,
    original_data_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, validation, and test DataLoaders.
    
    Uses temporal split (sorted by timestamp) to ensure test data comes
    after train/val data, preventing data leakage. Optional LOS stratification
    within temporal segments.
    
    Args:
        config: Configuration dictionary with data and training settings.
        labels: Optional dictionary mapping base_path to label (float for regression).
        preprocess: Optional preprocessing function.
        transform: Optional PyTorch transform.
        icu_mapper: Optional ICUStayMapper for automatic label generation.
        mortality_labels: Optional dictionary mapping base_path to mortality label.
        original_data_dir: Optional path to original data directory.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
        test_loader is None if test_split is 0 or not specified.
    """
    # Get data directory from config
    data_config = config.get("data", {})
    data_dir = data_config.get("data_dir", "data/raw/demo/ecg/mimic-iv-ecg-demo")
    window_seconds = data_config.get("window_seconds", 10.0)
    
    # Get task type and split strategy (only temporal split supported)
    task_type = data_config.get("task_type", "regression")
    split_strategy = data_config.get("split_strategy", "temporal_stratified")
    
    # Only temporal splits are supported
    if split_strategy not in ["temporal", "temporal_stratified"]:
        print(f"Warning: split_strategy '{split_strategy}' not supported. Using 'temporal_stratified' instead.")
        split_strategy = "temporal_stratified"
    
    # Get training settings
    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 64)
    num_workers = training_config.get("num_workers", 4)
    pin_memory = training_config.get("pin_memory", True)
    
    # Create augmentation transform if not provided and enabled in config
    if transform is None:
        from ..augmentation import create_augmentation_transform
        transform = create_augmentation_transform(config)
    
    # Get validation split
    val_config = config.get("validation", {})
    val_split = val_config.get("val_split", 0.2)
    test_split = config.get("test_split", 0.0)  # Optional test split
    
    # Load records - use build_npy_index for .npy files, build_demo_index for .hea/.dat files
    # Try .npy first (preprocessed data), fall back to .hea/.dat (raw data)
    using_npy = False
    try:
        records = build_npy_index(data_dir=data_dir)
        using_npy = True
    except (FileNotFoundError, RuntimeError):
        # Fall back to demo index if .npy files not found
        records = build_demo_index(data_dir=data_dir)
        using_npy = False
    
    # Load or create timestamp mapping if using .npy files
    timestamp_mapping = None
    if using_npy:
        require_timestamp_mapping = data_config.get("require_timestamp_mapping", True)

        # Get original data directory
        if original_data_dir is None:
            # Try to get from config
            original_data_dir = data_config.get("original_data_dir")
        
        if original_data_dir is None:
            # Try auto-detection
            original_data_dir_path = auto_detect_original_path(data_dir)
            if original_data_dir_path:
                original_data_dir = str(original_data_dir_path)
        
        # Get mapping file path
        mapping_path = get_timestamp_mapping_path(data_dir)
        
        # Try to load existing mapping
        if mapping_path.exists():
            try:
                print(f"Loading timestamp mapping from: {mapping_path}")
                timestamp_mapping = load_timestamp_mapping(str(mapping_path))
                print(f"Loaded timestamp mapping with {len(timestamp_mapping):,} entries")
            except Exception as e:
                print(f"Warning: Failed to load timestamp mapping: {e}")
                timestamp_mapping = None
        
        # Do NOT create mapping automatically - it should be created before training
        if timestamp_mapping is None:
            msg = (
                f"Timestamp mapping not found at: {mapping_path}\n"
                f"Create it before training using:\n"
                f"  python scripts/data/create_timestamp_mapping.py --preprocessed_dir {data_dir}\n"
            )
            if require_timestamp_mapping:
                raise FileNotFoundError(msg)
            print(f"Warning: {msg}")
            print("  Continuing without timestamps (will use fallback to first stay's intime)")
    
    # Patient-level split: extract subject_id from each record
    # Group records by subject_id to avoid data leakage
    from .ecg_dataset import extract_subject_id_from_path
    
    records_with_subject = []
    for record in records:
        try:
            subject_id = extract_subject_id_from_path(record["base_path"])
            records_with_subject.append((subject_id, record))
        except ValueError:
            # Skip records without valid subject_id
            continue
    
    # Get unique subject_ids
    subject_to_records = {}
    for subject_id, record in records_with_subject:
        if subject_id not in subject_to_records:
            subject_to_records[subject_id] = []
        subject_to_records[subject_id].append(record)
    
    unique_subjects = list(subject_to_records.keys())
    
    # Use temporal split (with optional LOS stratification)
    if icu_mapper is None:
        raise ValueError("Temporal split requires icu_mapper. Please provide ICUStayMapper instance.")
    
    stratify = (split_strategy == "temporal_stratified")
    print(f"Using temporal split (stratify={stratify}) for {task_type} task...")
    
    # Get timestamps for temporal sorting
    subject_timestamps = get_subject_timestamps(
        subjects=unique_subjects,
        subject_to_records=subject_to_records,
        icu_mapper=icu_mapper,
        timestamp_mapping=timestamp_mapping,
        data_dir=data_dir,
    )
    
    if len(subject_timestamps) == 0:
        raise ValueError("No subjects with valid timestamps found. Cannot perform temporal split.")
    
    # Get LOS values if stratification is enabled
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
    
    # Convert subject splits back to record lists
    train_records = []
    for subject_id in train_subjects:
        train_records.extend(subject_to_records[subject_id])
    
    val_records = []
    for subject_id in val_subjects:
        val_records.extend(subject_to_records[subject_id])
    
    test_records = []
    for subject_id in test_subjects:
        test_records.extend(subject_to_records[subject_id])
    
    # Get demographic features config if enabled
    demographic_features_config = data_config.get("demographic_features", {})
    
    # Regression sample weighting: compute weights from train LOS distribution
    regression_weights = None
    los_binning = data_config.get("los_binning", {})
    if task_type == "regression" and training_config.get("loss", {}).get("weighted", False):
        los_values = get_los_values_for_records(
            train_records,
            icu_mapper,
            timestamp_mapping=timestamp_mapping,
            data_dir=data_dir,
        )
        regression_weights = compute_regression_weights(los_values, config)
        if regression_weights is not None:
            print(f"Regression sample weighting: enabled (from {len(los_values):,} train samples)")
        else:
            print("Regression sample weighting: disabled (no matched samples or empty)")
    
    # Create datasets
    # Augmentation only for training (transform is None for val/test)
    train_dataset = ECGDataset(
        records=train_records,
        labels=labels,
        preprocess=preprocess,
        window_seconds=window_seconds,
        transform=transform,  # Augmentation applied here
        icu_mapper=icu_mapper,
        mortality_labels=mortality_labels,
        timestamp_mapping=timestamp_mapping,
        data_dir=data_dir,
        demographic_features_config=demographic_features_config,
        config=config,
        regression_weights=regression_weights,
        los_binning=los_binning,
    )
    
    # No augmentation for validation/test (no regression weighting for eval)
    val_dataset = ECGDataset(
        records=val_records,
        labels=labels,
        preprocess=preprocess,
        window_seconds=window_seconds,
        transform=None,  # No augmentation for validation
        icu_mapper=icu_mapper,
        mortality_labels=mortality_labels,
        timestamp_mapping=timestamp_mapping,
        data_dir=data_dir,
        demographic_features_config=demographic_features_config,
        config=config,
        regression_weights=None,
        los_binning=None,
    )
    
    # Create DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
    )
    
    # Test loader (optional)
    test_loader = None
    if test_records:
        test_dataset = ECGDataset(
            records=test_records,
            labels=labels,
            preprocess=preprocess,
            window_seconds=window_seconds,
            transform=None,  # No augmentation for test
            icu_mapper=icu_mapper,
            mortality_labels=mortality_labels,
            timestamp_mapping=timestamp_mapping,
            data_dir=data_dir,
            demographic_features_config=demographic_features_config,
            config=config,
            regression_weights=None,
            los_binning=None,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn,
        )
    
    return train_loader, val_loader, test_loader
