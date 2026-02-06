"""ECG data loading and processing modules.

This package contains:
- ecg_loader: Low-level ECG file loading (PhysioNet format)
- ecg_dataset: PyTorch Dataset wrapper for ECG data
- dataloader_factory: Factory for creating PyTorch DataLoaders
- ecg_metadata: Utilities for ECG metadata extraction
"""

from .ecg_loader import ECGDemoDataset, build_demo_index, ECGRecord, ECGNPYDataset, build_npy_index
from .ecg_dataset import ECGDataset, extract_subject_id_from_path, construct_ecg_time
from .dataloader_factory import create_dataloaders
from .ecg_metadata import (
    extract_timestamp_from_record,
    extract_timestamps_from_directory,
    get_wfdb_record_metadata,
)

__all__ = [
    # ecg_loader
    "ECGDemoDataset",
    "build_demo_index",
    "ECGNPYDataset",
    "build_npy_index",
    "ECGRecord",
    # ecg_dataset
    "ECGDataset",
    "extract_subject_id_from_path",
    "construct_ecg_time",
    # dataloader_factory
    "create_dataloaders",
    # ecg_metadata
    "extract_timestamp_from_record",
    "extract_timestamps_from_directory",
    "get_wfdb_record_metadata",
]

