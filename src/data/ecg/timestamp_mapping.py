"""Timestamp mapping utilities for ECG data.

Extracts timestamps from original .hea/.dat files and creates a CSV mapping
that can be used when loading preprocessed .npy files.
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# tqdm optional (for progress bars)
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

from .ecg_loader import build_demo_index
from .ecg_metadata import extract_timestamp_from_record


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


def create_timestamp_mapping(
    original_data_dir: str,
    output_path: str,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Create timestamp mapping CSV from original .hea/.dat files.
    
    Extracts base_date and base_time from all ECG records in the original
    data directory and creates a CSV mapping from relative base_path to timestamps.
    
    Args:
        original_data_dir: Path to directory containing .hea/.dat files.
        output_path: Path where CSV mapping will be saved.
        limit: Optional limit on number of records to process (for testing).
    
    Returns:
        DataFrame with columns: base_path, base_date, base_time, subject_id, study_id
    """
    original_data_dir = Path(original_data_dir)
    if not original_data_dir.exists():
        raise FileNotFoundError(f"Original data directory not found: {original_data_dir}")
    
    print(f"Creating timestamp mapping from: {original_data_dir}")
    print(f"Output will be saved to: {output_path}")
    
    # Find all .hea/.dat files
    records = build_demo_index(data_dir=str(original_data_dir), limit=limit)
    print(f"Found {len(records):,} ECG records")
    
    # Create list of records for DataFrame
    mapping_data = []
    failed_count = 0
    
    for record in tqdm(records, desc="Extracting timestamps"):
        base_path = Path(record["base_path"])
        
        # Get relative path from original_data_dir
        try:
            rel_path = base_path.relative_to(original_data_dir)
            # Convert to string and normalize (use forward slashes for portability)
            rel_path_str = str(rel_path).replace("\\", "/")
        except ValueError:
            # If base_path is not relative to original_data_dir, use just the name
            rel_path_str = base_path.name
        
        # Extract subject_id and study_id from path
        subject_id = None
        study_id = None
        try:
            subject_id = extract_subject_id_from_path(str(base_path))
            # Extract study_id from path (s{ID} segment)
            path_parts = base_path.parts
            for part in path_parts:
                if part.startswith('s') and len(part) > 1 and part[1:].isdigit():
                    study_id = int(part[1:])
                    break
        except Exception:
            pass  # Keep as None if extraction fails
        
        # Extract timestamp from .hea file
        base_date = None
        base_time = None
        try:
            timestamp_info = extract_timestamp_from_record(str(base_path))
            base_date = timestamp_info.get("date")
            base_time = timestamp_info.get("time")
        except Exception as e:
            # If extraction fails, keep as None
            print(f"Warning: Failed to extract timestamp from {base_path}: {e}")
            failed_count += 1
        
        mapping_data.append({
            "base_path": rel_path_str,
            "base_date": base_date,
            "base_time": base_time,
            "subject_id": subject_id,
            "study_id": study_id,
        })
    
    # Create DataFrame
    df = pd.DataFrame(mapping_data)
    
    # Save as CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    successful_count = df["base_date"].notna().sum()
    
    print(f"\nTimestamp mapping created:")
    print(f"  Total records: {len(df):,}")
    print(f"  Successful: {successful_count:,}")
    print(f"  Failed: {failed_count:,}")
    print(f"  Saved to: {output_path}")
    
    return df


def load_timestamp_mapping(mapping_path: str) -> Dict[str, Dict[str, Optional[str]]]:
    """Load timestamp mapping from CSV file.
    
    Args:
        mapping_path: Path to CSV mapping file.
    
    Returns:
        Dictionary mapping base_path (relative) -> {"base_date": str, "base_time": str}
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Timestamp mapping file not found: {mapping_path}")
    
    # Load CSV
    df = pd.read_csv(mapping_path)
    
    # Convert to dictionary format for compatibility
    mapping = {}
    for _, row in df.iterrows():
        base_path = str(row["base_path"])
        mapping[base_path] = {
            "base_date": row.get("base_date") if pd.notna(row.get("base_date")) else None,
            "base_time": row.get("base_time") if pd.notna(row.get("base_time")) else None,
        }
    
    return mapping


def auto_detect_original_path(preprocessed_data_dir: str) -> Optional[Path]:
    """Auto-detect original data directory from preprocessed directory path.
    
    Tries various naming conventions to find the original .hea/.dat files.
    
    Args:
        preprocessed_data_dir: Path to preprocessed data directory (e.g., preprocessed_24h_1).
    
    Returns:
        Path to original data directory if found, None otherwise.
    """
    preprocessed_path = Path(preprocessed_data_dir)
    
    # Try different naming patterns
    candidates = []
    
    # Pattern 1: preprocessed_24h_1 -> original_24h_1
    if "preprocessed" in preprocessed_path.name:
        original_name = preprocessed_path.name.replace("preprocessed", "original")
        candidates.append(preprocessed_path.parent / original_name)
    
    # Pattern 2: data/icu_ecgs_24h/P1 -> data/icu_ecgs_24h/original
    if preprocessed_path.name in ["P1", "P2", "P3"] and "icu_ecgs_24h" in str(preprocessed_path):
        candidates.append(preprocessed_path.parent / "original")
    
    # Pattern 3: preprocessed_icu_ecgs_P1 -> icu_ecgs_24h_P1 or data/icu_ecgs_24h/original
    if "preprocessed_icu_ecgs" in preprocessed_path.name:
        # Try: icu_ecgs_24h_P1
        candidates.append(preprocessed_path.parent / "icu_ecgs_24h_P1")
        # Try: data/icu_ecgs_24h/original
        candidates.append(preprocessed_path.parent.parent / "icu_ecgs_24h" / "original")
    
    # Pattern 4: preprocessed_24h_1 -> data/icu_ecgs_24h/original
    if "24h" in preprocessed_path.name or "preprocessed_24h" in preprocessed_path.name:
        candidates.append(preprocessed_path.parent.parent / "icu_ecgs_24h" / "original")
        candidates.append(preprocessed_path.parent / "icu_ecgs_24h_P1")
    
    # Pattern 5: Generic fallback
    candidates.append(preprocessed_path.parent / "original")
    candidates.append(preprocessed_path.parent.parent / "original")
    
    # Check each candidate
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Check if it contains .hea files
            hea_files = list(candidate.rglob("*.hea"))
            if len(hea_files) > 0:
                print(f"Auto-detected original data directory: {candidate}")
                print(f"  Found {len(hea_files):,} .hea files")
                return candidate
    
    return None


def get_timestamp_mapping_path(data_dir: str) -> Path:
    """Get standard path for timestamp mapping CSV file.
    
    Args:
        data_dir: Path to data directory (used to generate unique mapping filename).
                  Examples: 
                  - data/icu_ecgs_24h/P1 -> timestamps_mapping_P1.csv
                  - preprocessed_24h_1 -> timestamps_mapping_24h-1.csv
    
    Returns:
        Path to timestamp mapping CSV file in data/labeling/labels_csv/
    """
    data_path = Path(data_dir)

    # Create a stable identifier from the data directory name.
    # Important: for the 24h datasets we standardize to the user-facing naming:
    #   timestamps_mapping_24h_P{1|2|3}.csv
    dataset_name = data_path.name

    # Case A: preprocessed_24h_1 -> 24h_P1 (and similarly for _2/_3)
    if dataset_name.startswith("preprocessed_24h_"):
        suffix = dataset_name.replace("preprocessed_24h_", "")
        if suffix.isdigit():
            dataset_name = f"24h_P{suffix}"

    # Case B: icu_ecgs_24h_P1 -> 24h_P1
    if dataset_name.startswith("icu_ecgs_24h_P"):
        dataset_name = dataset_name.replace("icu_ecgs_24h_", "")

    # Case C: data/icu_ecgs_24h/P1 -> 24h_P1
    if dataset_name in {"P1", "P2", "P3"} and data_path.parent.name == "icu_ecgs_24h":
        dataset_name = f"24h_{dataset_name}"

    # Fallback: keep directory name (but normalize)
    dataset_name = dataset_name.replace("-", "_")
    mapping_filename = f"timestamps_mapping_{dataset_name}.csv"
    
    # Save in data/labeling/labels_csv/ directory at project root
    project_root = Path(__file__).parent.parent.parent.parent
    mapping_path = project_root / "data" / "labeling" / "labels_csv" / mapping_filename
    
    return mapping_path

