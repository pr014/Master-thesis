"""
Copy ECGs recorded within the first 24 hours of ICU stay.

Workflow:
1) Load ICU stays (subject_id, stay_id, intime, outtime) from icustays.csv
2) Load ECG metadata (subject_id, study_id, waveform_path, ecg_time) from waveform_note_links.csv
3) Find all ECG files in source directory (icustay_ecgs\files)
4) For each ECG file:
   - Extract subject_id from path
   - Find ecg_time from CSV (via waveform_path match)
   - Match to ICU stay: subject_id + ecg_time within [intime, outtime]
   - Filter to first 24h: intime <= ecg_time <= (intime + 24 hours)
5) Copy matched ECGs preserving directory structure

Uses CSV timestamps (waveform_note_links.csv) for reliable matching.
Only processes ECGs that exist in source directory.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import shutil
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.labeling.icu_los_labels import load_icustays

# ECG time column names to try
ECG_TIME_COLUMNS = ["ecg_time", "charttime", "record_time"]


def load_ecg_index(ecg_index_path: str) -> pd.DataFrame:
    """
    Load ECG metadata with subject_id, study_id, waveform_path, ecg_time, optional hadm_id.
    Accepts CSV or Excel. Tries known time columns: ecg_time, charttime, record_time.
    """
    print(f"Loading ECG index from: {ecg_index_path}")
    ecg_path = Path(ecg_index_path)
    if not ecg_path.exists():
        raise FileNotFoundError(f"ECG index file not found: {ecg_index_path}")
    
    try:
        if ecg_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(ecg_index_path)
        else:
            df = pd.read_csv(ecg_index_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load ECG index: {e}")
    
    print(f"  Loaded {len(df)} ECG metadata rows")
    print(f"  Columns: {list(df.columns)}")
    
    required_cols = {'subject_id', 'study_id', 'waveform_path'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in ECG index: {missing}")
    
    # Pick time column
    time_col: Optional[str] = None
    for cand in ECG_TIME_COLUMNS:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        raise ValueError(f"No ECG time column found. Tried: {ECG_TIME_COLUMNS}")
    
    df = df.copy()
    df['subject_id'] = df['subject_id'].astype(int)
    df['study_id'] = df['study_id'].astype(int)
    if 'hadm_id' in df.columns:
        df['hadm_id'] = df['hadm_id'].astype('Int64')
    df['ecg_time'] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
    df = df.dropna(subset=['ecg_time'])
    
    # Normalize waveform_path to POSIX-style separators for consistency
    df['waveform_path'] = df['waveform_path'].astype(str).str.replace('\\', '/')
    
    print(f"  ECGs with valid timestamps: {len(df):,}")
    return df


def find_ecg_files(source_dir: Path) -> List[Path]:
    """Find all ECG .hea files (with corresponding .dat) in source directory.
    
    Args:
        source_dir: Root directory containing ECG files.
    
    Returns:
        List of base paths (without .hea/.dat extension).
    """
    print(f"Scanning for ECG files in: {source_dir}")
    
    # Find all .hea files
    hea_files = list(source_dir.rglob("*.hea"))
    base_paths = []
    
    for hea_path in hea_files:
        base_path = hea_path.with_suffix("")
        dat_path = base_path.with_suffix(".dat")
        
        # Only include if both .hea and .dat exist
        if dat_path.exists():
            base_paths.append(base_path)
    
    print(f"  Found {len(base_paths):,} ECG file pairs (.hea + .dat)")
    return base_paths


def extract_subject_id_from_path(base_path: Path) -> int:
    """Extract subject_id from ECG file path (longest p{ID} segment)."""
    from src.data.ecg import extract_subject_id_from_path as extract_subject_id
    return extract_subject_id(str(base_path))


def match_file_to_csv(base_path: Path, ecg_df: pd.DataFrame, source_dir: Path) -> Optional[pd.Series]:
    """Match ECG file to CSV entry by waveform_path.
    
    Returns:
        Series from ecg_df if match found, None otherwise.
    """
    # Get relative path from source_dir
    try:
        rel_path = base_path.relative_to(source_dir)
    except ValueError:
        return None
    
    # Normalize path (POSIX style, remove .hea/.dat)
    rel_path_str = str(rel_path).replace('\\', '/')
    if rel_path_str.endswith('.hea') or rel_path_str.endswith('.dat'):
        rel_path_str = rel_path_str.rsplit('.', 1)[0]
    
    # Try to match in CSV
    # waveform_path might be absolute or relative, try both
    matches = ecg_df[
        (ecg_df['waveform_path'].str.replace('\\', '/').str.endswith(rel_path_str, na=False)) |
        (ecg_df['waveform_path'].str.replace('\\', '/') == rel_path_str)
    ]
    
    if len(matches) == 0:
        return None
    
    # If multiple matches, pick first (should be unique)
    return matches.iloc[0]


def filter_24h_ecgs_from_files(
    ecg_files: List[Path],
    icu_df: pd.DataFrame,
    ecg_df: pd.DataFrame,
    source_dir: Path,
) -> List[Dict]:
    """Filter ECGs from files to those within first 24h of ICU stay.
    
    Args:
        ecg_files: List of base paths (without extension).
        icu_df: ICU stays DataFrame.
        ecg_df: ECG metadata DataFrame from CSV.
        source_dir: Source directory root.
    
    Returns:
        List of dicts with matched ECG info.
    """
    print(f"\nFiltering ECGs within first 24h of ICU stay...")
    
    # Prepare ICU data
    icu = icu_df[['subject_id', 'intime', 'outtime']].copy()
    if 'stay_id' in icu_df.columns:
        icu['stay_id'] = icu_df['stay_id']
    
    # Ensure intime/outtime are timezone-aware (UTC)
    if icu['intime'].dt.tz is None:
        icu['intime'] = pd.to_datetime(icu['intime'], utc=True)
    if icu['outtime'].dt.tz is None:
        icu['outtime'] = pd.to_datetime(icu['outtime'], utc=True)
    
    # Build lookup: subject_id -> list of stays
    subject_to_stays = {}
    for _, row in icu.iterrows():
        subject_id = row['subject_id']
        if subject_id not in subject_to_stays:
            subject_to_stays[subject_id] = []
        subject_to_stays[subject_id].append(row)
    
    matched_ecgs = []
    unmatched_no_csv = 0
    unmatched_no_stay = 0
    outside_24h = 0
    
    for idx, base_path in enumerate(ecg_files):
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,}/{len(ecg_files):,} ECGs...")
        
        try:
            # Extract subject_id from path
            try:
                subject_id = extract_subject_id_from_path(base_path)
            except ValueError:
                unmatched_no_csv += 1
                continue
            
            # Match file to CSV entry
            csv_row = match_file_to_csv(base_path, ecg_df, source_dir)
            if csv_row is None:
                unmatched_no_csv += 1
                continue
            
            ecg_time = csv_row['ecg_time']
            if pd.isna(ecg_time):
                unmatched_no_csv += 1
                continue
            
            # Find matching ICU stay
            if subject_id not in subject_to_stays:
                unmatched_no_stay += 1
                continue
            
            # Check all stays for this subject
            best_match = None
            matched_outside_24h = False
            for stay_row in subject_to_stays[subject_id]:
                intime = stay_row['intime']
                outtime = stay_row['outtime']
                
                # Check if ecg_time is within stay window
                if intime <= ecg_time <= outtime:
                    # Check if within first 24h
                    intime_24h = intime + timedelta(hours=24)
                    if intime <= ecg_time <= intime_24h:
                        # This is a match within 24h
                        if best_match is None:
                            best_match = stay_row
                        else:
                            # If multiple matches, pick closest to intime
                            current_diff = abs((ecg_time - intime).total_seconds())
                            best_diff = abs((ecg_time - best_match['intime']).total_seconds())
                            if current_diff < best_diff:
                                best_match = stay_row
                    else:
                        # Matched stay but outside 24h (don't break, might have other stays)
                        matched_outside_24h = True
            
            if best_match is not None:
                rel_path = base_path.relative_to(source_dir)
                matched_ecgs.append({
                    'base_path': str(base_path),
                    'rel_path': str(rel_path),
                    'stay_id': best_match['stay_id'],
                    'subject_id': subject_id,
                    'ecg_time': ecg_time,
                    'intime': best_match['intime'],
                })
            elif matched_outside_24h:
                # Had matching stay but outside 24h
                outside_24h += 1
            elif subject_id in subject_to_stays:
                # Had stays but none matched (ecg_time outside all stay windows)
                unmatched_no_stay += 1
                
        except Exception as e:
            unmatched_no_csv += 1
            if unmatched_no_csv <= 5:
                print(f"  Error processing {base_path}: {e}")
    
    print(f"\nFiltering results:")
    print(f"  Total ECGs processed: {len(ecg_files):,}")
    print(f"  Matched within 24h: {len(matched_ecgs):,}")
    print(f"  Unmatched (no CSV entry): {unmatched_no_csv:,}")
    print(f"  Unmatched (no ICU stay): {unmatched_no_stay:,}")
    print(f"  Matched but outside 24h: {outside_24h:,}")
    
    return matched_ecgs


def copy_ecg_files(
    matched_ecgs: List[Dict],
    source_dir: Path,
    target_dir: Path,
) -> Dict:
    """Copy ECG files preserving directory structure.
    
    Args:
        matched_ecgs: List of dicts with base_path and rel_path.
        source_dir: Source directory root.
        target_dir: Target directory root.
    
    Returns:
        Dictionary with copy statistics.
    """
    print(f"\nCopying ECG files...")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    
    copied = 0
    errors = 0
    missing_files = 0
    
    for idx, ecg_info in enumerate(matched_ecgs):
        if (idx + 1) % 1000 == 0:
            print(f"  Copied {idx + 1:,}/{len(matched_ecgs):,} ECGs...")
        
        base_path = Path(ecg_info['base_path'])
        rel_path = Path(ecg_info['rel_path'])
        
        # Source files
        hea_source = base_path.with_suffix('.hea')
        dat_source = base_path.with_suffix('.dat')
        
        # Check source files exist
        if not hea_source.exists() or not dat_source.exists():
            missing_files += 1
            if missing_files <= 5:
                print(f"  Missing: {rel_path}")
            continue
        
        # Target paths
        target_base = target_dir / rel_path
        hea_target = target_base.with_suffix('.hea')
        dat_target = target_base.with_suffix('.dat')
        
        try:
            # Create target directory
            hea_target.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy both files
            shutil.copy2(hea_source, hea_target)
            shutil.copy2(dat_source, dat_target)
            
            copied += 1
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error copying {rel_path}: {e}")
    
    print(f"\nCopy complete!")
    print(f"  Successfully copied: {copied:,} ECG pairs")
    print(f"  Errors: {errors:,}")
    print(f"  Missing files: {missing_files:,}")
    print(f"  Target directory: {target_dir}")
    
    return {'copied': copied, 'errors': errors, 'missing_files': missing_files}


def print_statistics(matched_ecgs: List[Dict]):
    """Print statistics about matched ECGs."""
    if len(matched_ecgs) == 0:
        print("\nNo ECGs matched - no statistics available.")
        return
    
    import numpy as np
    from collections import Counter
    
    df = pd.DataFrame(matched_ecgs)
    
    print("\n" + "=" * 70)
    print("24h ECG Dataset Statistics")
    print("=" * 70)
    
    print(f"\nTotal ECGs within first 24h: {len(df):,}")
    print(f"Unique stays: {df['stay_id'].nunique():,}")
    print(f"Unique subjects: {df['subject_id'].nunique():,}")
    
    # ECGs per stay
    stay_counts = df.groupby('stay_id').size()
    print(f"\nECGs per stay:")
    print(f"  Mean: {stay_counts.mean():.2f}")
    print(f"  Median: {stay_counts.median():.1f}")
    print(f"  Min: {stay_counts.min()}")
    print(f"  Max: {stay_counts.max()}")
    
    # Distribution
    count_dist = Counter(stay_counts.values)
    print(f"\nDistribution:")
    print(f"  1 ECG/stay: {count_dist[1]:,} stays ({100*count_dist[1]/len(stay_counts):.1f}%)")
    print(f"  2 ECGs/stay: {count_dist[2]:,} stays ({100*count_dist[2]/len(stay_counts):.1f}%)")
    print(f"  3 ECGs/stay: {count_dist[3]:,} stays ({100*count_dist[3]/len(stay_counts):.1f}%)")
    print(f"  4+ ECGs/stay: {sum(count_dist[i] for i in count_dist if i >= 4):,} stays ({100*sum(count_dist[i] for i in count_dist if i >= 4)/len(stay_counts):.1f}%)")
    
    # Time from intime
    df['hours_from_intime'] = (df['ecg_time'] - df['intime']).dt.total_seconds() / 3600
    hours_from_intime = df['hours_from_intime']
    
    # VALIDATION: Ensure all ECGs are within 24h window
    invalid_before = (hours_from_intime < 0).sum()
    invalid_after = (hours_from_intime > 24).sum()
    
    if invalid_before > 0 or invalid_after > 0:
        print(f"\n[WARNING] Validation failed!")
        print(f"  ECGs before intime: {invalid_before}")
        print(f"  ECGs after 24h: {invalid_after}")
        print(f"  This should not happen - check filtering logic!")
    else:
        print(f"\n[OK] Validation passed: All ECGs within first 24h")
    
    print(f"\nTime from ICU admission (intime):")
    print(f"  Mean: {hours_from_intime.mean():.2f} hours")
    print(f"  Median: {hours_from_intime.median():.2f} hours")
    print(f"  Min: {hours_from_intime.min():.2f} hours")
    print(f"  Max: {hours_from_intime.max():.2f} hours")
    
    # Additional validation: Max should be <= 24h
    if hours_from_intime.max() > 24.0:
        print(f"\n[ERROR] Max time exceeds 24h! This should not happen.")
    elif hours_from_intime.max() <= 24.0 and hours_from_intime.min() >= 0:
        print(f"\n[OK] All ECGs confirmed within [0, 24] hours window")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Copy ECGs recorded within first 24h of ICU stay"
    )
    parser.add_argument(
        '--icustays',
        type=str,
        default=r"D:\MA\data\mimic-iv-ecg\icustays.csv",
        help='Path to icustays.csv'
    )
    parser.add_argument(
        '--ecg_index',
        type=str,
        default=r"D:\MA\data\mimic-iv-ecg\ecgs_all\waveform_note_links.csv",
        help='ECG metadata CSV (waveform_note_links.csv) with ecg_time'
    )
    parser.add_argument(
        '--source',
        type=str,
        default=r"D:\MA\data\mimic-iv-ecg\icustay_ecgs\files",
        help='Source directory containing ECG files'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=r"D:\MA\data\mimic-iv-ecg\icustay_ecgs_24h\files",
        help='Target directory for 24h ECGs (will preserve directory structure)'
    )
    
    args = parser.parse_args()
    
    # Paths
    icustays_path = Path(args.icustays)
    ecg_index_path = Path(args.ecg_index)
    source_dir = Path(args.source)
    target_dir = Path(args.target)
    
    # Verify paths
    if not icustays_path.exists():
        print(f"ERROR: ICU stays file not found: {icustays_path}")
        return
    
    if not ecg_index_path.exists():
        print(f"ERROR: ECG index file not found: {ecg_index_path}")
        return
    
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return
    
    print("=" * 70)
    print("24h ICU ECG Dataset Creation")
    print("=" * 70)
    print(f"ICU stays: {icustays_path}")
    print(f"ECG index: {ecg_index_path}")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print("=" * 70)
    
    # Step 1: Load ICU stays
    print("\nLoading ICU stays...")
    icustays_df = load_icustays(str(icustays_path))
    print(f"  Loaded {len(icustays_df):,} ICU stays")
    print(f"  Unique subjects: {icustays_df['subject_id'].nunique():,}")
    
    # Step 2: Load ECG metadata
    ecg_df = load_ecg_index(str(ecg_index_path))
    print(f"  ECGs in index: {len(ecg_df):,}")
    print(f"  ECG subjects: {ecg_df['subject_id'].nunique():,}")
    
    # Step 3: Find all ECG files in source directory
    ecg_files = find_ecg_files(source_dir)
    
    if len(ecg_files) == 0:
        print("\nNo ECG files found. Exiting.")
        return
    
    # Step 4: Filter to 24h ECGs (matches files to CSV and ICU stays)
    matched_24h = filter_24h_ecgs_from_files(ecg_files, icustays_df, ecg_df, source_dir)
    
    if len(matched_24h) == 0:
        print("\nNo ECGs within first 24h. Exiting.")
        return
    
    # Step 5: Print statistics
    print_statistics(matched_24h)
    
    # Step 6: Copy files
    copy_results = copy_ecg_files(matched_24h, source_dir, target_dir)
    
    print("\n" + "=" * 70)
    print("Dataset creation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()