"""
Copy ECGs that were recorded **during** an ICU stay into a target directory.

Refined workflow:
1) Load ICU stays (subject_id, hadm_id, stay_id, intime, outtime)
2) Load ECG index/metadata (subject_id, study_id, waveform_path, ecg_time, optional hadm_id)
3) Keep only ECGs whose subject_id matches and ecg_time lies within an ICU stay
   (if hadm_id is available, require it to match the stay)
4) Report counts (stays, subjects, ECGs after filter, ECGs per stay distribution)
5) Copy only matched .hea/.dat pairs, preserving directory structure
"""

import sys
import os
from pathlib import Path
from typing import Set, Dict, List, Optional
import shutil

import pandas as pd

ECG_TIME_COLUMNS = ["ecg_time", "charttime", "record_time"]

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def load_icustays(icustays_path: str) -> pd.DataFrame:
    """Load icustays.csv or Excel file with required ICU stay columns."""
    print(f"Loading ICU stays from: {icustays_path}")
    
    icustays_path_obj = Path(icustays_path)
    if not icustays_path_obj.exists():
        raise FileNotFoundError(f"ICU stays file not found: {icustays_path}")
    
    # Try to load as CSV first, then Excel
    try:
        if icustays_path_obj.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(icustays_path)
        else:
            df = pd.read_csv(icustays_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load ICU stays file: {e}")
    
    print(f"  Loaded {len(df)} ICU stay records")
    print(f"  Columns: {list(df.columns)}")
    
    required_cols = {'subject_id', 'hadm_id', 'intime', 'outtime'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in ICU file: {missing}")
    
    # Normalize dtypes
    df = df.copy()
    df['subject_id'] = df['subject_id'].astype(int)
    if 'stay_id' in df.columns:
        df['stay_id'] = df['stay_id'].astype(int)
    df['hadm_id'] = df['hadm_id'].astype('Int64')
    df['intime'] = pd.to_datetime(df['intime'], utc=True, errors='coerce')
    df['outtime'] = pd.to_datetime(df['outtime'], utc=True, errors='coerce')
    df = df.dropna(subset=['intime', 'outtime'])
    
    print(f"  Unique subject_ids: {df['subject_id'].nunique()}")
    print(f"  Unique stays: {df['stay_id'].nunique() if 'stay_id' in df.columns else 'n/a'}")
    return df


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
    
    return df


def extract_subject_id_from_path(base_path: str) -> str:
    """Extract subject_id from ECG file path (longest p{ID} segment)."""
    path_parts = Path(base_path).parts
    subject_ids = []
    for part in path_parts:
        if part.startswith('p') and len(part) > 1 and part[1:].isdigit():
            subject_ids.append(part[1:])
    
    if not subject_ids:
        raise ValueError(f"Could not extract subject_id from path: {base_path}")
    
    return max(subject_ids, key=len)


def normalize_relative_path(waveform_path: str, source_dir: Path) -> Path:
    """
    Normalize waveform_path relative to source_dir.
    Handles cases where source_dir already points to 'files' and waveform_path starts with 'files/'.
    """
    rel_path = Path(waveform_path)
    if rel_path.is_absolute():
        # Make absolute path relative to source_dir if possible
        try:
            rel_path = rel_path.relative_to(source_dir)
        except ValueError:
            rel_path = rel_path
    rel_parts = rel_path.parts
    if source_dir.name == 'files' and len(rel_parts) > 0 and rel_parts[0] == 'files':
        rel_path = Path(*rel_parts[1:])
    return rel_path


def match_ecgs_to_icustays(
    icu_df: pd.DataFrame,
    ecg_df: pd.DataFrame,
    precision_mode: str = 'strict'
) -> pd.DataFrame:
    """
    Return ECG rows that fall within an ICU stay window, with UNIQUE stay assignment per ECG.
    
    Args:
        precision_mode: 'strict' (maximize precision) or 'balanced' (allow ambiguous matches)
    
    Strict mode rules:
    - subject_id must match
    - ecg_time within [intime, outtime]
    - If ecg.hadm_id exists and is not NA: require exact match with icu.hadm_id
    - If ecg.hadm_id is NA: only keep if ECG matches EXACTLY ONE ICU stay (drop if 0 or >1 matches)
    
    Balanced mode rules:
    - subject_id must match
    - ecg_time within [intime, outtime]
    - hadm_id match if both present, otherwise allow time-window match
    - Resolve multiple matches by selecting closest stay
    
    Returns:
        DataFrame with matched ECGs, including stay_id, intime, outtime columns.
        Each ECG (waveform_path) appears exactly once.
        Also stores reporting counters in matched.attrs dictionary.
    """
    icu = icu_df[['subject_id', 'hadm_id', 'intime', 'outtime']].copy()
    if 'stay_id' in icu_df.columns:
        icu['stay_id'] = icu_df['stay_id']
    icu = icu.rename(columns={'hadm_id': 'icu_hadm_id'})
    
    ecg = ecg_df.copy()
    ecg_has_hadm = 'hadm_id' in ecg.columns
    if ecg_has_hadm:
        ecg['ecg_hadm_id'] = ecg['hadm_id']
    else:
        ecg['ecg_hadm_id'] = pd.Series(pd.NA, index=ecg.index)
    
    # Merge on subject_id to get all potential matches
    merged = ecg.merge(icu, on='subject_id', how='inner', suffixes=('', '_icu'))
    
    # Filter by time window first
    time_mask = (merged['ecg_time'] >= merged['intime']) & (merged['ecg_time'] <= merged['outtime'])
    candidates_after_time = merged[time_mask].copy()
    candidates_after_time = candidates_after_time.dropna(subset=['intime', 'outtime'])
    
    # Initialize reporting counters
    stats = {
        'total_candidates_after_merge': len(merged),
        'candidates_after_time_filter': len(candidates_after_time),
        'kept_strict_hadm_match': 0,
        'kept_unique_time_match': 0,
        'dropped_hadm_mismatch': 0,
        'dropped_hadm_missing_ambiguous': 0,
        'dropped_hadm_missing_no_match': 0,
        'waveform_paths_with_multiple_stays': 0
    }
    
    if len(candidates_after_time) == 0:
        matched = pd.DataFrame()
        matched.attrs = stats
        return matched
    
    if precision_mode == 'strict':
        # STRICT MODE: Maximize precision
        
        # Split into two groups: ECGs with hadm_id and ECGs without
        ecg_has_hadm_mask = candidates_after_time['ecg_hadm_id'].notna()
        ecgs_with_hadm = candidates_after_time[ecg_has_hadm_mask].copy()
        ecgs_without_hadm = candidates_after_time[~ecg_has_hadm_mask].copy()
        
        # Group 1: ECGs with hadm_id - require exact match
        if len(ecgs_with_hadm) > 0:
            hadm_match_mask = (ecgs_with_hadm['ecg_hadm_id'] == ecgs_with_hadm['icu_hadm_id'])
            matched_with_hadm = ecgs_with_hadm[hadm_match_mask].copy()
            dropped_hadm_mismatch = ecgs_with_hadm[~hadm_match_mask].copy()
            
            stats['kept_strict_hadm_match'] = len(matched_with_hadm)
            stats['dropped_hadm_mismatch'] = len(dropped_hadm_mismatch)
        else:
            matched_with_hadm = pd.DataFrame()
        
        # Group 2: ECGs without hadm_id - only keep if exactly ONE match
        matched_without_hadm = pd.DataFrame()
        if len(ecgs_without_hadm) > 0:
            # Count matches per waveform_path
            match_counts = ecgs_without_hadm.groupby('waveform_path').size()
            stats['waveform_paths_with_multiple_stays'] = (match_counts > 1).sum()
            
            # Keep only waveform_paths with exactly 1 match
            unique_matches = match_counts[match_counts == 1].index
            matched_without_hadm = ecgs_without_hadm[ecgs_without_hadm['waveform_path'].isin(unique_matches)].copy()
            
            # Count dropped ambiguous and no-match cases
            ambiguous_paths = match_counts[match_counts > 1].index
            no_match_paths = set(ecgs_without_hadm['waveform_path'].unique()) - set(unique_matches) - set(ambiguous_paths)
            
            stats['kept_unique_time_match'] = len(matched_without_hadm)
            stats['dropped_hadm_missing_ambiguous'] = len(ambiguous_paths)
            stats['dropped_hadm_missing_no_match'] = len(no_match_paths)
        
        # Combine results
        if len(matched_with_hadm) > 0 and len(matched_without_hadm) > 0:
            matched = pd.concat([matched_with_hadm, matched_without_hadm], ignore_index=True)
        elif len(matched_with_hadm) > 0:
            matched = matched_with_hadm
        elif len(matched_without_hadm) > 0:
            matched = matched_without_hadm
        else:
            matched = pd.DataFrame()
        
        # Ensure waveform_path uniqueness (should already be unique, but double-check)
        matched = matched.drop_duplicates(subset=['waveform_path'])
        
    else:
        # BALANCED MODE: Current behavior (allow ambiguous matches, resolve by ranking)
        hadm_mask = candidates_after_time['ecg_hadm_id'].isna() | candidates_after_time['icu_hadm_id'].isna() | (candidates_after_time['ecg_hadm_id'] == candidates_after_time['icu_hadm_id'])
        matched = candidates_after_time[hadm_mask].copy()
        
        if len(matched) > 0:
            # Count ECGs with multiple matching stays (before resolution)
            multi_match_counts = matched.groupby('waveform_path').size()
            stats['waveform_paths_with_multiple_stays'] = (multi_match_counts > 1).sum()
            
            # Resolve multiple matches: select stay with smallest positive time difference
            matched['time_diff'] = (matched['ecg_time'] - matched['intime']).dt.total_seconds()
            matched['time_diff_positive'] = matched['time_diff'].where(matched['time_diff'] >= 0, pd.NA)
            
            matched['rank_positive'] = matched.groupby('waveform_path')['time_diff_positive'].rank(method='min', na_option='keep')
            matched['rank_absolute'] = matched.groupby('waveform_path')['time_diff'].abs().rank(method='min')
            matched['rank'] = matched['rank_positive'].fillna(matched['rank_absolute'])
            
            matched = matched[matched['rank'] == 1].copy()
            matched = matched.drop(columns=['time_diff', 'time_diff_positive', 'rank', 'rank_positive', 'rank_absolute'])
        
        # Count kept matches
        if len(matched) > 0:
            with_hadm = matched['ecg_hadm_id'].notna().sum()
            without_hadm = (matched['ecg_hadm_id'].isna()).sum()
            stats['kept_strict_hadm_match'] = with_hadm
            stats['kept_unique_time_match'] = without_hadm
    
    # Clean up temporary columns
    if 'ecg_hadm_id' in matched.columns:
        matched = matched.drop(columns=['ecg_hadm_id'])
    if 'icu_hadm_id' in matched.columns:
        matched = matched.drop(columns=['icu_hadm_id'])
    
    # Store reporting stats
    matched.attrs = stats
    
    return matched


def filter_one_ecg_per_stay(
    matched_ecgs: pd.DataFrame,
    method: str = 'first'
) -> pd.DataFrame:
    """
    Filter to exactly one ECG per stay.
    
    Args:
        matched_ecgs: DataFrame with stay_id and ecg_time columns
        method: 'first' (first ECG after intime), 'last' (last ECG before outtime),
                'closest_to_intime' (ECG closest to intime)
    
    Returns:
        Filtered DataFrame with one ECG per stay
    """
    if 'stay_id' not in matched_ecgs.columns:
        print("Warning: stay_id not available, cannot filter to one ECG per stay")
        return matched_ecgs
    
    if method == 'first':
        # First ECG after intime (or closest if all before)
        matched_ecgs = matched_ecgs.sort_values(['stay_id', 'ecg_time'])
        matched_ecgs = matched_ecgs.groupby('stay_id').first().reset_index()
    elif method == 'last':
        # Last ECG before outtime (or closest if all after)
        matched_ecgs = matched_ecgs.sort_values(['stay_id', 'ecg_time'])
        matched_ecgs = matched_ecgs.groupby('stay_id').last().reset_index()
    elif method == 'closest_to_intime':
        # ECG with smallest absolute difference from intime
        matched_ecgs['time_from_intime'] = (matched_ecgs['ecg_time'] - matched_ecgs['intime']).dt.total_seconds().abs()
        matched_ecgs = matched_ecgs.sort_values(['stay_id', 'time_from_intime'])
        matched_ecgs = matched_ecgs.groupby('stay_id').first().reset_index()
        matched_ecgs = matched_ecgs.drop(columns=['time_from_intime'])
    else:
        raise ValueError(f"Unknown method: {method}. Use 'first', 'last', or 'closest_to_intime'")
    
    return matched_ecgs


def copy_ecg_files(
    ecg_df: pd.DataFrame,
    source_dir: Path,
    target_dir: Path,
    progress_interval: int = 1000
) -> Dict[str, int]:
    """
    Copy ECG files (.hea and .dat) to target directory maintaining structure.
    
    Returns:
        Dictionary with 'copied' and 'errors' counts
    """
    print(f"\nCopying {len(ecg_df)} ECG file pairs to: {target_dir}")
    print("This may take time depending on disk speed...")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    errors = 0
    missing_files = 0
    
    for i, row in enumerate(ecg_df.itertuples(index=False), 1):
        if i % progress_interval == 0:
            print(f"  Progress: {i:,}/{len(ecg_df):,} ({100*i/len(ecg_df):.1f}%), "
                  f"Copied: {copied:,}, Errors: {errors}")
        
        rel_path = normalize_relative_path(row.waveform_path, source_dir)
        source_base = (source_dir / rel_path).with_suffix('')
        target_base = (target_dir / rel_path).with_suffix('')
        
        source_hea = source_base.with_suffix('.hea')
        source_dat = source_base.with_suffix('.dat')
        target_hea = target_base.with_suffix('.hea')
        target_dat = target_base.with_suffix('.dat')
        
        # Verify source files exist
        if not source_hea.exists():
            errors += 1
            missing_files += 1
            if errors <= 5:
                print(f"  Error: Source .hea file not found: {source_hea}")
            continue
        if not source_dat.exists():
            errors += 1
            missing_files += 1
            if errors <= 5:
                print(f"  Error: Source .dat file not found: {source_dat}")
            continue
        
        try:
            target_base.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_hea, target_hea)
            shutil.copy2(source_dat, target_dat)
            copied += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error copying {rel_path}: {e}")
    
    print(f"\nCopy complete!")
    print(f"  Successfully copied: {copied:,} ECG pairs")
    print(f"  Errors: {errors}")
    print(f"  Missing files: {missing_files}")
    print(f"  Target directory: {target_dir}")
    
    return {'copied': copied, 'errors': errors, 'missing_files': missing_files}


def main():
    """Filter ECGs to ICU stay windows and copy matched records."""
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Copy ECGs recorded during ICU stays")
    parser.add_argument(
        '--icustays',
        type=str,
        default=r"D:\MA\data\mimic-iv-ecg\icustays.csv",
        help='Path to icustays.csv or Excel file (must contain subject_id column)'
    )
    parser.add_argument(
        '--source',
        type=str,
        default=r"D:\MA\data\mimic-iv-ecg\ecgs_all",
        help='Source directory root with all ECGs (directory containing `files/`)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=r"D:\MA\data\mimic-iv-ecg\icustay_ecgs",
        help='Target directory root for ICU ECGs (will mirror relative paths)'
    )
    parser.add_argument(
        '--ecg_index',
        type=str,
        default=r"D:\MA\data\mimic-iv-ecg\ecgs_all\waveform_note_links.csv",
        help='ECG metadata file containing subject_id, study_id, waveform_path, and time column'
    )
    parser.add_argument(
        '--one_ecg_per_stay',
        type=str,
        choices=['first', 'last', 'closest_to_intime'],
        default=None,
        help='If set, keep only one ECG per stay: first (after intime), last (before outtime), or closest_to_intime'
    )
    parser.add_argument(
        '--precision_mode',
        type=str,
        choices=['strict', 'balanced'],
        default='strict',
        help='Matching precision mode: strict (maximize precision, drop ambiguous matches) or balanced (allow ambiguous matches, resolve by ranking)'
    )
    
    args = parser.parse_args()
    
    # Paths
    icustays_path = args.icustays
    source_ecg_dir = Path(args.source)
    target_ecg_dir = Path(args.target)
    ecg_index_path = args.ecg_index
    
    # Verify source directory exists
    if not source_ecg_dir.exists():
        print(f"ERROR: Source directory not found: {source_ecg_dir}")
        return
    
    print("=" * 70)
    print("ICU ECG Data Split - ECGs during ICU stays")
    print("=" * 70)
    print(f"ICU CSV path: {icustays_path}")
    print(f"ECG index: {ecg_index_path}")
    print(f"Source root: {source_ecg_dir}")
    print(f"Target root: {target_ecg_dir}")
    print(f"Precision mode: {args.precision_mode}")
    if args.one_ecg_per_stay:
        print(f"Filter mode: one ECG per stay ({args.one_ecg_per_stay})")
    print("=" * 70)
    
    # Step 1: Load ICU stays
    icu_df = load_icustays(icustays_path)
    print(f"\nICU stays: {len(icu_df):,}")
    print(f"Unique subjects: {icu_df['subject_id'].nunique():,}")
    
    # Step 2: Load ECG metadata
    ecg_df = load_ecg_index(ecg_index_path)
    print(f"ECGs in index: {len(ecg_df):,}")
    print(f"ECG subjects: {ecg_df['subject_id'].nunique():,}")
    
    # Step 3: Match ECGs to ICU stays by time (and hadm_id when available)
    # Each ECG is assigned to exactly one ICU stay
    matched_ecgs = match_ecgs_to_icustays(icu_df, ecg_df, precision_mode=args.precision_mode)
    
    if len(matched_ecgs) == 0:
        print("\nNo ECGs matched ICU stay windows. Exiting.")
        return
    
    # Extract reporting stats
    stats = matched_ecgs.attrs if hasattr(matched_ecgs, 'attrs') else {}
    
    # Report matching statistics
    print("\n" + "=" * 70)
    print("MATCHING STATISTICS")
    print("=" * 70)
    print(f"Total candidates after merge: {stats.get('total_candidates_after_merge', 0):,}")
    print(f"Candidates after time filter: {stats.get('candidates_after_time_filter', 0):,}")
    print(f"\nKept via strict hadm_id match: {stats.get('kept_strict_hadm_match', 0):,}")
    print(f"Kept via unique time-window match (hadm missing, unambiguous): {stats.get('kept_unique_time_match', 0):,}")
    print(f"\nDropped due to hadm_id mismatch: {stats.get('dropped_hadm_mismatch', 0):,}")
    print(f"Dropped due to hadm_id missing AND ambiguous (>1 ICU stay match): {stats.get('dropped_hadm_missing_ambiguous', 0):,}")
    print(f"Dropped due to hadm_id missing AND no time match: {stats.get('dropped_hadm_missing_no_match', 0):,}")
    print(f"Waveform paths with multiple ICU stay matches (before resolution): {stats.get('waveform_paths_with_multiple_stays', 0):,}")
    
    # Step 4: Optional filtering to one ECG per stay
    if args.one_ecg_per_stay:
        print(f"\nFiltering to one ECG per stay (method: {args.one_ecg_per_stay})...")
        before_count = len(matched_ecgs)
        matched_ecgs = filter_one_ecg_per_stay(matched_ecgs, method=args.one_ecg_per_stay)
        after_count = len(matched_ecgs)
        print(f"  Reduced from {before_count:,} to {after_count:,} ECGs")
    
    # Step 5: Validation and reporting
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    print(f"Matched ECGs: {len(matched_ecgs):,}")
    print(f"Matched subjects: {matched_ecgs['subject_id'].nunique():,}")
    
    if 'stay_id' in matched_ecgs.columns:
        stay_counts = matched_ecgs.groupby('stay_id').size()
        matched_stays = stay_counts.index.nunique()
        print(f"Matched stays: {matched_stays:,}")
        
        # ECGs per stay distribution
        print("\nECGs per stay distribution:")
        desc = stay_counts.describe()
        print(desc.to_string())
        print(f"\nPercentiles:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = stay_counts.quantile(p / 100.0)
            print(f"  {p}th percentile: {val:.1f} ECGs")
    else:
        print("Stay_id not available; reporting by subject only.")
    
    # Step 6: Copy files and report missing files
    copy_results = copy_ecg_files(matched_ecgs, source_ecg_dir, target_ecg_dir)
    
    print("\n" + "=" * 70)
    print("COPY SUMMARY")
    print("=" * 70)
    print(f"Successfully copied: {copy_results['copied']:,} ECG pairs")
    print(f"Errors (missing files): {copy_results['missing_files']:,}")
    print(f"Other errors: {copy_results['errors'] - copy_results['missing_files']:,}")
    
    # Step 7: Sanity check - print 5 random matched rows
    print("\n" + "=" * 70)
    print("SANITY CHECK - 5 Random Matched ECGs")
    print("=" * 70)
    if len(matched_ecgs) > 0:
        sample_size = min(5, len(matched_ecgs))
        sample_indices = random.sample(range(len(matched_ecgs)), sample_size)
        sample_df = matched_ecgs.iloc[sample_indices].copy()
        
        # Select relevant columns for display
        display_cols = ['subject_id', 'ecg_time', 'waveform_path']
        if 'stay_id' in sample_df.columns:
            display_cols.insert(1, 'stay_id')
        if 'intime' in sample_df.columns:
            display_cols.insert(-1, 'intime')
        if 'outtime' in sample_df.columns:
            display_cols.insert(-1, 'outtime')
        
        available_cols = [c for c in display_cols if c in sample_df.columns]
        print(sample_df[available_cols].to_string(index=False))
    else:
        print("No matched ECGs to display.")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()