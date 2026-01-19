"""
Filter ECG records to only include those with:
1. Available labels (from records_w_diag_icd10.csv)
2. Matching ICU stay (from icustays.csv)

The script matches records based on:
- subject_id
- hadm_id (hosp_hadm_id in labels -> hadm_id in ICU stays)
- ecg_time falls within ICU stay period (intime to outtime)
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Set, Tuple


def load_icu_stays(icustays_path: str) -> pd.DataFrame:
    """Load ICU stays and prepare for matching."""
    print(f"Loading ICU stays from {icustays_path}...")
    icu_df = pd.read_csv(icustays_path)
    
    # Convert time columns to datetime
    icu_df['intime'] = pd.to_datetime(icu_df['intime'])
    icu_df['outtime'] = pd.to_datetime(icu_df['outtime'])
    
    print(f"Loaded {len(icu_df)} ICU stays")
    print(f"Unique subjects: {icu_df['subject_id'].nunique()}")
    print(f"Unique hadm_ids: {icu_df['hadm_id'].nunique()}")
    
    return icu_df


def filter_labels_with_icu(
    labels_path: str,
    icu_df: pd.DataFrame,
    chunk_size: int = 100000,
    require_time_match: bool = True,
    min_total_days: float = 0.0
) -> pd.DataFrame:
    """
    Filter labels to only include records with matching ICU stays.
    
    Filtering steps:
    1. First filter by subject_id: Patient must have been in ICU (subject_id in ICU stays)
    2. Then filter by hadm_id: hosp_hadm_id (from labels) must match hadm_id (from ICU stays)
    3. Optionally filter by time: ecg_time must fall within ICU stay period
    4. Optionally filter by minimum total ICU days: Only patients with >= min_total_days total ICU days
    
    Args:
        labels_path: Path to labels CSV file
        icu_df: DataFrame with ICU stays
        chunk_size: Size of chunks for processing large files
        require_time_match: If True, ECG time must be within ICU stay period
        min_total_days: Minimum total ICU days per patient (across all stays). Default 0.0 (no filter)
    """
    print(f"\nFiltering labels from {labels_path}...")
    print(f"Processing in chunks of {chunk_size} rows...")
    print(f"Time matching required: {require_time_match}")
    
    # Step 1: Create set of subject_ids that have ICU stays (primary filter)
    # If min_total_days is set, calculate total days per patient first
    if min_total_days > 0:
        patient_total_days = icu_df.groupby('subject_id')['los'].sum()
        eligible_patients = patient_total_days[patient_total_days >= min_total_days].index
        icu_subject_ids = set(eligible_patients)
        print(f"Patients with ICU stays: {icu_df['subject_id'].nunique():,}")
        print(f"Patients with >= {min_total_days} total ICU days: {len(icu_subject_ids):,}")
    else:
        icu_subject_ids = set(icu_df['subject_id'].unique())
        print(f"Patients with ICU stays: {len(icu_subject_ids):,}")
    
    # Step 2: Create a set of (subject_id, hadm_id) tuples from ICU stays for fast lookup
    icu_subject_hadm = set(zip(icu_df['subject_id'], icu_df['hadm_id']))
    
    # Step 3: Create a dictionary for time range lookup (if needed)
    # Key: (subject_id, hadm_id), Value: list of (intime, outtime) tuples
    icu_time_ranges = {}
    if require_time_match:
        for _, row in icu_df.iterrows():
            key = (row['subject_id'], row['hadm_id'])
            if key not in icu_time_ranges:
                icu_time_ranges[key] = []
            icu_time_ranges[key].append((row['intime'], row['outtime']))
    
    filtered_chunks = []
    total_processed = 0
    total_filtered_subject = 0
    total_filtered_hadm = 0
    total_filtered_time = 0
    
    # Process labels in chunks
    for chunk in pd.read_csv(labels_path, chunksize=chunk_size):
        total_processed += len(chunk)
        
        # Convert ecg_time to datetime
        chunk['ecg_time'] = pd.to_datetime(chunk['ecg_time'])
        
        # Filter Step 1: subject_id must be in ICU stays
        mask_has_subject = chunk['subject_id'].isin(icu_subject_ids)
        chunk_subject = chunk[mask_has_subject].copy()
        
        if len(chunk_subject) == 0:
            print(f"  Chunk: {total_processed:,} processed, 0 with subject_id in ICU")
            continue
        
        total_filtered_subject += len(chunk_subject)
        
        # Filter Step 2: must have hosp_hadm_id (not NaN) and match ICU stay
        chunk_with_hadm = chunk_subject[chunk_subject['hosp_hadm_id'].notna()].copy()
        
        if len(chunk_with_hadm) == 0:
            print(f"  Chunk: {total_processed:,} processed, {len(chunk_subject)} with subject_id, 0 with hadm_id")
            continue
        
        # Create matching key for (subject_id, hadm_id)
        chunk_with_hadm['match_key'] = list(
            zip(chunk_with_hadm['subject_id'], chunk_with_hadm['hosp_hadm_id'].astype(int))
        )
        
        # Filter: must have matching (subject_id, hadm_id) in ICU stays
        mask_has_icu = chunk_with_hadm['match_key'].isin(icu_subject_hadm)
        chunk_matched = chunk_with_hadm[mask_has_icu].copy()
        
        if len(chunk_matched) == 0:
            print(f"  Chunk: {total_processed:,} processed, {len(chunk_subject)} with subject_id, 0 matched hadm_id")
            continue
        
        total_filtered_hadm += len(chunk_matched)
        
        # Filter Step 3: ecg_time must fall within ICU stay period (if required)
        if require_time_match:
            def time_within_icu_stay(row):
                key = (row['subject_id'], int(row['hosp_hadm_id']))
                if key not in icu_time_ranges:
                    return False
                ecg_time = row['ecg_time']
                for intime, outtime in icu_time_ranges[key]:
                    if intime <= ecg_time <= outtime:
                        return True
                return False
            
            chunk_matched['within_icu'] = chunk_matched.apply(time_within_icu_stay, axis=1)
            chunk_final = chunk_matched[chunk_matched['within_icu']].copy()
            chunk_final = chunk_final.drop(columns=['within_icu'])
            total_filtered_time += len(chunk_final)
        else:
            chunk_final = chunk_matched.copy()
            total_filtered_time += len(chunk_final)
        
        # Drop helper columns
        chunk_final = chunk_final.drop(columns=['match_key'])
        
        filtered_chunks.append(chunk_final)
        
        print(f"  Chunk: {total_processed:,} processed -> {len(chunk_subject)} subject_id match -> {len(chunk_matched)} hadm_id match -> {len(chunk_final)} final")
    
    if not filtered_chunks:
        print("\nNo records matched the criteria!")
        return pd.DataFrame()
    
    # Combine all filtered chunks
    result = pd.concat(filtered_chunks, ignore_index=True)
    
    print(f"\nFiltering complete:")
    print(f"  Total processed: {total_processed:,}")
    print(f"  After subject_id filter: {total_filtered_subject:,}")
    print(f"  After hadm_id filter: {total_filtered_hadm:,}")
    print(f"  Final (after time filter): {len(result):,}")
    print(f"  Unique subjects: {result['subject_id'].nunique():,}")
    print(f"  Unique study_ids: {result['study_id'].nunique():,}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Filter ECG records to only include those with labels and ICU stays"
    )
    parser.add_argument(
        '--labels',
        type=str,
        default='data/labels/records_w_diag_icd10.csv',
        help='Path to labels CSV file'
    )
    parser.add_argument(
        '--icustays',
        type=str,
        default='data/icustay.csv/icustays.csv',
        help='Path to ICU stays CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/labels/records_w_diag_icd10_filtered_icu.csv',
        help='Output path for filtered labels CSV'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Chunk size for reading large labels file'
    )
    parser.add_argument(
        '--no-time-match',
        action='store_true',
        help='Do not require ECG time to be within ICU stay period (only match by subject_id and hadm_id)'
    )
    parser.add_argument(
        '--min-total-days',
        type=float,
        default=0.0,
        help='Minimum total ICU days per patient (across all stays). Default: 0.0 (no filter). Use 3.0 to filter for patients with >=3 days.'
    )
    args = parser.parse_args()
    
    # Load ICU stays
    icu_df = load_icu_stays(args.icustays)
    
    # Filter labels
    filtered_df = filter_labels_with_icu(
        args.labels, 
        icu_df, 
        chunk_size=args.chunk_size,
        require_time_match=not args.no_time_match,
        min_total_days=args.min_total_days
    )
    
    if len(filtered_df) == 0:
        print("\nNo records to save. Exiting.")
        return
    
    # Save filtered results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving filtered records to {output_path}...")
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved {len(filtered_df):,} records to {output_path}")
    
    # Also save a summary
    summary_path = output_path.with_suffix('.summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"ECG Records Filtered for ICU Stays\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Total records: {len(filtered_df):,}\n")
        f.write(f"Unique subjects: {filtered_df['subject_id'].nunique():,}\n")
        f.write(f"Unique study_ids: {filtered_df['study_id'].nunique():,}\n")
        f.write(f"Unique hadm_ids: {filtered_df['hosp_hadm_id'].nunique():,}\n")
        f.write(f"\nFilter criteria:\n")
        f.write(f"  1. Must have labels (from {args.labels})\n")
        f.write(f"  2. subject_id must be in ICU stays\n")
        if args.min_total_days > 0:
            f.write(f"  2b. Patient must have >= {args.min_total_days} total ICU days (across all stays)\n")
        f.write(f"  3. Must have hosp_hadm_id (not NaN)\n")
        f.write(f"  4. Must match ICU stay (subject_id + hadm_id)\n")
        if not args.no_time_match:
            f.write(f"  5. ECG time must fall within ICU stay period\n")
        else:
            f.write(f"  5. Time matching: DISABLED\n")
    
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()

