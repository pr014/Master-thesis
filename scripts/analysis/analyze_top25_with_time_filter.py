#!/usr/bin/env python3
"""
Analyze Top 25 diagnoses with time filtering (admittime < ecg_time).
Compares with current Top 25 to see if they change.
"""

import pandas as pd
from pathlib import Path
import sys
from collections import Counter
import ast
from typing import Dict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.labeling import load_icustays

def extract_diagnoses_from_column(df: pd.DataFrame, col_name: str) -> Dict[str, int]:
    """Extract individual diagnoses from a diagnosis column."""
    if col_name not in df.columns:
        return Counter()
    
    diagnosis_counter = Counter()
    
    for idx, row in df.iterrows():
        diag_str = row[col_name]
        
        if pd.isna(diag_str) or diag_str == '' or diag_str == '[]':
            continue
        
        diagnoses = []
        
        # Try to parse as Python list string first
        diag_str_clean = str(diag_str).strip()
        if diag_str_clean.startswith('[') and diag_str_clean.endswith(']'):
            try:
                parsed_list = ast.literal_eval(diag_str_clean)
                if isinstance(parsed_list, list):
                    diagnoses = [str(d).strip().strip("'\"") for d in parsed_list if d]
            except:
                pass
        
        # If not a list, try semicolon-separated
        if not diagnoses:
            diagnoses = str(diag_str).split(';')
        
        # Clean and count diagnoses
        for diag in diagnoses:
            diag = diag.strip().strip("'\"[]")
            if diag and diag != 'nan' and diag != '':
                icd10_code = diag.split()[0] if ' ' in diag else diag
                icd10_code = icd10_code.strip("'\"")
                if icd10_code:
                    diagnosis_counter[icd10_code] += 1
    
    return diagnosis_counter

def load_diagnosis_data_with_time_filter(csv_path, icustays_df, admissions_path):
    """Load diagnosis data with time filtering."""
    
    print(f"Loading diagnosis data from: {csv_path}")
    print("  Applying time filter: admittime < ecg_time")
    
    # Load CSV
    diagnosis_cols = [
        'file_name', 'ecg_time', 'subject_id', 'hosp_hadm_id', 'gender', 'age',
        'ed_diag_ed', 'ed_diag_hosp', 'hosp_diag_hosp', 
        'all_diag_hosp', 'all_diag_all'
    ]
    
    df_header = pd.read_csv(csv_path, nrows=0)
    available_cols = [col for col in diagnosis_cols if col in df_header.columns]
    
    df = pd.read_csv(csv_path, usecols=available_cols, low_memory=False)
    print(f"  Loaded {len(df):,} records")
    
    # Convert ecg_time
    df['ecg_time'] = pd.to_datetime(df['ecg_time'], utc=True, errors='coerce')
    df = df.dropna(subset=['ecg_time'])
    print(f"  Records with valid ecg_time: {len(df):,}")
    
    # Match ECGs to ICU stays
    print("  Matching ECGs to ICU stays...")
    df_with_stays = df.merge(
        icustays_df[['stay_id', 'subject_id', 'intime', 'outtime', 'hadm_id']],
        on='subject_id',
        how='inner'
    )
    
    # Filter by time window
    time_mask = (df_with_stays['ecg_time'] >= df_with_stays['intime']) & (df_with_stays['ecg_time'] <= df_with_stays['outtime'])
    df_matched = df_with_stays[time_mask].copy()
    
    # Handle multiple matches
    if len(df_matched) > 0:
        df_matched['time_diff'] = (df_matched['ecg_time'] - df_matched['intime']).dt.total_seconds()
        df_matched['rank'] = df_matched.groupby('file_name')['time_diff'].rank(method='min')
        df_matched = df_matched[df_matched['rank'] == 1].drop(columns=['time_diff', 'rank'])
    
    print(f"  Matched {len(df_matched):,} ECGs to ICU stays")
    
    df_matched = df_matched.dropna(subset=['hadm_id'])
    print(f"  Records with valid hadm_id: {len(df_matched):,}")
    
    # Load admissions
    print("  Loading admissions...")
    admissions_df = pd.read_csv(admissions_path, usecols=['subject_id', 'hadm_id', 'admittime'], low_memory=False)
    admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'], utc=True, errors='coerce')
    admissions_df = admissions_df.dropna(subset=['admittime'])
    
    # Merge with admissions
    df_merged = df_matched.merge(
        admissions_df,
        on=['subject_id', 'hadm_id'],
        how='inner'
    )
    
    # Apply time filter: admittime < ecg_time
    before_filter = len(df_merged)
    df_filtered = df_merged[df_merged['admittime'] < df_merged['ecg_time']].copy()
    after_filter = len(df_filtered)
    
    print(f"  Time filtering: {before_filter:,} -> {after_filter:,} records ({after_filter/before_filter*100:.1f}% kept)")
    
    # Remove duplicates
    df_filtered = df_filtered.drop_duplicates(subset=['file_name'], keep='first')
    
    return df_filtered

def get_top25_by_frequency(df_filtered):
    """Get Top 25 diagnoses by frequency from filtered data."""
    
    print("\n" + "="*80)
    print("EXTRACTING TOP 25 DIAGNOSES BY FREQUENCY")
    print("="*80)
    
    diagnosis_columns = ['ed_diag_ed', 'ed_diag_hosp', 'hosp_diag_hosp', 'all_diag_hosp', 'all_diag_all']
    available_cols = [col for col in diagnosis_columns if col in df_filtered.columns]
    
    all_diagnoses = Counter()
    
    for col in available_cols:
        print(f"\n  Extracting from {col}...")
        col_diagnoses = extract_diagnoses_from_column(df_filtered, col)
        all_diagnoses.update(col_diagnoses)
        print(f"    Found {len(col_diagnoses)} unique diagnoses, {sum(col_diagnoses.values()):,} total occurrences")
    
    print(f"\nTotal unique diagnoses: {len(all_diagnoses)}")
    print(f"Total diagnosis occurrences: {sum(all_diagnoses.values()):,}")
    
    # Get top 25
    top25 = dict(all_diagnoses.most_common(25))
    
    return top25, all_diagnoses

def compare_with_current_top25(new_top25, current_top25_list):
    """Compare new Top 25 with current Top 25."""
    
    print("\n" + "="*80)
    print("COMPARISON: NEW TOP 25 vs CURRENT TOP 25")
    print("="*80)
    
    new_set = set(new_top25.keys())
    current_set = set(current_top25_list)
    
    # Diagnoses that are in both
    in_both = new_set.intersection(current_set)
    print(f"\n✓ In both lists: {len(in_both)}/{25}")
    for diag in sorted(in_both):
        print(f"  {diag}")
    
    # Diagnoses only in new list
    only_new = new_set - current_set
    print(f"\n+ Only in NEW list: {len(only_new)}")
    for diag in sorted(only_new):
        count = new_top25[diag]
        print(f"  {diag:10s} (count: {count:,})")
    
    # Diagnoses only in current list
    only_current = current_set - new_set
    print(f"\n- Only in CURRENT list: {len(only_current)}")
    for diag in sorted(only_current):
        print(f"  {diag}")
    
    return {
        'in_both': in_both,
        'only_new': only_new,
        'only_current': only_current
    }

def main():
    """Main function."""
    
    print("="*80)
    print("ANALYZING TOP 25 DIAGNOSES WITH TIME FILTERING")
    print("="*80)
    
    # Current Top 25 from config
    current_top25 = [
        'R6521', 'J9690', 'Z66', 'R6520', 'N170', 'A419', 'E872', 'J690', 'N179', 'J189',
        'J449', 'D696', 'E871', 'N390', 'D62', 'I10', 'E785', 'I2510', 'I4891', 'K219',
        'Z87891', 'E119', 'I509', 'F329', 'Z7901'
    ]
    
    # Paths
    csv_path = project_root / "data/labeling/labels_csv/records_w_diag_icd10.csv"
    icustays_path = project_root / "data/labeling/labels_csv/icustays.csv"
    admissions_path = project_root / "data/labeling/labels_csv/admissions.csv"
    
    # Load ICU stays
    print("\nLoading ICU stays...")
    icustays_df = load_icustays(str(icustays_path))
    icustays_df['intime'] = pd.to_datetime(icustays_df['intime'], utc=True, errors='coerce')
    icustays_df['outtime'] = pd.to_datetime(icustays_df['outtime'], utc=True, errors='coerce')
    icustays_df = icustays_df.dropna(subset=['intime', 'outtime'])
    print(f"Loaded {len(icustays_df):,} ICU stays")
    
    # Load diagnosis data with time filtering
    print("\n" + "="*80)
    print("LOADING DIAGNOSES WITH TIME FILTERING")
    print("="*80)
    df_filtered = load_diagnosis_data_with_time_filter(csv_path, icustays_df, admissions_path)
    
    if len(df_filtered) == 0:
        print("\nERROR: No diagnosis data after filtering!")
        return
    
    # Get Top 25 by frequency
    new_top25, all_diagnoses = get_top25_by_frequency(df_filtered)
    
    # Print new Top 25
    print("\n" + "="*80)
    print("NEW TOP 25 DIAGNOSES (by frequency, with time filtering)")
    print("="*80)
    for i, (diag, count) in enumerate(new_top25.items(), 1):
        print(f"{i:2d}. {diag:10s}: {count:8,} occurrences")
    
    # Compare with current
    comparison = compare_with_current_top25(new_top25, current_top25)
    
    # Load descriptions
    print("\n" + "="*80)
    print("NEW TOP 25 WITH DESCRIPTIONS")
    print("="*80)
    icd_path = project_root / "data/labeling/labels_csv/d_icd_diagnoses.csv"
    icd_lookup = pd.read_csv(icd_path)
    icd_lookup = icd_lookup[icd_lookup['icd_version'] == 10]
    icd_dict = dict(zip(icd_lookup['icd_code'], icd_lookup['long_title']))
    
    for i, (diag, count) in enumerate(new_top25.items(), 1):
        desc = icd_dict.get(diag, "Description not found")
        marker = "✓" if diag in current_top25 else "+"
        print(f"{marker} {i:2d}. {diag:10s} - {desc} ({count:,} occurrences)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total unique diagnoses found: {len(all_diagnoses)}")
    print(f"Total diagnosis occurrences: {sum(all_diagnoses.values()):,}")
    print(f"\nDiagnoses unchanged: {len(comparison['in_both'])}/{25}")
    print(f"Diagnoses added: {len(comparison['only_new'])}")
    print(f"Diagnoses removed: {len(comparison['only_current'])}")
    
    if len(comparison['only_new']) == 0 and len(comparison['only_current']) == 0:
        print("\n✓ Top 25 diagnoses remain the same after time filtering!")
    else:
        print(f"\n⚠ Top 25 diagnoses changed after time filtering!")

if __name__ == "__main__":
    main()
