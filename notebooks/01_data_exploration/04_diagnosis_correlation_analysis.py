"""Analyze which factors (diagnoses, demographics) correlate most strongly with LOS and Mortality.

This notebook analyzes:
1. All ECGs from data/all_icu_ecgs/original dataset
2. Matches ECGs with ICU stays via subject_id
3. Loads diagnoses from records_w_diag_icd10.csv (if available)
4. Analyzes which factors influence LOS and Mortality:
   - ICD-10 diagnoses
   - Demographics (age, gender)
   - Other available factors

Usage:
    # Analyze top 15 diagnoses from config (default)
    python notebooks/01_data_exploration/04_diagnosis_correlation_analysis.py --mode top15
    
    # Analyze ALL diagnoses and select top 50 based on combined influence
    python notebooks/01_data_exploration/04_diagnosis_correlation_analysis.py --mode combined
"""

from pathlib import Path
import sys
import argparse
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data.labeling import load_icustays, load_mortality_mapping, ICUStayMapper
from src.data.ecg.ecg_loader import build_npy_index, build_demo_index
from src.data.ecg.ecg_dataset import extract_subject_id_from_path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_ecg_records_from_directory(data_dir: Path) -> pd.DataFrame:
    """Load all ECG records from directory and extract subject_ids.
    
    Args:
        data_dir: Path to directory containing ECG files (.npy format)
        
    Returns:
        DataFrame with columns: base_path, subject_id
    """
    print(f"Loading ECG records from: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Build index of all ECG files - try .npy first, then WFDB format
    records = None
    try:
        records = build_npy_index(data_dir=str(data_dir))
        print(f"Found {len(records):,} ECG files (.npy format)")
    except (FileNotFoundError, RuntimeError):
        try:
            records = build_demo_index(data_dir=str(data_dir))
            print(f"Found {len(records):,} ECG files (WFDB format)")
        except Exception as e:
            raise RuntimeError(f"Failed to build ECG index (tried both .npy and WFDB): {e}")
    
    # Extract subject_ids from paths
    ecg_data = []
    failed_extractions = 0
    
    for record in records:
        base_path = record.get('base_path', '')
        try:
            subject_id = extract_subject_id_from_path(base_path)
            ecg_data.append({
                'base_path': base_path,
                'subject_id': int(subject_id)
            })
        except Exception as e:
            failed_extractions += 1
            continue
    
    if failed_extractions > 0:
        print(f"Warning: Failed to extract subject_id from {failed_extractions} ECGs")
    
    df = pd.DataFrame(ecg_data)
    print(f"Successfully loaded {len(df):,} ECG records with subject_ids")
    print(f"Unique subjects: {df['subject_id'].nunique():,}")
    
    return df


def load_diagnosis_data_per_stay(
    csv_path: Path, 
    icustays_df: pd.DataFrame,
    stay_ids: Optional[pd.Series] = None,
    apply_time_filter: bool = True,
    admissions_path: Optional[Path] = None
) -> pd.DataFrame:
    """Load diagnosis data from CSV file and aggregate per ICU stay.
    
    Args:
        csv_path: Path to records_w_diag_icd10.csv
        icustays_df: DataFrame with ICU stays (for matching via hadm_id)
        stay_ids: Optional Series of stay_ids to filter
        apply_time_filter: If True, filter diagnoses where admittime < ecg_time
        admissions_path: Path to admissions.csv (required if apply_time_filter=True)
        
    Returns:
        DataFrame with diagnosis columns aggregated per stay_id
    """
    print(f"Loading diagnosis data from: {csv_path}")
    if apply_time_filter:
        print("  Applying time filter: admittime < ecg_time")
    
    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}. Continuing without diagnosis data.")
        return pd.DataFrame()
    
    # Load CSV - we need diagnosis columns, identifiers, and ecg_time for filtering
    diagnosis_cols = [
        'file_name', 'ecg_time', 'subject_id', 'hosp_hadm_id', 'gender', 'age',
        'ed_diag_ed', 'ed_diag_hosp', 'hosp_diag_hosp', 
        'all_diag_hosp', 'all_diag_all'
    ]
    
    # Check which columns exist
    df_header = pd.read_csv(csv_path, nrows=0)
    available_cols = [col for col in diagnosis_cols if col in df_header.columns]
    
    # Read CSV with available columns
    df = pd.read_csv(csv_path, usecols=available_cols, low_memory=False)
    print(f"Loaded {len(df):,} records from CSV")
    
    # Filter ICU stays if provided
    if stay_ids is not None:
        icustays_filtered = icustays_df[icustays_df['stay_id'].isin(stay_ids)].copy()
    else:
        icustays_filtered = icustays_df.copy()
    
    # TIME-BASED FILTERING: Match ECGs to ICU stays and filter admittime < ecg_time
    if apply_time_filter and 'ecg_time' in df.columns:
        if admissions_path is None:
            admissions_path = csv_path.parent / "admissions.csv"
        
        if not admissions_path.exists():
            print(f"Warning: admissions.csv not found at {admissions_path}. Skipping time filter.")
            apply_time_filter = False
        else:
            # Convert ecg_time to datetime
            df['ecg_time'] = pd.to_datetime(df['ecg_time'], utc=True, errors='coerce')
            df = df.dropna(subset=['ecg_time'])
            print(f"  Records with valid ecg_time: {len(df):,}")
            
            # Match ECGs to ICU stays: subject_id + ecg_time within intime/outtime
            print("  Matching ECGs to ICU stays...")
            df_with_stays = df.merge(
                icustays_filtered[['stay_id', 'subject_id', 'intime', 'outtime'] + (['hadm_id'] if 'hadm_id' in icustays_filtered.columns else [])],
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
            
            # Get hadm_id from icustays if not present
            if 'hadm_id' not in df_matched.columns and 'hadm_id' in icustays_filtered.columns:
                df_matched = df_matched.merge(
                    icustays_filtered[['stay_id', 'hadm_id']],
                    on='stay_id',
                    how='left'
                )
            
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
            df = df_filtered
            
            # Drop time-related columns that we don't need for aggregation
            df = df.drop(columns=['ecg_time', 'intime', 'outtime', 'admittime', 'file_name'], errors='ignore')
    
    # If no time filtering, use original matching logic
    if not apply_time_filter or 'ecg_time' not in available_cols:
        # Match diagnoses to ICU stays via subject_id + hadm_id
        if 'hadm_id' in icustays_filtered.columns:
            df_with_hadm = df.merge(
                icustays_filtered[['stay_id', 'subject_id', 'hadm_id']],
                on='subject_id',
                how='inner'
            )
            # Filter to matching hadm_id (if available in CSV)
            if 'hosp_hadm_id' in df_with_hadm.columns:
                df_with_hadm = df_with_hadm[
                    (df_with_hadm['hosp_hadm_id'] == df_with_hadm['hadm_id']) | 
                    (df_with_hadm['hosp_hadm_id'].isna())
                ]
        else:
            # Fallback: match only by subject_id
            df_with_hadm = df.merge(
                icustays_filtered[['stay_id', 'subject_id']],
                on='subject_id',
                how='inner'
            )
        
        print(f"Matched {len(df_with_hadm):,} diagnosis records to ICU stays")
        df = df_with_hadm
    
    # Aggregate diagnoses per stay_id
    agg_dict = {}
    for col in df.columns:
        if col in ['stay_id', 'subject_id', 'hadm_id', 'hosp_hadm_id', 'file_name', 'ecg_time', 'intime', 'outtime', 'admittime']:
            continue
        elif col in ['gender', 'age']:
            # For demographics, take first non-null
            agg_dict[col] = 'first'
        else:
            # For diagnosis columns, combine all non-null values
            agg_dict[col] = lambda x: ';'.join(x.dropna().astype(str).unique()) if len(x.dropna()) > 0 else None
    
    df_agg = df.groupby('stay_id').agg(agg_dict).reset_index()
    print(f"Aggregated to {len(df_agg):,} unique ICU stays")
    
    return df_agg


def extract_diagnoses_from_column(df: pd.DataFrame, col_name: str) -> Dict[str, int]:
    """Extract individual diagnoses from a diagnosis column.
    
    Diagnosis columns may contain:
    - Python list string: "['R4182', 'G9340']"
    - Single diagnosis: "I21.9"
    - Multiple diagnoses (semicolon-separated): "I21.9;I50.9;E11.9"
    - Empty/NaN values or empty lists: "[]"
    
    Args:
        df: DataFrame with diagnosis column
        col_name: Name of diagnosis column
        
    Returns:
        Counter with diagnosis codes as keys and counts as values
    """
    if col_name not in df.columns:
        return Counter()
    
    diagnosis_counter = Counter()
    
    for idx, row in df.iterrows():
        diag_str = row[col_name]
        
        if pd.isna(diag_str) or diag_str == '' or diag_str == '[]':
            continue
        
        diagnoses = []
        
        # Try to parse as Python list string first (most common format)
        diag_str_clean = str(diag_str).strip()
        if diag_str_clean.startswith('[') and diag_str_clean.endswith(']'):
            try:
                # Parse Python list string: "['R4182', 'G9340']"
                import ast
                parsed_list = ast.literal_eval(diag_str_clean)
                if isinstance(parsed_list, list):
                    diagnoses = [str(d).strip().strip("'\"") for d in parsed_list if d]
            except:
                # If parsing fails, try other methods
                pass
        
        # If not a list, try semicolon-separated
        if not diagnoses:
            diagnoses = str(diag_str).split(';')
        
        # Clean and count diagnoses
        for diag in diagnoses:
            diag = diag.strip().strip("'\"[]")
            if diag and diag != 'nan' and diag != '':
                # Extract ICD-10 code (first part before space, if any)
                icd10_code = diag.split()[0] if ' ' in diag else diag
                # Remove quotes if present
                icd10_code = icd10_code.strip("'\"")
                if icd10_code:
                    diagnosis_counter[icd10_code] += 1
    
    return diagnosis_counter


def get_top_diagnoses(df: pd.DataFrame, diagnosis_columns: List[str], top_n: int = 50) -> Dict[str, int]:
    """Get top N most frequent diagnoses across all diagnosis columns.
    
    Args:
        df: DataFrame with diagnosis columns
        diagnosis_columns: List of column names containing diagnoses
        top_n: Number of top diagnoses to return
        
    Returns:
        Dictionary mapping diagnosis code to total count
    """
    all_diagnoses = Counter()
    
    for col in diagnosis_columns:
        if col in df.columns:
            print(f"  Extracting from {col}...")
            col_diagnoses = extract_diagnoses_from_column(df, col)
            all_diagnoses.update(col_diagnoses)
            print(f"    Found {len(col_diagnoses)} unique diagnoses")
    
    print(f"\nTotal unique diagnoses: {len(all_diagnoses)}")
    print(f"Total diagnosis occurrences: {sum(all_diagnoses.values()):,}")
    
    # Get top N
    top_diagnoses = dict(all_diagnoses.most_common(top_n))
    
    return top_diagnoses


def create_diagnosis_features(df: pd.DataFrame, top_diagnoses: Dict[str, int]) -> pd.DataFrame:
    """Create binary features for each top diagnosis.
    
    Args:
        df: Original DataFrame
        top_diagnoses: Dictionary of top diagnosis codes
        
    Returns:
        DataFrame with added binary columns for each diagnosis
    """
    df_features = df.copy()
    
    # Get all diagnosis columns
    diag_cols = ['ed_diag_ed', 'ed_diag_hosp', 'hosp_diag_hosp', 'all_diag_hosp', 'all_diag_all']
    available_diag_cols = [col for col in diag_cols if col in df.columns]
    
    # Create binary features for each top diagnosis
    for diag_code in top_diagnoses.keys():
        feature_name = f"has_{diag_code.replace('.', '_')}"
        df_features[feature_name] = 0
        
        # Check if diagnosis appears in any diagnosis column
        # Diagnoses are stored as Python list strings: "['R4182', 'G9340']"
        for col in available_diag_cols:
            # Check if diagnosis code appears in the string (handles both list format and plain strings)
            mask = df_features[col].astype(str).str.contains(
                f"'{diag_code}'|\\b{diag_code}\\b",  # Match 'CODE' or word boundary CODE
                na=False, 
                regex=True
            )
            df_features.loc[mask, feature_name] = 1
    
    return df_features


def match_with_icu_stays(df: pd.DataFrame, icustays_df: pd.DataFrame, icu_mapper: ICUStayMapper) -> pd.DataFrame:
    """Match ECG records with ICU stays via subject_id + ecg_time (within first 24h).
    
    Since all ECGs in icu_ecgs_24h dataset are within first 24h of ICU stay,
    we match ECGs to ICU stays using time-based matching.
    
    Args:
        df: DataFrame with ECG records (must have subject_id, base_path)
        icustays_df: DataFrame with ICU stays
        icu_mapper: ICUStayMapper for time-based matching
        
    Returns:
        DataFrame with added 'los_days', 'stay_id', aggregated per ICU stay
    """
    print("\nMatching ECG records with ICU stays via subject_id + ecg_time (24h window)...")
    
    # Try to extract timestamps from ECG files (WFDB format)
    from datetime import timedelta
    import wfdb
    
    df_with_stays = df.copy()
    df_with_stays['stay_id'] = None
    df_with_stays['los_days'] = np.nan
    df_with_stays['ecg_time'] = None
    
    matched_count = 0
    unmatched_count = 0
    
    print("  Extracting timestamps and matching ECGs to ICU stays...")
    for idx, row in df_with_stays.iterrows():
        if (idx + 1) % 5000 == 0:
            print(f"    Processed {idx + 1:,}/{len(df_with_stays):,} ECGs...")
        
        base_path = row['base_path']
        subject_id = row['subject_id']
        
        try:
            # Try to read timestamp from WFDB file
            try:
                record = wfdb.rdrecord(base_path)
                base_date = getattr(record, 'base_date', None)
                base_time = getattr(record, 'base_time', None)
                
                if base_date and base_time:
                    ecg_time = pd.to_datetime(f"{base_date} {base_time}", utc=True, errors='coerce')
                    if pd.isna(ecg_time):
                        ecg_time = None
                else:
                    ecg_time = None
            except:
                ecg_time = None
            
            # Match to ICU stay
            if ecg_time is not None:
                if ecg_time.tz is not None:
                    ecg_time = ecg_time.tz_localize(None)
                stay_id = icu_mapper.map_ecg_to_stay(subject_id, ecg_time)
            else:
                # Fallback: use first ICU stay for this subject
                subject_stays = icustays_df[icustays_df['subject_id'] == subject_id]
                if len(subject_stays) > 0:
                    stay_id = int(subject_stays.iloc[0]['stay_id'])
                else:
                    stay_id = None
            
            if stay_id is not None:
                los_days = icu_mapper.get_los(stay_id)
                if los_days is not None:
                    df_with_stays.loc[idx, 'stay_id'] = stay_id
                    df_with_stays.loc[idx, 'los_days'] = los_days
                    if ecg_time is not None:
                        df_with_stays.loc[idx, 'ecg_time'] = ecg_time
                    matched_count += 1
                else:
                    unmatched_count += 1
            else:
                unmatched_count += 1
                
        except Exception:
            unmatched_count += 1
            continue
    
    # Aggregate to stay level (one row per ICU stay)
    print("\n  Aggregating to stay level...")
    df_stays = df_with_stays[df_with_stays['stay_id'].notna()].copy()
    
    # Group by stay_id and aggregate
    stay_agg = df_stays.groupby('stay_id').agg({
        'subject_id': 'first',
        'los_days': 'first',
        'base_path': 'count'  # Count ECGs per stay
    }).reset_index()
    stay_agg = stay_agg.rename(columns={'base_path': 'ecg_count'})
    
    # Get unique stay_ids with their info
    stay_info = icustays_df[['stay_id', 'subject_id', 'los']].copy()
    stay_info = stay_info.merge(stay_agg[['stay_id', 'ecg_count']], on='stay_id', how='inner')
    stay_info['los_days'] = stay_info['los']
    stay_info = stay_info.drop(columns=['los'])
    
    matched_stays = stay_info['stay_id'].nunique()
    print(f"  Matched: {matched_count:,} ECGs from {matched_stays:,} ICU stays ({matched_count/len(df)*100:.2f}%)")
    print(f"  Unmatched: {unmatched_count:,} ECGs ({unmatched_count/len(df)*100:.2f}%)")
    
    return stay_info


def match_with_mortality(df: pd.DataFrame, icustays_df: pd.DataFrame, mortality_mapping: Dict[int, int]) -> pd.DataFrame:
    """Match ECG records with mortality labels.
    
    Args:
        df: DataFrame with ECG records (must have stay_id column)
        mortality_mapping: Dictionary mapping stay_id to mortality (0/1)
        
    Returns:
        DataFrame with added 'mortality' column
    """
    df_with_mortality = df.copy()
    df_with_mortality['mortality'] = np.nan
    
    print("\nMatching ECG records with mortality...")
    
    # Match via stay_id
    df_with_mortality['mortality'] = df_with_mortality['stay_id'].map(mortality_mapping)
    
    matched_count = df_with_mortality['mortality'].notna().sum()
    unmatched_count = len(df) - matched_count
    print(f"  Matched: {matched_count:,} ({matched_count/len(df)*100:.2f}%)")
    print(f"  Unmatched: {unmatched_count:,} ({unmatched_count/len(df)*100:.2f}%)")
    
    return df_with_mortality


def analyze_diagnosis_los_correlation(df: pd.DataFrame, top_diagnoses: Dict[str, int]) -> pd.DataFrame:
    """Analyze correlation between diagnoses and LOS (per ICU stay).
    
    Args:
        df: DataFrame with diagnosis features and los_days (one row per ICU stay)
        top_diagnoses: Dictionary of top diagnosis codes
        
    Returns:
        DataFrame with correlation statistics for each diagnosis
    """
    results = []
    
    print("\n" + "="*80)
    print("ANALYZING DIAGNOSIS-LOS CORRELATIONS (per ICU stay)")
    print("="*80)
    
    # Filter to records with valid LOS
    df_valid = df[df['los_days'].notna()].copy()
    print(f"\nICU stays with valid LOS: {len(df_valid):,}")
    
    if len(df_valid) == 0:
        print("ERROR: No records with valid LOS found!")
        return pd.DataFrame()
    
    for diag_code in top_diagnoses.keys():
        feature_name = f"has_{diag_code.replace('.', '_')}"
        
        if feature_name not in df_valid.columns:
            continue
        
        # Get groups
        has_diag = df_valid[df_valid[feature_name] == 1]
        no_diag = df_valid[df_valid[feature_name] == 0]
        
        if len(has_diag) == 0 or len(no_diag) == 0:
            continue
        
        # Calculate statistics
        los_with = has_diag['los_days'].mean()
        los_without = no_diag['los_days'].mean()
        los_diff = los_with - los_without
        
        # Statistical test (Mann-Whitney U test for non-normal distributions)
        from scipy.stats import mannwhitneyu
        try:
            stat, p_value = mannwhitneyu(
                has_diag['los_days'].dropna(),
                no_diag['los_days'].dropna(),
                alternative='two-sided'
            )
        except:
            p_value = 1.0
        
        # Spearman correlation
        try:
            corr, corr_p = spearmanr(df_valid[feature_name], df_valid['los_days'])
        except:
            corr, corr_p = 0.0, 1.0
        
        results.append({
            'diagnosis': diag_code,
            'count': len(has_diag),
            'count_pct': len(has_diag) / len(df_valid) * 100,
            'los_with_diag': los_with,
            'los_without_diag': los_without,
            'los_difference': los_diff,
            'los_difference_pct': (los_diff / los_without * 100) if los_without > 0 else 0,
            'p_value': p_value,
            'spearman_corr': corr,
            'spearman_p': corr_p,
            'significant': p_value < 0.05
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('los_difference', ascending=False)
    
    return results_df


def analyze_diagnosis_mortality_correlation(df: pd.DataFrame, top_diagnoses: Dict[str, int]) -> pd.DataFrame:
    """Analyze correlation between diagnoses and mortality (per ICU stay).
    
    Args:
        df: DataFrame with diagnosis features and mortality (one row per ICU stay)
        top_diagnoses: Dictionary of top diagnosis codes
        
    Returns:
        DataFrame with correlation statistics for each diagnosis
    """
    results = []
    
    print("\n" + "="*80)
    print("ANALYZING DIAGNOSIS-MORTALITY CORRELATIONS (per ICU stay)")
    print("="*80)
    
    # Filter to records with valid mortality
    df_valid = df[df['mortality'].notna()].copy()
    print(f"\nICU stays with valid mortality: {len(df_valid):,}")
    
    if len(df_valid) == 0:
        print("ERROR: No records with valid mortality found!")
        return pd.DataFrame()
    
    # Overall mortality rate
    overall_mortality = df_valid['mortality'].mean()
    print(f"Overall mortality rate: {overall_mortality*100:.2f}%")
    
    for diag_code in top_diagnoses.keys():
        feature_name = f"has_{diag_code.replace('.', '_')}"
        
        if feature_name not in df_valid.columns:
            continue
        
        # Get groups
        has_diag = df_valid[df_valid[feature_name] == 1]
        no_diag = df_valid[df_valid[feature_name] == 0]
        
        if len(has_diag) == 0 or len(no_diag) == 0:
            continue
        
        # Calculate mortality rates
        mortality_with = has_diag['mortality'].mean()
        mortality_without = no_diag['mortality'].mean()
        mortality_diff = mortality_with - mortality_without
        mortality_ratio = mortality_with / mortality_without if mortality_without > 0 else float('inf')
        
        # Chi-square test
        contingency = pd.crosstab(df_valid[feature_name], df_valid['mortality'])
        if contingency.shape == (2, 2):
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
            except:
                p_value = 1.0
        else:
            p_value = 1.0
        
        # Odds ratio
        if contingency.shape == (2, 2):
            try:
                a = contingency.loc[1, 1] if 1 in contingency.index and 1 in contingency.columns else 0
                b = contingency.loc[1, 0] if 1 in contingency.index and 0 in contingency.columns else 0
                c = contingency.loc[0, 1] if 0 in contingency.index and 1 in contingency.columns else 0
                d = contingency.loc[0, 0] if 0 in contingency.index and 0 in contingency.columns else 0
                
                if b > 0 and c > 0:
                    odds_ratio = (a * d) / (b * c)
                else:
                    odds_ratio = float('inf') if a > 0 else 0.0
            except:
                odds_ratio = 1.0
        else:
            odds_ratio = 1.0
        
        results.append({
            'diagnosis': diag_code,
            'count': len(has_diag),
            'count_pct': len(has_diag) / len(df_valid) * 100,
            'mortality_with_diag': mortality_with,
            'mortality_without_diag': mortality_without,
            'mortality_difference': mortality_diff,
            'mortality_ratio': mortality_ratio,
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mortality_difference', ascending=False)
    
    return results_df


def plot_top15_diagnoses_los(results_df: pd.DataFrame, diagnosis_list: List[str], save_path: Optional[Path] = None):
    """Plot LOS correlation for top 15 diagnoses.
    
    Args:
        results_df: DataFrame with LOS correlation results
        diagnosis_list: List of top 15 diagnosis codes to plot
        save_path: Optional path to save figure
    """
    if len(results_df) == 0:
        print("No data to plot")
        return
    
    # Filter to only top 15 diagnoses
    top15_results = results_df[results_df['diagnosis'].isin(diagnosis_list)].copy()
    
    if len(top15_results) == 0:
        print("None of the top 15 diagnoses found in results")
        return
    
    # Sort by LOS difference (descending)
    top15_results = top15_results.sort_values('los_difference', ascending=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create color map: Red = significant positive (bad), Green = significant negative (good), Gray = not significant
    def get_color_los(p_value, los_diff):
        if p_value >= 0.05:
            return '#808080'  # Gray = not significant
        elif los_diff > 0:
            return '#d62728'  # Red = significant positive (longer LOS = bad)
        else:
            return '#2ca02c'  # Green = significant negative (shorter LOS = good)
    
    colors = [get_color_los(row['p_value'], row['los_difference']) 
              for _, row in top15_results.iterrows()]
    
    # Horizontal bar plot
    bars = ax.barh(range(len(top15_results)), top15_results['los_difference'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize y-axis labels (diagnosis codes)
    ax.set_yticks(range(len(top15_results)))
    ax.set_yticklabels(top15_results['diagnosis'], fontsize=11)
    
    # Labels and title
    ax.set_xlabel('LOS Difference (days)', fontsize=13, fontweight='bold')
    ax.set_title('Top Diagnoses: Influence on Length of Stay (LOS)\n(Red = significant positive/bad, Green = significant negative/good, Gray = not significant)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top15_results.iterrows()):
        value = row['los_difference']
        label_x = value + (0.1 if value >= 0 else -0.1)
        ax.text(label_x, i, f'{value:.2f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved LOS plot to: {save_path}")
    
    plt.close()


def plot_top15_diagnoses_mortality(results_df: pd.DataFrame, diagnosis_list: List[str], save_path: Optional[Path] = None):
    """Plot mortality correlation for top 15 diagnoses.
    
    Args:
        results_df: DataFrame with mortality correlation results
        diagnosis_list: List of top 15 diagnosis codes to plot
        save_path: Optional path to save figure
    """
    if len(results_df) == 0:
        print("No data to plot")
        return
    
    # Filter to only top 15 diagnoses
    top15_results = results_df[results_df['diagnosis'].isin(diagnosis_list)].copy()
    
    if len(top15_results) == 0:
        print("None of the top 15 diagnoses found in results")
        return
    
    # Sort by mortality difference (descending)
    top15_results = top15_results.sort_values('mortality_difference', ascending=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create color map: Red = significant positive (bad), Green = significant negative (good), Gray = not significant
    def get_color_mortality(p_value, mort_diff):
        if p_value >= 0.05:
            return '#808080'  # Gray = not significant
        elif mort_diff > 0:
            return '#d62728'  # Red = significant positive (higher mortality = bad)
        else:
            return '#2ca02c'  # Green = significant negative (lower mortality = good)
    
    colors = [get_color_mortality(row['p_value'], row['mortality_difference']) 
              for _, row in top15_results.iterrows()]
    
    # Horizontal bar plot (mortality difference in percentage)
    bars = ax.barh(range(len(top15_results)), top15_results['mortality_difference'] * 100, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize y-axis labels (diagnosis codes)
    ax.set_yticks(range(len(top15_results)))
    ax.set_yticklabels(top15_results['diagnosis'], fontsize=11)
    
    # Labels and title
    ax.set_xlabel('Mortality Difference (%)', fontsize=13, fontweight='bold')
    ax.set_title('Top Diagnoses: Influence on Mortality\n(Red = significant positive/bad, Green = significant negative/good, Gray = not significant)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top15_results.iterrows()):
        value = row['mortality_difference'] * 100
        label_x = value + (0.5 if value >= 0 else -0.5)
        ax.text(label_x, i, f'{value:.2f}%', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved mortality plot to: {save_path}")
    
    plt.close()


def analyze_demographic_los_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlation between demographics (age, gender) and LOS.
    
    Args:
        df: DataFrame with 'age', 'gender', 'los_days' columns
        
    Returns:
        DataFrame with correlation statistics for demographics
    """
    results = []
    
    print("\n" + "="*80)
    print("ANALYZING DEMOGRAPHIC-LOS CORRELATIONS (per ICU stay)")
    print("="*80)
    
    # Filter valid data
    df_valid = df[df['los_days'].notna()].copy()
    print(f"\nICU stays with valid LOS: {len(df_valid):,}")
    
    if len(df_valid) == 0:
        print("ERROR: No records with valid LOS found!")
        return pd.DataFrame()
    
    # Age analysis
    if 'age' in df_valid.columns:
        df_age = df_valid[df_valid['age'].notna()].copy()
        if len(df_age) > 0:
            print(f"\nAge analysis (n={len(df_age):,}):")
            
            # Spearman correlation
            try:
                corr, corr_p = spearmanr(df_age['age'], df_age['los_days'])
            except:
                corr, corr_p = 0.0, 1.0
            
            # Age groups analysis (quartiles)
            age_q25 = df_age['age'].quantile(0.25)
            age_q50 = df_age['age'].quantile(0.50)
            age_q75 = df_age['age'].quantile(0.75)
            
            # Compare high age (>= Q75) vs low age (<= Q25)
            high_age = df_age[df_age['age'] >= age_q75]
            low_age = df_age[df_age['age'] <= age_q25]
            
            los_high = None
            los_low = None
            los_diff = None
            p_value = None
            
            if len(high_age) > 0 and len(low_age) > 0:
                los_high = high_age['los_days'].mean()
                los_low = low_age['los_days'].mean()
                los_diff = los_high - los_low
                
                # Mann-Whitney U test
                from scipy.stats import mannwhitneyu
                try:
                    stat, p_value = mannwhitneyu(
                        high_age['los_days'].dropna(),
                        low_age['los_days'].dropna(),
                        alternative='two-sided'
                    )
                except:
                    p_value = 1.0
            
            results.append({
                'factor': 'age',
                'type': 'continuous',
                'mean': df_age['age'].mean(),
                'std': df_age['age'].std(),
                'min': df_age['age'].min(),
                'max': df_age['age'].max(),
                'q25': age_q25,
                'q50': age_q50,
                'q75': age_q75,
                'los_correlation': corr,
                'los_correlation_p': corr_p,
                'los_high_age': los_high,
                'los_low_age': los_low,
                'los_difference': los_diff,
                'p_value': p_value,
                'significant': p_value < 0.05 if p_value is not None else False,
                'count': len(df_age)
            })
            
            print(f"  Age-LOS Spearman correlation: {corr:.4f} (p={corr_p:.4f})")
            if los_diff is not None:
                print(f"  LOS difference (high age vs low age): {los_diff:.2f} days (p={p_value:.4f})")
    
    # Gender analysis
    if 'gender' in df_valid.columns:
        df_gender = df_valid[df_valid['gender'].notna()].copy()
        if len(df_gender) > 0:
            print(f"\nGender analysis (n={len(df_gender):,}):")
            
            # Get unique genders (usually M/F)
            genders = df_gender['gender'].unique()
            if len(genders) >= 2:
                # Compare each gender group
                for gender in genders:
                    gender_group = df_gender[df_gender['gender'] == gender]
                    other_groups = df_gender[df_gender['gender'] != gender]
                    
                    if len(gender_group) > 0 and len(other_groups) > 0:
                        los_with = gender_group['los_days'].mean()
                        los_without = other_groups['los_days'].mean()
                        los_diff = los_with - los_without
                        
                        # Mann-Whitney U test
                        from scipy.stats import mannwhitneyu
                        try:
                            stat, p_value = mannwhitneyu(
                                gender_group['los_days'].dropna(),
                                other_groups['los_days'].dropna(),
                                alternative='two-sided'
                            )
                        except:
                            p_value = 1.0
                        
                        # Spearman correlation (binary encoding)
                        try:
                            gender_binary = (df_gender['gender'] == gender).astype(int)
                            corr, corr_p = spearmanr(gender_binary, df_gender['los_days'])
                        except:
                            corr, corr_p = 0.0, 1.0
                        
                        results.append({
                            'factor': f'gender_{gender}',
                            'type': 'categorical',
                            'los_with': los_with,
                            'los_without': los_without,
                            'los_difference': los_diff,
                            'los_correlation': corr,
                            'los_correlation_p': corr_p,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'count': len(gender_group),
                            'count_pct': len(gender_group) / len(df_gender) * 100
                        })
                        
                        print(f"  Gender {gender}: LOS={los_with:.2f} days (n={len(gender_group):,}, {len(gender_group)/len(df_gender)*100:.1f}%)")
                        print(f"    LOS difference vs others: {los_diff:.2f} days (p={p_value:.4f})")
    
    results_df = pd.DataFrame(results)
    if 'los_difference' in results_df.columns:
        # Sort by absolute difference
        results_df['abs_los_difference'] = results_df['los_difference'].abs()
        results_df = results_df.sort_values('abs_los_difference', ascending=False)
        results_df = results_df.drop('abs_los_difference', axis=1)
    
    return results_df


def analyze_demographic_mortality_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlation between demographics (age, gender) and mortality.
    
    Args:
        df: DataFrame with 'age', 'gender', 'mortality' columns
        
    Returns:
        DataFrame with correlation statistics for demographics
    """
    results = []
    
    print("\n" + "="*80)
    print("ANALYZING DEMOGRAPHIC-MORTALITY CORRELATIONS (per ICU stay)")
    print("="*80)
    
    # Filter valid data
    df_valid = df[df['mortality'].notna()].copy()
    print(f"\nICU stays with valid mortality: {len(df_valid):,}")
    
    if len(df_valid) == 0:
        print("ERROR: No records with valid mortality found!")
        return pd.DataFrame()
    
    # Overall mortality rate
    overall_mortality = df_valid['mortality'].mean()
    print(f"Overall mortality rate: {overall_mortality*100:.2f}%")
    
    # Age analysis
    if 'age' in df_valid.columns:
        df_age = df_valid[df_valid['age'].notna()].copy()
        if len(df_age) > 0:
            print(f"\nAge analysis (n={len(df_age):,}):")
            
            # Compare high age (>= Q75) vs low age (<= Q25)
            age_q25 = df_age['age'].quantile(0.25)
            age_q75 = df_age['age'].quantile(0.75)
            
            high_age = df_age[df_age['age'] >= age_q75]
            low_age = df_age[df_age['age'] <= age_q25]
            
            mortality_high = None
            mortality_low = None
            mortality_diff = None
            odds_ratio = None
            p_value = None
            
            if len(high_age) > 0 and len(low_age) > 0:
                mortality_high = high_age['mortality'].mean()
                mortality_low = low_age['mortality'].mean()
                mortality_diff = mortality_high - mortality_low
                
                # Chi-square test
                high_age_mort = (high_age['age'] >= age_q75).astype(int)
                low_age_mort = (low_age['age'] <= age_q25).astype(int)
                
                # Create binary age variable
                df_age_binary = df_age.copy()
                df_age_binary['high_age'] = (df_age_binary['age'] >= age_q75).astype(int)
                
                contingency = pd.crosstab(df_age_binary['high_age'], df_age_binary['mortality'])
                if contingency.shape == (2, 2):
                    try:
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                    except:
                        p_value = 1.0
                    
                    # Odds ratio
                    try:
                        a = contingency.loc[1, 1] if 1 in contingency.index and 1 in contingency.columns else 0
                        b = contingency.loc[1, 0] if 1 in contingency.index and 0 in contingency.columns else 0
                        c = contingency.loc[0, 1] if 0 in contingency.index and 1 in contingency.columns else 0
                        d = contingency.loc[0, 0] if 0 in contingency.index and 0 in contingency.columns else 0
                        
                        if b > 0 and c > 0:
                            odds_ratio = (a * d) / (b * c)
                        else:
                            odds_ratio = float('inf') if a > 0 else 0.0
                    except:
                        odds_ratio = 1.0
                else:
                    p_value = 1.0
                    odds_ratio = 1.0
            
            # Mean age comparison
            age_mort = df_age[df_age['mortality'] == 1]['age'].mean()
            age_surv = df_age[df_age['mortality'] == 0]['age'].mean()
            
            results.append({
                'factor': 'age',
                'type': 'continuous',
                'mean_age_died': age_mort,
                'mean_age_survived': age_surv,
                'age_difference': age_mort - age_surv,
                'mortality_high_age': mortality_high,
                'mortality_low_age': mortality_low,
                'mortality_difference': mortality_diff,
                'odds_ratio': odds_ratio,
                'p_value': p_value,
                'significant': p_value < 0.05 if p_value is not None else False,
                'count_high_age': len(high_age) if 'high_age' in locals() else 0,
                'count_low_age': len(low_age) if 'low_age' in locals() else 0,
                'count_total': len(df_age)
            })
            
            print(f"  Mean age - Died: {age_mort:.1f}, Survived: {age_surv:.1f}")
            if mortality_diff is not None:
                print(f"  Mortality difference (high age vs low age): {mortality_diff*100:.2f}% (p={p_value:.4f}, OR={odds_ratio:.2f})")
    
    # Gender analysis
    if 'gender' in df_valid.columns:
        df_gender = df_valid[df_valid['gender'].notna()].copy()
        if len(df_gender) > 0:
            print(f"\nGender analysis (n={len(df_gender):,}):")
            
            # Get unique genders (usually M/F)
            genders = df_gender['gender'].unique()
            if len(genders) >= 2:
                # Compare each gender group
                for gender in genders:
                    gender_group = df_gender[df_gender['gender'] == gender]
                    other_groups = df_gender[df_gender['gender'] != gender]
                    
                    if len(gender_group) > 0 and len(other_groups) > 0:
                        mortality_with = gender_group['mortality'].mean()
                        mortality_without = other_groups['mortality'].mean()
                        mortality_diff = mortality_with - mortality_without
                        mortality_ratio = mortality_with / mortality_without if mortality_without > 0 else float('inf')
                        
                        # Chi-square test
                        contingency = pd.crosstab(df_gender['gender'] == gender, df_gender['mortality'])
                        if contingency.shape == (2, 2):
                            try:
                                chi2, p_value, dof, expected = chi2_contingency(contingency)
                            except:
                                p_value = 1.0
                            
                            # Odds ratio
                            try:
                                a = contingency.loc[True, 1] if True in contingency.index and 1 in contingency.columns else 0
                                b = contingency.loc[True, 0] if True in contingency.index and 0 in contingency.columns else 0
                                c = contingency.loc[False, 1] if False in contingency.index and 1 in contingency.columns else 0
                                d = contingency.loc[False, 0] if False in contingency.index and 0 in contingency.columns else 0
                                
                                if b > 0 and c > 0:
                                    odds_ratio = (a * d) / (b * c)
                                else:
                                    odds_ratio = float('inf') if a > 0 else 0.0
                            except:
                                odds_ratio = 1.0
                        else:
                            p_value = 1.0
                            odds_ratio = 1.0
                        
                        results.append({
                            'factor': f'gender_{gender}',
                            'type': 'categorical',
                            'mortality_with': mortality_with,
                            'mortality_without': mortality_without,
                            'mortality_difference': mortality_diff,
                            'mortality_ratio': mortality_ratio,
                            'odds_ratio': odds_ratio,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'count': len(gender_group),
                            'count_pct': len(gender_group) / len(df_gender) * 100
                        })
                        
                        print(f"  Gender {gender}: Mortality={mortality_with*100:.2f}% (n={len(gender_group):,}, {len(gender_group)/len(df_gender)*100:.1f}%)")
                        print(f"    Mortality difference vs others: {mortality_diff*100:.2f}% (p={p_value:.4f}, OR={odds_ratio:.2f})")
    
    results_df = pd.DataFrame(results)
    if 'mortality_difference' in results_df.columns:
        # Sort by absolute difference
        results_df['abs_mortality_difference'] = results_df['mortality_difference'].abs()
        results_df = results_df.sort_values('abs_mortality_difference', ascending=False)
        results_df = results_df.drop('abs_mortality_difference', axis=1)
    
    return results_df


def plot_demographic_influence(demographic_los_df: pd.DataFrame, demographic_mortality_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot demographic factors (Age & Sex) influence on LOS and Mortality.
    
    Args:
        demographic_los_df: DataFrame with demographic LOS correlation results
        demographic_mortality_df: DataFrame with demographic mortality correlation results
        save_path: Optional path to save figure
    """
    if len(demographic_los_df) == 0 and len(demographic_mortality_df) == 0:
        print("No demographic data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # === LOS Analysis ===
    # Age LOS
    if 'age' in demographic_los_df['factor'].values:
        age_row = demographic_los_df[demographic_los_df['factor'] == 'age'].iloc[0]
        ax1 = axes[0, 0]
        
        # Create age groups visualization
        age_data = {
            'High Age (≥Q75)': age_row.get('los_high_age', 0),
            'Low Age (≤Q25)': age_row.get('los_low_age', 0)
        }
        
        colors_age = ['#ff7f0e', '#1f77b4']
        bars = ax1.bar(age_data.keys(), age_data.values(), color=colors_age, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_ylabel('LOS (days)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Age Influence on LOS\n(Spearman r={age_row.get("los_correlation", 0):.4f}, p={age_row.get("los_correlation_p", 1):.4f})', 
                     fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Gender LOS
    gender_los = demographic_los_df[demographic_los_df['factor'].str.startswith('gender_')]
    if len(gender_los) > 0:
        ax2 = axes[0, 1]
        
        genders = gender_los['factor'].str.replace('gender_', '').tolist()
        los_values = gender_los['los_with'].tolist()
        
        colors_gender = ['#e377c2', '#17becf', '#bcbd22'][:len(genders)]
        bars = ax2.bar(genders, los_values, color=colors_gender, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_ylabel('LOS (days)', fontsize=12, fontweight='bold')
        ax2.set_title('Gender Influence on LOS', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels and significance
        for i, (bar, row) in enumerate(zip(bars, gender_los.itertuples())):
            height = bar.get_height()
            sig = '*' if row.significant else 'ns'
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}\n({sig})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # === Mortality Analysis ===
    # Age Mortality
    if 'age' in demographic_mortality_df['factor'].values:
        age_mort_row = demographic_mortality_df[demographic_mortality_df['factor'] == 'age'].iloc[0]
        ax3 = axes[1, 0]
        
        age_mort_data = {
            'High Age (≥Q75)': age_mort_row.get('mortality_high_age', 0) * 100,
            'Low Age (≤Q25)': age_mort_row.get('mortality_low_age', 0) * 100
        }
        
        colors_age = ['#ff7f0e', '#1f77b4']
        bars = ax3.bar(age_mort_data.keys(), age_mort_data.values(), color=colors_age, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_ylabel('Mortality Rate (%)', fontsize=12, fontweight='bold')
        ax3.set_title(f'Age Influence on Mortality\n(OR={age_mort_row.get("odds_ratio", 1):.2f}, p={age_mort_row.get("p_value", 1):.4f})', 
                     fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Gender Mortality
    gender_mort = demographic_mortality_df[demographic_mortality_df['factor'].str.startswith('gender_')]
    if len(gender_mort) > 0:
        ax4 = axes[1, 1]
        
        genders = gender_mort['factor'].str.replace('gender_', '').tolist()
        mort_values = [x * 100 for x in gender_mort['mortality_with'].tolist()]
        
        colors_gender = ['#e377c2', '#17becf', '#bcbd22'][:len(genders)]
        bars = ax4.bar(genders, mort_values, color=colors_gender, alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_ylabel('Mortality Rate (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Gender Influence on Mortality', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels and significance
        for i, (bar, row) in enumerate(zip(bars, gender_mort.itertuples())):
            height = bar.get_height()
            sig = '*' if row.significant else 'ns'
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%\n({sig})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Demographic Factors: Influence on LOS and Mortality', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved demographic plot to: {save_path}")
    
    plt.close()


def plot_mortality_correlations(results_df: pd.DataFrame, top_n: int = 20, save_path: Optional[Path] = None):
    """Plot mortality correlation results.
    
    Args:
        results_df: DataFrame with mortality correlation results
        top_n: Number of top diagnoses to plot
        save_path: Optional path to save figure
    """
    if len(results_df) == 0:
        print("No data to plot")
        return
    
    # Get top N by absolute difference
    top_results = results_df.nlargest(top_n, 'mortality_difference')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Mortality difference
    ax1 = axes[0]
    colors = ['red' if x < 0.05 else 'blue' for x in top_results['p_value']]
    ax1.barh(range(len(top_results)), top_results['mortality_difference'] * 100, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_results)))
    ax1.set_yticklabels(top_results['diagnosis'])
    ax1.set_xlabel('Mortality Difference (%)', fontsize=12)
    ax1.set_title(f'Top {top_n} Diagnoses by Mortality Difference\n(Red = significant p<0.05, Blue = not significant)', fontsize=14)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Odds ratio
    ax2 = axes[1]
    top_results_or = results_df[results_df['odds_ratio'] < 100].nlargest(top_n, 'odds_ratio')  # Filter extreme values
    colors2 = ['red' if x < 0.05 else 'blue' for x in top_results_or['p_value']]
    ax2.barh(range(len(top_results_or)), top_results_or['odds_ratio'], color=colors2, alpha=0.7)
    ax2.set_yticks(range(len(top_results_or)))
    ax2.set_yticklabels(top_results_or['diagnosis'])
    ax2.set_xlabel('Odds Ratio', fontsize=12)
    ax2.set_title(f'Top {top_n} Diagnoses by Odds Ratio for Mortality\n(Red = significant p<0.05, Blue = not significant)', fontsize=14)
    ax2.axvline(x=1.0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved mortality correlation plot to: {save_path}")
    
    plt.show()


def select_top50_combined_influence(
    los_results: pd.DataFrame, 
    mortality_results: pd.DataFrame, 
    n_los: int = 25, 
    n_mortality: int = 25
) -> Tuple[List[str], pd.DataFrame]:
    """Select top 50 diagnoses based on combined influence on LOS and Mortality.
    
    Args:
        los_results: DataFrame with LOS correlation results
        mortality_results: DataFrame with mortality correlation results
        n_los: Number of top diagnoses to select for LOS (default: 25)
        n_mortality: Number of top diagnoses to select for Mortality (default: 25)
        
    Returns:
        Tuple of (selected_diagnosis_list, combined_results_df)
    """
    print(f"\n{'='*80}")
    print("SELECTING TOP 50 DIAGNOSES BASED ON COMBINED INFLUENCE")
    print(f"{'='*80}")
    
    # Create influence scores for LOS (combining effect size and significance)
    # Higher score = more influence
    los_results = los_results.copy()
    los_results['los_influence_score'] = (
        los_results['los_difference'].abs() * (1 - los_results['p_value']) * 
        (los_results['significant'].astype(int) * 2 + 1)  # Bonus for significance
    )
    
    # Create influence scores for Mortality (combining effect size and significance)
    mortality_results = mortality_results.copy()
    mortality_results['mortality_influence_score'] = (
        mortality_results['mortality_difference'].abs() * (1 - mortality_results['p_value']) *
        (mortality_results['significant'].astype(int) * 2 + 1)  # Bonus for significance
    )
    
    # Select top N for LOS (by influence score)
    top_los = los_results.nlargest(n_los, 'los_influence_score')
    top_los_diagnoses = set(top_los['diagnosis'].tolist())
    print(f"\nTop {n_los} diagnoses for LOS:")
    for i, (idx, row) in enumerate(top_los.iterrows(), 1):
        print(f"  {i:2d}. {row['diagnosis']:15s} - LOS diff: {row['los_difference']:6.2f} days, "
              f"p={row['p_value']:.4f}, significant={row['significant']}")
    
    # Select top N for Mortality (by influence score)
    top_mortality = mortality_results.nlargest(n_mortality, 'mortality_influence_score')
    top_mortality_diagnoses = set(top_mortality['diagnosis'].tolist())
    print(f"\nTop {n_mortality} diagnoses for Mortality:")
    for i, (idx, row) in enumerate(top_mortality.iterrows(), 1):
        print(f"  {i:2d}. {row['diagnosis']:15s} - Mort diff: {row['mortality_difference']*100:6.2f}%, "
              f"p={row['p_value']:.4f}, significant={row['significant']}")
    
    # Combine (allows overlaps)
    combined_diagnoses = top_los_diagnoses.union(top_mortality_diagnoses)
    print(f"\nCombined selection: {len(combined_diagnoses)} unique diagnoses")
    print(f"  - From LOS list: {len(top_los_diagnoses)}")
    print(f"  - From Mortality list: {len(top_mortality_diagnoses)}")
    print(f"  - Overlaps: {len(top_los_diagnoses.intersection(top_mortality_diagnoses))}")
    
    # Create combined results DataFrame
    combined_results = []
    for diag in combined_diagnoses:
        los_row = los_results[los_results['diagnosis'] == diag]
        mort_row = mortality_results[mortality_results['diagnosis'] == diag]
        
        combined_row = {
            'diagnosis': diag,
            'in_los_top25': diag in top_los_diagnoses,
            'in_mortality_top25': diag in top_mortality_diagnoses,
            'in_both': diag in top_los_diagnoses and diag in top_mortality_diagnoses,
        }
        
        if len(los_row) > 0:
            los_row = los_row.iloc[0]
            combined_row.update({
                'los_difference': los_row['los_difference'],
                'los_p_value': los_row['p_value'],
                'los_significant': los_row['significant'],
                'los_influence_score': los_row['los_influence_score'],
            })
        else:
            combined_row.update({
                'los_difference': np.nan,
                'los_p_value': np.nan,
                'los_significant': False,
                'los_influence_score': 0.0,
            })
        
        if len(mort_row) > 0:
            mort_row = mort_row.iloc[0]
            combined_row.update({
                'mortality_difference': mort_row['mortality_difference'],
                'mortality_p_value': mort_row['p_value'],
                'mortality_significant': mort_row['significant'],
                'mortality_influence_score': mort_row['mortality_influence_score'],
                'odds_ratio': mort_row.get('odds_ratio', np.nan),
            })
        else:
            combined_row.update({
                'mortality_difference': np.nan,
                'mortality_p_value': np.nan,
                'mortality_significant': False,
                'mortality_influence_score': 0.0,
                'odds_ratio': np.nan,
            })
        
        # Combined influence score
        combined_row['combined_influence_score'] = (
            combined_row.get('los_influence_score', 0.0) + 
            combined_row.get('mortality_influence_score', 0.0)
        )
        
        combined_results.append(combined_row)
    
    combined_df = pd.DataFrame(combined_results)
    combined_df = combined_df.sort_values('combined_influence_score', ascending=False)
    
    return list(combined_diagnoses), combined_df


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description='Analyze diagnosis correlations with LOS and Mortality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  top15    - Analyze top 15 diagnoses from config (default)
  combined - Analyze ALL diagnoses and select top 50 based on combined influence
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['top15', 'combined'],
        default='top15',
        help='Analysis mode: top15 (from config) or combined (top 50 by influence)'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("FACTOR ANALYSIS: LOS AND MORTALITY CORRELATIONS")
    print(f"Mode: {args.mode}")
    print("="*80)
    print()
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data/icu_ecgs_24h/P1"  # 24h ICU ECG dataset
    csv_path = project_root / "data/labeling/labels_csv/records_w_diag_icd10.csv"
    icustays_path = project_root / "data/labeling/labels_csv/icustays.csv"
    admissions_path = project_root / "data/labeling/labels_csv/admissions.csv"
    
    # Step 1: Load all ECGs from directory
    print("="*80)
    print("STEP 1: LOADING ECG RECORDS")
    print("="*80)
    df_ecgs = load_ecg_records_from_directory(data_dir)
    
    # Step 2: Load ICU stays and mortality
    print("\n" + "="*80)
    print("STEP 2: LOADING ICU STAYS AND MORTALITY")
    print("="*80)
    print(f"Loading ICU stays from: {icustays_path}")
    icustays_df = load_icustays(str(icustays_path))
    print(f"Loaded {len(icustays_df):,} ICU stays")
    
    print(f"\nLoading admissions from: {admissions_path}")
    mortality_mapping = load_mortality_mapping(str(admissions_path), icustays_df)
    print(f"Loaded mortality mapping: {sum(mortality_mapping.values())} died, {len(mortality_mapping) - sum(mortality_mapping.values())} survived")
    
    # Create ICU mapper for time-based matching
    icu_mapper = ICUStayMapper(icustays_df, mortality_mapping=mortality_mapping)
    
    # Step 3: Match ECGs with ICU stays (via subject_id + ecg_time, within 24h)
    print("\n" + "="*80)
    print("STEP 3: MATCHING ECGs WITH ICU STAYS (24h window)")
    print("="*80)
    df_stays = match_with_icu_stays(df_ecgs, icustays_df, icu_mapper)
    
    # Step 4: Match with mortality
    df_with_mortality = match_with_mortality(df_stays, icustays_df, mortality_mapping)
    
    # Step 5: Load diagnosis data per stay (if available)
    print("\n" + "="*80)
    print("STEP 4: LOADING DIAGNOSIS DATA (per ICU stay)")
    print("="*80)
    print("  Applying time filter: admittime < ecg_time")
    df_diagnoses = load_diagnosis_data_per_stay(
        csv_path, 
        icustays_df,
        stay_ids=df_with_mortality['stay_id'].unique(),
        apply_time_filter=True,
        admissions_path=admissions_path
    )
    
    # Merge diagnoses with stay data
    if len(df_diagnoses) > 0:
        df_merged = df_with_mortality.merge(df_diagnoses, on='stay_id', how='left', suffixes=('', '_diag'))
        print(f"Merged diagnosis data: {df_merged['stay_id'].notna().sum():,} stays with diagnosis info")
    else:
        df_merged = df_with_mortality.copy()
        print("No diagnosis data available, continuing with demographics only")
    
    # Step 6: Extract diagnoses (if available)
    top_diagnoses = {}
    all_diagnoses_dict = {}
    if len(df_diagnoses) > 0:
        print("\n" + "="*80)
        print("STEP 5: EXTRACTING DIAGNOSES")
        print("="*80)
        diagnosis_columns = ['ed_diag_ed', 'ed_diag_hosp', 'hosp_diag_hosp', 'all_diag_hosp', 'all_diag_all']
        available_diag_cols = [col for col in diagnosis_columns if col in df_merged.columns]
        print(f"Available diagnosis columns: {available_diag_cols}")
        
        if available_diag_cols:
            if args.mode == 'combined':
                # Extract ALL diagnoses (not limited to top 50 by frequency)
                print("Mode: combined - Extracting ALL diagnoses...")
                all_diagnoses = Counter()
                for col in available_diag_cols:
                    if col in df_merged.columns:
                        print(f"  Extracting from {col}...")
                        col_diagnoses = extract_diagnoses_from_column(df_merged, col)
                        all_diagnoses.update(col_diagnoses)
                        print(f"    Found {len(col_diagnoses)} unique diagnoses")
                
                print(f"\nTotal unique diagnoses: {len(all_diagnoses)}")
                print(f"Total diagnosis occurrences: {sum(all_diagnoses.values()):,}")
                all_diagnoses_dict = dict(all_diagnoses)
                top_diagnoses = all_diagnoses_dict  # Use all for analysis
            else:
                # Extract top 50 by frequency (original behavior)
                print("Mode: top15 - Extracting top 50 most frequent diagnoses...")
            top_diagnoses = get_top_diagnoses(df_merged, available_diag_cols, top_n=50)
            print(f"\nTop 20 most frequent diagnoses:")
            for i, (diag, count) in enumerate(list(top_diagnoses.items())[:20], 1):
                print(f"  {i:2d}. {diag:15s}: {count:6,} occurrences")
            
            # Create diagnosis features
            print("\n" + "="*80)
            print("STEP 6: CREATING DIAGNOSIS FEATURES")
            print("="*80)
            df_merged = create_diagnosis_features(df_merged, top_diagnoses)
            print(f"Created {len(top_diagnoses)} binary diagnosis features")
    
    # Step 7: Analyze correlations
    output_dir = project_root / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import yaml
    feature_config_path = project_root / "configs/features/demographic_features.yaml"
    top15_diagnoses = []
    if feature_config_path.exists():
        with open(feature_config_path, 'r') as f:
            feature_config = yaml.safe_load(f)
        top15_diagnoses = feature_config.get('data', {}).get('diagnosis_features', {}).get('diagnosis_list', [])
    
    # Analyze LOS correlations (diagnoses + demographics)
    if len(top_diagnoses) > 0:
        los_results = analyze_diagnosis_los_correlation(df_merged, top_diagnoses)
        
        if len(los_results) > 0:
            print(f"\nTop 20 diagnoses by LOS difference:")
            print(los_results[['diagnosis', 'count', 'los_difference', 'p_value', 'significant']].head(20).to_string())
            
            if args.mode == 'top15':
                # Plot top 15 diagnoses LOS (from config)
                if len(top15_diagnoses) > 0:
                    plot_path = output_dir / "demographic_and_EHR_data" / "top15_diagnoses_los.png"
                    plot_path.parent.mkdir(parents=True, exist_ok=True)
                    plot_top15_diagnoses_los(los_results, top15_diagnoses, save_path=plot_path)
    
    # Analyze mortality correlations (diagnoses + demographics)
    if len(top_diagnoses) > 0:
        mortality_results = analyze_diagnosis_mortality_correlation(df_merged, top_diagnoses)
        
        if len(mortality_results) > 0:
            print(f"\nTop 20 diagnoses by mortality difference:")
            print(mortality_results[['diagnosis', 'count', 'mortality_difference', 'odds_ratio', 'p_value', 'significant']].head(20).to_string())
            
            if args.mode == 'top15':
                # Plot top 15 diagnoses Mortality (from config)
                if len(top15_diagnoses) > 0:
                    plot_path = output_dir / "demographic_and_EHR_data" / "top15_diagnoses_mortality.png"
                    plot_path.parent.mkdir(parents=True, exist_ok=True)
                    plot_top15_diagnoses_mortality(mortality_results, top15_diagnoses, save_path=plot_path)
    
    # Combined mode: Select top 50 based on combined influence
    if args.mode == 'combined' and len(los_results) > 0 and len(mortality_results) > 0:
        print("\n" + "="*80)
        print("STEP 8: SELECTING TOP 50 DIAGNOSES BASED ON COMBINED INFLUENCE")
        print("="*80)
        
        selected_diagnoses, combined_results_df = select_top50_combined_influence(
            los_results, 
            mortality_results, 
            n_los=25, 
            n_mortality=25
        )
        
        print(f"\n{'='*80}")
        print(f"FINAL SELECTION: {len(selected_diagnoses)} DIAGNOSES")
        print(f"{'='*80}")
        print(f"\nSelected diagnoses (sorted by combined influence score):")
        print(combined_results_df[['diagnosis', 'in_los_top25', 'in_mortality_top25', 'in_both', 
                                    'los_difference', 'mortality_difference', 'combined_influence_score']].to_string())
        
        # Save visualizations
        combined_output_dir = output_dir / "diagnosis_selection_combined"
        combined_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot top 50 diagnoses
        plot_path_los = combined_output_dir / "top50_diagnoses_los.png"
        plot_top15_diagnoses_los(los_results, selected_diagnoses, save_path=plot_path_los)
        
        plot_path_mortality = combined_output_dir / "top50_diagnoses_mortality.png"
        plot_top15_diagnoses_mortality(mortality_results, selected_diagnoses, save_path=plot_path_mortality)
        
        # Create combined visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        
        los_selected = los_results[los_results['diagnosis'].isin(selected_diagnoses)].copy()
        mort_selected = mortality_results[mortality_results['diagnosis'].isin(selected_diagnoses)].copy()
        
        # Rename columns to avoid conflicts
        los_selected = los_selected.rename(columns={'p_value': 'los_p_value', 'significant': 'los_significant'})
        mort_selected = mort_selected.rename(columns={'p_value': 'mortality_p_value', 'significant': 'mortality_significant'})
        
        merged_viz = los_selected[['diagnosis', 'los_difference', 'los_p_value', 'los_significant']].merge(
            mort_selected[['diagnosis', 'mortality_difference', 'mortality_p_value', 'mortality_significant']],
            on='diagnosis',
            how='outer'
        )
        
        merged_viz = merged_viz.set_index('diagnosis').reindex(combined_results_df['diagnosis']).reset_index()
        
        # LOS Plot
        ax1 = axes[0]
        los_sorted = merged_viz.sort_values('los_difference', ascending=True)
        
        def get_color_los(p_value, los_diff):
            if pd.isna(p_value) or p_value >= 0.05:
                return '#808080'  # Gray = not significant
            elif los_diff > 0:
                return '#d62728'  # Red = significant positive (longer LOS = bad)
            else:
                return '#2ca02c'  # Green = significant negative (shorter LOS = good)
        
        colors_los = [get_color_los(row['los_p_value'], row['los_difference']) 
                      for _, row in los_sorted.iterrows()]
        bars1 = ax1.barh(range(len(los_sorted)), los_sorted['los_difference'].fillna(0), 
                         color=colors_los, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(range(len(los_sorted)))
        ax1.set_yticklabels(los_sorted['diagnosis'], fontsize=10)
        ax1.set_xlabel('LOS Difference (days)', fontsize=13, fontweight='bold')
        ax1.set_title('Top Diagnoses: Influence on LOS\n(Red = significant positive/bad, Green = significant negative/good, Gray = not significant)', 
                      fontsize=14, fontweight='bold', pad=20)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # Mortality Plot
        ax2 = axes[1]
        mort_sorted = merged_viz.sort_values('mortality_difference', ascending=True)
        
        def get_color_mortality(p_value, mort_diff):
            if pd.isna(p_value) or p_value >= 0.05:
                return '#808080'  # Gray = not significant
            elif mort_diff > 0:
                return '#d62728'  # Red = significant positive (higher mortality = bad)
            else:
                return '#2ca02c'  # Green = significant negative (lower mortality = good)
        
        colors_mort = [get_color_mortality(row['mortality_p_value'], row['mortality_difference']) 
                       for _, row in mort_sorted.iterrows()]
        bars2 = ax2.barh(range(len(mort_sorted)), mort_sorted['mortality_difference'].fillna(0) * 100, 
                         color=colors_mort, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_yticks(range(len(mort_sorted)))
        ax2.set_yticklabels(mort_sorted['diagnosis'], fontsize=10)
        ax2.set_xlabel('Mortality Difference (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Top Diagnoses: Influence on Mortality\n(Red = significant positive/bad, Green = significant negative/good, Gray = not significant)', 
                      fontsize=14, fontweight='bold', pad=20)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        plt.suptitle('Top Diagnoses: Combined Influence on LOS and Mortality', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        plot_path_combined = combined_output_dir / "top50_diagnoses_combined.png"
        plt.savefig(plot_path_combined, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {plot_path_combined}")
        plt.close()
        
        print(f"\nAll visualizations saved to: {combined_output_dir}")
        print(f"  - top50_diagnoses_los.png")
        print(f"  - top50_diagnoses_mortality.png")
        print(f"  - top50_diagnoses_combined.png")
    
    # Analyze demographic factors (per ICU stay) - Detailed analysis
    print("\n" + "="*80)
    print("STEP 7: ANALYZING DEMOGRAPHIC FACTORS (per ICU stay)")
    print("="*80)
    
    if 'age' in df_merged.columns or 'gender' in df_merged.columns:
        # LOS correlations
        demographic_los_results = analyze_demographic_los_correlation(df_merged)
        
        if len(demographic_los_results) > 0:
            print(f"\nDemographic-LOS correlation results:")
            print(demographic_los_results.to_string())
        
        # Mortality correlations
        demographic_mortality_results = analyze_demographic_mortality_correlation(df_merged)
        
        if len(demographic_mortality_results) > 0:
            print(f"\nDemographic-Mortality correlation results:")
            print(demographic_mortality_results.to_string())
        
        # Plot demographic influence
        if len(demographic_los_results) > 0 and len(demographic_mortality_results) > 0:
            plot_path = output_dir / "demographic_and_EHR_data" / "demographic_influence.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_demographic_influence(demographic_los_results, demographic_mortality_results, save_path=plot_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

