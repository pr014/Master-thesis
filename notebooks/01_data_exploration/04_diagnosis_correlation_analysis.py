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
    python notebooks/01_data_exploration/04_diagnosis_correlation_analysis.py
"""

from pathlib import Path
import sys
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
    stay_ids: Optional[pd.Series] = None
) -> pd.DataFrame:
    """Load diagnosis data from CSV file and aggregate per ICU stay.
    
    Args:
        csv_path: Path to records_w_diag_icd10.csv
        icustays_df: DataFrame with ICU stays (for matching via hadm_id)
        stay_ids: Optional Series of stay_ids to filter
        
    Returns:
        DataFrame with diagnosis columns aggregated per stay_id
    """
    print(f"Loading diagnosis data from: {csv_path}")
    
    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}. Continuing without diagnosis data.")
        return pd.DataFrame()
    
    # Load CSV - we need diagnosis columns and identifiers
    diagnosis_cols = [
        'subject_id', 'hosp_hadm_id', 'gender', 'age',
        'ed_diag_ed', 'ed_diag_hosp', 'hosp_diag_hosp', 
        'all_diag_hosp', 'all_diag_all'
    ]
    
    # Read CSV with diagnosis columns
    df = pd.read_csv(csv_path, usecols=diagnosis_cols, low_memory=False)
    print(f"Loaded {len(df):,} records from CSV")
    
    # Filter ICU stays if provided
    if stay_ids is not None:
        icustays_filtered = icustays_df[icustays_df['stay_id'].isin(stay_ids)].copy()
    else:
        icustays_filtered = icustays_df.copy()
    
    # Match diagnoses to ICU stays via subject_id + hadm_id
    # First, merge with ICU stays to get stay_id
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
    
    # Aggregate diagnoses per stay_id
    agg_dict = {}
    for col in df_with_hadm.columns:
        if col in ['stay_id', 'subject_id', 'hadm_id', 'hosp_hadm_id']:
            continue
        elif col in ['gender', 'age']:
            # For demographics, take first non-null
            agg_dict[col] = 'first'
        else:
            # For diagnosis columns, combine all non-null values
            agg_dict[col] = lambda x: ';'.join(x.dropna().astype(str).unique()) if len(x.dropna()) > 0 else None
    
    df_agg = df_with_hadm.groupby('stay_id').agg(agg_dict).reset_index()
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


def plot_los_correlations(results_df: pd.DataFrame, top_n: int = 20, save_path: Optional[Path] = None):
    """Plot LOS correlation results.
    
    Args:
        results_df: DataFrame with LOS correlation results
        top_n: Number of top diagnoses to plot
        save_path: Optional path to save figure
    """
    if len(results_df) == 0:
        print("No data to plot")
        return
    
    # Get top N by absolute difference
    top_results = results_df.nlargest(top_n, 'los_difference')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: LOS difference
    ax1 = axes[0]
    colors = ['red' if x < 0.05 else 'blue' for x in top_results['p_value']]
    ax1.barh(range(len(top_results)), top_results['los_difference'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_results)))
    ax1.set_yticklabels(top_results['diagnosis'])
    ax1.set_xlabel('LOS Difference (days)', fontsize=12)
    ax1.set_title(f'Top {top_n} Diagnoses by LOS Difference\n(Red = significant p<0.05, Blue = not significant)', fontsize=14)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Spearman correlation
    ax2 = axes[1]
    top_results_corr = results_df.nlargest(top_n, 'spearman_corr')
    colors2 = ['red' if x < 0.05 else 'blue' for x in top_results_corr['spearman_p']]
    ax2.barh(range(len(top_results_corr)), top_results_corr['spearman_corr'], color=colors2, alpha=0.7)
    ax2.set_yticks(range(len(top_results_corr)))
    ax2.set_yticklabels(top_results_corr['diagnosis'])
    ax2.set_xlabel('Spearman Correlation', fontsize=12)
    ax2.set_title(f'Top {top_n} Diagnoses by Spearman Correlation with LOS\n(Red = significant p<0.05, Blue = not significant)', fontsize=14)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved LOS correlation plot to: {save_path}")
    
    plt.show()


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


def main():
    """Main analysis function."""
    print("="*80)
    print("FACTOR ANALYSIS: LOS AND MORTALITY CORRELATIONS")
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
    df_diagnoses = load_diagnosis_data_per_stay(
        csv_path, 
        icustays_df,
        stay_ids=df_with_mortality['stay_id'].unique()
    )
    
    # Merge diagnoses with stay data
    if len(df_diagnoses) > 0:
        df_merged = df_with_mortality.merge(df_diagnoses, on='stay_id', how='left', suffixes=('', '_diag'))
        print(f"Merged diagnosis data: {df_merged['stay_id'].notna().sum():,} stays with diagnosis info")
    else:
        df_merged = df_with_mortality.copy()
        print("No diagnosis data available, continuing with demographics only")
    
    # Step 6: Extract top diagnoses (if available)
    top_diagnoses = {}
    if len(df_diagnoses) > 0:
        print("\n" + "="*80)
        print("STEP 5: EXTRACTING TOP DIAGNOSES")
        print("="*80)
        diagnosis_columns = ['ed_diag_ed', 'ed_diag_hosp', 'hosp_diag_hosp', 'all_diag_hosp', 'all_diag_all']
        available_diag_cols = [col for col in diagnosis_columns if col in df_merged.columns]
        print(f"Available diagnosis columns: {available_diag_cols}")
        
        if available_diag_cols:
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
    
    # Analyze LOS correlations (diagnoses + demographics)
    if len(top_diagnoses) > 0:
        los_results = analyze_diagnosis_los_correlation(df_merged, top_diagnoses)
        
        if len(los_results) > 0:
            print(f"\nTop 20 diagnoses by LOS difference:")
            print(los_results[['diagnosis', 'count', 'los_difference', 'p_value', 'significant']].head(20).to_string())
            
            los_output_path = output_dir / "diagnosis_los_correlations.csv"
            los_results.to_csv(los_output_path, index=False)
            print(f"\nSaved LOS correlation results to: {los_output_path}")
            
            plot_los_correlations(los_results, top_n=20, save_path=output_dir / "diagnosis_los_correlations.png")
    
    # Analyze mortality correlations (diagnoses + demographics)
    if len(top_diagnoses) > 0:
        mortality_results = analyze_diagnosis_mortality_correlation(df_merged, top_diagnoses)
        
        if len(mortality_results) > 0:
            print(f"\nTop 20 diagnoses by mortality difference:")
            print(mortality_results[['diagnosis', 'count', 'mortality_difference', 'odds_ratio', 'p_value', 'significant']].head(20).to_string())
            
            mortality_output_path = output_dir / "diagnosis_mortality_correlations.csv"
            mortality_results.to_csv(mortality_output_path, index=False)
            print(f"\nSaved mortality correlation results to: {mortality_output_path}")
            
            plot_mortality_correlations(mortality_results, top_n=20, save_path=output_dir / "diagnosis_mortality_correlations.png")
    
    # Analyze demographic factors (per ICU stay)
    print("\n" + "="*80)
    print("STEP 7: ANALYZING DEMOGRAPHIC FACTORS (per ICU stay)")
    print("="*80)
    if 'age' in df_merged.columns and 'gender' in df_merged.columns:
        # Age correlation with LOS
        df_valid = df_merged[df_merged['los_days'].notna() & df_merged['age'].notna()].copy()
        if len(df_valid) > 0:
            age_los_corr, age_los_p = spearmanr(df_valid['age'], df_valid['los_days'])
            print(f"Age-LOS correlation (per stay): {age_los_corr:.4f} (p={age_los_p:.4f})")
        
        # Age correlation with mortality
        df_valid = df_merged[df_merged['mortality'].notna() & df_merged['age'].notna()].copy()
        if len(df_valid) > 0:
            age_mort = df_valid[df_valid['mortality'] == 1]['age'].mean()
            age_surv = df_valid[df_valid['mortality'] == 0]['age'].mean()
            print(f"Mean age (per stay) - Died: {age_mort:.1f}, Survived: {age_surv:.1f}")
        
        # Gender correlation
        if 'gender' in df_merged.columns:
            df_valid = df_merged[df_merged['mortality'].notna() & df_merged['gender'].notna()].copy()
            if len(df_valid) > 0:
                gender_mort = pd.crosstab(df_valid['gender'], df_valid['mortality'])
                print(f"\nGender-Mortality distribution (per stay):")
                print(gender_mort)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

