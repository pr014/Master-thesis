#!/usr/bin/env python3
"""Quick script to analyze demographic factors (Age & Sex) influence on LOS and Mortality.

This script extracts the demographic analysis from the full correlation analysis
and shows the results.
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import only what we need (avoid matplotlib/seaborn for quick analysis)
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, chi2_contingency

from src.data.labeling import load_icustays, load_mortality_mapping, ICUStayMapper

def analyze_demographic_los_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlation between demographics (age, gender) and LOS."""
    results = []
    
    print("\n" + "="*80)
    print("DEMOGRAPHIC-LOS CORRELATIONS")
    print("="*80)
    
    df_valid = df[df['los_days'].notna()].copy()
    print(f"\nICU stays with valid LOS: {len(df_valid):,}")
    
    if len(df_valid) == 0:
        return pd.DataFrame()
    
    # Age analysis
    if 'age' in df_valid.columns:
        df_age = df_valid[df_valid['age'].notna()].copy()
        if len(df_age) > 0:
            # Spearman correlation
            try:
                corr, corr_p = spearmanr(df_age['age'], df_age['los_days'])
            except:
                corr, corr_p = 0.0, 1.0
            
            # Age quartiles
            age_q25 = df_age['age'].quantile(0.25)
            age_q75 = df_age['age'].quantile(0.75)
            
            high_age = df_age[df_age['age'] >= age_q75]
            low_age = df_age[df_age['age'] <= age_q25]
            
            los_high = los_low = los_diff = p_value = None
            
            if len(high_age) > 0 and len(low_age) > 0:
                los_high = high_age['los_days'].mean()
                los_low = low_age['los_days'].mean()
                los_diff = los_high - los_low
                
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
            
            print(f"\nAge Analysis (n={len(df_age):,}):")
            print(f"  Mean age: {df_age['age'].mean():.1f} ± {df_age['age'].std():.1f} years")
            print(f"  Range: {df_age['age'].min():.0f} - {df_age['age'].max():.0f} years")
            print(f"  Age-LOS Spearman correlation: {corr:.4f} (p={corr_p:.4f})")
            if los_diff is not None:
                print(f"  LOS difference (high age ≥{age_q75:.0f} vs low age ≤{age_q25:.0f}): {los_diff:.2f} days")
                print(f"    High age LOS: {los_high:.2f} days, Low age LOS: {los_low:.2f} days")
                print(f"    Statistical significance: p={p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
    
    # Gender analysis
    if 'gender' in df_valid.columns:
        df_gender = df_valid[df_valid['gender'].notna()].copy()
        if len(df_gender) > 0:
            genders = df_gender['gender'].unique()
            if len(genders) >= 2:
                print(f"\nGender Analysis (n={len(df_gender):,}):")
                for gender in genders:
                    gender_group = df_gender[df_gender['gender'] == gender]
                    other_groups = df_gender[df_gender['gender'] != gender]
                    
                    if len(gender_group) > 0 and len(other_groups) > 0:
                        los_with = gender_group['los_days'].mean()
                        los_without = other_groups['los_days'].mean()
                        los_diff = los_with - los_without
                        
                        try:
                            stat, p_value = mannwhitneyu(
                                gender_group['los_days'].dropna(),
                                other_groups['los_days'].dropna(),
                                alternative='two-sided'
                            )
                        except:
                            p_value = 1.0
                        
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
                        
                        print(f"  Gender {gender}:")
                        print(f"    LOS: {los_with:.2f} days (n={len(gender_group):,}, {len(gender_group)/len(df_gender)*100:.1f}%)")
                        print(f"    LOS difference vs others: {los_diff:.2f} days")
                        print(f"    Statistical significance: p={p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
    
    return pd.DataFrame(results)


def analyze_demographic_mortality_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlation between demographics (age, gender) and mortality."""
    results = []
    
    print("\n" + "="*80)
    print("DEMOGRAPHIC-MORTALITY CORRELATIONS")
    print("="*80)
    
    df_valid = df[df['mortality'].notna()].copy()
    print(f"\nICU stays with valid mortality: {len(df_valid):,}")
    
    if len(df_valid) == 0:
        return pd.DataFrame()
    
    overall_mortality = df_valid['mortality'].mean()
    print(f"Overall mortality rate: {overall_mortality*100:.2f}%")
    
    # Age analysis
    if 'age' in df_valid.columns:
        df_age = df_valid[df_valid['age'].notna()].copy()
        if len(df_age) > 0:
            age_q75 = df_age['age'].quantile(0.75)
            age_q25 = df_age['age'].quantile(0.25)
            
            high_age = df_age[df_age['age'] >= age_q75]
            low_age = df_age[df_age['age'] <= age_q25]
            
            mortality_high = mortality_low = mortality_diff = odds_ratio = p_value = None
            
            if len(high_age) > 0 and len(low_age) > 0:
                mortality_high = high_age['mortality'].mean()
                mortality_low = low_age['mortality'].mean()
                mortality_diff = mortality_high - mortality_low
                
                df_age_binary = df_age.copy()
                df_age_binary['high_age'] = (df_age_binary['age'] >= age_q75).astype(int)
                
                contingency = pd.crosstab(df_age_binary['high_age'], df_age_binary['mortality'])
                if contingency.shape == (2, 2):
                    try:
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                    except:
                        p_value = 1.0
                    
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
            
            print(f"\nAge Analysis (n={len(df_age):,}):")
            print(f"  Mean age - Died: {age_mort:.1f} years, Survived: {age_surv:.1f} years")
            print(f"  Age difference: {age_mort - age_surv:.1f} years")
            if mortality_diff is not None:
                print(f"  Mortality (high age ≥{age_q75:.0f} vs low age ≤{age_q25:.0f}):")
                print(f"    High age: {mortality_high*100:.2f}%, Low age: {mortality_low*100:.2f}%")
                print(f"    Mortality difference: {mortality_diff*100:.2f}%")
                print(f"    Odds Ratio: {odds_ratio:.2f}")
                print(f"    Statistical significance: p={p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
    
    # Gender analysis
    if 'gender' in df_valid.columns:
        df_gender = df_valid[df_valid['gender'].notna()].copy()
        if len(df_gender) > 0:
            genders = df_gender['gender'].unique()
            if len(genders) >= 2:
                print(f"\nGender Analysis (n={len(df_gender):,}):")
                for gender in genders:
                    gender_group = df_gender[df_gender['gender'] == gender]
                    other_groups = df_gender[df_gender['gender'] != gender]
                    
                    if len(gender_group) > 0 and len(other_groups) > 0:
                        mortality_with = gender_group['mortality'].mean()
                        mortality_without = other_groups['mortality'].mean()
                        mortality_diff = mortality_with - mortality_without
                        mortality_ratio = mortality_with / mortality_without if mortality_without > 0 else float('inf')
                        
                        contingency = pd.crosstab(df_gender['gender'] == gender, df_gender['mortality'])
                        if contingency.shape == (2, 2):
                            try:
                                chi2, p_value, dof, expected = chi2_contingency(contingency)
                            except:
                                p_value = 1.0
                            
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
                        
                        print(f"  Gender {gender}:")
                        print(f"    Mortality: {mortality_with*100:.2f}% (n={len(gender_group):,}, {len(gender_group)/len(df_gender)*100:.1f}%)")
                        print(f"    Mortality difference vs others: {mortality_diff*100:.2f}%")
                        print(f"    Odds Ratio: {odds_ratio:.2f}")
                        print(f"    Statistical significance: p={p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
    
    return pd.DataFrame(results)


def main():
    """Load data and run demographic analysis."""
    print("="*80)
    print("DEMOGRAPHIC FACTORS ANALYSIS: Age & Sex influence on LOS and Mortality")
    print("="*80)
    
    # Load data - we need the merged data with demographics
    # For now, let's try to load from existing analysis or create a simple test
    project_root = Path(__file__).parent.parent.parent
    
    # Try to load from records CSV directly
    csv_path = project_root / "data/labeling/labels_csv/records_w_diag_icd10.csv"
    icustays_path = project_root / "data/labeling/labels_csv/icustays.csv"
    admissions_path = project_root / "data/labeling/labels_csv/admissions.csv"
    
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return
    
    print(f"\nLoading data from: {csv_path}")
    df_records = pd.read_csv(csv_path, usecols=['subject_id', 'hosp_hadm_id', 'gender', 'age'], low_memory=False)
    print(f"Loaded {len(df_records):,} records")
    
    # Load ICU stays
    icustays_df = load_icustays(str(icustays_path))
    print(f"Loaded {len(icustays_df):,} ICU stays")
    
    # Load mortality
    mortality_mapping = load_mortality_mapping(str(admissions_path), icustays_df)
    print(f"Loaded mortality mapping: {sum(mortality_mapping.values())} died, {len(mortality_mapping) - sum(mortality_mapping.values())} survived")
    
    # Merge demographics with ICU stays
    df_merged = icustays_df.merge(
        df_records[['subject_id', 'hosp_hadm_id', 'gender', 'age']],
        on='subject_id',
        how='left'
    )
    
    # Add mortality
    df_merged['mortality'] = df_merged['stay_id'].map(mortality_mapping)
    
    # Get LOS from icustays
    if 'los' in df_merged.columns:
        df_merged['los_days'] = df_merged['los']
    elif 'los_days' not in df_merged.columns:
        # Calculate from intime/outtime if available
        if 'intime' in df_merged.columns and 'outtime' in df_merged.columns:
            df_merged['intime'] = pd.to_datetime(df_merged['intime'])
            df_merged['outtime'] = pd.to_datetime(df_merged['outtime'])
            df_merged['los_days'] = (df_merged['outtime'] - df_merged['intime']).dt.total_seconds() / 86400
    
    print(f"\nMerged data: {len(df_merged):,} ICU stays")
    print(f"  With age: {df_merged['age'].notna().sum():,}")
    print(f"  With gender: {df_merged['gender'].notna().sum():,}")
    print(f"  With LOS: {df_merged['los_days'].notna().sum():,}")
    print(f"  With mortality: {df_merged['mortality'].notna().sum():,}")
    
    # Run analyses
    los_results = analyze_demographic_los_correlation(df_merged)
    mortality_results = analyze_demographic_mortality_correlation(df_merged)
    
    # Save results
    output_dir = project_root / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(los_results) > 0:
        los_output_path = output_dir / "demographic_los_correlations.csv"
        los_results.to_csv(los_output_path, index=False)
        print(f"\n✅ Saved demographic LOS results to: {los_output_path}")
    
    if len(mortality_results) > 0:
        mortality_output_path = output_dir / "demographic_mortality_correlations.csv"
        mortality_results.to_csv(mortality_output_path, index=False)
        print(f"✅ Saved demographic mortality results to: {mortality_output_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

