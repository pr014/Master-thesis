"""
Analyze the influence of ICU unit (first_careunit) on LOS and Mortality.
Analysis is done per stay_id (not per patient), as patients can have multiple stays.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, Tuple
import sys

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIS = True
except ImportError:
    HAS_VIS = False
    print("Warning: matplotlib/seaborn not available, skipping visualizations")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_data() -> pd.DataFrame:
    """Load icustays and admissions data."""
    print("Loading data...")
    
    # Load icustays
    icustays_path = project_root / "data/labeling/labels_csv/icustays.csv"
    icustays = pd.read_csv(icustays_path)
    print(f"  Loaded {len(icustays):,} ICU stays")
    
    # Load admissions for mortality
    admissions_path = project_root / "data/labeling/labels_csv/admissions.csv"
    admissions = pd.read_csv(admissions_path, usecols=['hadm_id', 'hospital_expire_flag'])
    print(f"  Loaded {len(admissions):,} admissions")
    
    # Merge
    df = icustays.merge(admissions, on='hadm_id', how='left')
    print(f"  Merged data: {len(df):,} stays")
    
    return df


def analyze_los_by_icu(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze LOS by ICU unit."""
    print("\n" + "="*80)
    print("LENGTH OF STAY (LOS) ANALYSIS BY ICU UNIT")
    print("="*80)
    
    # Group by ICU unit
    los_stats = df.groupby('first_careunit').agg({
        'los': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'stay_id': 'nunique'
    }).round(2)
    
    los_stats.columns = ['count', 'mean_los', 'median_los', 'std_los', 'min_los', 'max_los', 'unique_stays']
    los_stats = los_stats.sort_values('mean_los', ascending=False)
    
    print("\nLOS Statistics by ICU Unit:")
    print(los_stats)
    
    # Additional analysis: distribution percentiles
    print("\nLOS Percentiles (25%, 50%, 75%) by ICU Unit:")
    percentiles = df.groupby('first_careunit')['los'].quantile([0.25, 0.5, 0.75]).unstack()
    percentiles.columns = ['p25', 'p50', 'p75']
    percentiles = percentiles.sort_values('p50', ascending=False)
    print(percentiles.round(2))
    
    # Very short stays (< 1 day)
    print("\nVery Short Stays (< 1 day) by ICU Unit:")
    short_stays = df[df['los'] < 1.0].groupby('first_careunit').size()
    total_stays = df.groupby('first_careunit').size()
    short_stay_rate = (short_stays / total_stays * 100).sort_values(ascending=False)
    short_stay_df = pd.DataFrame({
        'count': short_stays,
        'total': total_stays,
        'rate_percent': short_stay_rate
    }).round(2)
    print(short_stay_df)
    
    # Long stays (> 7 days)
    print("\nLong Stays (> 7 days) by ICU Unit:")
    long_stays = df[df['los'] > 7.0].groupby('first_careunit').size()
    long_stay_rate = (long_stays / total_stays * 100).sort_values(ascending=False)
    long_stay_df = pd.DataFrame({
        'count': long_stays,
        'total': total_stays,
        'rate_percent': long_stay_rate
    }).round(2)
    print(long_stay_df)
    
    return los_stats


def analyze_mortality_by_icu(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze mortality by ICU unit."""
    print("\n" + "="*80)
    print("MORTALITY ANALYSIS BY ICU UNIT")
    print("="*80)
    
    # Filter out missing mortality data
    df_mort = df[df['hospital_expire_flag'].notna()].copy()
    
    # Group by ICU unit
    mortality_stats = df_mort.groupby('first_careunit').agg({
        'hospital_expire_flag': ['count', 'sum', 'mean'],
        'stay_id': 'nunique'
    }).round(4)
    
    mortality_stats.columns = ['total', 'deaths', 'mortality_rate', 'unique_stays']
    mortality_stats = mortality_stats.sort_values('mortality_rate', ascending=False)
    
    print("\nMortality Statistics by ICU Unit:")
    print(mortality_stats)
    
    return mortality_stats


def statistical_tests(df: pd.DataFrame) -> None:
    """Perform statistical tests to compare ICU units."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    # Get top 10 most common ICU units
    top_units = df['first_careunit'].value_counts().head(10).index.tolist()
    
    print(f"\nComparing Top 10 ICU Units (Mann-Whitney U Test):")
    print(f"Units: {[u[:30] for u in top_units]}")
    
    # LOS comparisons
    print("\n1. LOS Comparisons (Mann-Whitney U Test):")
    los_results = []
    for i, unit1 in enumerate(top_units):
        for unit2 in top_units[i+1:]:
            los1 = df[df['first_careunit'] == unit1]['los']
            los2 = df[df['first_careunit'] == unit2]['los']
            
            if len(los1) > 0 and len(los2) > 0:
                stat, pval = stats.mannwhitneyu(los1, los2, alternative='two-sided')
                los_results.append({
                    'unit1': unit1,
                    'unit2': unit2,
                    'p_value': pval,
                    'mean_los1': los1.mean(),
                    'mean_los2': los2.mean(),
                    'significant': pval < 0.05
                })
    
    los_df = pd.DataFrame(los_results)
    significant_los = los_df[los_df['significant']].sort_values('p_value')
    
    if len(significant_los) > 0:
        print(f"\nSignificant LOS differences (p < 0.05):")
        print(significant_los[['unit1', 'unit2', 'p_value', 'mean_los1', 'mean_los2']].head(10))
    else:
        print("\nNo significant LOS differences found between top 10 units.")
    
    # Mortality comparisons (Chi-square test)
    print("\n2. Mortality Comparisons (Chi-square Test):")
    df_mort = df[df['hospital_expire_flag'].notna()].copy()
    
    mortality_results = []
    for i, unit1 in enumerate(top_units):
        for unit2 in top_units[i+1:]:
            mort1 = df_mort[df_mort['first_careunit'] == unit1]['hospital_expire_flag']
            mort2 = df_mort[df_mort['first_careunit'] == unit2]['hospital_expire_flag']
            
            if len(mort1) > 0 and len(mort2) > 0:
                # Create contingency table
                contingency = pd.crosstab(
                    pd.concat([pd.Series([0]*len(mort1), index=mort1.index), 
                              pd.Series([1]*len(mort2), index=mort2.index)]),
                    pd.concat([mort1, mort2])
                )
                
                if contingency.shape == (2, 2):
                    chi2, pval, dof, expected = stats.chi2_contingency(contingency)
                    mortality_results.append({
                        'unit1': unit1,
                        'unit2': unit2,
                        'p_value': pval,
                        'mortality_rate1': mort1.mean(),
                        'mortality_rate2': mort2.mean(),
                        'significant': pval < 0.05
                    })
    
    mort_df = pd.DataFrame(mortality_results)
    significant_mort = mort_df[mort_df['significant']].sort_values('p_value')
    
    if len(significant_mort) > 0:
        print(f"\nSignificant mortality differences (p < 0.05):")
        print(significant_mort[['unit1', 'unit2', 'p_value', 'mortality_rate1', 'mortality_rate2']].head(10))
    else:
        print("\nNo significant mortality differences found between top 10 units.")


def create_visualizations(df: pd.DataFrame, los_stats: pd.DataFrame, mortality_stats: pd.DataFrame) -> None:
    """Create visualizations."""
    if not HAS_VIS:
        print("\nSkipping visualizations (matplotlib/seaborn not available)")
        return
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    output_dir = project_root / "outputs/analysis/icu_unit_influence"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Filter to top 10 most common ICU units for readability
    top_units = df['first_careunit'].value_counts().head(10).index.tolist()
    df_top = df[df['first_careunit'].isin(top_units)].copy()
    
    # 1. LOS Boxplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # LOS Boxplot
    ax1 = axes[0, 0]
    df_top.boxplot(column='los', by='first_careunit', ax=ax1, rot=45)
    ax1.set_title('LOS Distribution by ICU Unit (Top 10)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('ICU Unit', fontsize=12)
    ax1.set_ylabel('Length of Stay (days)', fontsize=12)
    ax1.set_xticklabels([u[:30] for u in top_units], rotation=45, ha='right')
    plt.suptitle('')  # Remove default title
    
    # LOS Mean Bar Plot
    ax2 = axes[0, 1]
    los_means = los_stats.loc[top_units, 'mean_los'].sort_values(ascending=True)
    ax2.barh(range(len(los_means)), los_means.values)
    ax2.set_yticks(range(len(los_means)))
    ax2.set_yticklabels([u[:40] for u in los_means.index], fontsize=9)
    ax2.set_xlabel('Mean LOS (days)', fontsize=12)
    ax2.set_title('Mean LOS by ICU Unit', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Mortality Rate Bar Plot
    ax3 = axes[1, 0]
    df_mort = df_top[df_top['hospital_expire_flag'].notna()].copy()
    mortality_rates = df_mort.groupby('first_careunit')['hospital_expire_flag'].mean().loc[top_units].sort_values(ascending=True)
    ax3.barh(range(len(mortality_rates)), mortality_rates.values * 100)
    ax3.set_yticks(range(len(mortality_rates)))
    ax3.set_yticklabels([u[:40] for u in mortality_rates.index], fontsize=9)
    ax3.set_xlabel('Mortality Rate (%)', fontsize=12)
    ax3.set_title('Mortality Rate by ICU Unit', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Sample Size
    ax4 = axes[1, 1]
    sample_sizes = df_top['first_careunit'].value_counts().loc[top_units].sort_values(ascending=True)
    ax4.barh(range(len(sample_sizes)), sample_sizes.values)
    ax4.set_yticks(range(len(sample_sizes)))
    ax4.set_yticklabels([u[:40] for u in sample_sizes.index], fontsize=9)
    ax4.set_xlabel('Number of Stays', fontsize=12)
    ax4.set_title('Sample Size by ICU Unit', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'icu_unit_influence_overview.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'icu_unit_influence_overview.png'}")
    
    # 2. Detailed LOS Distribution
    fig, ax = plt.subplots(figsize=(14, 8))
    for unit in top_units[:5]:  # Top 5 for readability
        los_data = df_top[df_top['first_careunit'] == unit]['los']
        ax.hist(los_data, bins=50, alpha=0.6, label=unit[:40], density=True)
    ax.set_xlabel('Length of Stay (days)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('LOS Distribution by ICU Unit (Top 5)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'los_distribution_by_icu.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'los_distribution_by_icu.png'}")
    
    plt.close('all')


def main():
    """Main analysis function."""
    print("="*80)
    print("ICU UNIT INFLUENCE ANALYSIS")
    print("Analysis per stay_id (patients can have multiple stays)")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Analyze LOS
    los_stats = analyze_los_by_icu(df)
    
    # Analyze mortality
    mortality_stats = analyze_mortality_by_icu(df)
    
    # Statistical tests
    statistical_tests(df)
    
    # Create visualizations
    create_visualizations(df, los_stats, mortality_stats)
    
    # Save summary
    output_dir = project_root / "outputs/analysis/icu_unit_influence"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_dir / "icu_unit_influence_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("ICU UNIT INFLUENCE ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write("LOS Statistics:\n")
        f.write(str(los_stats))
        f.write("\n\nMortality Statistics:\n")
        f.write(str(mortality_stats))
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Summary saved to: {summary_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

