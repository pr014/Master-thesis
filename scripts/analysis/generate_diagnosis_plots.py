#!/usr/bin/env python3
"""Generate plots for top 15 diagnoses and demographics.

This script loads existing CSV results and generates the plots.
It requires matplotlib to be available.
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from typing import List, Optional
except ImportError as e:
    print(f"ERROR: Required packages not available: {e}")
    print("Please install: matplotlib, pandas, numpy")
    sys.exit(1)

import yaml


def plot_top15_diagnoses_los(results_df: pd.DataFrame, diagnosis_list: List[str], save_path: Optional[Path] = None):
    """Plot LOS correlation for top 15 diagnoses."""
    if len(results_df) == 0:
        print("No data to plot")
        return
    
    # Filter to only top 15 diagnoses
    top15_results = results_df[results_df['diagnosis'].isin(diagnosis_list)].copy()
    
    if len(top15_results) == 0:
        print("None of the top 15 diagnoses found in results")
        return
    
    # Sort by LOS difference (ascending for horizontal bar)
    top15_results = top15_results.sort_values('los_difference', ascending=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create color map: red for significant, green for not significant
    colors = ['#d62728' if x < 0.05 else '#2ca02c' for x in top15_results['p_value']]
    
    # Horizontal bar plot
    bars = ax.barh(range(len(top15_results)), top15_results['los_difference'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize y-axis labels (diagnosis codes)
    ax.set_yticks(range(len(top15_results)))
    ax.set_yticklabels(top15_results['diagnosis'], fontsize=11)
    
    # Labels and title
    ax.set_xlabel('LOS Difference (days)', fontsize=13, fontweight='bold')
    ax.set_title('Top 15 Diagnoses: Influence on Length of Stay (LOS)\n(Red = significant p<0.05, Green = not significant)', 
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
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved LOS plot to: {save_path}")
    
    plt.close()


def plot_top15_diagnoses_mortality(results_df: pd.DataFrame, diagnosis_list: List[str], save_path: Optional[Path] = None):
    """Plot mortality correlation for top 15 diagnoses."""
    if len(results_df) == 0:
        print("No data to plot")
        return
    
    # Filter to only top 15 diagnoses
    top15_results = results_df[results_df['diagnosis'].isin(diagnosis_list)].copy()
    
    if len(top15_results) == 0:
        print("None of the top 15 diagnoses found in results")
        return
    
    # Sort by mortality difference (ascending for horizontal bar)
    top15_results = top15_results.sort_values('mortality_difference', ascending=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create color map: red for significant, green for not significant
    colors = ['#d62728' if x < 0.05 else '#2ca02c' for x in top15_results['p_value']]
    
    # Horizontal bar plot (mortality difference in percentage)
    bars = ax.barh(range(len(top15_results)), top15_results['mortality_difference'] * 100, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize y-axis labels (diagnosis codes)
    ax.set_yticks(range(len(top15_results)))
    ax.set_yticklabels(top15_results['diagnosis'], fontsize=11)
    
    # Labels and title
    ax.set_xlabel('Mortality Difference (%)', fontsize=13, fontweight='bold')
    ax.set_title('Top 15 Diagnoses: Influence on Mortality\n(Red = significant p<0.05, Green = not significant)', 
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
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved mortality plot to: {save_path}")
    
    plt.close()


def plot_demographic_influence(demographic_los_df: pd.DataFrame, demographic_mortality_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot demographic factors (Age & Sex) influence on LOS and Mortality."""
    if len(demographic_los_df) == 0 and len(demographic_mortality_df) == 0:
        print("No demographic data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # === LOS Analysis ===
    # Age LOS
    if len(demographic_los_df) > 0 and 'age' in demographic_los_df['factor'].values:
        age_row = demographic_los_df[demographic_los_df['factor'] == 'age'].iloc[0]
        ax1 = axes[0, 0]
        
        # Create age groups visualization
        age_data = {
            'High Age\n(≥Q75)': age_row.get('los_high_age', 0),
            'Low Age\n(≤Q25)': age_row.get('los_low_age', 0)
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
    else:
        axes[0, 0].text(0.5, 0.5, 'No Age LOS data', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Age Influence on LOS', fontsize=12, fontweight='bold')
    
    # Gender LOS
    gender_los = demographic_los_df[demographic_los_df['factor'].str.startswith('gender_')] if len(demographic_los_df) > 0 else pd.DataFrame()
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
    else:
        axes[0, 1].text(0.5, 0.5, 'No Gender LOS data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Gender Influence on LOS', fontsize=12, fontweight='bold')
    
    # === Mortality Analysis ===
    # Age Mortality
    if len(demographic_mortality_df) > 0 and 'age' in demographic_mortality_df['factor'].values:
        age_mort_row = demographic_mortality_df[demographic_mortality_df['factor'] == 'age'].iloc[0]
        ax3 = axes[1, 0]
        
        age_mort_data = {
            'High Age\n(≥Q75)': age_mort_row.get('mortality_high_age', 0) * 100,
            'Low Age\n(≤Q25)': age_mort_row.get('mortality_low_age', 0) * 100
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
    else:
        axes[1, 0].text(0.5, 0.5, 'No Age Mortality data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Age Influence on Mortality', fontsize=12, fontweight='bold')
    
    # Gender Mortality
    gender_mort = demographic_mortality_df[demographic_mortality_df['factor'].str.startswith('gender_')] if len(demographic_mortality_df) > 0 else pd.DataFrame()
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
    else:
        axes[1, 1].text(0.5, 0.5, 'No Gender Mortality data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Gender Influence on Mortality', fontsize=12, fontweight='bold')
    
    plt.suptitle('Demographic Factors: Influence on LOS and Mortality', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved demographic plot to: {save_path}")
    
    plt.close()


def main():
    """Load existing CSV results and generate plots."""
    print("="*80)
    print("GENERATING DIAGNOSIS AND DEMOGRAPHIC PLOTS")
    print("="*80)
    
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "outputs" / "analysis" / "demographic_and_EHR_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get top 15 diagnosis list from config
    feature_config_path = project_root / "configs/features/demographic_features.yaml"
    top15_diagnoses = []
    if feature_config_path.exists():
        with open(feature_config_path, 'r') as f:
            feature_config = yaml.safe_load(f)
        top15_diagnoses = feature_config.get('data', {}).get('diagnosis_features', {}).get('diagnosis_list', [])
        print(f"\nLoaded top 15 diagnoses from config: {len(top15_diagnoses)} diagnoses")
    else:
        print(f"\nWARNING: Config file not found: {feature_config_path}")
    
    # Load diagnosis results from CSV (if they exist)
    los_csv_path = project_root / "outputs" / "analysis" / "diagnosis_los_correlations.csv"
    mortality_csv_path = project_root / "outputs" / "analysis" / "diagnosis_mortality_correlations.csv"
    
    if los_csv_path.exists():
        print(f"\nLoading LOS results from: {los_csv_path}")
        los_results = pd.read_csv(los_csv_path)
        print(f"Loaded {len(los_results)} diagnosis results")
        
        if len(top15_diagnoses) > 0:
            plot_path = output_dir / "top15_diagnoses_los.png"
            plot_top15_diagnoses_los(los_results, top15_diagnoses, save_path=plot_path)
    else:
        print(f"\nWARNING: LOS results CSV not found: {los_csv_path}")
    
    if mortality_csv_path.exists():
        print(f"\nLoading mortality results from: {mortality_csv_path}")
        mortality_results = pd.read_csv(mortality_csv_path)
        print(f"Loaded {len(mortality_results)} diagnosis results")
        
        if len(top15_diagnoses) > 0:
            plot_path = output_dir / "top15_diagnoses_mortality.png"
            plot_top15_diagnoses_mortality(mortality_results, top15_diagnoses, save_path=plot_path)
    else:
        print(f"\nWARNING: Mortality results CSV not found: {mortality_csv_path}")
    
    # Load demographic results from CSV (if they exist)
    demo_los_csv_path = project_root / "outputs" / "analysis" / "demographic_los_correlations.csv"
    demo_mort_csv_path = project_root / "outputs" / "analysis" / "demographic_mortality_correlations.csv"
    
    demographic_los_df = pd.DataFrame()
    demographic_mortality_df = pd.DataFrame()
    
    if demo_los_csv_path.exists():
        print(f"\nLoading demographic LOS results from: {demo_los_csv_path}")
        demographic_los_df = pd.read_csv(demo_los_csv_path)
        print(f"Loaded {len(demographic_los_df)} demographic LOS results")
    else:
        print(f"\nWARNING: Demographic LOS CSV not found: {demo_los_csv_path}")
    
    if demo_mort_csv_path.exists():
        print(f"\nLoading demographic mortality results from: {demo_mort_csv_path}")
        demographic_mortality_df = pd.read_csv(demo_mort_csv_path)
        print(f"Loaded {len(demographic_mortality_df)} demographic mortality results")
    else:
        print(f"\nWARNING: Demographic mortality CSV not found: {demo_mort_csv_path}")
    
    # Plot demographic influence
    if len(demographic_los_df) > 0 or len(demographic_mortality_df) > 0:
        plot_path = output_dir / "demographic_influence.png"
        plot_demographic_influence(demographic_los_df, demographic_mortality_df, save_path=plot_path)
    
    print("\n" + "="*80)
    print("PLOT GENERATION COMPLETE")
    print("="*80)
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()

