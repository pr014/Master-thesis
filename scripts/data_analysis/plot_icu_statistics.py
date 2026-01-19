"""
Plot ICU statistics: number of patients and length of stay.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_and_analyze_icu_data(icustays_path: str) -> pd.DataFrame:
    """Load ICU stays data and calculate statistics."""
    print(f"Loading ICU stays from {icustays_path}...")
    df = pd.read_csv(icustays_path)
    
    # Convert time columns to datetime
    df['intime'] = pd.to_datetime(df['intime'])
    df['outtime'] = pd.to_datetime(df['outtime'])
    
    # Calculate length of stay in days (los is already in days, but we can verify)
    # df['los_days'] = (df['outtime'] - df['intime']).dt.total_seconds() / 86400
    
    print(f"Loaded {len(df)} ICU stays")
    print(f"Unique patients (subject_id): {df['subject_id'].nunique()}")
    print(f"Average length of stay: {df['los'].mean():.2f} days")
    print(f"Median length of stay: {df['los'].median():.2f} days")
    
    # Calculate patients with 3+ days total (across all stays) for quick summary
    patient_stats_temp = df.groupby('subject_id')['los'].sum()
    patients_3plus = (patient_stats_temp >= 3).sum()
    print(f"Patients with >=3 days total in ICU (across all stays): {patients_3plus:,} ({patients_3plus/len(patient_stats_temp)*100:.1f}%)")
    
    return df


def create_plots(df: pd.DataFrame, output_dir: str):
    """Create visualization plots for ICU statistics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics per patient
    patient_stats = df.groupby('subject_id').agg({
        'stay_id': 'count',  # Number of ICU stays per patient
        'los': 'sum'  # Total days in ICU per patient
    }).rename(columns={'stay_id': 'num_stays', 'los': 'total_days'})
    
    # Calculate patients with 3+ days
    patients_3plus_days = (patient_stats['total_days'] >= 3).sum()
    patients_less_3_days = (patient_stats['total_days'] < 3).sum()
    pct_3plus = (patients_3plus_days / len(patient_stats)) * 100
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ICU Statistics: Patients and Length of Stay', fontsize=16, fontweight='bold')
    
    # Plot 1: Distribution of number of ICU stays per patient
    ax1 = axes[0, 0]
    stay_counts = patient_stats['num_stays'].value_counts().sort_index()
    ax1.bar(stay_counts.index, stay_counts.values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Number of ICU Stays per Patient', fontsize=11)
    ax1.set_ylabel('Number of Patients', fontsize=11)
    ax1.set_title('Distribution: ICU Stays per Patient', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for x, y in zip(stay_counts.index, stay_counts.values):
        ax1.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Distribution of total days in ICU per patient
    ax2 = axes[0, 1]
    # Use histogram with reasonable bins
    bins = np.linspace(0, patient_stats['total_days'].max(), 50)
    ax2.hist(patient_stats['total_days'], bins=bins, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Total Days in ICU per Patient (sum across all stays)', fontsize=11)
    ax2.set_ylabel('Number of Patients', fontsize=11)
    ax2.set_title('Distribution: Total ICU Days per Patient (across all stays)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    # Add statistics text
    median_days = patient_stats['total_days'].median()
    mean_days = patient_stats['total_days'].mean()
    ax2.axvline(median_days, color='red', linestyle='--', linewidth=2, label=f'Median: {median_days:.1f} days')
    ax2.axvline(mean_days, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_days:.1f} days')
    # Add 3-day threshold line
    ax2.axvline(3, color='purple', linestyle=':', linewidth=2, label='3 days threshold')
    ax2.legend()
    
    # Plot 3: Distribution of length of stay per ICU stay
    ax3 = axes[1, 0]
    bins_los = np.linspace(0, df['los'].quantile(0.95), 50)  # Show up to 95th percentile
    ax3.hist(df['los'], bins=bins_los, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Length of Stay (days)', fontsize=11)
    ax3.set_ylabel('Number of ICU Stays', fontsize=11)
    ax3.set_title('Distribution: Length of Stay per ICU Stay', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    # Add statistics
    median_los = df['los'].median()
    mean_los = df['los'].mean()
    ax3.axvline(median_los, color='red', linestyle='--', linewidth=2, label=f'Median: {median_los:.2f} days')
    ax3.axvline(mean_los, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_los:.2f} days')
    ax3.legend()
    
    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary statistics
    summary_data = [
        ['Total ICU Stays', f"{len(df):,}"],
        ['Unique Patients', f"{df['subject_id'].nunique():,}"],
        ['Patients with 1 stay', f"{(patient_stats['num_stays'] == 1).sum():,}"],
        ['Patients with >1 stay', f"{(patient_stats['num_stays'] > 1).sum():,}"],
        ['', ''],
        ['Patients by Total ICU Days:', ''],
        ['  < 3 days (total)', f"{patients_less_3_days:,} ({100-pct_3plus:.1f}%)"],
        ['  >= 3 days (total)', f"{patients_3plus_days:,} ({pct_3plus:.1f}%)"],
        ['', ''],
        ['Length of Stay (per stay):', ''],
        ['  Mean', f"{df['los'].mean():.2f} days"],
        ['  Median', f"{df['los'].median():.2f} days"],
        ['  Min', f"{df['los'].min():.2f} days"],
        ['  Max', f"{df['los'].max():.2f} days"],
        ['  Q25', f"{df['los'].quantile(0.25):.2f} days"],
        ['  Q75', f"{df['los'].quantile(0.75):.2f} days"],
        ['', ''],
        ['Total ICU Days (per patient):', ''],
        ['  Mean', f"{patient_stats['total_days'].mean():.2f} days"],
        ['  Median', f"{patient_stats['total_days'].median():.2f} days"],
        ['  Min', f"{patient_stats['total_days'].min():.2f} days"],
        ['  Max', f"{patient_stats['total_days'].max():.2f} days"],
    ]
    
    table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.65, 0.35], bbox=[0, 0, 1, 0.95])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # Style all cells for better readability
    for i in range(len(summary_data)):
        for j in range(2):
            cell = table[(i, j)]
            cell.set_edgecolor('lightgray')
            cell.set_linewidth(0.5)
            # Make numbers and section headers more readable
            if summary_data[i][j] and (summary_data[i][j][0].isdigit() or summary_data[i][j].startswith('  ')):
                cell.set_text_props(fontsize=10, weight='normal')
            elif summary_data[i][j] and ':' in summary_data[i][j]:
                # Section headers
                cell.set_text_props(fontsize=11, weight='bold')
                cell.set_facecolor('#E8F5E9')
            elif summary_data[i][j]:
                cell.set_text_props(fontsize=10, weight='normal')
    
    # Style the first row (if it's a header)
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'icu_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Also save as PDF
    output_file_pdf = output_path / 'icu_statistics.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Plot saved to: {output_file_pdf}")
    
    plt.close()
    
    # Create a separate large table-only figure
    create_table_only_plot(df, patient_stats, patients_3plus_days, patients_less_3_days, pct_3plus, output_path)
    
    return patient_stats


def create_table_only_plot(df: pd.DataFrame, patient_stats: pd.DataFrame, 
                          patients_3plus_days: int, patients_less_3_days: int, 
                          pct_3plus: float, output_path: Path):
    """Create a large table-only figure."""
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.axis('off')
    
    # Create summary statistics
    summary_data = [
        ['Total ICU Stays', f"{len(df):,}"],
        ['Unique Patients', f"{df['subject_id'].nunique():,}"],
        ['Patients with 1 stay', f"{(patient_stats['num_stays'] == 1).sum():,}"],
        ['Patients with >1 stay', f"{(patient_stats['num_stays'] > 1).sum():,}"],
        ['', ''],
        ['Patients by Total ICU Days:', ''],
        ['  < 3 days (total)', f"{patients_less_3_days:,} ({100-pct_3plus:.1f}%)"],
        ['  >= 3 days (total)', f"{patients_3plus_days:,} ({pct_3plus:.1f}%)"],
        ['', ''],
        ['Length of Stay (per stay):', ''],
        ['  Mean', f"{df['los'].mean():.2f} days"],
        ['  Median', f"{df['los'].median():.2f} days"],
        ['  Min', f"{df['los'].min():.2f} days"],
        ['  Max', f"{df['los'].max():.2f} days"],
        ['  Q25', f"{df['los'].quantile(0.25):.2f} days"],
        ['  Q75', f"{df['los'].quantile(0.75):.2f} days"],
        ['', ''],
        ['Total ICU Days (per patient):', ''],
        ['  Mean', f"{patient_stats['total_days'].mean():.2f} days"],
        ['  Median', f"{patient_stats['total_days'].median():.2f} days"],
        ['  Min', f"{patient_stats['total_days'].min():.2f} days"],
        ['  Max', f"{patient_stats['total_days'].max():.2f} days"],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.65, 0.35], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    # Style all cells for better readability
    for i in range(len(summary_data)):
        for j in range(2):
            cell = table[(i, j)]
            cell.set_edgecolor('lightgray')
            cell.set_linewidth(0.8)
            # Make numbers and section headers more readable
            if summary_data[i][j] and (summary_data[i][j][0].isdigit() or summary_data[i][j].startswith('  ')):
                cell.set_text_props(fontsize=13, weight='normal')
            elif summary_data[i][j] and ':' in summary_data[i][j]:
                # Section headers
                cell.set_text_props(fontsize=14, weight='bold')
                cell.set_facecolor('#E8F5E9')
            elif summary_data[i][j]:
                cell.set_text_props(fontsize=13, weight='normal')
    
    # Style the first row (header)
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=14)
    
    # Save table-only figure
    output_file_table = output_path / 'icu_statistics_table_only.png'
    plt.savefig(output_file_table, dpi=300, bbox_inches='tight')
    print(f"Table-only plot saved to: {output_file_table}")
    
    # Also save as PDF
    output_file_table_pdf = output_path / 'icu_statistics_table_only.pdf'
    plt.savefig(output_file_table_pdf, bbox_inches='tight')
    print(f"Table-only plot saved to: {output_file_table_pdf}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create plots showing ICU patient statistics"
    )
    parser.add_argument(
        '--icustays',
        type=str,
        default='data/icustay.csv/icustays.csv',
        help='Path to ICU stays CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/data_analysis',
        help='Output directory for plots'
    )
    args = parser.parse_args()
    
    # Load and analyze data
    df = load_and_analyze_icu_data(args.icustays)
    
    # Create plots
    patient_stats = create_plots(df, args.output_dir)
    
    # Save patient statistics to CSV
    stats_file = Path(args.output_dir) / 'patient_icu_statistics.csv'
    patient_stats.to_csv(stats_file)
    print(f"Patient statistics saved to: {stats_file}")


if __name__ == '__main__':
    main()

