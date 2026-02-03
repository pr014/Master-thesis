"""
Analyze records_w_diag_icd10.csv to identify:
1. Available columns
2. Timestamp columns
3. Diagnosis-related columns
4. Relevant columns for filtering diagnoses before ECG time
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def analyze_labels_csv(csv_path: Path, sample_rows: int = 10):
    """Analyze the labels CSV file structure."""
    
    print("="*80)
    print("ANALYZING LABELS CSV FILE")
    print("="*80)
    print(f"File: {csv_path}")
    print(f"File size: {csv_path.stat().st_size / (1024**2):.2f} MB")
    print()
    
    # Read header only
    print("Reading header...")
    df_header = pd.read_csv(csv_path, nrows=0)
    all_columns = df_header.columns.tolist()
    
    print(f"\nTotal columns: {len(all_columns)}")
    print(f"\nAll columns:")
    for i, col in enumerate(all_columns, 1):
        print(f"  {i:3d}. {col}")
    
    # Read sample rows
    print(f"\n{'='*80}")
    print(f"Reading {sample_rows} sample rows...")
    print(f"{'='*80}")
    df_sample = pd.read_csv(csv_path, nrows=sample_rows, low_memory=False)
    
    # Analyze column types and non-null counts
    print(f"\nColumn analysis (from {sample_rows} sample rows):")
    print(f"{'='*80}")
    for col in all_columns:
        non_null = df_sample[col].notna().sum()
        dtype = df_sample[col].dtype
        sample_values = df_sample[col].dropna().head(3).tolist()
        
        print(f"\n{col}:")
        print(f"  Type: {dtype}")
        print(f"  Non-null in sample: {non_null}/{sample_rows}")
        if len(sample_values) > 0:
            print(f"  Sample values: {sample_values}")
    
    # Identify diagnosis-related columns
    print(f"\n{'='*80}")
    print("DIAGNOSIS-RELATED COLUMNS")
    print(f"{'='*80}")
    diag_cols = [col for col in all_columns if 'diag' in col.lower()]
    if diag_cols:
        print(f"Found {len(diag_cols)} diagnosis-related columns:")
        for col in diag_cols:
            non_null = df_sample[col].notna().sum()
            print(f"  - {col} (non-null: {non_null}/{sample_rows})")
            
            # Show sample diagnosis values
            sample_diag = df_sample[col].dropna().head(2).tolist()
            if sample_diag:
                print(f"    Sample: {sample_diag[0][:100] if len(str(sample_diag[0])) > 100 else sample_diag[0]}")
    else:
        print("No diagnosis-related columns found!")
    
    # Identify timestamp/date columns
    print(f"\n{'='*80}")
    print("TIMESTAMP/DATE COLUMNS")
    print(f"{'='*80}")
    time_keywords = ['time', 'date', 'chart', 'admit', 'disch', 'intime', 'outtime']
    time_cols = [col for col in all_columns 
                 if any(keyword in col.lower() for keyword in time_keywords)]
    
    if time_cols:
        print(f"Found {len(time_cols)} timestamp/date-related columns:")
        for col in time_cols:
            non_null = df_sample[col].notna().sum()
            sample_vals = df_sample[col].dropna().head(2).tolist()
            print(f"  - {col} (non-null: {non_null}/{sample_rows})")
            if sample_vals:
                print(f"    Sample: {sample_vals}")
    else:
        print("No timestamp/date columns found!")
    
    # Identify identifier columns
    print(f"\n{'='*80}")
    print("IDENTIFIER COLUMNS")
    print(f"{'='*80}")
    id_keywords = ['id', 'subject', 'hadm', 'stay', 'record']
    id_cols = [col for col in all_columns 
               if any(keyword in col.lower() for keyword in id_keywords)]
    
    if id_cols:
        print(f"Found {len(id_cols)} identifier columns:")
        for col in id_cols:
            print(f"  - {col}")
    
    # Identify demographic columns
    print(f"\n{'='*80}")
    print("DEMOGRAPHIC COLUMNS")
    print(f"{'='*80}")
    demo_keywords = ['age', 'gender', 'sex', 'race', 'marital', 'insurance']
    demo_cols = [col for col in all_columns 
                 if any(keyword in col.lower() for keyword in demo_keywords)]
    
    if demo_cols:
        print(f"Found {len(demo_cols)} demographic columns:")
        for col in demo_cols:
            print(f"  - {col}")
    
    # Summary and recommendations
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print("\n✓ Relevant columns for diagnosis filtering:")
    print("  Identifier columns:")
    for col in id_cols:
        print(f"    - {col}")
    
    print("\n  Diagnosis columns:")
    for col in diag_cols:
        print(f"    - {col}")
    
    if time_cols:
        print("\n  Timestamp columns (for filtering by time):")
        for col in time_cols:
            print(f"    - {col}")
    else:
        print("\n  ⚠️  No timestamp columns found in CSV!")
        print("     Recommendation: Use icustays.intime or admissions.admittime")
        print("     to determine if diagnoses were before ECG time")
    
    # Check for icu_diag specifically
    if 'icu_diag' in all_columns:
        print("\n  ✓ Found 'icu_diag' column!")
        print(f"     Non-null in sample: {df_sample['icu_diag'].notna().sum()}/{sample_rows}")
    else:
        print("\n  ⚠️  'icu_diag' column not found")
        print("     Available diagnosis columns:")
        for col in diag_cols:
            print(f"       - {col}")
    
    return {
        'all_columns': all_columns,
        'diagnosis_columns': diag_cols,
        'timestamp_columns': time_cols,
        'identifier_columns': id_cols,
        'demographic_columns': demo_cols,
        'has_icu_diag': 'icu_diag' in all_columns,
        'sample_df': df_sample
    }


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "data/labeling/labels_csv/records_w_diag_icd10.csv"
    
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return
    
    results = analyze_labels_csv(csv_path, sample_rows=10)
    
    # Save summary to file
    output_path = project_root / "outputs/analysis/labels_csv_analysis.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("LABELS CSV ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total columns: {len(results['all_columns'])}\n\n")
        f.write(f"Diagnosis columns: {len(results['diagnosis_columns'])}\n")
        for col in results['diagnosis_columns']:
            f.write(f"  - {col}\n")
        f.write(f"\nTimestamp columns: {len(results['timestamp_columns'])}\n")
        for col in results['timestamp_columns']:
            f.write(f"  - {col}\n")
        f.write(f"\nHas icu_diag: {results['has_icu_diag']}\n")
    
    print(f"\n{'='*80}")
    print(f"Analysis summary saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

