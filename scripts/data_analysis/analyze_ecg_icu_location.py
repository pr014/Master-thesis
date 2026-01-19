"""
Analyze how many ECGs were actually taken in ICU vs other locations.
"""

import pandas as pd

# Load filtered data
print("Loading filtered ECG data...")
filtered = pd.read_csv('data/labels/records_w_diag_icd10_filtered_icu.csv')

print(f"\nTotal filtered records: {len(filtered):,}")

# Check available columns
print("\nAvailable location columns:")
location_cols = [c for c in filtered.columns if 'ed' in c.lower() or 'hosp' in c.lower() or 'icu' in c.lower()]
print(location_cols)

# Analyze location columns
if 'ecg_taken_in_ed' in filtered.columns:
    ed_count = filtered['ecg_taken_in_ed'].sum()
    print(f"\nECGs taken in ED: {ed_count:,} ({ed_count/len(filtered)*100:.1f}%)")

if 'ecg_taken_in_hosp' in filtered.columns:
    hosp_count = filtered['ecg_taken_in_hosp'].sum()
    print(f"ECGs taken in hospital: {hosp_count:,} ({hosp_count/len(filtered)*100:.1f}%)")

if 'ecg_taken_in_ed_or_hosp' in filtered.columns:
    ed_or_hosp_count = filtered['ecg_taken_in_ed_or_hosp'].sum()
    print(f"ECGs taken in ED or hospital: {ed_or_hosp_count:,} ({ed_or_hosp_count/len(filtered)*100:.1f}%)")

# The current filter already ensures ecg_time is within ICU stay period
# So theoretically all filtered ECGs should be during ICU stay
print(f"\nCurrent filter ensures:")
print(f"  - ECG time is within ICU stay period (intime <= ecg_time <= outtime)")
print(f"  - This means ECGs were taken DURING ICU stay")
print(f"  - However, they might not have been taken ON the ICU unit itself")

# Check if there's a way to determine if ECG was actually on ICU
# We can check if ecg_time falls within ICU stay period (which we already do)
# But we might want to be more strict

print(f"\nAll {len(filtered):,} filtered ECGs have:")
print(f"  - Matching ICU stay (subject_id + hadm_id)")
print(f"  - ECG time within ICU stay period")
print(f"  - Patient with >= 3 days total ICU stay")

# Show sample
print("\nSample of filtered data:")
print(filtered[['subject_id', 'study_id', 'ecg_time', 'ecg_taken_in_ed', 'ecg_taken_in_hosp', 'ecg_taken_in_ed_or_hosp']].head(10))

