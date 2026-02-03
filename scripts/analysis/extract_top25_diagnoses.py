"""
Extract Top 25 diagnoses from analysis and get their descriptions from d_icd_diagnoses.csv.
"""

import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def get_top25_diagnoses_with_descriptions():
    """Get Top 25 diagnoses sorted by combined influence score and their descriptions."""
    
    # Load ICD-10 descriptions
    icd_path = project_root / "data/labeling/labels_csv/d_icd_diagnoses.csv"
    icd_lookup = pd.read_csv(icd_path)
    icd_lookup = icd_lookup[icd_lookup['icd_version'] == 10]  # Only ICD-10
    icd_dict = dict(zip(icd_lookup['icd_code'], icd_lookup['long_title']))
    
    print(f"Loaded {len(icd_dict)} ICD-10 codes")
    
    # Since we don't have the saved combined_results_df, we need to either:
    # 1. Re-run the analysis (complex)
    # 2. Extract from the visualization images (not possible)
    # 3. Use a manual list based on the analysis
    
    # For now, let's create a script that can extract top 25 from combined results
    # if they were saved, or we'll need to manually specify them
    
    # Check if there are saved results
    los_csv = project_root / "outputs/analysis/diagnosis_los_correlations.csv"
    mort_csv = project_root / "outputs/analysis/diagnosis_mortality_correlations.csv"
    
    if los_csv.exists() and mort_csv.exists():
        print("Found saved analysis results, extracting top 25...")
        los_results = pd.read_csv(los_csv)
        mort_results = pd.read_csv(mort_csv)
        
        # Recreate the combined influence score logic
        los_results['los_influence_score'] = (
            los_results['los_difference'].abs() * (1 - los_results['p_value']) * 
            (los_results['significant'].astype(int) * 2 + 1)
        )
        
        mort_results['mortality_influence_score'] = (
            mort_results['mortality_difference'].abs() * (1 - mort_results['p_value']) *
            (mort_results['significant'].astype(int) * 2 + 1)
        )
        
        # Get top 25 for each
        top_los = los_results.nlargest(25, 'los_influence_score')
        top_mort = mort_results.nlargest(25, 'mortality_influence_score')
        
        # Combine
        combined_diagnoses = set(top_los['diagnosis'].tolist()) | set(top_mort['diagnosis'].tolist())
        
        # Create combined results
        combined_results = []
        for diag in combined_diagnoses:
            los_row = los_results[los_results['diagnosis'] == diag]
            mort_row = mort_results[mort_results['diagnosis'] == diag]
            
            combined_row = {
                'diagnosis': diag,
                'los_influence_score': los_row['los_influence_score'].iloc[0] if len(los_row) > 0 else 0.0,
                'mortality_influence_score': mort_row['mortality_influence_score'].iloc[0] if len(mort_row) > 0 else 0.0,
            }
            combined_row['combined_influence_score'] = (
                combined_row['los_influence_score'] + combined_row['mortality_influence_score']
            )
            combined_results.append(combined_row)
        
        combined_df = pd.DataFrame(combined_results)
        combined_df = combined_df.sort_values('combined_influence_score', ascending=False)
        
        # Get top 25
        top25 = combined_df.head(25)
        
        print(f"\nTop 25 Diagnoses (by combined influence score):")
        print("="*80)
        
        result = []
        for idx, row in top25.iterrows():
            diag_code = row['diagnosis']
            description = icd_dict.get(diag_code, "Description not found")
            result.append({
                'code': diag_code,
                'description': description,
                'combined_score': row['combined_influence_score']
            })
            print(f"{diag_code:10s} - {description}")
        
        return result
    else:
        print("Analysis results not found. Please run the analysis first.")
        return None

if __name__ == "__main__":
    result = get_top25_diagnoses_with_descriptions()
    if result:
        print(f"\n✅ Extracted {len(result)} diagnoses")
