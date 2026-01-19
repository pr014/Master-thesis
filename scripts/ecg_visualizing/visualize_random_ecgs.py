"""Visualize 10 specific ECGs from the dataset (full 10s length)."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.ecg import ECGDemoDataset, build_demo_index
from src.visualization.plot_ecg import plot_12lead_ecg


def main():
    data_dir = r"D:\MA\data\mimic-iv-ecg\files"
    output_dir = Path("outputs/ecg_visualizing/12lead/random_sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Patient IDs from first run (to recreate same ECGs)
    patient_ids = ["1915", "1324", "1704", "1295", "1751", "1779", "1822", "1361", "1110", "1712"]
    
    print(f"Discovering ECG records in {data_dir}...")
    print("This may take a while for large datasets (scanning all .hea files)...")
    records = build_demo_index(data_dir=data_dir)
    print(f"Found {len(records)} ECG records")
    
    # Find records matching the patient IDs
    print(f"\nFinding ECGs for patient IDs: {patient_ids}...")
    selected_records = []
    for patient_id in patient_ids:
        # Find first record for this patient
        for record in records:
            if f"p{patient_id}" in record["base_path"]:
                selected_records.append(record)
                break
        else:
            print(f"  Warning: No record found for patient {patient_id}")
    
    print(f"Found {len(selected_records)} matching records")
    print(f"\nVisualizing {len(selected_records)} ECGs (full 10s)...")
    dataset = ECGDemoDataset(selected_records, window_seconds=None)  # Full ECG (10s)
    
    for i in range(len(selected_records)):
        try:
            item = dataset[i]
            base_path = item["meta"]["base_path"]
            record_name = Path(base_path).name
            patient_id = patient_ids[i]
            
            filename = f"random_{i+1:02d}_p{patient_id}.png"
            output_path = output_dir / filename
            
            plot_12lead_ecg(
                signal=item["signal"],
                fs=item["meta"]["fs"],
                title=f"ECG {i+1}/{len(selected_records)}: {record_name}",
                output_path=output_path,
                show=False,
            )
            print(f"  [{i+1}/{len(selected_records)}] Saved: {output_path}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_records)}] Error: {e}")
    
    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

