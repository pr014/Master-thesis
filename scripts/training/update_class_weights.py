"""Update class weights in config files based on class distribution.

This script:
1. Analyzes class distribution in the dataset
2. Calculates balanced and sqrt weights
3. Updates the config files with the calculated weights
"""

from pathlib import Path
import sys
import yaml
import numpy as np
from collections import Counter

# Add project root and src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data.ecg.ecg_loader import build_npy_index
from src.data.ecg.ecg_dataset import extract_subject_id_from_path
from src.data.labeling import load_icustays, ICUStayMapper, los_to_bin
import pandas as pd

def analyze_class_distribution(
    data_dir: str,
    icustays_path: str = None,
    dataset_name: str = "dataset"
):
    """Analyze class distribution in ECG dataset."""
    print("="*80)
    print(f"Class Distribution Analysis: {dataset_name}")
    print("="*80)
    
    # Load ICU stays
    if icustays_path is None:
        icustays_path = Path(data_dir).parent.parent / "labeling" / "labels_csv" / "icustays.csv"
        if not Path(icustays_path).exists():
            icustays_path = Path("data/labeling/labels_csv/icustays.csv")
    
    icustays_path = Path(icustays_path)
    if not icustays_path.exists():
        raise FileNotFoundError(f"icustays.csv not found at: {icustays_path}")
    
    print(f"\nLoading ICU stays from: {icustays_path}")
    icustays_df = load_icustays(str(icustays_path))
    icu_mapper = ICUStayMapper(icustays_df)
    print(f"Loaded {len(icustays_df)} ICU stays")
    
    # Find all ECG files
    print(f"\nScanning ECG files in: {data_dir}")
    records = build_npy_index(data_dir=data_dir)
    print(f"Found {len(records)} ECG files")
    
    # Analyze class distribution
    class_counts = Counter()
    matched_count = 0
    unmatched_count = 0
    
    print("\nAnalyzing class labels...")
    for i, record in enumerate(records):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(records)} files...")
        
        base_path = record["base_path"]
        try:
            subject_id = extract_subject_id_from_path(base_path)
            subject_stays = icu_mapper.icustays_df[
                icu_mapper.icustays_df['subject_id'] == subject_id
            ]
            
            if len(subject_stays) == 0:
                unmatched_count += 1
                continue
            
            first_stay = subject_stays.iloc[0]
            ecg_time = pd.to_datetime(first_stay['intime'])
            stay_id = icu_mapper.map_ecg_to_stay(subject_id, ecg_time)
            
            if stay_id is None:
                unmatched_count += 1
                continue
            
            los_days = icu_mapper.get_los(stay_id)
            if los_days is None:
                unmatched_count += 1
                continue
            
            class_label = los_to_bin(los_days)
            class_counts[class_label] += 1
            matched_count += 1
        except (ValueError, KeyError):
            unmatched_count += 1
            continue
    
    print(f"\nTotal ECG files: {len(records):,}")
    print(f"Matched: {matched_count:,} ({matched_count/len(records)*100:.2f}%)")
    print(f"Unmatched: {unmatched_count:,} ({unmatched_count/len(records)*100:.2f}%)")
    
    if matched_count == 0:
        print("\n⚠️  WARNING: No matched samples found!")
        return None
    
    return class_counts


def calculate_weights(class_counts: Counter, n_classes: int = 10):
    """Calculate balanced and sqrt weights from class counts.
    
    Args:
        class_counts: Counter with class indices as keys and counts as values
        n_classes: Number of classes (default: 10)
    
    Returns:
        Tuple of (balanced_weights, sqrt_weights) as lists
    """
    # Get counts for all classes
    counts = [class_counts.get(i, 0) for i in range(n_classes)]
    n_total = sum(counts)
    
    if n_total == 0:
        raise ValueError("No samples found in class_counts")
    
    # Method 1: Balanced weights
    # Formula: w_i = n_samples / (n_classes * count[i])
    balanced_weights = []
    for count in counts:
        if count > 0:
            weight = n_total / (n_classes * count)
        else:
            weight = 0.0
        balanced_weights.append(weight)
    
    # Normalize so mean = 1.0
    mean_weight = np.mean([w for w in balanced_weights if w > 0])
    balanced_weights = [w / mean_weight if w > 0 else 0.0 for w in balanced_weights]
    
    # Method 2: SQRT weights
    # Formula: w_i = sqrt(n_total) / sqrt(n_i)
    sqrt_weights = []
    for count in counts:
        if count > 0:
            weight = np.sqrt(n_total) / np.sqrt(count)
        else:
            weight = 0.0
        sqrt_weights.append(weight)
    
    # Normalize so mean = 1.0
    mean_weight = np.mean([w for w in sqrt_weights if w > 0])
    sqrt_weights = [w / mean_weight if w > 0 else 0.0 for w in sqrt_weights]
    
    return balanced_weights, sqrt_weights, counts


def update_config_file(config_path: Path, weights: list, counts: list, method: str):
    """Update config file with calculated weights.
    
    Args:
        config_path: Path to config YAML file
        weights: List of weights for each class
        counts: List of counts for each class
        method: "balanced" or "sqrt"
    """
    # Read file
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Find loss section and update weights
    n_total = sum(counts)
    weight_str = "[" + ", ".join([str(round(w, 1)) for w in weights]) + "]"
    
    # Build comment lines
    comment_lines = []
    comment_lines.append(f"  # Weights calculated based on training set distribution ({n_total:,} samples):")
    comment_lines.append(f"  # Class distribution: {counts}")
    if method == "balanced":
        comment_lines.append("  # Method: balanced weights (n_samples / (n_classes * count[i]))")
    else:
        comment_lines.append("  # Method: Square-Root Weighting (SQINV)")
        comment_lines.append("  #   Formula: w_i = sqrt(n_total) / sqrt(n_i)")
    comment_lines.append("  # Normalized so mean = 1.0")
    
    # Add per-class comments
    for i, (count, weight) in enumerate(zip(counts, weights)):
        if count > 0:
            percentage = count / n_total * 100
            comment_lines.append(f"  # Class {i}: {weight:.2f} ({count:,} samples, {percentage:.2f}%)")
    
    # Find and replace weight line
    new_lines = []
    in_loss_section = False
    weight_line_found = False
    comment_section_started = False
    
    for i, line in enumerate(lines):
        # Check if we're in the loss section
        if 'loss:' in line and 'type:' not in line:
            in_loss_section = True
        
        # Find weight line
        if in_loss_section and 'weight:' in line and not weight_line_found:
            # Replace weight line
            indent = len(line) - len(line.lstrip())
            new_lines.append(' ' * indent + f'weight: {weight_str}  # TO_BE_UPDATED\n')
            weight_line_found = True
            comment_section_started = True
            # Add comments after weight line
            for comment in comment_lines:
                new_lines.append(comment + '\n')
            continue
        
        # Skip old comments after weight line
        if comment_section_started and line.strip().startswith('#') and ('Class' in line or 'Method' in line or 'Weights calculated' in line or 'Class distribution' in line or 'Normalized' in line):
            continue
        
        # Stop skipping comments when we hit a non-comment, non-empty line
        if comment_section_started and line.strip() and not line.strip().startswith('#'):
            comment_section_started = False
            in_loss_section = False
        
        new_lines.append(line)
    
    # Write updated file
    with open(config_path, 'w') as f:
        f.writelines(new_lines)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update class weights in config files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/all_icu_ecgs/P1",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--icustays-path",
        type=str,
        default=None,
        help="Path to icustays.csv (default: data/labeling/labels_csv/icustays.csv)"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/all_icu_ecgs/weighted",
        help="Directory containing config files"
    )
    
    args = parser.parse_args()
    
    # Analyze class distribution
    print("Analyzing class distribution...")
    class_counts = analyze_class_distribution(
        data_dir=args.data_dir,
        icustays_path=args.icustays_path,
        dataset_name="all_icu_ecgs"
    )
    
    if not class_counts or sum(class_counts.values()) == 0:
        print("ERROR: No samples found. Cannot calculate weights.")
        return
    
    # Calculate weights
    print("\nCalculating weights...")
    balanced_weights, sqrt_weights, counts = calculate_weights(class_counts)
    
    # Update config files
    config_dir = Path(args.config_dir)
    
    balanced_config = config_dir / "balanced_weights.yaml"
    sqrt_config = config_dir / "sqrt_weights.yaml"
    
    print(f"\nUpdating {balanced_config}...")
    update_config_file(balanced_config, balanced_weights, counts, "balanced")
    
    print(f"Updating {sqrt_config}...")
    update_config_file(sqrt_config, sqrt_weights, counts, "sqrt")
    
    print("\n✅ Config files updated successfully!")
    print(f"\nBalanced weights: {[round(w, 1) for w in balanced_weights]}")
    print(f"SQRT weights: {[round(w, 1) for w in sqrt_weights]}")


if __name__ == "__main__":
    main()

