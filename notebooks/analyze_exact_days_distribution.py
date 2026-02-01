"""Analyze class distribution with exact day bins (1 day, 2 days, 3 days, etc.)"""

from pathlib import Path
import sys
import pandas as pd
from collections import Counter
import numpy as np

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data.ecg.ecg_loader import build_npy_index
from src.data.ecg.ecg_dataset import extract_subject_id_from_path
from src.data.labeling import load_icustays, ICUStayMapper

def los_to_bin_exact_days(los_days: float, max_days: int = 9) -> int:
    """Convert LOS to bin class using exact days.
    
    Class 0: exactly 1 day (rounded: [0.5, 1.5))
    Class 1: exactly 2 days (rounded: [1.5, 2.5))
    Class 2: exactly 3 days (rounded: [2.5, 3.5))
    ...
    Class max_days-1: exactly max_days days (rounded: [max_days-0.5, max_days+0.5))
    Class max_days: >= max_days+1 days
    
    Args:
        los_days: Length of stay in days (float)
        max_days: Maximum number of exact day classes (default: 9)
    
    Returns:
        Class index in [0..max_days]
    """
    if los_days < 0.5:
        return 0  # Very short stays (< 0.5 days) -> Class 0
    
    # Round to nearest integer day
    rounded_days = int(np.round(los_days))
    
    if rounded_days < 1:
        return 0
    elif rounded_days > max_days:
        return max_days  # >= max_days+1 days -> last class
    else:
        return rounded_days - 1  # Class 0 = 1 day, Class 1 = 2 days, etc.


def analyze_exact_day_distribution(
    data_dir: str,
    icustays_path: str = None,
    max_days: int = 9
):
    """Analyze class distribution with exact day bins."""
    print("="*80)
    print(f"Class Distribution Analysis: Exact Days (1 day, 2 days, ..., {max_days}+ days)")
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
    los_values = []  # Store all LOS values for statistics
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
            
            los_values.append(los_days)
            class_label = los_to_bin_exact_days(los_days, max_days=max_days)
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
    
    # Print distribution
    print("\n" + "-"*80)
    print("Class Distribution (Exact Days):")
    print("-"*80)
    print(f"{'Class':<8} {'LOS (days)':<20} {'Count':<12} {'Percentage':<12} {'Cumulative %':<12}")
    print("-"*80)
    
    total = matched_count
    cumulative = 0
    
    for class_idx in range(max_days + 1):
        count = class_counts.get(class_idx, 0)
        percentage = (count / total * 100) if total > 0 else 0
        cumulative += percentage
        
        if class_idx == max_days:
            los_range = f">= {max_days+1} days"
        else:
            los_range = f"exactly {class_idx+1} day{'s' if class_idx+1 > 1 else ''}"
        
        print(f"{class_idx:<8} {los_range:<20} {count:<12,} {percentage:<12.2f}% {cumulative:<12.2f}%")
    
    print("-"*80)
    print(f"{'Total':<8} {'':<20} {total:<12,} {100.0:<12.2f}% {100.0:<12.2f}%")
    
    # Statistics
    print("\n" + "-"*80)
    print("Statistics:")
    print("-"*80)
    
    if los_values:
        los_array = np.array(los_values)
        print(f"Mean LOS: {los_array.mean():.2f} days")
        print(f"Median LOS: {np.median(los_array):.2f} days")
        print(f"Min LOS: {los_array.min():.2f} days")
        print(f"Max LOS: {los_array.max():.2f} days")
        print(f"Std LOS: {los_array.std():.2f} days")
    
    most_frequent = class_counts.most_common(1)[0] if class_counts else None
    least_frequent = min(class_counts.items(), key=lambda x: x[1]) if class_counts else None
    
    if most_frequent:
        print(f"\nMost frequent class: {most_frequent[0]} ({most_frequent[1]:,} samples, {most_frequent[1]/total*100:.2f}%)")
    if least_frequent:
        print(f"Least frequent class: {least_frequent[0]} ({least_frequent[1]:,} samples, {least_frequent[1]/total*100:.2f}%)")
    
    # Calculate imbalance ratio
    if class_counts:
        counts = [class_counts.get(i, 0) for i in range(max_days + 1)]
        max_count = max(counts)
        min_count = min([c for c in counts if c > 0])
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1 (max/min)")
    
    return class_counts, los_values


def calculate_and_update_weights(config_path: str, class_counts: Counter, max_days: int = 7):
    """Calculate balanced weights and update config file.
    
    Args:
        config_path: Path to config YAML file to update
        class_counts: Counter with class indices as keys and counts as values
        max_days: Maximum number of exact day classes (for 8 classes, max_days=7)
    """
    import yaml
    from pathlib import Path
    
    n_classes = max_days + 1  # 8 classes for max_days=7
    counts = [class_counts.get(i, 0) for i in range(n_classes)]
    n_total = sum(counts)
    
    if n_total == 0:
        raise ValueError("No samples found in class_counts")
    
    # Calculate balanced weights
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
    
    # Read config file
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Build weight string and comments
    weight_str = "[" + ", ".join([str(round(w, 1)) for w in balanced_weights]) + "]"
    comment_lines = []
    comment_lines.append(f"  # Weights calculated based on training set distribution ({n_total:,} samples):")
    comment_lines.append(f"  # Class distribution: {counts}")
    comment_lines.append("  # Method: balanced weights (n_samples / (n_classes * count[i]))")
    comment_lines.append("  # Normalized so mean = 1.0")
    
    # Add per-class comments
    for i, (count, weight) in enumerate(zip(counts, balanced_weights)):
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
            new_lines.append(' ' * indent + f'weight: {weight_str}\n')
            weight_line_found = True
            comment_section_started = True
            # Add comments after weight line
            for comment in comment_lines:
                new_lines.append(comment + '\n')
            continue
        
        # Skip old comments after weight line
        if comment_section_started and line.strip().startswith('#') and ('Class' in line or 'Method' in line or 'Weights calculated' in line or 'Class distribution' in line or 'Normalized' in line or 'TO_BE_UPDATED' in line or 'TO_BE_CALCULATED' in line):
            continue
        
        # Stop skipping comments when we hit a non-comment, non-empty line
        if comment_section_started and line.strip() and not line.strip().startswith('#'):
            comment_section_started = False
            in_loss_section = False
        
        new_lines.append(line)
    
    # Write updated file
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"\n✅ Config file updated: {config_path}")
    print(f"Balanced weights: {[round(w, 1) for w in balanced_weights]}")
    
    return balanced_weights, counts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze class distribution with exact day bins")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/icu_ecgs_24h/P1",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--icustays-path",
        type=str,
        default=None,
        help="Path to icustays.csv"
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=9,
        help="Maximum number of exact day classes (default: 9, so classes 0-9 = 1-9 days, class 9 = 10+ days)"
    )
    
    parser.add_argument(
        "--update-config",
        type=str,
        default=None,
        help="Path to config file to update with calculated weights (optional)"
    )
    
    args = parser.parse_args()
    
    class_counts, los_values = analyze_exact_day_distribution(
        data_dir=args.data_dir,
        icustays_path=args.icustays_path,
        max_days=args.max_days
    )
    
    # Update config file if requested
    if args.update_config and class_counts:
        print("\n" + "="*80)
        print("Calculating and updating class weights in config file...")
        print("="*80)
        calculate_and_update_weights(
            config_path=args.update_config,
            class_counts=class_counts,
            max_days=args.max_days
        )

