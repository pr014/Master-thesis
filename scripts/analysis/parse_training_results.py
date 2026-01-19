#!/usr/bin/env python3
"""
Parse and display training results from SLURM log files in a formatted way.
Usage: python scripts/analysis/parse_training_results.py [JOB_ID]
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_log_file(log_path: Path) -> Dict:
    """Parse SLURM log file and extract key metrics."""
    results = {
        'job_id': None,
        'node': None,
        'start_time': None,
        'end_time': None,
        'duration': None,
        'train_samples': None,
        'val_samples': None,
        'test_samples': None,
        'train_class_dist': {},
        'val_class_dist': {},
        'test_class_dist': {},
        'best_val_loss': None,
        'test_loss': None,
        'test_accuracy': None,
        'test_icu_stays': None,
        'epochs': [],
    }
    
    if not log_path.exists():
        print(f"❌ Log-Datei nicht gefunden: {log_path}")
        return results
    
    content = log_path.read_text()
    
    # Job Info
    job_match = re.search(r'Job ID: (\d+)', content)
    if job_match:
        results['job_id'] = job_match.group(1)
    
    node_match = re.search(r'Node: (\S+)', content)
    if node_match:
        results['node'] = node_match.group(1)
    
    start_match = re.search(r'Start time: (.+)', content)
    if start_match:
        results['start_time'] = start_match.group(1)
    
    end_match = re.search(r'End time: (.+)', content)
    if end_match:
        results['end_time'] = end_match.group(1)
    
    # Duration from job feedback
    duration_match = re.search(r'Job Wall-clock time: (.+)', content)
    if duration_match:
        results['duration'] = duration_match.group(1)
    
    # Sample counts
    train_match = re.search(r'ECGDataset Statistics:.*?Total samples: (\d+)', content, re.DOTALL)
    if train_match:
        results['train_samples'] = int(train_match.group(1))
    
    # Class distributions (find all three)
    class_dist_matches = re.findall(r'Class distribution \(after filtering\): (\{[^}]+\})', content)
    if len(class_dist_matches) >= 1:
        results['train_class_dist'] = eval(class_dist_matches[0])
    if len(class_dist_matches) >= 2:
        results['val_class_dist'] = eval(class_dist_matches[1])
    if len(class_dist_matches) >= 3:
        results['test_class_dist'] = eval(class_dist_matches[2])
    
    # Additional sample counts from class distributions
    if results['train_class_dist']:
        results['train_samples'] = sum(results['train_class_dist'].values())
    if results['val_class_dist']:
        results['val_samples'] = sum(results['val_class_dist'].values())
    if results['test_class_dist']:
        results['test_samples'] = sum(results['test_class_dist'].values())
    
    # Best validation loss
    val_loss_match = re.search(r'Best validation loss: ([\d.]+)', content)
    if val_loss_match:
        results['best_val_loss'] = float(val_loss_match.group(1))
    
    # Test results
    test_loss_match = re.search(r'Test Loss: ([\d.]+)', content)
    if test_loss_match:
        results['test_loss'] = float(test_loss_match.group(1))
    
    test_acc_match = re.search(r'Test Accuracy:\s+([\d.]+)', content)
    if test_acc_match:
        results['test_accuracy'] = float(test_acc_match.group(1))
    
    # New detailed metrics
    # Balanced Accuracy (may have percentage in parentheses)
    test_balanced_acc_match = re.search(r'Balanced Accuracy:\s*([\d.]+)', content)
    if test_balanced_acc_match:
        results['test_balanced_accuracy'] = float(test_balanced_acc_match.group(1))
    
    test_macro_precision_match = re.search(r'Macro Precision:\s*([\d.]+)', content)
    if test_macro_precision_match:
        results['test_macro_precision'] = float(test_macro_precision_match.group(1))
    
    test_macro_recall_match = re.search(r'Macro Recall:\s*([\d.]+)', content)
    if test_macro_recall_match:
        results['test_macro_recall'] = float(test_macro_recall_match.group(1))
    
    test_macro_f1_match = re.search(r'Macro F1-Score:\s*([\d.]+)', content)
    if test_macro_f1_match:
        results['test_macro_f1'] = float(test_macro_f1_match.group(1))
    
    # Per-class metrics
    per_class_section = re.search(r'Per-Class Metrics:(.*?)(?:\n\s*\n|Confusion Matrix:)', content, re.DOTALL)
    if per_class_section:
        per_class_lines = per_class_section.group(1).strip().split('\n')
        results['per_class_metrics'] = {}
        for line in per_class_lines[2:]:  # Skip header lines
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    cls = int(parts[0])
                    results['per_class_metrics'][cls] = {
                        'precision': float(parts[1]),
                        'recall': float(parts[2]),
                        'f1': float(parts[3])
                    }
                except (ValueError, IndexError):
                    continue
    
    # Confusion Matrix
    cm_section = re.search(r'Confusion Matrix:\s*\n(.*?)(?:\n\s*\n|End time|Job completed)', content, re.DOTALL)
    if cm_section:
        cm_lines = [line.strip() for line in cm_section.group(1).strip().split('\n') if line.strip()]
        if cm_lines:
            results['confusion_matrix'] = []
            # Skip header row (starts with spaces/numbers for column headers)
            for line in cm_lines:
                line = line.strip()
                if not line:
                    continue
                # Check if line starts with a digit (row label)
                if line and line[0].isdigit():
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            # Skip first element (row label), convert rest to int
                            row = [int(x) for x in parts[1:]]
                            results['confusion_matrix'].append(row)
                        except (ValueError, IndexError):
                            continue
    
    test_stays_match = re.search(r'Number of ICU stays evaluated: (\d+)', content)
    if test_stays_match:
        results['test_icu_stays'] = int(test_stays_match.group(1))
    
    # Epoch metrics (last few epochs)
    epoch_matches = re.findall(r'Epoch (\d+)/(\d+).*?Train Loss: ([\d.]+).*?Val Loss: ([\d.]+)', content)
    if epoch_matches:
        results['epochs'] = [
            {
                'epoch': int(e[0]),
                'total': int(e[1]),
                'train_loss': float(e[2]),
                'val_loss': float(e[3])
            }
            for e in epoch_matches[-5:]  # Last 5 epochs
        ]
    
    return results


def format_class_distribution(dist: Dict[int, int]) -> str:
    """Format class distribution as a compact string."""
    if not dist:
        return "N/A"
    total = sum(dist.values())
    percentages = {k: (v / total * 100) for k, v in dist.items()}
    return f"Total: {total} | " + " | ".join([f"C{k}: {v} ({p:.1f}%)" for k, v, p in 
                                               sorted([(k, v, percentages[k]) for k, v in dist.items()])])


def print_results(results: Dict):
    """Print results in a formatted way."""
    print("=" * 80)
    print("📊 TRAINING RESULTS SUMMARY")
    print("=" * 80)
    
    # Job Info
    print("\n🔹 Job Information:")
    print(f"   Job ID:      {results['job_id'] or 'N/A'}")
    print(f"   Node:        {results['node'] or 'N/A'}")
    print(f"   Start:       {results['start_time'] or 'N/A'}")
    print(f"   End:         {results['end_time'] or 'N/A'}")
    print(f"   Duration:    {results['duration'] or 'N/A'}")
    
    # Dataset Info
    print("\n🔹 Dataset Split:")
    print(f"   Train:       {results['train_samples'] or 'N/A':,} samples")
    print(f"   Validation:  {results['val_samples'] or 'N/A':,} samples")
    print(f"   Test:        {results['test_samples'] or 'N/A':,} samples")
    total = (results['train_samples'] or 0) + (results['val_samples'] or 0) + (results['test_samples'] or 0)
    print(f"   Total:       {total:,} samples")
    
    # Class Distributions
    print("\n🔹 Class Distribution:")
    print(f"   Train:       {format_class_distribution(results['train_class_dist'])}")
    print(f"   Validation:  {format_class_distribution(results['val_class_dist'])}")
    print(f"   Test:        {format_class_distribution(results['test_class_dist'])}")
    
    # Model Performance
    print("\n🔹 Model Performance:")
    if results['best_val_loss'] is not None:
        print(f"   Best Validation Loss: {results['best_val_loss']:.4f}")
    if results['test_loss'] is not None:
        print(f"   Test Loss:            {results['test_loss']:.4f}")
    if results['test_accuracy'] is not None:
        print(f"   Test Accuracy:        {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
    if results.get('test_balanced_accuracy') is not None:
        print(f"   Balanced Accuracy:    {results['test_balanced_accuracy']:.4f} ({results['test_balanced_accuracy']*100:.2f}%)")
    if results.get('test_macro_precision') is not None:
        print(f"   Macro Precision:      {results['test_macro_precision']:.4f}")
    if results.get('test_macro_recall') is not None:
        print(f"   Macro Recall:         {results['test_macro_recall']:.4f}")
    if results.get('test_macro_f1') is not None:
        print(f"   Macro F1-Score:       {results['test_macro_f1']:.4f}")
    if results['test_icu_stays'] is not None:
        print(f"   Test ICU Stays:       {results['test_icu_stays']:,}")
    
    # Per-Class Metrics
    if results.get('per_class_metrics'):
        print("\n🔹 Per-Class Metrics:")
        print(f"   {'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("   " + "-" * 44)
        for cls in sorted(results['per_class_metrics'].keys()):
            metrics = results['per_class_metrics'][cls]
            print(f"   {cls:<8} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    # Confusion Matrix
    if results.get('confusion_matrix'):
        print("\n🔹 Confusion Matrix:")
        cm = results['confusion_matrix']
        if cm and len(cm) > 0:
            num_rows = len(cm)
            num_cols = len(cm[0]) if cm[0] else 0
            # Use the maximum of rows and cols to determine number of classes
            num_classes = max(num_rows, num_cols)
            
            # Print header
            print("   " + " ".join([f"{i:>6}" for i in range(num_classes)]))
            
            # Print rows
            for i in range(num_classes):
                if i < num_rows and len(cm[i]) >= num_classes:
                    row_str = f"   {i} " + " ".join([f"{cm[i][j]:>6}" for j in range(num_classes)])
                    print(row_str)
                elif i < num_rows:
                    # Row exists but is shorter - pad with zeros
                    row = cm[i] + [0] * (num_classes - len(cm[i]))
                    row_str = f"   {i} " + " ".join([f"{row[j]:>6}" for j in range(num_classes)])
                    print(row_str)
                else:
                    # Row doesn't exist - print zeros
                    row_str = f"   {i} " + " ".join(["     0"] * num_classes)
                    print(row_str)
    
    # Last Epochs
    if results['epochs']:
        print("\n🔹 Last 5 Epochs:")
        print(f"   {'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12}")
        print("   " + "-" * 32)
        for ep in results['epochs']:
            print(f"   {ep['epoch']:<8} {ep['train_loss']:<12.4f} {ep['val_loss']:<12.4f}")
    
    print("\n" + "=" * 80)
    if results.get('job_id'):
        checkpoint_path = f"outputs/checkpoints/CNNScratch_best_{results['job_id']}.pt"
        print(f"✅ Checkpoints: {checkpoint_path}")
    else:
        print("✅ Checkpoints: outputs/checkpoints/CNNScratch_best_<JOB_ID>.pt (Job ID not found)")
    print("=" * 80)


def main():
    """Main function."""
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    else:
        # Find latest log file
        log_dir = Path("outputs/logs")
        if not log_dir.exists():
            print("❌ Log-Verzeichnis nicht gefunden: outputs/logs/")
            sys.exit(1)
        
        log_files = sorted(log_dir.glob("slurm_*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print("❌ Keine Log-Dateien gefunden in outputs/logs/")
            sys.exit(1)
        
        job_id = log_files[0].stem.replace("slurm_", "")
        print(f"📄 Verwende neueste Log-Datei: {log_files[0].name}\n")
    
    log_path = Path(f"outputs/logs/slurm_{job_id}.out")
    
    if not log_path.exists():
        print(f"❌ Log-Datei nicht gefunden: {log_path}")
        print(f"   Verfügbare Log-Dateien:")
        log_dir = Path("outputs/logs")
        if log_dir.exists():
            for f in sorted(log_dir.glob("slurm_*.out")):
                print(f"   - {f.name}")
        sys.exit(1)
    
    results = parse_log_file(log_path)
    print_results(results)


if __name__ == "__main__":
    main()

