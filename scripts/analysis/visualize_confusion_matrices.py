"""Visualize confusion matrices from two training runs."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_confusion_matrix_from_log(log_path: Path) -> np.ndarray:
    """Parse confusion matrix from log file."""
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    content = log_path.read_text()
    
    # Find confusion matrix section (try both with and without emoji)
    cm_patterns = [
        r'🔹 Confusion Matrix:\s*\n(.*?)(?=\n\n|$)',
        r'Confusion Matrix:\s*\n(.*?)(?=\n\n|$)',
        r'Confusion Matrix:\s*\n.*?\n(.*?)(?=\n\n|$)',
    ]
    
    cm_section = None
    for pattern in cm_patterns:
        cm_match = re.search(pattern, content, re.DOTALL)
        if cm_match:
            cm_section = cm_match.group(1)
            break
    
    if not cm_section:
        raise ValueError(f"Could not find confusion matrix in log file: {log_path}")
    
    cm_lines = [line.strip() for line in cm_section.split('\n') if line.strip()]
    
    # Parse matrix
    cm_rows = []
    for line in cm_lines:
        # Skip header line (starts with spaces and numbers for column headers)
        if line.startswith(' ') and not line[1:].strip()[0].isdigit():
            continue
        
        # Extract row - should start with a digit (row index)
        parts = line.split()
        if len(parts) > 1:
            try:
                # First element is row index, rest are values
                row = [int(x) for x in parts[1:]]
                # Remove any extra columns (like column 10 if it exists)
                if len(row) > 10:
                    row = row[:10]
                elif len(row) < 10:
                    row = row + [0] * (10 - len(row))
                cm_rows.append(row)
            except (ValueError, IndexError):
                continue
    
    if not cm_rows:
        raise ValueError(f"Could not parse confusion matrix from: {log_path}")
    
    # Ensure we have 10x10 matrix
    cm = np.array(cm_rows)
    if cm.shape[0] < 10:
        # Pad with zeros
        padding = np.zeros((10 - cm.shape[0], cm.shape[1]), dtype=int)
        cm = np.vstack([cm, padding])
    if cm.shape[1] < 10:
        # Pad with zeros
        padding = np.zeros((cm.shape[0], 10 - cm.shape[1]), dtype=int)
        cm = np.hstack([cm, padding])
    if cm.shape[0] > 10:
        cm = cm[:10, :]
    if cm.shape[1] > 10:
        cm = cm[:, :10]
    
    return cm


def create_comparison_visualization(
    cm1: np.ndarray,
    cm2: np.ndarray,
    name1: str,
    name2: str,
    save_path: Path
):
    """Create comparison visualization of two confusion matrices."""
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Confusion Matrix 1
    ax1 = axes[0]
    sns.heatmap(
        cm1,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax1,
        cbar_kws={'label': 'Count'},
        xticklabels=range(10),
        yticklabels=range(10)
    )
    ax1.set_title(f'{name1}\nConfusion Matrix', fontsize=14, weight='bold', pad=15)
    ax1.set_xlabel('Predicted Class', fontsize=12)
    ax1.set_ylabel('Actual Class', fontsize=12)
    
    # Confusion Matrix 2
    ax2 = axes[1]
    sns.heatmap(
        cm2,
        annot=True,
        fmt='d',
        cmap='Greens',
        ax=ax2,
        cbar_kws={'label': 'Count'},
        xticklabels=range(10),
        yticklabels=range(10)
    )
    ax2.set_title(f'{name2}\nConfusion Matrix', fontsize=14, weight='bold', pad=15)
    ax2.set_xlabel('Predicted Class', fontsize=12)
    ax2.set_ylabel('Actual Class', fontsize=12)
    
    # Delta Matrix
    ax3 = axes[2]
    delta_matrix = cm2.astype(float) - cm1.astype(float)
    vmax = max(abs(delta_matrix.max()), abs(delta_matrix.min())) if delta_matrix.size > 0 else 1.0
    if vmax == 0:
        vmax = 1.0
    
    sns.heatmap(
        delta_matrix,
        annot=True,
        fmt='.0f',
        cmap='RdBu_r',
        center=0,
        ax=ax3,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={'label': 'Difference (Run 2 - Run 1)'},
        xticklabels=range(10),
        yticklabels=range(10)
    )
    ax3.set_title(f'Difference\n({name2} - {name1})', fontsize=14, weight='bold', pad=15)
    ax3.set_xlabel('Predicted Class', fontsize=12)
    ax3.set_ylabel('Actual Class', fontsize=12)
    
    # Overall title
    fig.suptitle('Confusion Matrix Comparison', fontsize=18, weight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualization saved to: {save_path}")
    plt.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize confusion matrices from two training runs')
    parser.add_argument('--job1', type=str, required=True, help='First job ID')
    parser.add_argument('--job2', type=str, required=True, help='Second job ID')
    parser.add_argument('--output', type=str, default=None, help='Output path (default: outputs/visualizations/cm_comparison_job{JOB1}_vs_job{JOB2}.png)')
    
    args = parser.parse_args()
    
    # Build paths
    log1_path = Path(f"outputs/logs/slurm_{args.job1}.out")
    log2_path = Path(f"outputs/logs/slurm_{args.job2}.out")
    
    if not log1_path.exists():
        print(f"Error: Log file not found: {log1_path}", file=sys.stderr)
        sys.exit(1)
    
    if not log2_path.exists():
        print(f"Error: Log file not found: {log2_path}", file=sys.stderr)
        sys.exit(1)
    
    # Parse confusion matrices
    print(f"Parsing confusion matrix from: {log1_path}")
    cm1 = parse_confusion_matrix_from_log(log1_path)
    
    print(f"Parsing confusion matrix from: {log2_path}")
    cm2 = parse_confusion_matrix_from_log(log2_path)
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"outputs/visualizations/cm_comparison_job{args.job1}_vs_job{args.job2}.png")
    
    # Create visualization
    print("Creating visualization...")
    create_comparison_visualization(
        cm1=cm1,
        cm2=cm2,
        name1=f"Job {args.job1}",
        name2=f"Job {args.job2}",
        save_path=output_path
    )
    
    print("Done!")


if __name__ == '__main__':
    main()

