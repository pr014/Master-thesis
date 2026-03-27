#!/usr/bin/env python3
"""
Plot Mortality Recall (positive class = died) per model.

Recall = 0 means the model never predicted 'died'.

Usage:
  python scripts/analysis/plot_mortality_recall.py
  python scripts/analysis/plot_mortality_recall.py --output outputs/analysis/mortality_recall.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot Mortality Recall per model")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Models in requested order; recall values (positive class = died)
    # Update values if you have actual recall from evaluation logs
    models = ["XGBoost", "CNN scratch", "unidirect. LSTM", "bidirect. LSTM", "Hybrid CNN-LSTM", "deepecg_sl"]
    recall_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.75]  # deepecg_sl ~0.75; others 0 (never predicted 'died')

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, recall_values, color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Mortality Recall (positive class = died)", fontsize=12, fontweight="bold")
    ax.set_title("Recall = 0 → Model never predicted 'died'", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    out_path = Path(args.output) if args.output else PROJECT_ROOT / "outputs/analysis/mortality_recall.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
