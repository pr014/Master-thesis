#!/usr/bin/env python3
"""
Plot mortality rate by LOS (1-15 days) - upper chart only with x-axis label.

Usage:
  python scripts/analysis/plot_mortality_by_los.py
  python scripts/analysis/plot_mortality_by_los.py --output outputs/analysis/24h_demographics/mortality_rate_by_los_1_15.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.labeling import load_icustays, load_mortality_mapping


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot mortality rate by LOS (1-15 days)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--upper-only", action="store_true", default=True, help="Plot only mortality rate (upper chart)")
    args = parser.parse_args()

    icustays_path = PROJECT_ROOT / "data/labeling/labels_csv/icustays.csv"
    admissions_path = PROJECT_ROOT / "data/labeling/labels_csv/admissions.csv"

    if not icustays_path.exists():
        print(f"Error: {icustays_path} not found")
        sys.exit(1)
    if not admissions_path.exists():
        print(f"Error: {admissions_path} not found")
        sys.exit(1)

    icustays_df = load_icustays(str(icustays_path))
    mortality_mapping = load_mortality_mapping(str(admissions_path), icustays_df)

    # Add mortality to icustays
    icustays_df["mortality"] = icustays_df["stay_id"].map(mortality_mapping)
    df = icustays_df[icustays_df["mortality"].notna()].copy()
    df = df[df["los"].notna() & np.isfinite(df["los"])]
    df["los_day"] = np.floor(df["los"].astype(float)).astype(int)

    # Filter LOS 1-15 days
    df = df[(df["los_day"] >= 1) & (df["los_day"] <= 15)]

    # Mortality rate per LOS day
    los_days = list(range(1, 16))
    mortality_rates = []
    for d in los_days:
        subset = df[df["los_day"] == d]
        if len(subset) > 0:
            rate_pct = 100 * subset["mortality"].mean()
        else:
            rate_pct = 0.0
        mortality_rates.append(rate_pct)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = "#e74c3c"  # light red/salmon
    bars = ax.bar(los_days, mortality_rates, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(los_days)
    ax.set_xticklabels([str(d) for d in los_days])
    ax.set_xlabel("LOS (days)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mortality rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Mortality rate by LOS (1-15 days)", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(mortality_rates) * 1.15 if mortality_rates else 25)

    out_path = Path(args.output) if args.output else PROJECT_ROOT / "outputs/analysis/24h_demographics/mortality_by_los_1_15.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
