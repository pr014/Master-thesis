#!/usr/bin/env python3
"""
Plot LOS (Length of Stay) class distribution to visualize dataset imbalance.

Shows the percentage of ECG records per LOS class: <1 day, 1 day, 2 days, ..., 15+ days.
Useful for the Data chapter to illustrate class imbalance.

Usage:
  python scripts/analysis/plot_los_imbalance.py
  python scripts/analysis/plot_los_imbalance.py --from-csv [csv_path] [output.png]

Without --from-csv: loads data, saves CSV + PNG to outputs/analysis/los_imbalance/ (~5 min for full dataset).
With --from-csv: generates PNG from existing CSV only (fast, for re-plotting after style changes).
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_VIS = True
except ImportError:
    HAS_VIS = False


def load_config(config_path: Path) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_los_values(config: dict, data_dir_override: str = None, limit: int = None) -> list:
    """Get LOS values for all ECG records in the dataset."""
    from src.data.ecg.ecg_loader import build_npy_index
    from src.data.labeling import load_icustays, ICUStayMapper
    from src.data.ecg.dataloader_factory import get_los_values_for_records
    from src.data.ecg.timestamp_mapping import load_timestamp_mapping, get_timestamp_mapping_path

    data_config = config.get("data", {})
    data_dir = data_dir_override or data_config.get("data_dir", "data/icu_ecgs_24h/P1")
    data_dir = Path(project_root) / data_dir if not Path(data_dir).is_absolute() else Path(data_dir)

    # ICU mapper
    icustays_path = project_root / "data" / "labeling" / "labels_csv" / "icustays.csv"
    if not icustays_path.exists():
        icustays_path = Path(data_dir).parent.parent / "labeling" / "labels_csv" / "icustays.csv"
    if not icustays_path.exists():
        raise FileNotFoundError(f"icustays.csv not found. Tried: {icustays_path}")
    icustays_df = load_icustays(str(icustays_path))
    icu_mapper = ICUStayMapper(icustays_df)

    # Records
    records = build_npy_index(data_dir=str(data_dir), limit=limit)
    print(f"Found {len(records):,} ECG records in {data_dir}" + (f" (limit={limit})" if limit else ""))

    # Timestamp mapping (optional)
    timestamp_mapping = None
    mapping_path = get_timestamp_mapping_path(str(data_dir))
    if mapping_path.exists():
        timestamp_mapping = load_timestamp_mapping(str(mapping_path))
        print(f"Loaded timestamp mapping: {len(timestamp_mapping):,} entries")

    los_values = get_los_values_for_records(
        records=records,
        icu_mapper=icu_mapper,
        timestamp_mapping=timestamp_mapping,
        data_dir=str(data_dir),
    )
    return los_values


def bin_los(los_values: list, n_classes: int = 15) -> tuple:
    """
    Bin LOS into classes: [0,1), [1,2), [2,3), ..., [n_classes-1, n_classes), [n_classes, inf).
    Returns (counts, labels) where labels are e.g. "<1 day", "1 day", "2 days", ..., "15+ days".
    """
    counts = [0] * (n_classes + 1)
    for los in los_values:
        if los < 0:
            continue
        day_bin = int(np.floor(los))
        if day_bin >= n_classes:
            counts[n_classes] += 1
        else:
            counts[day_bin] += 1

    labels = ["<1 day"]
    labels += [f"{i} day" if i == 1 else f"{i} days" for i in range(1, n_classes)]
    labels.append(f"{n_classes}+ days")
    return counts, labels


# German → English label mapping (for CSV compatibility)
_GERMAN_TO_ENGLISH = {
    "< 1 Tag": "<1 day",
    "1 Tag": "1 day",
    **{f"{i} Tage": f"{i} days" for i in range(2, 16)},
    "15+ Tage": "15+ days",
}


def _ensure_english_labels(labels: list) -> list:
    """Translate German labels to English if present."""
    return [_GERMAN_TO_ENGLISH.get(lbl, lbl) for lbl in labels]


def plot_los_imbalance(
    counts: list,
    labels: list,
    save_path: Path,
    n_classes: int = 15,
) -> None:
    """Create bar plot of LOS class distribution (percentage)."""
    if not HAS_VIS:
        print("matplotlib not available, skipping plot")
        return

    labels = _ensure_english_labels(labels)
    n_total = sum(counts)
    percentages = [100 * c / n_total for c in counts] if n_total > 0 else counts

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))
    bars = ax.bar(range(len(labels)), percentages, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Length of stay (days)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage of ECG records (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Length-of-stay distribution (n={n_total:,} ECG segments)\n"
        "Class imbalance across LOS bins",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Value labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        if pct > 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        elif pct > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot LOS class distribution (imbalance)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "model" / "CNN" / "cnn_scratch.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data_dir from config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: outputs/analysis/los_imbalance/los_imbalance_distribution.png)",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=15,
        help="Number of LOS classes (1 to n, plus n+ for remainder)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records (for quick testing)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    config = load_config(config_path)

    print("Loading LOS values...")
    try:
        los_values = get_los_values(config, args.data_dir, limit=args.limit)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure data/icu_ecgs_24h/P1 exists and contains .npy files.")
        sys.exit(1)

    if len(los_values) == 0:
        print("No LOS values found. Check data path and icustays.")
        sys.exit(1)

    print(f"Matched {len(los_values):,} records with LOS")
    print(f"LOS range: {min(los_values):.2f} - {max(los_values):.2f} days")
    print(f"LOS mean: {np.mean(los_values):.2f}, median: {np.median(los_values):.2f}")

    counts, labels = bin_los(los_values, n_classes=args.n_classes)
    n_total = sum(counts)
    print("\nDistribution:")
    for lbl, cnt in zip(labels, counts):
        pct = 100 * cnt / n_total
        print(f"  {lbl}: {cnt:,} ({pct:.1f}%)")

    if args.output:
        save_path = Path(args.output)
    else:
        save_path = project_root / "outputs" / "analysis" / "los_imbalance" / "los_imbalance_distribution.png"

    # Save distribution as CSV (always)
    csv_path = save_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["los_class", "count", "percent"])
        for lbl, cnt in zip(labels, counts):
            pct = 100 * cnt / n_total
            writer.writerow([lbl, cnt, f"{pct:.2f}"])
    print(f"Saved distribution CSV: {csv_path}")

    if HAS_VIS:
        plot_los_imbalance(counts, labels, save_path, n_classes=args.n_classes)
    else:
        print("Install matplotlib to generate the PNG plot.")


def plot_from_csv(csv_path: Path, output_path: Path = None) -> None:
    """Generate plot from existing CSV (fast, no data loading)."""
    import csv
    labels, counts = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            labels.append(row["los_class"])
            counts.append(int(row["count"]))
    n_total = sum(counts)
    if output_path is None:
        output_path = csv_path.with_suffix(".png")
    if HAS_VIS:
        plot_los_imbalance(counts, labels, output_path, n_classes=len(labels) - 1)
    else:
        print("Install matplotlib to generate the plot.")


if __name__ == "__main__":
    args_list = sys.argv[1:]
    if args_list and args_list[0] == "--from-csv":
        # Quick plot from existing CSV (no data loading)
        csv_path = Path(args_list[1]) if len(args_list) > 1 else project_root / "outputs" / "analysis" / "los_imbalance" / "los_imbalance_distribution.csv"
        out_path = Path(args_list[2]) if len(args_list) > 2 else None
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}. Run without --from-csv first to generate it.")
            sys.exit(1)
        plot_from_csv(Path(csv_path), out_path)
    else:
        main()
