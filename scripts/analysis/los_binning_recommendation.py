#!/usr/bin/env python3
"""
Analyze LOS distribution and recommend clinically interpretable binning.

Uses existing los_imbalance CSV. Output: recommended boundaries + class distribution.
"""

import csv
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
CSV_PATH = project_root / "outputs" / "analysis" / "los_imbalance" / "los_imbalance_distribution.csv"


def load_distribution() -> tuple[list[str], list[int], list[float]]:
    """Load LOS distribution from CSV. Returns (labels, counts, percents)."""
    labels, counts, percents = [], [], []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            labels.append(row["los_class"])
            counts.append(int(row["count"]))
            percents.append(float(row["percent"]))
    return labels, counts, percents


def bin_to_los_range(label: str) -> tuple[float, float]:
    """Parse label like '<1 day', '1 day', '2 days', '15+ days' to (low, high) range."""
    if label == "<1 day":
        return (0.0, 1.0)
    if label.endswith("+ days"):
        n = int(label.split("+")[0].strip())
        return (float(n), float("inf"))
    # "1 day" or "2 days"
    n = int("".join(c for c in label if c.isdigit()))
    return (float(n), float(n + 1))


def evaluate_custom_boundaries(
    labels: list[str],
    counts: list[int],
    boundaries: list[float],
) -> list[tuple[str, int, float]]:
    """
    boundaries define bins: [b0,b1), [b1,b2), ..., [b_{n-1}, b_n)
    boundaries = [0, 1, 2, 4, 7, 14, inf] -> 6 bins
    """
    n_total = sum(counts)
    result = []
    for i in range(len(boundaries) - 1):
        low, high = boundaries[i], boundaries[i + 1]
        cnt = 0
        for lbl, c in zip(labels, counts):
            rng = bin_to_los_range(lbl)
            # Overlap with [low, high)
            if rng[1] > low and rng[0] < high:
                overlap_low = max(rng[0], low)
                overlap_high = min(rng[1], high) if high != float("inf") else rng[1]
                span = rng[1] - rng[0]
                if span == float("inf") or span <= 0:
                    fraction = 1.0 if overlap_high > overlap_low else 0.0
                else:
                    fraction = (overlap_high - overlap_low) / span
                cnt += int(round(c * fraction))
        pct = 100 * cnt / n_total if n_total > 0 else 0
        if high == float("inf"):
            label = f"[{int(low)}+ days"
        else:
            label = f"[{int(low)},{int(high)}) days"
        result.append((label, cnt, pct))
    return result


def main():
    if not CSV_PATH.exists():
        print(f"Run plot_los_imbalance.py first to generate {CSV_PATH}")
        return

    labels, counts, percents = load_distribution()
    n_total = sum(counts)
    print("=" * 70)
    print("LOS DISTRIBUTION (24h ICU ECG dataset)")
    print("=" * 70)
    print(f"Total records: {n_total:,}")
    print()
    print("Raw distribution (per integer day):")
    for lbl, cnt, pct in zip(labels, counts, percents):
        print(f"  {lbl:>8}: {cnt:>6,} ({pct:5.2f}%)")
    print()

    # Cumulative for percentiles
    cum = 0
    cum_pcts = []
    for c in counts:
        cum += c
        cum_pcts.append(100 * cum / n_total)

    print("Cumulative percentiles (approximate):")
    for i, (lbl, cp) in enumerate(zip(labels, cum_pcts)):
        print(f"  up to {lbl:>8}: {cp:5.1f}%")
    print()

    # Recommendation: 7 bins, clinically interpretable
    # Boundaries chosen to balance size while keeping clinical meaning
    # [0,1) <24h, [1,2) 1d, [2,4) 2-3d short, [4,7) 4-6d medium, [7,14) 1-2w, [14,+) 2w+
    boundaries_v1 = [0, 1, 2, 4, 7, 14, float("inf")]  # 6 bins
    boundaries_v2 = [0, 1, 2, 3, 5, 7, 10, float("inf")]  # 7 bins - finer

    print("=" * 70)
    print("RECOMMENDED BINNING (clinical interpretability focus)")
    print("=" * 70)

    print("\n--- Option A: 6 Bins (EMPFOHLEN – beste klinische Lesbarkeit) ---")
    bins_a = evaluate_custom_boundaries(labels, counts, boundaries_v1)
    for label, cnt, pct in bins_a:
        print(f"  {label:>18}: {cnt:>6,} ({pct:5.2f}%)")
    print("  Labels: <24h | 1 day | 2-3 days (short) | 4-6 days (medium) | 1-2 weeks | 2+ weeks")

    print("\n--- Option B: 7 bins (feiner, trotzdem interpretierbar) ---")
    bins_b = evaluate_custom_boundaries(labels, counts, boundaries_v2)
    for label, cnt, pct in bins_b:
        print(f"  {label:>18}: {cnt:>6,} ({pct:5.2f}%)")
    print("  Labels: <24h | 1 day | 2 days | 3-4 days | 5-6 days | 7-9 days | 10+ days")

    print("\n" + "=" * 70)
    print("EMPFOHLEN: Option A (6 Bins) – beste Balance aus Klinik und Verteilung")
    print("  Boundaries: [0, 1, 2, 4, 7, 14, +inf)")
    print("  Implementierung: neue Strategie 'custom_boundaries' oder fixe 6-Bin-Logik")
    print("=" * 70)


if __name__ == "__main__":
    main()
