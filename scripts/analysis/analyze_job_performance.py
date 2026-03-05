#!/usr/bin/env python3
"""
Analyze job IDs from model_overview.csv by performance metrics.
Metrics: MAE (lower better), R² (higher better), AUC (higher better).
"""

import csv
from pathlib import Path
from typing import Optional


def _parse_float(s: str) -> Optional[float]:
    """Parse European (comma decimal) or US (dot decimal) format."""
    if not s or s in ("N/A", "", " "):
        return None
    s = s.strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def load_csv_rows(csv_path: Path) -> list[dict]:
    """Load data rows from model_overview.csv.
    Fixed column layout: 1=model, 5=job_id, 9=MAE, 10=R², 11=AUC, 12=notes
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) < 10 or not any(row):
                continue
            job_id_s = row[5].strip() if len(row) > 5 else ""
            if not job_id_s.isdigit():
                continue
            mae = _parse_float(row[9]) if len(row) > 9 else None
            if mae is None:
                continue
            r2 = _parse_float(row[10]) if len(row) > 10 else None
            auc = _parse_float(row[11]) if len(row) > 11 else None
            model = row[1].strip() if len(row) > 1 else ""
            notes = row[12].strip() if len(row) > 12 else ""
            rows.append({
                "job_id": int(job_id_s),
                "model": model,
                "mae": mae,
                "r2": r2,
                "auc": auc,
                "notes": notes,
            })
    return rows


def main() -> None:
    csv_path = Path(__file__).resolve().parents[2] / "docs" / "results" / "model_overview.csv"
    rows = load_csv_rows(csv_path)

    # Filter rows with at least MAE
    valid = [r for r in rows if r["mae"] is not None]
    print(f"Loaded {len(valid)} jobs with metrics from {csv_path.name}\n")

    def _r2(r): return f"{r['r2']:.4f}" if r['r2'] is not None else "N/A"
    def _auc(r): return f"{r['auc']:.4f}" if r['auc'] is not None else "N/A"

    # --- 1. Best by MAE (LOS regression - primary task) ---
    by_mae = sorted(valid, key=lambda x: x["mae"])
    print("=" * 70)
    print("TOP 10 by MAE (LOS regression, lower = better)")
    print("=" * 70)
    for i, r in enumerate(by_mae[:10], 1):
        print(f"  {i:2}. job {r['job_id']:7} | MAE {r['mae']:.4f} | R² {_r2(r):>6} | AUC {_auc(r):>6} | {r['model'][:25]:25} | {r['notes'][:30]}")

    # --- 2. Best by R² (among those with R²) ---
    with_r2 = [r for r in valid if r["r2"] is not None]
    by_r2 = sorted(with_r2, key=lambda x: x["r2"], reverse=True)
    print("\n" + "=" * 70)
    print("TOP 10 by R² (LOS regression, higher = better)")
    print("=" * 70)
    for i, r in enumerate(by_r2[:10], 1):
        print(f"  {i:2}. job {r['job_id']:7} | MAE {r['mae']:.4f} | R² {r['r2']:.4f} | AUC {_auc(r):>6} | {r['model'][:25]:25} | {r['notes'][:30]}")

    # --- 3. Best by AUC (mortality) ---
    with_auc = [r for r in valid if r["auc"] is not None]
    by_auc = sorted(with_auc, key=lambda x: x["auc"], reverse=True)
    print("\n" + "=" * 70)
    print("TOP 10 by AUC (mortality, higher = better)")
    print("=" * 70)
    for i, r in enumerate(by_auc[:10], 1):
        print(f"  {i:2}. job {r['job_id']:7} | MAE {r['mae']:.4f} | R² {_r2(r):>6} | AUC {r['auc']:.4f} | {r['model'][:25]:25} | {r['notes'][:30]}")

    # --- 4. Balanced composite (MAE primary, R² and AUC secondary) ---
    def composite(r):
        mae_norm = 1.0 - min(r["mae"] / 4.0, 1.0) if r["mae"] else 0
        r2_val = r["r2"] if r["r2"] is not None else 0
        auc_val = r["auc"] if r["auc"] else 0.5
        return 0.5 * mae_norm + 0.25 * max(0, r2_val) + 0.25 * (auc_val - 0.5) * 2  # scale AUC 0.5-1 to 0-1

    by_composite = sorted(valid, key=composite, reverse=True)
    print("\n" + "=" * 70)
    print("TOP 10 by composite score (0.5*MAE_norm + 0.25*R² + 0.25*AUC_norm)")
    print("=" * 70)
    for i, r in enumerate(by_composite[:10], 1):
        print(f"  {i:2}. job {r['job_id']:7} | MAE {r['mae']:.4f} | R² {_r2(r):>6} | AUC {_auc(r):>6} | {r['model'][:25]:25} | {r['notes'][:30]}")

    # --- Summary ---
    best_mae_job = by_mae[0]
    best_auc_job = by_auc[0] if by_auc else None
    best_composite_job = by_composite[0]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Best LOS (MAE):  job {best_mae_job['job_id']} ({best_mae_job['model']}) – MAE {best_mae_job['mae']:.4f} days")
    if best_auc_job:
        print(f"  Best Mortality (AUC): job {best_auc_job['job_id']} ({best_auc_job['model']}) – AUC {best_auc_job['auc']:.4f}")
    print(f"  Best balanced:   job {best_composite_job['job_id']} ({best_composite_job['model']})")


if __name__ == "__main__":
    main()
