#!/usr/bin/env python3
"""
Plot MAE (best job per architecture) vs parameter count.

Scans outputs/logs/slurm_*.out, extracts model type and Test LOS MAE per job,
keeps the best (lowest) MAE per architecture, and plots against parameter count.

Usage:
  python scripts/analysis/plot_architecture_params_mae.py
  python scripts/analysis/plot_architecture_params_mae.py --logs-dir outputs/logs --out outputs/analysis/mae_vs_params.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Import parser
from scripts.analysis.parse_training_results import parse_log_text

MODEL_ORDER = [
    "XGBoost",
    "CNN Scratch",
    "LSTM Uni",
    "LSTM Bi",
    "Hybrid CNN-LSTM",
    "DeepECG-SL",
    "HuBERT-ECG",
]

# Map log "Model type: X" to display name
MODEL_TYPE_TO_NAME = {
    "CNNScratch": "CNN Scratch",
    "LSTM1D": "LSTM Uni",
    "LSTM1D_Unidirectional": "LSTM Uni",
    "LSTM1D_Bidirectional": "LSTM Bi",
    "HybridCNNLSTM": "Hybrid CNN-LSTM",
    "DeepECG_SL": "DeepECG-SL",
    "HuBERT_ECG": "HuBERT-ECG",
    "HuBERT-ECG": "HuBERT-ECG",
}

# Config path hints for XGBoost (no "Model type" in classical ML logs)
XGBOOST_PATTERNS = ["xgboost", "XGBoost", "train_xgboost"]


def _extract_model_type(text: str, log_path: Path) -> str | None:
    """Extract model type from log. Returns display name or None."""
    # 1. Try "Model type: X"
    m = re.search(r"Model\s+type:\s*(\S+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        return MODEL_TYPE_TO_NAME.get(raw, raw)

    # 2. Try "Model config: path" for XGBoost
    m = re.search(r"Model\s+config:\s*([^\s]+)", text, re.IGNORECASE)
    if m:
        path = m.group(1)
        if "xgboost" in path.lower():
            return "XGBoost"
        if "cnn_scratch" in path:
            return "CNN Scratch"
        if "lstm/unidirectional" in path or "lstm_2layer" in path and "bi" not in path:
            return "LSTM Uni"
        if "lstm_bi" in path or "bidirectional" in path:
            return "LSTM Bi"
        if "hybrid_cnn_lstm" in path:
            return "Hybrid CNN-LSTM"
        if "deepecg_sl" in path:
            return "DeepECG-SL"
        if "hubert" in path.lower():
            return "HuBERT-ECG"

    # 3. Training script title
    if "XGBoost" in text and "Training" in text:
        return "XGBoost"
    if "Training CNN" in text or "CNN from scratch" in text:
        return "CNN Scratch"
    if "Training LSTM1D Unidirectional" in text:
        return "LSTM Uni"
    if "Training LSTM1D Bidirectional" in text:
        return "LSTM Bi"
    if "Training Hybrid CNN-LSTM" in text or "Hybrid CNN-LSTM" in text:
        return "Hybrid CNN-LSTM"
    if "Training DeepECG-SL" in text or "DeepECG-SL" in text:
        return "DeepECG-SL"
    if "HuBERT" in text or "HuBERT-ECG" in text:
        return "HuBERT-ECG"

    return None


def get_best_mae_per_architecture(
    logs_dir: Path,
    subgroup_leq10: bool = False,
) -> dict[str, float]:
    """Scan slurm_*.out, parse each, keep best (lowest) MAE per architecture.

    If subgroup_leq10=True, use MAE for LOS ≤ 10 days subgroup instead of overall.
    """
    metric_key = "test_los_mae_leq10" if subgroup_leq10 else "test_los_mae"
    results: dict[str, float] = {}
    jobs_per_arch: dict[str, list[tuple[int, float]]] = {}

    for p in sorted(logs_dir.glob("slurm_*.out")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        parsed = parse_log_text(text)
        mae = parsed.test_metrics.get(metric_key)
        if mae is None and not subgroup_leq10:
            mae = parsed.best_val_mae
        if mae is None:
            continue
        arch = _extract_model_type(text, p)
        if arch is None:
            continue
        job_id = parsed.job_id or 0
        jobs_per_arch.setdefault(arch, []).append((job_id, mae))

    for arch, pairs in jobs_per_arch.items():
        best_mae = min(mae for _, mae in pairs)
        results[arch] = best_mae

    return results


def get_parameter_counts() -> dict[str, int]:
    """Parameter counts per architecture (fallback values, no heavy model loading)."""
    return {
        "XGBoost": 25_000,  # heuristic (tree-based; no direct param count)
        "CNN Scratch": 46_627,  # from CNNScratch + MultiTask wrapper
        "LSTM Uni": 233_000,
        "LSTM Bi": 596_000,
        "Hybrid CNN-LSTM": 900_000,
        "DeepECG-SL": 100_000_000,
        "HuBERT-ECG": 93_000_000,
    }


def main() -> None:
    matplotlib.use("Agg")  # Non-interactive backend for CLI; avoids affecting notebook when imported
    parser = argparse.ArgumentParser(description="Plot MAE (best job) vs parameter count")
    parser.add_argument("--logs-dir", type=str, default="outputs/logs")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--leq10", action="store_true", help="Use MAE for LOS ≤ 10 days subgroup")
    parser.add_argument("--exclude", type=str, default=None, help="Comma-separated architectures to exclude (e.g. HuBERT-ECG)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Logs dir not found: {logs_dir}")
        sys.exit(1)

    best_mae = get_best_mae_per_architecture(logs_dir, subgroup_leq10=args.leq10)
    params = get_parameter_counts()

    exclude = set()
    if args.exclude:
        exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}

    # Build plot data: only architectures with MAE (and not excluded)
    names = [a for a in MODEL_ORDER if a in best_mae and a in params and a not in exclude]
    if not names:
        metric = "MAE (LOS ≤ 10 days)" if args.leq10 else "MAE"
        print(f"No architectures with {metric} found in logs. Check --logs-dir.")
        sys.exit(1)

    x = [params[n] for n in names]
    y = [best_mae[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    for i, (name, xi, yi) in enumerate(zip(names, x, y)):
        ax.scatter(xi, yi, s=150, c=[colors[i]], edgecolors="black", linewidths=0.5, zorder=2)
        # Rightmost points: place label to the left to avoid clipping at plot border
        if xi >= 50_000_000:
            ax.annotate(name, (xi, yi), xytext=(-8, 4), textcoords="offset points", fontsize=10, ha="right")
        else:
            ax.annotate(name, (xi, yi), xytext=(8, 4), textcoords="offset points", fontsize=10)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    if args.leq10:
        ax.set_ylabel("Test LOS MAE (days, LOS ≤ 10)")
        ax.set_title("Best Job MAE (LOS ≤ 10 days) per Architecture vs Parameter Count")
    else:
        ax.set_ylabel("Test LOS MAE (days)")
        ax.set_title("Best Job MAE per Architecture vs Parameter Count")

    out_path = args.out
    if out_path is None:
        out_path = "outputs/analysis/architecture_diagrams/mae_vs_params_leq10.png" if args.leq10 else "outputs/analysis/architecture_diagrams/mae_vs_params.png"
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.1, y=0.1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    for n in names:
        print(f"  {n}: MAE={best_mae[n]:.4f} days, Params={params[n]:,}")


if __name__ == "__main__":
    main()
