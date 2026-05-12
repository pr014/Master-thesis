#!/usr/bin/env python3
"""
Plot diagrams connecting parameter count with accuracy for each model architecture.

Generates one diagram per architecture: XGBoost, CNN Scratch, LSTM Uni, LSTM Bi, Hybrid CNN-LSTM, DeepECG-SL, HuBERT-ECG.
Each diagram shows parameter count and accuracy (or placeholder if not yet evaluated).

Usage:
  python scripts/analysis/plot_architecture_params_accuracy.py
  python scripts/analysis/plot_architecture_params_accuracy.py --accuracy outputs/analysis/architecture_accuracy.csv

To provide accuracy values, create a CSV with columns: model, mortality_acc, los_mae (optional).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np


# Model order as requested
MODEL_ORDER = [
    "XGBoost",
    "CNN Scratch",
    "LSTM Uni",
    "LSTM Bi",
    "Hybrid CNN-LSTM",
    "DeepECG-SL",
    "HuBERT-ECG",
]


def get_parameter_counts(load_heavy_models: bool = False) -> dict[str, int]:
    """Get parameter counts for each model by loading them (or use known values).

    Set load_heavy_models=False to skip DeepECG-SL and HuBERT-ECG (avoids slow HuggingFace loads).
    """
    try:
        from src.utils.config_loader import load_config
        from src.models import CNNScratch, HybridCNNLSTM
        from src.models.lstm import LSTM1D_Unidirectional, LSTM1D_Bidirectional
    except ImportError as e:
        print(f"Warning: Could not import models, using fallback: {e}")
        return _fallback_params()

    params: dict[str, int] = {}

    # XGBoost: tree-based, use proxy (n_estimators * nodes per tree)
    try:
        cfg = load_config(model_config_path=Path("configs/classical_ml/xgboost_handcrafted.yaml"))
        n_est = cfg.get("xgboost", {}).get("n_estimators", 200)
        max_d = cfg.get("xgboost", {}).get("max_depth", 6)
        params["XGBoost"] = int(n_est * (2 ** (max_d + 1) - 1))
    except Exception:
        params["XGBoost"] = 25_000

    # CNN Scratch
    try:
        cfg = load_config(model_config_path=Path("configs/model/CNN/cnn_scratch.yaml"))
        m = CNNScratch(cfg)
        params["CNN Scratch"] = m.count_parameters()
    except Exception:
        params["CNN Scratch"] = 90_000

    # LSTM Uni, LSTM Bi
    try:
        cfg = load_config(model_config_path=Path("configs/model/lstm/unidirectional/lstm_2layer.yaml"))
        m = LSTM1D_Unidirectional(cfg)
        params["LSTM Uni"] = m.count_parameters()
    except Exception:
        params["LSTM Uni"] = 233_000
    try:
        cfg = load_config(model_config_path=Path("configs/model/lstm/bidirectional/lstm_bi_2layer.yaml"))
        m = LSTM1D_Bidirectional(cfg)
        params["LSTM Bi"] = m.count_parameters()
    except Exception:
        params["LSTM Bi"] = 596_000

    # Hybrid CNN-LSTM
    try:
        cfg = load_config(model_config_path=Path("configs/model/hybrid_cnn_lstm/hybrid_cnn_lstm.yaml"))
        m = HybridCNNLSTM(cfg)
        params["Hybrid CNN-LSTM"] = m.count_parameters()
    except Exception:
        params["Hybrid CNN-LSTM"] = 900_000

    # DeepECG-SL, HuBERT-ECG: skip by default (require HuggingFace/checkpoints)
    if load_heavy_models:
        try:
            from src.models.deepecg_sl import DeepECG_SL
            cfg = load_config(model_config_path=Path("configs/model/deepecg_sl/deepecg_sl.yaml"))
            m = DeepECG_SL(cfg)
            params["DeepECG-SL"] = m.count_parameters()
        except Exception:
            params["DeepECG-SL"] = 100_000_000
        try:
            from src.models.hubert_ecg import HuBERT_ECG
            cfg = load_config(model_config_path=Path("configs/model/hubert_ecg/hubert_ecg.yaml"))
            m = HuBERT_ECG(cfg)
            params["HuBERT-ECG"] = m.count_parameters()
        except Exception:
            params["HuBERT-ECG"] = 93_000_000
    else:
        params["DeepECG-SL"] = 100_000_000
        params["HuBERT-ECG"] = 93_000_000

    return params


def _fallback_params() -> dict[str, int]:
    """Fallback parameter counts when models cannot be loaded."""
    return {
        "XGBoost": 25_000,       # proxy (tree nodes)
        "CNN Scratch": 90_000,
        "LSTM Uni": 233_000,
        "LSTM Bi": 596_000,
        "Hybrid CNN-LSTM": 900_000,
        "DeepECG-SL": 100_000_000,
        "HuBERT-ECG": 93_000_000,
    }


def load_accuracy_from_csv(path: Path) -> dict[str, float]:
    """Load accuracy values from CSV. Columns: model, mortality_acc (or los_r2)."""
    import csv
    acc: dict[str, float] = {}
    name_map = {
        "xgboost": "XGBoost",
        "cnn_scratch": "CNN Scratch",
        "cnn from scratch": "CNN Scratch",
        "lstm_uni": "LSTM Uni",
        "lstm uni": "LSTM Uni",
        "lstm_bi": "LSTM Bi",
        "lstm bi": "LSTM Bi",
        "hybrid_cnn_lstm": "Hybrid CNN-LSTM",
        "hybrid cnn lstm": "Hybrid CNN-LSTM",
        "hybrid": "Hybrid CNN-LSTM",
        "deepecg_sl": "DeepECG-SL",
        "deepecg-sl": "DeepECG-SL",
        "hubert_ecg": "HuBERT-ECG",
        "hubert-ecg": "HuBERT-ECG",
    }
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            m = row.get("model", "").strip().lower()
            m = name_map.get(m, row.get("model", "").strip())
            if not m:
                continue
            v = row.get("mortality_acc") or row.get("accuracy")
            if v:
                acc[m] = float(v)
    return acc


def format_params(n: int) -> str:
    """Format parameter count for display."""
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    if n >= 1_000:
        return f"{n/1e3:.0f}K"
    return str(n)


def plot_single_architecture(
    ax: plt.Axes,
    name: str,
    params: int,
    accuracy: float | None,
) -> None:
    """Create one diagram for a single architecture: params and accuracy connected."""
    ax.set_title(name, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    param_str = format_params(params)
    acc_str = f"{accuracy:.2%}" if accuracy is not None else "—"

    # Top row: two boxes connected
    # Left box: Parameters
    rect1 = plt.Rectangle((0.05, 0.45), 0.35, 0.35, fill=True, facecolor="#E8F4F8", edgecolor="#2E86AB", lw=2)
    ax.add_patch(rect1)
    ax.text(0.225, 0.7, "Parameters", fontsize=10, color="#2E86AB", ha="center", fontweight="bold")
    ax.text(0.225, 0.55, param_str, fontsize=18, ha="center", color="#1a5276", fontweight="bold")

    # Right box: Accuracy
    rect2 = plt.Rectangle((0.6, 0.45), 0.35, 0.35, fill=True, facecolor="#F8E8F4", edgecolor="#A23B72", lw=2)
    ax.add_patch(rect2)
    ax.text(0.775, 0.7, "Mortality Acc.", fontsize=10, color="#A23B72", ha="center", fontweight="bold")
    ax.text(0.775, 0.55, acc_str, fontsize=18, ha="center", color="#6B2D5C", fontweight="bold")

    # Connecting arrow between boxes
    ax.annotate(
        "",
        xy=(0.58, 0.625),
        xytext=(0.42, 0.625),
        arrowprops=dict(arrowstyle="->", color="#555", lw=2.5),
    )

    # Bottom: dual bar chart (params on log scale, accuracy 0-1)
    param_max = 120_000_000
    param_norm = np.log10(params + 1) / np.log10(param_max) if params > 0 else 0
    bar_max_width = 0.35
    ax.barh(0.18, param_norm * bar_max_width, left=0.05, height=0.1, color="#2E86AB", alpha=0.8)
    ax.text(0.225, 0.28, f"Params: {param_str}", fontsize=9, ha="center", va="bottom")
    if accuracy is not None:
        ax.barh(0.02, accuracy * bar_max_width, left=0.55, height=0.08, color="#A23B72", alpha=0.8)
        ax.text(0.775, 0.10, f"Acc: {acc_str}", fontsize=9, ha="center", va="bottom")


def plot_combined_scatter(
    fig: plt.Figure,
    params_dict: dict[str, int],
    accuracy_dict: dict[str, float],
) -> None:
    """Create combined scatter: x=params (log), y=accuracy."""
    ax = fig.add_subplot(111)
    names = [m for m in MODEL_ORDER if m in params_dict]
    x = [params_dict[m] for m in names]
    y = [accuracy_dict.get(m, np.nan) for m in names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    for i, n in enumerate(names):
        xi = x[i]
        yi = y[i]
        if np.isnan(yi):
            ax.scatter(xi, 0, s=120, c=[colors[i]], label=n, zorder=2)
            ax.annotate(n, (xi, 0), xytext=(5, 0), textcoords="offset points", fontsize=9)
        else:
            ax.scatter(xi, yi, s=120, c=[colors[i]], label=n, zorder=2)
            ax.annotate(n, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Mortality Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="#ccc", ls="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot architecture diagrams: params vs accuracy")
    parser.add_argument("--accuracy", type=str, default=None, help="CSV with model, mortality_acc")
    parser.add_argument("--out-dir", type=str, default="outputs/analysis/architecture_diagrams")
    parser.add_argument("--combined", action="store_true", help="Also create combined scatter plot")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params_dict = get_parameter_counts()
    accuracy_dict: dict[str, float] = {}
    if args.accuracy:
        acc_path = Path(args.accuracy)
        if acc_path.exists():
            accuracy_dict = load_accuracy_from_csv(acc_path)

    # Create one figure per architecture
    for name in MODEL_ORDER:
        if name not in params_dict:
            continue
        fig, ax = plt.subplots(figsize=(5, 3))
        plot_single_architecture(ax, name, params_dict[name], accuracy_dict.get(name))
        fig.tight_layout()
        fname = name.lower().replace(" ", "_").replace("-", "_") + "_params_accuracy.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / fname}")

    # Combined overview
    if args.combined:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_combined_scatter(fig, params_dict, accuracy_dict)
        fig.tight_layout()
        fig.savefig(out_dir / "combined_params_vs_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / 'combined_params_vs_accuracy.png'}")

    print("Done. To add accuracy values, create a CSV with columns: model, mortality_acc")


if __name__ == "__main__":
    main()
