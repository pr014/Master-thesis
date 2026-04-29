#!/usr/bin/env python3
"""Bland-Altman plot for LOS prediction by job ID.

Loads the checkpoint for a given SLURM job, runs inference on the test set,
and produces a Bland-Altman plot (predicted LOS vs. true LOS) coloured by
the ECG recording day relative to ICU admission.

Usage
-----
# Activate venv first, then run from project root:
source venv/bin/activate
python scripts/analysis/results/bland_altman.py --job 3346714

# Save to a custom output path:
python scripts/analysis/results/bland_altman.py --job 3346714 --out /tmp/ba.png
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so src.* imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ecg import create_dataloaders
from src.data.ecg.ecg_dataset import construct_ecg_time
from src.data.labeling import load_icustays, ICUStayMapper, load_mortality_mapping
from src.training.train_loop import _forward_aux_kwargs
from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# Helper: find checkpoint
# ---------------------------------------------------------------------------

def _find_checkpoint(job_id: str) -> Path:
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    candidates = sorted(ckpt_dir.glob(f"*_best_{job_id}.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching '*_best_{job_id}.pt' found in {ckpt_dir}"
        )
    if len(candidates) > 1:
        print(f"[warn] Multiple checkpoints found for job {job_id}, using: {candidates[0].name}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Helper: load config from checkpoint
# ---------------------------------------------------------------------------

def _get_config_from_checkpoint(checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" in ckpt and ckpt["config"]:
        return ckpt["config"]
    model_config_path = (ckpt.get("config_paths") or {}).get("model")
    if not model_config_path:
        raise ValueError("Checkpoint contains neither 'config' nor 'config_paths.model'.")
    path = (
        PROJECT_ROOT / model_config_path
        if not Path(model_config_path).is_absolute()
        else Path(model_config_path)
    )
    return load_config(model_config_path=path)


# ---------------------------------------------------------------------------
# Helper: build model from config + checkpoint
# ---------------------------------------------------------------------------

def _build_model(config: dict, checkpoint_path: Path, device: torch.device):
    from src.models import (
        CNNScratch,
        HybridCNNLSTM,
        DeepECG_SL,
        HuBERT_ECG,
    )
    from src.models.lstm import LSTM1D_Unidirectional, LSTM1D_Bidirectional
    from src.models.core.multi_task_model import MultiTaskECGModel

    model_type = config.get("model", {}).get("type", "").strip().lower().replace("-", "_")
    mapping = {
        "lstm1d": LSTM1D_Unidirectional,
        "lstm1d_bidirectional": LSTM1D_Bidirectional,
        "cnnscratch": CNNScratch,
        "hybridcnnlstm": HybridCNNLSTM,
        "deepecg_sl": DeepECG_SL,
        "hubert_ecg": HuBERT_ECG,
    }
    if model_type not in mapping:
        raise ValueError(f"Unknown model type: {model_type!r}. Supported: {list(mapping)}")

    base_model = mapping[model_type](config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    is_multitask = any(
        k.startswith("base_model.") or k.startswith("los_head.") or k.startswith("mortality_head.")
        for k in state.keys()
    )
    if is_multitask:
        model = MultiTaskECGModel(base_model, config)
        model.load_state_dict(state)
    else:
        model = base_model
        model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Helper: setup ICU mapper
# ---------------------------------------------------------------------------

def _setup_icu_mapper(config: dict) -> ICUStayMapper:
    import os

    icustays_path = (
        Path(os.getenv("ICUSTAYS_PATH")) if os.getenv("ICUSTAYS_PATH") else None
    )
    if not icustays_path or not icustays_path.exists():
        data_dir = config.get("data", {}).get("data_dir", "")
        icustays_path = (
            Path(data_dir).resolve().parent.parent / "labeling" / "labels_csv" / "icustays.csv"
            if data_dir
            else PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "icustays.csv"
        )
    if not icustays_path.exists():
        icustays_path = PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "icustays.csv"
    if not icustays_path.exists():
        raise FileNotFoundError(f"icustays.csv not found: {icustays_path}")

    icustays_df = load_icustays(str(icustays_path))
    mortality_mapping = None
    if config.get("multi_task", {}).get("enabled", False):
        adm_path = config.get("multi_task", {}).get("admissions_path", "")
        if adm_path:
            p = (
                PROJECT_ROOT / adm_path
                if not Path(adm_path).is_absolute()
                else Path(adm_path)
            )
            if p.exists():
                mortality_mapping = load_mortality_mapping(str(p), icustays_df)
    return ICUStayMapper(icustays_df, mortality_mapping=mortality_mapping)


# ---------------------------------------------------------------------------
# Inference: collect predictions + recording day metadata
# ---------------------------------------------------------------------------

def _los_tensor_from_output(out):
    if isinstance(out, dict):
        return out.get("los", out)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _collect_predictions_and_meta(
    model,
    test_loader,
    device: torch.device,
    icu_mapper: ICUStayMapper,
    config: dict,
) -> pd.DataFrame:
    stay_id_to_intime = pd.Series(
        icu_mapper.icustays_df.set_index("stay_id")["intime"]
    )
    rows = []
    model.eval()
    fwd_params = inspect.signature(model.forward).parameters
    with torch.no_grad():
        for batch in test_loader:
            signals = batch["signal"].to(device)
            labels = batch["label"].cpu().numpy()
            meta_list = batch["meta"]
            valid = labels >= 0
            if not np.any(valid):
                continue
            signals, labels = signals[valid], labels[valid]
            meta_list = [m for m, v in zip(meta_list, valid) if v]

            demographic_features = None
            if batch.get("demographic_features") is not None:
                demographic_features = batch["demographic_features"].to(device)[valid]
            icu_unit_features = None
            if batch.get("icu_unit_features") is not None:
                icu_unit_features = batch["icu_unit_features"].to(device)[valid]
            sofa_features = None
            if batch.get("sofa_features") is not None:
                sofa_features = batch["sofa_features"].to(device)[valid]
            icu_therapy_support_features = None
            if batch.get("icu_therapy_support_features") is not None:
                icu_therapy_support_features = batch["icu_therapy_support_features"].to(
                    device
                )[valid]
            ehr_window_features = None
            if batch.get("ehr_window_features") is not None:
                ehr_window_features = batch["ehr_window_features"].to(device)[valid]

            aux = _forward_aux_kwargs(
                config,
                demographic_features,
                icu_unit_features,
                sofa_features,
                icu_therapy_support_features,
                ehr_window_features,
            )
            aux = {k: v for k, v in aux.items() if k in fwd_params}
            out = model(signals, **aux)
            los_out = _los_tensor_from_output(out)
            preds = (
                (los_out.squeeze(-1) if los_out.dim() > 1 else los_out).cpu().numpy()
            )

            for i in range(len(labels)):
                m = meta_list[i]
                stay_id = m.get("stay_id")
                base_date = m.get("base_date")
                base_time = m.get("base_time")
                recording_day = None
                if stay_id is not None and base_date is not None and base_time is not None:
                    ecg_time = construct_ecg_time(base_date, base_time)
                    if ecg_time is not None and stay_id in stay_id_to_intime.index:
                        recording_day = (
                            pd.Timestamp(ecg_time)
                            - pd.Timestamp(stay_id_to_intime[stay_id])
                        ).total_seconds() / 86400.0
                rows.append(
                    {
                        "y_true": float(labels[i]),
                        "y_pred": float(preds[i]),
                        "recording_day": recording_day,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_bland_altman(df: pd.DataFrame, job_id: str, out_path: Path) -> None:
    df_ba = df.copy()
    df_ba["mean_ba"] = (df_ba["y_true"] + df_ba["y_pred"]) / 2.0
    df_ba["diff_ba"] = df_ba["y_pred"] - df_ba["y_true"]
    has_day = df_ba["recording_day"].notna()
    df_ba["day_bin"] = np.where(
        has_day,
        np.clip(np.floor(df_ba["recording_day"]).astype(int), 0, 99),
        -1,
    )

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    mean_diff = df_ba["diff_ba"].mean()
    std_diff = df_ba["diff_ba"].std()
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    if has_day.any():
        day_order = sorted(df_ba.loc[has_day, "day_bin"].unique())
        try:
            cmap = plt.colormaps["viridis"].resampled(max(len(day_order), 2))
        except AttributeError:
            cmap = plt.cm.get_cmap("viridis", max(len(day_order), 2))
        for k, day in enumerate(day_order):
            mask = (df_ba["day_bin"] == day) & has_day
            sub = df_ba.loc[mask]
            color = (
                cmap(k / max(len(day_order) - 1, 1)) if len(day_order) > 1 else cmap(0)
            )
            ax.scatter(
                sub["mean_ba"],
                sub["diff_ba"],
                label=f"Day {day} (n={len(sub)})",
                alpha=0.6,
                s=24,
                color=color,
            )
        if (df_ba["day_bin"] == -1).any():
            u = df_ba["day_bin"] == -1
            ax.scatter(
                df_ba.loc[u, "mean_ba"],
                df_ba.loc[u, "diff_ba"],
                label="Unknown",
                alpha=0.5,
                s=24,
                color="gray",
            )
    else:
        ax.scatter(df_ba["mean_ba"], df_ba["diff_ba"], alpha=0.5, s=24, label="Test ECGs")

    ax.axhline(
        mean_diff,
        color="k",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean difference: {mean_diff:.3f}",
    )
    ax.axhline(
        loa_upper,
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"+1.96 SD: {loa_upper:.3f}",
    )
    ax.axhline(
        loa_lower,
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"-1.96 SD: {loa_lower:.3f}",
    )
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    ax.set_xlabel("Mean (true LOS + predicted LOS) / 2 [days]")
    ax.set_ylabel("Difference (predicted LOS − true LOS) [days]")
    ax.set_title(f"Bland-Altman: LOS Prediction (Job ID {job_id})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bland-Altman plot for LOS prediction."
    )
    parser.add_argument(
        "--job",
        type=str,
        required=True,
        help="SLURM job ID of the training run (e.g. 3346714)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output path for the plot (default: "
            "outputs/bland_altman/bland_altman_job_<job>.png)"
        ),
    )
    args = parser.parse_args()

    job_id = args.job.strip()
    out_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "outputs" / "bland_altman" / f"bland_altman_job_{job_id}.png"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Job ID: {job_id}, Device: {device}")

    checkpoint_path = _find_checkpoint(job_id)
    print(f"Checkpoint: {checkpoint_path}")

    config = _get_config_from_checkpoint(checkpoint_path)
    icu_mapper = _setup_icu_mapper(config)

    _, _, test_loader = create_dataloaders(
        config=config,
        labels=None,
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,
    )
    if test_loader is None:
        raise RuntimeError("No test set found (set test_split > 0 in config).")

    model = _build_model(config, checkpoint_path, device)

    print("Running inference on test set...")
    df = _collect_predictions_and_meta(model, test_loader, device, icu_mapper, config)
    print(f"Test ECGs: {len(df)}, with recording day: {df['recording_day'].notna().sum()}")

    _plot_bland_altman(df, job_id, out_path)


if __name__ == "__main__":
    main()
