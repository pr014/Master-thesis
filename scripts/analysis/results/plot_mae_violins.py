#!/usr/bin/env python3
"""Create MAE violin plots from saved model jobs.

The YAML config controls which jobs are evaluated, how many plots are allowed,
labels, output filenames, and whether errors are aggregated per ECG window or
per ICU stay.

Optional **ICU LOS cap** (evaluation-time filter): set ``evaluation.max_true_los_days``
to a float (e.g. ``10``) to keep only samples whose **true** LOS label (days) is
``<=`` that value, before stay-level aggregation. Set to ``null`` or omit the key
to use the full test set.

Optional **multiple figures**: define ``plot_groups`` (list of ``pdf_path`` +
``jobs``). Each group produces one PDF; jobs are deduplicated for inference so
each checkpoint runs once. If ``plot_groups`` is absent, the legacy single
``jobs`` + ``output.pdf_path`` layout is used.

CLI **``--plot-group-index N``** (with ``plot_groups``): run only the Nth group
(0-based)—one PDF and only that group's jobs—so you can submit one short SLURM
job per figure and avoid walltime timeouts on the all-in-one run.

Example
-------
python scripts/analysis/results/plot_mae_violins.py \
  --config configs/analysis/mae_violin_baseline.yaml
"""

from __future__ import annotations

import argparse
import copy
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ecg import create_dataloaders
from src.features import extract_handcrafted_features
from src.models import CNNScratch, DeepECG_SL, HuBERT_ECG, HybridCNNLSTM, MultiTaskECGModel
from src.models.classical_ml import XGBoostECGModel
from src.models.lstm import LSTM1D_Bidirectional, LSTM1D_Unidirectional
from src.training import setup_icustays_mapper
from src.training.train_loop import _forward_aux_kwargs
from src.utils.config_loader import load_config, load_yaml


TORCH_MODEL_REGISTRY = {
    "cnnscratch": CNNScratch,
    "cnn_scratch": CNNScratch,
    "lstm1d": LSTM1D_Unidirectional,
    "lstm1d_unidirectional": LSTM1D_Unidirectional,
    "lstm1d_bidirectional": LSTM1D_Bidirectional,
    "hybridcnnlstm": HybridCNNLSTM,
    "hybrid_cnn_lstm": HybridCNNLSTM,
    "deepecg_sl": DeepECG_SL,
    "hubert_ecg": HuBERT_ECG,
}


def _project_path(path_like: str | Path | None) -> Optional[Path]:
    if path_like is None:
        return None
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _pick_device(name: Optional[str]) -> torch.device:
    if name and str(name).lower() not in {"auto", "none", ""}:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_active_job_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["job_id"] = str(row.get("job_id", "")).strip()
    if not row["job_id"]:
        raise ValueError(f"Active job is missing job_id: {row}")
    row["plot_label"] = str(row.get("plot_label") or row.get("label") or f"job {row['job_id']}")
    row["feature_stage"] = row["plot_label"]
    return row


def _active_jobs_from_list(jobs_raw: list[dict[str, Any]], *, max_plots: int) -> list[dict[str, Any]]:
    jobs = [_normalize_active_job_row(dict(row)) for row in jobs_raw if row.get("enabled", True)]
    if not jobs:
        raise ValueError("No active jobs configured. Set at least one jobs[].enabled=true.")
    if len(jobs) > max_plots:
        raise ValueError(f"Configured {len(jobs)} active jobs, but max_plots={max_plots}.")
    return jobs


def _active_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    max_plots = int(config.get("plot", {}).get("max_plots", 7))
    return _active_jobs_from_list(config.get("jobs", []), max_plots=max_plots)


def _evaluation_max_true_los_days(evaluation_cfg: dict[str, Any]) -> Optional[float]:
    """If set, keep only rows with 0 <= true LOS (target) <= this value."""
    raw = evaluation_cfg.get("max_true_los_days")
    if raw is None or raw is False:
        return None
    if isinstance(raw, str) and not str(raw).strip():
        return None
    return float(raw)


def _dedupe_jobs_for_collection(job_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run each job_id once; first occurrence wins (plot_label must stay consistent if a job is reused)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in job_rows:
        if not row.get("enabled", True):
            continue
        jid = str(row.get("job_id", "")).strip()
        if not jid or jid in seen:
            continue
        seen.add(jid)
        out.append(_normalize_active_job_row(dict(row)))
    return out


def _plot_groups_from_config(config: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    groups = config.get("plot_groups")
    if not groups:
        return None
    if not isinstance(groups, list):
        raise ValueError("plot_groups must be a list of objects with pdf_path and jobs.")
    return [g for g in groups if g.get("enabled", True)]


def _collection_jobs_for_plot_groups(plot_groups: list[dict[str, Any]], default_max_plots: int) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for group in plot_groups:
        jobs_raw = group.get("jobs") or []
        max_plots = int(group.get("max_plots", default_max_plots))
        _active_jobs_from_list(jobs_raw, max_plots=max_plots)
        for row in jobs_raw:
            flattened.append(dict(row))
    return _dedupe_jobs_for_collection(flattened)


def _subset_errors_for_jobs(error_df: pd.DataFrame, jobs: list[dict[str, Any]]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for row in jobs:
        jid = str(row["job_id"])
        pl = str(row["plot_label"])
        m = (error_df["job_id"].astype(str) == jid) & (error_df["plot_label"] == pl)
        sub = error_df.loc[m]
        if sub.empty:
            print(f"[warn] No error rows for job_id={jid!r} plot_label={pl!r} (check plot_groups vs collection labels).")
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True)


def _apply_loader_overrides(config: dict[str, Any], batch_size: int, num_workers: int) -> dict[str, Any]:
    out = copy.deepcopy(config)
    out.setdefault("training", {})["batch_size"] = int(batch_size)
    out["training"]["num_workers"] = int(num_workers)
    out["training"]["pin_memory"] = False
    return out


def _is_xgboost_job(row: dict[str, Any]) -> bool:
    return str(row.get("model_type", "")).strip().lower() in {"xgboost", "xgb"}


def _find_torch_checkpoint(row: dict[str, Any]) -> Path:
    explicit = _project_path(row.get("checkpoint_path"))
    if explicit is not None:
        return explicit

    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    candidates = sorted(ckpt_dir.glob(f"*_best_{row['job_id']}.pt"))
    if not candidates:
        raise FileNotFoundError(f"No PyTorch checkpoint matching '*_best_{row['job_id']}.pt' in {ckpt_dir}")
    if len(candidates) > 1:
        print(f"[warn] Multiple checkpoints for job {row['job_id']}; using {candidates[0].name}")
    return candidates[0]


def _load_torch_model(row: dict[str, Any], device: torch.device, batch_size: int, num_workers: int):
    checkpoint_path = _find_torch_checkpoint(row)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or not checkpoint["config"]:
        raise ValueError(f"Checkpoint has no embedded config: {checkpoint_path}")

    config = _apply_loader_overrides(checkpoint["config"], batch_size, num_workers)
    model_type = str(config.get("model", {}).get("type", "")).strip()
    model_key = model_type.lower().replace("-", "_")
    model_cls = TORCH_MODEL_REGISTRY.get(model_key)
    if model_cls is None:
        raise ValueError(f"Unsupported model.type={model_type!r} in {checkpoint_path.name}")

    base_model = model_cls(config)
    if config.get("multi_task", {}).get("enabled", False):
        model = MultiTaskECGModel(base_model, config)
    else:
        model = base_model

    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config, checkpoint_path


def _batch_optional_tensor(batch: dict[str, Any], key: str, valid_mask: torch.Tensor, device: torch.device):
    value = batch.get(key)
    if value is None:
        return None
    return value.to(device)[valid_mask]


def _los_predictions_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        output = output["los"]
    elif isinstance(output, (tuple, list)):
        output = output[0]
    return output.squeeze(-1) if output.dim() > 1 else output


def _forward_torch_model(model: torch.nn.Module, signals: torch.Tensor, config: dict[str, Any], **features):
    aux = _forward_aux_kwargs(
        config,
        features.get("demographic_features"),
        features.get("icu_unit_features"),
        features.get("sofa_features"),
        features.get("icu_therapy_support_features"),
        features.get("ehr_window_features"),
    )
    forward_params = inspect.signature(model.forward).parameters
    aux = {key: value for key, value in aux.items() if key in forward_params}
    return model(signals, **aux)


def _create_test_loader(config: dict[str, Any]):
    icu_mapper = setup_icustays_mapper(config)
    _, _, test_loader = create_dataloaders(
        config=config,
        labels=None,
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,
    )
    if test_loader is None:
        raise RuntimeError("No test loader was created.")
    return test_loader


def _rows_to_requested_level(
    ecg_rows: list[dict[str, Any]],
    row: dict[str, Any],
    error_level: str,
) -> pd.DataFrame:
    if error_level == "ecg":
        return pd.DataFrame(ecg_rows)

    stay_predictions: dict[object, list[float]] = {}
    stay_labels: dict[object, float] = {}
    stay_n_ecgs: dict[object, int] = {}
    for item in ecg_rows:
        stay_id = item.get("stay_id")
        if stay_id is None:
            continue
        stay_predictions.setdefault(stay_id, []).append(float(item["prediction"]))
        stay_labels.setdefault(stay_id, float(item["target"]))
        stay_n_ecgs[stay_id] = stay_n_ecgs.get(stay_id, 0) + 1

    stay_rows = []
    for stay_id, preds in stay_predictions.items():
        pred_mean = float(np.mean(preds))
        target = stay_labels[stay_id]
        stay_rows.append(
            {
                "job_id": row["job_id"],
                "feature_stage": row["feature_stage"],
                "plot_label": row["plot_label"],
                "unit": "stay",
                "stay_id": stay_id,
                "prediction": pred_mean,
                "target": target,
                "abs_error": abs(pred_mean - target),
                "n_ecgs": stay_n_ecgs[stay_id],
            }
        )
    return pd.DataFrame(stay_rows)


def collect_torch_absolute_errors(
    row: dict[str, Any],
    *,
    error_level: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_true_los_days: Optional[float] = None,
) -> pd.DataFrame:
    model, config, checkpoint_path = _load_torch_model(row, device, batch_size, num_workers)
    print(f"  checkpoint: {checkpoint_path.name} ({config.get('model', {}).get('type', 'unknown')})")
    test_loader = _create_test_loader(config)

    ecg_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in test_loader:
            signals = batch["signal"].to(device)
            labels = batch["label"].to(device)
            meta = batch["meta"]
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue

            signals = signals[valid_mask]
            labels = labels[valid_mask]
            meta = [meta[i] for i in range(len(meta)) if bool(valid_mask[i].item())]

            output = _forward_torch_model(
                model,
                signals,
                config,
                demographic_features=_batch_optional_tensor(batch, "demographic_features", valid_mask, device),
                icu_unit_features=_batch_optional_tensor(batch, "icu_unit_features", valid_mask, device),
                sofa_features=_batch_optional_tensor(batch, "sofa_features", valid_mask, device),
                icu_therapy_support_features=_batch_optional_tensor(
                    batch, "icu_therapy_support_features", valid_mask, device
                ),
                ehr_window_features=_batch_optional_tensor(batch, "ehr_window_features", valid_mask, device),
            )
            predictions = _los_predictions_from_output(output).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            for pred, target, sample_meta in zip(predictions, labels_np, meta):
                pred_f = float(pred)
                target_f = float(target)
                ecg_rows.append(
                    {
                        "job_id": row["job_id"],
                        "feature_stage": row["feature_stage"],
                        "plot_label": row["plot_label"],
                        "unit": "ecg",
                        "stay_id": sample_meta.get("stay_id"),
                        "prediction": pred_f,
                        "target": target_f,
                        "abs_error": abs(pred_f - target_f),
                        "ecg_path": sample_meta.get("base_path") or sample_meta.get("path"),
                    }
                )

    if max_true_los_days is not None:
        ecg_rows = [r for r in ecg_rows if 0.0 <= float(r["target"]) <= float(max_true_los_days)]

    return _rows_to_requested_level(ecg_rows, row, error_level)


def _find_xgboost_model_path(row: dict[str, Any]) -> Path:
    explicit = _project_path(row.get("model_path") or row.get("checkpoint_path"))
    if explicit is not None:
        return explicit
    return PROJECT_ROOT / "outputs" / "checkpoints" / f"xgboost_los_best_{row['job_id']}.pkl"


def _xgboost_config_path(row: dict[str, Any]) -> Path:
    return _project_path(row.get("config_path", "configs/classical_ml/xgboost_handcrafted.yaml"))  # type: ignore[return-value]


def collect_xgboost_absolute_errors(
    row: dict[str, Any],
    *,
    error_level: str,
    batch_size: int,
    num_workers: int,
    max_true_los_days: Optional[float] = None,
) -> pd.DataFrame:
    model_path = _find_xgboost_model_path(row)
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")

    config = load_config(model_config_path=_xgboost_config_path(row))
    config = _apply_loader_overrides(config, batch_size, num_workers)
    feature_type = config.get("features", {}).get("feature_type", "handcrafted")
    if feature_type != "handcrafted":
        raise ValueError("XGBoost violin plot currently supports feature_type='handcrafted' only.")

    print(f"  model: {model_path.name} (xgboost)")
    test_loader = _create_test_loader(config)
    model = XGBoostECGModel(config, random_state=config.get("seed", 42)).load_model(str(model_path))
    fs = config.get("data", {}).get("sampling_rate", 500.0)
    use_demographics = config.get("features", {}).get("use_demographics", False)

    ecg_rows: list[dict[str, Any]] = []
    for batch in test_loader:
        signals = batch["signal"].numpy()
        labels = batch["label"].numpy()
        meta = batch["meta"]
        valid_mask = labels >= 0
        if not valid_mask.any():
            continue

        signals = signals[valid_mask]
        labels = labels[valid_mask]
        meta = [meta[i] for i in range(len(meta)) if bool(valid_mask[i])]

        x_ecg = np.vstack([extract_handcrafted_features(signal, fs=fs) for signal in signals])
        if use_demographics and batch.get("demographic_features") is not None:
            demo = batch["demographic_features"].numpy()[valid_mask]
            x_features = np.hstack([x_ecg, demo])
        else:
            x_features = x_ecg
        predictions = model.predict(x_features)

        for pred, target, sample_meta in zip(predictions, labels, meta):
            pred_f = float(pred)
            target_f = float(target)
            ecg_rows.append(
                {
                    "job_id": row["job_id"],
                    "feature_stage": row["feature_stage"],
                    "plot_label": row["plot_label"],
                    "unit": "ecg",
                    "stay_id": sample_meta.get("stay_id"),
                    "prediction": pred_f,
                    "target": target_f,
                    "abs_error": abs(pred_f - target_f),
                    "ecg_path": sample_meta.get("base_path") or sample_meta.get("path"),
                }
            )

    if max_true_los_days is not None:
        ecg_rows = [r for r in ecg_rows if 0.0 <= float(r["target"]) <= float(max_true_los_days)]

    return _rows_to_requested_level(ecg_rows, row, error_level)


def collect_absolute_errors(
    jobs: list[dict[str, Any]],
    *,
    error_level: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_true_los_days: Optional[float] = None,
) -> pd.DataFrame:
    frames = []
    for idx, row in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {row['plot_label']} (job {row['job_id']})")
        if _is_xgboost_job(row):
            frame = collect_xgboost_absolute_errors(
                row,
                error_level=error_level,
                batch_size=batch_size,
                num_workers=num_workers,
                max_true_los_days=max_true_los_days,
            )
        else:
            frame = collect_torch_absolute_errors(
                row,
                error_level=error_level,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                max_true_los_days=max_true_los_days,
            )
        frames.append(frame)
        mae = frame["abs_error"].mean()
        print(f"  n={len(frame):,}  mae={mae:.4f}\n")

    return pd.concat(frames, ignore_index=True)


def summarize_absolute_errors(error_df: pd.DataFrame) -> pd.DataFrame:
    return (
        error_df.groupby(["plot_label", "job_id", "unit"], sort=False)["abs_error"]
        .agg(n="count", mae="mean", median="median", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
        .reset_index()
    )


def plot_mae_violins(error_df: pd.DataFrame, jobs: list[dict[str, Any]], config: dict[str, Any]) -> Path:
    plot_cfg = config.get("plot", {})
    output_cfg = config.get("output", {})

    plot_labels = [row["plot_label"] for row in jobs]
    violin_data = [
        error_df.loc[
            (error_df["plot_label"] == row["plot_label"]) & (error_df["job_id"].astype(str) == row["job_id"]),
            "abs_error",
        ]
        .dropna()
        .to_numpy()
        for row in jobs
    ]
    means = np.array([values.mean() if len(values) else np.nan for values in violin_data], dtype=float)
    x = np.arange(len(plot_labels))

    fig_width = max(7.0, 1.35 * len(plot_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    parts = ax.violinplot(
        violin_data,
        positions=x,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    color = plot_cfg.get("color", "#8AD185")
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("0.35")
        body.set_alpha(0.88)

    if bool(plot_cfg.get("show_mean_marker", True)):
        ax.scatter(x, means, marker="o", s=44, color="0.08", zorder=3, edgecolors="white", linewidths=0.9)

    non_empty = [values for values in violin_data if len(values)]
    label_offset = 0.05
    if non_empty:
        flat_values = np.concatenate(non_empty)
        y_min = float(np.min(flat_values))
        y_max = float(np.max(flat_values))
        y_span = max(y_max - y_min, 1e-6)
        ax.set_ylim(max(0.0, y_min - 0.06 * y_span), y_max + 0.14 * y_span)
        label_offset = 0.028 * y_span

    decimals = int(plot_cfg.get("annotate_mae_decimals", 2))
    ann_fmt = str(plot_cfg.get("annotate_format", "value_only")).strip().lower()
    for xi, mean in zip(x, means):
        if np.isfinite(mean):
            if ann_fmt == "mae":
                ann = f"MAE {mean:.{decimals}f}"
            else:
                ann = f"{mean:.{decimals}f}"
            ax.text(
                xi,
                mean + label_offset,
                ann,
                ha="center",
                va="bottom",
                fontsize=float(plot_cfg.get("annotate_fontsize", 12)),
                color="black",
                fontweight="bold",
            )

    rotation = float(plot_cfg.get("rotation", 22))
    if all(str(label).strip().upper().startswith("D") for label in plot_labels):
        rotation = 0.0
    xtick_fs = float(plot_cfg.get("xtick_fontsize", 12))
    ytick_fs = float(plot_cfg.get("ytick_fontsize", 11.5))
    ylabel_fs = float(plot_cfg.get("ylabel_fontsize", 12.5))

    ax.set_xticks(x)
    xtick_ha = "center" if rotation == 0 else "right"
    ax.set_xticklabels(plot_labels, rotation=rotation, ha=xtick_ha, fontsize=xtick_fs)
    ylabel = plot_cfg.get("ylabel", "Absolute error (days)")
    ax.set_ylabel(ylabel, fontsize=ylabel_fs)
    ax.tick_params(axis="y", labelsize=ytick_fs)
    ax.tick_params(axis="x", length=0)

    title = plot_cfg.get("title")
    show_title = bool(plot_cfg.get("show_title", False))
    if show_title and title:
        ax.set_title(str(title), fontsize=float(plot_cfg.get("title_fontsize", 14)))

    ax.grid(axis="y", alpha=0.22, linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = _project_path(output_cfg.get("pdf_path", "outputs/results/models/violin/mae_violin.pdf"))
    assert out_path is not None
    if out_path.suffix.lower() != ".pdf":
        out_path = out_path.with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_optional_outputs(error_df: pd.DataFrame, summary: pd.DataFrame, config: dict[str, Any]) -> None:
    output_cfg = config.get("output", {})

    summary_path = _project_path(output_cfg.get("summary_csv_path"))
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"summary csv: {summary_path}")

    errors_path = _project_path(output_cfg.get("errors_csv_path"))
    if errors_path is not None:
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        error_df.to_csv(errors_path, index=False)
        print(f"errors csv: {errors_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MAE violin plots from saved model jobs.")
    parser.add_argument(
        "--config",
        default="configs/analysis/mae_violin_baseline.yaml",
        help="YAML config path.",
    )
    parser.add_argument("--device", default=None, help="Override config evaluation.device (cuda/cpu/auto).")
    parser.add_argument("--error-level", choices=["ecg", "stay"], default=None, help="Override config error_level.")
    parser.add_argument("--out", default=None, help="Override output.pdf_path.")
    parser.add_argument(
        "--plot-group-index",
        type=int,
        default=None,
        metavar="N",
        help=(
            "With a YAML that defines ``plot_groups``: run only the Nth group (0-based). "
            "Evaluates and plots just that figure's jobs—use one SLURM job per index to avoid long single runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _project_path(args.config)
    assert config_path is not None
    config = load_yaml(config_path)

    if args.out:
        config.setdefault("output", {})["pdf_path"] = args.out
    if args.error_level:
        config.setdefault("evaluation", {})["error_level"] = args.error_level
    if args.device:
        config.setdefault("evaluation", {})["device"] = args.device

    evaluation_cfg = config.get("evaluation", {})
    error_level = str(evaluation_cfg.get("error_level", "stay")).strip().lower()
    if error_level not in {"ecg", "stay"}:
        raise ValueError("evaluation.error_level must be 'ecg' or 'stay'.")

    batch_size = int(evaluation_cfg.get("batch_size", 64))
    num_workers = int(evaluation_cfg.get("num_workers", 0))
    device = _pick_device(evaluation_cfg.get("device", "auto"))

    os.environ.setdefault("ICUSTAYS_PATH", str(PROJECT_ROOT / "data" / "labeling" / "labels_csv" / "icustays.csv"))

    print(f"config: {config_path}")
    print(f"device: {device}")
    print(f"error_level: {error_level}")
    print(f"batch_size: {batch_size}, num_workers: {num_workers}")

    max_true_los_days = _evaluation_max_true_los_days(evaluation_cfg)
    if max_true_los_days is not None:
        print(f"max_true_los_days: {max_true_los_days} (only ECG/stay rows with true LOS <= this value)")
    else:
        print("max_true_los_days: off (full test set)")

    plot_groups_all = _plot_groups_from_config(config)
    default_max_plots = int(config.get("plot", {}).get("max_plots", 7))

    plot_groups = plot_groups_all
    selected_group_idx: Optional[int] = None
    if args.plot_group_index is not None:
        if not plot_groups_all:
            raise ValueError("--plot-group-index only applies when the YAML defines plot_groups.")
        gi = int(args.plot_group_index)
        if gi < 0 or gi >= len(plot_groups_all):
            raise ValueError(f"--plot-group-index must be in [0, {len(plot_groups_all) - 1}], got {gi}.")
        plot_groups = [plot_groups_all[gi]]
        selected_group_idx = gi
        print(f"plot_group_index: only group {gi} of {len(plot_groups_all)} in config (single figure)\n")

    if plot_groups:
        collection_jobs = _collection_jobs_for_plot_groups(plot_groups, default_max_plots)
        if not collection_jobs:
            raise ValueError("plot_groups produced no active jobs (check jobs[].enabled and job_id).")
        print(f"plot_groups: {len(plot_groups)} figure(s), {len(collection_jobs)} unique job(s) for inference\n")
        error_df = collect_absolute_errors(
            collection_jobs,
            error_level=error_level,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_true_los_days=max_true_los_days,
        )
        print(summarize_absolute_errors(error_df).to_string(index=False))

        err_combined = _project_path(config.get("output", {}).get("errors_csv_path"))
        if err_combined is not None:
            if selected_group_idx is not None:
                err_combined = err_combined.with_name(
                    f"{err_combined.stem}_group{selected_group_idx}{err_combined.suffix}"
                )
            err_combined.parent.mkdir(parents=True, exist_ok=True)
            error_df.to_csv(err_combined, index=False)
            print(f"errors csv (combined): {err_combined}")

        for idx, group in enumerate(plot_groups, start=1):
            pdf_g = group.get("pdf_path")
            if not pdf_g:
                raise ValueError(f"plot_groups entry {idx} is missing pdf_path")

            max_p = int(group.get("max_plots", default_max_plots))
            jobs_g = _active_jobs_from_list(group.get("jobs", []), max_plots=max_p)
            sub_df = _subset_errors_for_jobs(error_df, jobs_g)
            summary_g = summarize_absolute_errors(sub_df)
            print(f"\n--- plot group {idx}/{len(plot_groups)} ({len(jobs_g)} violins) ---")
            print(summary_g.to_string(index=False))

            merged_out = {**config.get("output", {})}
            merged_out["pdf_path"] = pdf_g
            merged_out["summary_csv_path"] = group.get("summary_csv_path")
            merged_out["errors_csv_path"] = None
            merged_cfg = {**config, "output": merged_out}
            write_optional_outputs(sub_df, summary_g, merged_cfg)
            out_path = plot_mae_violins(sub_df, jobs_g, merged_cfg)
            print(f"pdf: {out_path}")
        return

    jobs = _active_jobs(config)
    print(f"active plots: {len(jobs)}\n")

    error_df = collect_absolute_errors(
        jobs,
        error_level=error_level,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_true_los_days=max_true_los_days,
    )
    summary = summarize_absolute_errors(error_df)
    print(summary.to_string(index=False))

    write_optional_outputs(error_df, summary, config)
    out_path = plot_mae_violins(error_df, jobs, config)
    print(f"pdf: {out_path}")


if __name__ == "__main__":
    main()
