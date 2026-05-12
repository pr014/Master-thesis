#!/usr/bin/env python3
"""Batch evaluation on the **test split** checkpoint models.

Includes:

- Stay-level LOS + mortality (``evaluate_with_detailed_metrics``), same as training:
  threshold from validation max-F1 on stay-level probs unless the checkpoint config sets
  ``evaluation.mortality_test_threshold_mode`` to ``fixed_0_5`` (then 0.5).
- **ECG-window cohort** with **true stay LOS ≤ 10 days**: per-window LOS MAE and,
  for multi-task models, mortality ROC-AUC (one row per ECG window, same forward
  as ``bland_altman.py`` / ``_forward_aux_kwargs``).

Job IDs must be set explicitly: ``--preset`` (recommended), ``--job`` (repeatable), or
``--jobs-file``. There is **no** implicit default job list.

Preset groups (``--preset``) write to dedicated folders + CSVs::

  h4_baseline → ``outputs/results/test_10days_split/h4_baseline/h4_baseline_10days_split.csv``
  feature_development → ``.../feature_development/feature_development_10days_split.csv``

Writes JSON next to each group's CSV. Without ``--preset``, set ``--out-dir`` / ``--csv`` as needed.

Example
-------
  cd <repo-root>
  python scripts/test_10days/test_10days.py --preset both
  python scripts/test_10days/test_10days.py --device mps --job 3119696
  python scripts/test_10days/test_10days.py --out-dir outputs/results/test_10days_split \\
      --csv outputs/results/test_10days_split/meine_uebersicht.csv \\
      --job 3119696 --job 3217131

  # Job IDs from file (one integer per line; commas/spaces tolerated; # comments ok)
  python scripts/test_10days/test_10days.py --jobs-file configs/my_job_ids.txt

  # After an interrupt, same --job order plus --resume (uses subgroup_leq10_batch_progress.json).
  python scripts/test_10days/test_10days.py --resume \\
      --csv outputs/results/test_10days_split/baseline_10days_split.csv \\
      --job 3119696 --job 3217131

  # Re-run one job in a preset group: keep CSV rows with ``mae`` (skip full eval), evaluate only
  # job IDs that are missing or lack ``mae``. After updating GROUP_PRESETS (e.g. new Hybrid ID):
  python scripts/test_10days/test_10days.py --preset h4_baseline --reuse-partial-csv

  # Same without touching presets: full ordered ``--job`` list + existing CSV path/out-dir:
  python scripts/test_10days/test_10days.py --reuse-partial-csv \\
      --out-dir outputs/results/test_10days_split/h4_baseline \\
      --csv outputs/results/test_10days_split/h4_baseline/h4_baseline_10days_split.csv \\
      --job 4583928 --job 4584067 --job 4646227 --job 4584619 --job 4584798

  # Force validation max-F1 mortality threshold even if checkpoint YAML uses fixed_0_5:
  python scripts/test_10days/test_10days.py --preset both --val-max-f1-threshold
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import inspect
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Repo root: scripts/test_10days/test_10days.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.training.losses import get_loss, get_multi_task_loss
from src.training.train_loop import (
    _forward_aux_kwargs,
    evaluate_with_detailed_metrics,
    find_best_mortality_threshold_f1,
)

OUTPUT_DIR = REPO_ROOT / "outputs" / "results" / "test_10days_split"
DEFAULT_SUMMARY_CSV = "baseline_10days_split.csv"

# Named batches: fixed Slurm job IDs → separate out-dir + summary CSV (use ``--preset``).
GROUP_PRESETS: Dict[str, Dict[str, Any]] = {
    "h4_baseline": {
        # Hybrid CNN-LSTM slot: 4646227 (replaces earlier 4584305 checkpoint for A5 table).
        "job_ids": (4583928, 4584067, 4646227, 4584619, 4584798),
        "out_dir": REPO_ROOT / "outputs" / "results" / "test_10days_split" / "h4_baseline",
        "csv_name": "h4_baseline_10days_split.csv",
    },
    "feature_development": {
        "job_ids": (
            4587536,
            4587538,
            4587537,
            4587539,
            4587540,
            4587541,
        ),
        "out_dir": REPO_ROOT
        / "outputs"
        / "results"
        / "test_10days_split"
        / "feature_development",
        "csv_name": "feature_development_10days_split.csv",
    },
}


def _parse_jobs_file(path: Path) -> List[int]:
    """Parse job IDs: one per line; optional commas/spaces; ``#`` starts a comment."""
    if not path.is_file():
        raise FileNotFoundError(f"--jobs-file not found: {path}")
    out: List[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for tok in line.replace(",", " ").split():
            tok = tok.strip()
            if not tok:
                continue
            out.append(int(tok))
    return out


def _resolve_job_ids(jobs_file: Optional[str], jobs_cli: Optional[List[int]]) -> Tuple[int, ...]:
    parts: List[int] = []
    if jobs_file:
        parts.extend(_parse_jobs_file(Path(jobs_file).expanduser().resolve()))
    if jobs_cli:
        parts.extend(jobs_cli)
    seen: set[int] = set()
    uniq: List[int] = []
    for j in parts:
        if j not in seen:
            seen.add(j)
            uniq.append(j)
    return tuple(uniq)


def _load_evaluate_subgroup_module():
    """Load evaluate_subgroup as a module (same file as CLI; shared helpers)."""
    path = REPO_ROOT / "scripts" / "analysis" / "evaluate_subgroup.py"
    spec = importlib.util.spec_from_file_location("evaluate_subgroup", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _metrics_from_detailed_overall(m: Dict[str, Any]) -> Dict[str, Any]:
    """Shape compatible with previous ``overall`` / ``_regression_metrics`` JSON."""
    return {
        "mae": float(m["los_mae"]),
        "rmse": float(m["los_rmse"]),
        "r2": float(m["los_r2"]),
        "median_ae": float(m["los_median_ae"]),
        "p25_error": float(m["los_p25_error"]),
        "p50_error": float(m["los_p50_error"]),
        "p75_error": float(m["los_p75_error"]),
        "p90_error": float(m["los_p90_error"]),
        "n": int(m["num_stays"]),
    }


def _los_tensor_from_output(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        t = out.get("los", out)
        return t
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _mortality_tensor_from_output(out: Any) -> Optional[torch.Tensor]:
    if isinstance(out, dict):
        return out.get("mortality")
    if isinstance(out, (tuple, list)) and len(out) > 1:
        return out[1]
    return None


def _ecg_window_cohort_true_los_leq10(
    model: torch.nn.Module,
    test_loader: Any,
    device: torch.device,
    config: dict,
    is_multi_task: bool,
    max_true_los_days: float = 10.0,
) -> Dict[str, Any]:
    """
    One row per ECG window in the test loader; filter to windows whose **stay**
    has true LOS in [0, max_true_los_days].

    Forward path matches ``scripts/analysis/results/bland_altman.py`` (all aux tensors
    from config via ``_forward_aux_kwargs``).
    """
    y_true: List[float] = []
    y_pred: List[float] = []
    mort_prob: List[float] = []
    mort_label: List[float] = []

    model.eval()
    fwd_params = inspect.signature(model.forward).parameters
    with torch.no_grad():
        for batch in test_loader:
            signals = batch["signal"].to(device)
            labels = batch["label"].cpu().numpy()
            meta = batch["meta"]
            valid = labels >= 0
            if not np.any(valid):
                continue
            signals = signals[valid]
            labels_t = labels[valid]
            meta = [meta[i] for i in range(len(meta)) if valid[i]]

            mortality_labels_batch: Optional[torch.Tensor] = None
            if is_multi_task and batch.get("mortality_label") is not None:
                mortality_labels_batch = batch["mortality_label"].to(device)[valid]

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
            if (
                batch.get("icu_therapy_support_features") is not None
            ):
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

            los_t = _los_tensor_from_output(out)
            los_np = (
                (los_t.squeeze(-1) if los_t.dim() > 1 else los_t).cpu().numpy()
            )
            mort_t = _mortality_tensor_from_output(out) if is_multi_task else None
            mort_np = None
            if mort_t is not None:
                m = mort_t.squeeze(-1) if mort_t.dim() > 1 else mort_t
                mort_np = m.cpu().numpy()

            for i in range(len(labels_t)):
                yt = float(labels_t[i])
                y_true.append(yt)
                y_pred.append(float(los_np[i]))
                if mort_np is not None and mortality_labels_batch is not None:
                    mort_prob.append(float(mort_np[i]))
                    mort_label.append(float(mortality_labels_batch[i].item()))
                else:
                    mort_prob.append(float("nan"))
                    mort_label.append(float("nan"))

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    cohort = (yt >= 0.0) & (yt <= float(max_true_los_days))
    n_ecg = int(cohort.sum())
    out: Dict[str, Any] = {
        "max_true_los_days": float(max_true_los_days),
        "n_ecg_windows_test_all": int(len(yt)),
        "n_ecg_windows_cohort": n_ecg,
        "los_mae": None,
        "mortality_auc": None,
    }
    if n_ecg == 0:
        return out

    out["los_mae"] = float(np.mean(np.abs(yp[cohort] - yt[cohort])))

    if is_multi_task:
        mp = np.asarray(mort_prob, dtype=np.float64)
        ml = np.asarray(mort_label, dtype=np.float64)
        sub = cohort & np.isfinite(ml) & (ml >= 0.0)
        n_m = int(sub.sum())
        out["n_ecg_windows_cohort_mortality_valid"] = n_m
        if n_m > 0:
            try:
                from sklearn.metrics import roc_auc_score

                labs = ml[sub]
                probs = mp[sub]
                if len(np.unique(labs)) > 1:
                    out["mortality_auc"] = float(roc_auc_score(labs, probs))
            except ValueError:
                out["mortality_auc"] = None

    return out


def _metrics_from_detailed_leq10(m: Dict[str, Any]) -> Dict[str, Any]:
    n = int(m.get("los_n_leq10", 0))
    if n <= 0:
        return {}
    return {
        "mae": float(m["los_mae_leq10"]),
        "rmse": float(m["los_rmse_leq10"]),
        "r2": float(m["los_r2_leq10"]),
        "median_ae": float(m["los_median_ae_leq10"]),
        "p25_error": float(m["los_p25_error_leq10"]),
        "p50_error": float(m["los_p50_error_leq10"]),
        "p75_error": float(m["los_p75_error_leq10"]),
        "p90_error": float(m["los_p90_error_leq10"]),
        "n": n,
    }


def _pick_device(explicit: Optional[str]) -> torch.device:
    """Prefer CUDA, then Apple Metal (MPS), then CPU."""
    if explicit:
        return torch.device(explicit.strip())
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate_one_job(
    es: Any,
    job_id: int,
    device: torch.device,
    *,
    force_val_max_f1_threshold: bool = False,
) -> Dict[str, Any]:
    ckpt_path = es._find_checkpoint(job_id)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config: dict = ckpt.get("config", {})
    ckpt_job_id = ckpt.get("job_id", "unknown")
    epoch = ckpt.get("epoch", "?")
    model_type = config.get("model", {}).get("type", "unknown")

    base_model = es._build_model(config)
    multi_task_config = config.get("multi_task", {})
    is_multi_task = multi_task_config.get("enabled", False)

    if is_multi_task:
        model = es.MultiTaskECGModel(base_model, config)
    else:
        model = base_model

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    icu_mapper = es.setup_icustays_mapper(config)
    _, val_loader, test_loader = es.create_dataloaders(
        config=config,
        labels=None,
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,
    )

    # Mortality threshold: mirror training_utils.evaluate_and_print_results (stay-level max-F1 on val,
    # unless evaluation.mortality_test_threshold_mode is fixed_0_5 / h1_h4 / legacy).
    # With force_val_max_f1_threshold=True (--val-max-f1-threshold), always tune on val like f1_optimal
    # without editing checkpoint YAML/config.
    criterion = get_multi_task_loss(config) if is_multi_task else get_loss(config)
    eval_cfg = config.get("evaluation") or {}
    _mt_mode = str(eval_cfg.get("mortality_test_threshold_mode", "f1_optimal")).strip().lower()
    _fixed_modes = frozenset({"fixed_0_5", "h1_h4", "legacy"})
    use_val_max_f1_thr = is_multi_task and (
        force_val_max_f1_threshold or (_mt_mode not in _fixed_modes)
    )

    mortality_thr = 0.5
    if use_val_max_f1_thr and val_loader is not None:
        val_out = evaluate_with_detailed_metrics(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            config=config,
            mortality_threshold=0.5,
            return_mortality_arrays=True,
        )
        probs = val_out.get("mortality_probs_stay")
        labs = val_out.get("mortality_labels_stay")
        if probs is not None and labs is not None and len(probs) > 0:
            mortality_thr = float(find_best_mortality_threshold_f1(probs, labs))

    # Pass 1 (authoritative for LOS overall / LOS≤10 + mortality): test loader, same metrics as training test eval
    detailed = evaluate_with_detailed_metrics(
        model=model,
        val_loader=test_loader,
        criterion=criterion,
        device=device,
        config=config,
        mortality_threshold=mortality_thr,
        return_mortality_arrays=False,
    )

    overall = _metrics_from_detailed_overall(detailed)
    subgroup_leq10 = _metrics_from_detailed_leq10(detailed)

    # Pass 2: optional LOS bins (evaluate_subgroup forward; may differ if aux-only-in-detailed)
    predictions, labels = es._run_inference(model, test_loader, device, is_multi_task)
    mask_3_10 = (labels > 3.0) & (labels <= 10.0)
    mask_leq3 = labels <= 3.0
    mask_gt10 = labels > 10.0

    out: Dict[str, Any] = {
        "job_id": ckpt_job_id,
        "job_id_requested": job_id,
        "model_type": model_type,
        "epoch": epoch,
        "checkpoint": str(ckpt_path),
        "overall": overall,
        "subgroup_leq10": subgroup_leq10,
        "subgroup_leq3": es._regression_metrics(
            predictions[mask_leq3], labels[mask_leq3]
        )
        if mask_leq3.any()
        else {},
        "subgroup_3_10": es._regression_metrics(
            predictions[mask_3_10], labels[mask_3_10]
        )
        if mask_3_10.any()
        else {},
        "subgroup_gt10": es._regression_metrics(
            predictions[mask_gt10], labels[mask_gt10]
        )
        if mask_gt10.any()
        else {},
        "mortality": None,
        "ecg_windows_true_stay_los_leq10d": _ecg_window_cohort_true_los_leq10(
            model, test_loader, device, config, is_multi_task
        ),
        "los_cross_check": {
            "note": (
                "overall / subgroup_leq10 LOS from evaluate_with_detailed_metrics; "
                "subgroup_leq3 / 3_10 / gt10 from evaluate_subgroup._run_inference "
                "(narrower forward kwargs — compare if SOFA/EHR aux is enabled)."
            ),
            "overall_from_run_inference": es._regression_metrics(predictions, labels),
            "subgroup_leq10_from_run_inference": es._regression_metrics(
                predictions[labels <= 10.0], labels[labels <= 10.0]
            )
            if (labels <= 10.0).any()
            else {},
        },
    }

    if is_multi_task:
        out["mortality"] = {
            "threshold_used": float(detailed.get("mortality_threshold_used", mortality_thr)),
            "mortality_test_threshold_mode_config": _mt_mode,
            "forced_val_max_f1_by_cli": bool(force_val_max_f1_threshold),
            "threshold_selection": (
                "val_max_f1_then_apply_test"
                if use_val_max_f1_thr
                else "fixed_0_5_from_config"
            ),
            "auc_overall": float(detailed.get("mortality_auc", 0.0)),
            "accuracy": float(detailed.get("mortality_accuracy", 0.0)),
            "precision": float(detailed.get("mortality_precision", 0.0)),
            "recall": float(detailed.get("mortality_recall", 0.0)),
            "f1": float(detailed.get("mortality_f1", 0.0)),
            "n_stays_leq10_for_auc": int(detailed.get("mortality_n_stays_leq10", 0)),
            "auc_true_los_leq10_days": detailed.get("mortality_auc_leq10"),
        }
    else:
        out["mortality"] = None

    return out


def _csv_rows_from_job_entries(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for entry in rows:
        jid = int(entry["job_id_requested"])
        auc_s = ""
        mae_s = ""
        r2_s = ""
        f1_s = ""
        precision_s = ""
        status = "failed"
        err_s = ""
        if entry.get("ok") and isinstance(entry.get("result"), dict):
            res = entry["result"]
            if res.get("model_type") == "seeded_from_ecg_leq10_job_summary_csv":
                status = "seeded_csv"
            else:
                status = "ok"
            ecg = res.get("ecg_windows_true_stay_los_leq10d") or {}
            mae_v = ecg.get("los_mae")
            if mae_v is not None:
                mae_s = f"{float(mae_v):.6f}"
            auc_v = ecg.get("mortality_auc")
            if auc_v is not None:
                auc_s = f"{float(auc_v):.6f}"

            overall = res.get("overall") or {}
            if isinstance(overall, dict) and overall.get("r2") is not None:
                r2_s = f"{float(overall['r2']):.6f}"

            mort = res.get("mortality")
            if isinstance(mort, dict):
                if mort.get("f1") is not None:
                    f1_s = f"{float(mort['f1']):.6f}"
                if mort.get("precision") is not None:
                    precision_s = f"{float(mort['precision']):.6f}"
        elif entry.get("error"):
            err = str(entry["error"]).replace("\n", " ").strip()
            if len(err) > 200:
                err = err[:197] + "..."
            err_s = err
        out.append(
            {
                "job_id": str(jid),
                "status": status,
                "error": err_s,
                "auc": auc_s,
                "mae": mae_s,
                "r2": r2_s,
                "f1": f1_s,
                "precision": precision_s,
            }
        )
    return out


def _write_summary_csv(path: Path, csv_rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["job_id", "status", "error", "auc", "mae", "r2", "f1", "precision"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)


def _entries_from_existing_summary_csv(
    csv_path: Path, job_ids: Tuple[int, ...]
) -> Dict[int, Dict[str, Any]]:
    """
    Rows with non-empty ``mae`` become ok=true entries with minimal ``result``
    so the batch can skip re-evaluation (continue after interrupt).
    """
    out: Dict[int, Dict[str, Any]] = {}
    if not csv_path.is_file():
        return out
    want = {int(j) for j in job_ids}
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid_raw = row.get("job_id") or row.get("slurm_job_id") or ""
            jid = int(str(jid_raw).strip())
            if jid not in want:
                continue
            mae_s = (row.get("mae") or "").strip()
            if not mae_s:
                continue
            mae_v = float(mae_s)
            auc_s = (row.get("auc") or "").strip()
            auc_v: Optional[float] = float(auc_s) if auc_s else None

            r2_seed: Dict[str, float] = {}
            r2_csv = (row.get("r2") or "").strip()
            if r2_csv:
                r2_seed["r2"] = float(r2_csv)

            mort_seed: Optional[Dict[str, float]] = None
            f1_csv = (row.get("f1") or "").strip()
            prec_csv = (row.get("precision") or "").strip()
            if f1_csv or prec_csv:
                mort_seed = {}
                if f1_csv:
                    mort_seed["f1"] = float(f1_csv)
                if prec_csv:
                    mort_seed["precision"] = float(prec_csv)

            out[jid] = {
                "job_id_requested": jid,
                "ok": True,
                "result": {
                    "job_id": jid,
                    "job_id_requested": jid,
                    "model_type": "seeded_from_ecg_leq10_job_summary_csv",
                    "epoch": None,
                    "checkpoint": "",
                    "overall": r2_seed,
                    "subgroup_leq10": {},
                    "subgroup_leq3": {},
                    "subgroup_3_10": {},
                    "subgroup_gt10": {},
                    "mortality": mort_seed,
                    "ecg_windows_true_stay_los_leq10d": {
                        "max_true_los_days": 10.0,
                        "n_ecg_windows_test_all": 0,
                        "n_ecg_windows_cohort": 0,
                        "los_mae": mae_v,
                        "mortality_auc": auc_v,
                    },
                    "los_cross_check": {},
                    "_seed_note": (
                        "los_mae, mortality_auc, optional r2/f1/precision taken from existing "
                        "summary CSV; full metrics only for jobs evaluated in this run."
                    ),
                },
            }
    return out


def _run_batch(
    *,
    label: str,
    job_ids: Tuple[int, ...],
    out_dir: Path,
    csv_path_resolved: Path,
    device: torch.device,
    es: Any,
    resume: bool,
    reuse_partial_csv: bool,
    force_val_max_f1_threshold: bool,
) -> None:
    """Evaluate ``job_ids``; write progress JSON, timestamped + latest JSON, and summary CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    progress_path = out_dir / "subgroup_leq10_batch_progress.json"

    def _rows_in_job_order(by_jid: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [by_jid[j] for j in job_ids if j in by_jid]

    def _write_progress(by_jid: Dict[int, Dict[str, Any]]) -> None:
        rows_ord = _rows_in_job_order(by_jid)
        progress_path.write_text(
            json.dumps(
                {
                    "definition": (
                        "Partial batch progress; same job entries as final JSON ``jobs``."
                    ),
                    "preset": label,
                    "device": str(device),
                    "job_ids": [int(j) for j in job_ids],
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "jobs": rows_ord,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    prefix = f"[{label}] " if label != "manual" else ""

    by_jid: Dict[int, Dict[str, Any]] = {}
    if resume and progress_path.exists():
        try:
            prog = json.loads(progress_path.read_text(encoding="utf-8"))
            for r in prog.get("jobs", []):
                by_jid[int(r["job_id_requested"])] = r
            print(
                f"{prefix}Resume: loaded {len(by_jid)} job(s) from {progress_path} "
                f"({sum(1 for x in by_jid.values() if x.get('ok'))} ok)"
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
            print(f"{prefix}[warn] Could not load progress file: {e}; starting this run fresh.")
            by_jid = {}

    if reuse_partial_csv and csv_path_resolved.is_file():
        seeded = _entries_from_existing_summary_csv(csv_path_resolved, job_ids)
        for jid, entry in seeded.items():
            if by_jid.get(jid, {}).get("ok"):
                continue
            by_jid[jid] = entry
            print(f"{prefix}Seed job {jid} from existing CSV (skip full eval).")

    if by_jid:
        _write_progress(by_jid)
        rows_sync = _rows_in_job_order(by_jid)
        cr0 = _csv_rows_from_job_entries(rows_sync)
        _write_summary_csv(csv_path_resolved, cr0)

    for jid in job_ids:
        jid = int(jid)
        prev = by_jid.get(jid)
        if prev is not None and prev.get("ok"):
            print(f"{prefix}Skip job {jid} (already completed)")
            continue

        entry: Dict[str, Any] = {"job_id_requested": jid, "ok": False}
        try:
            entry["result"] = _evaluate_one_job(
                es,
                jid,
                device,
                force_val_max_f1_threshold=force_val_max_f1_threshold,
            )
            entry["ok"] = True
        except FileNotFoundError as e:
            entry["error"] = str(e)
        except ValueError as e:
            entry["error"] = str(e)
        except Exception as e:  # noqa: BLE001 — batch script: record and continue
            entry["error"] = f"{type(e).__name__}: {e}"
        by_jid[jid] = entry
        rows = _rows_in_job_order(by_jid)

        _write_progress(by_jid)

        cr = _csv_rows_from_job_entries(rows)
        _write_summary_csv(csv_path_resolved, cr)
        print(f"{prefix}Updated CSV ({len(cr)} rows): {csv_path_resolved}")

    rows = _rows_in_job_order(by_jid)

    _thr_def = (
        "Mortality threshold: same rule as training (val max-F1 on stay-level unless config "
        "evaluation.mortality_test_threshold_mode is fixed_0_5 / h1_h4 / legacy)."
    )
    if force_val_max_f1_threshold:
        _thr_def = (
            "Mortality threshold: validation max-F1 on stay-level was forced via "
            "--val-max-f1-threshold (ignores fixed_0_5 / h1_h4 / legacy in checkpoint config); "
            "differs from training_utils final test eval if training used fixed threshold."
        )
    payload = {
        "definition": (
            "Test split only (held-out subjects/stays). "
            "ecg_windows_true_stay_los_leq10d: each row is one ECG window; cohort = windows "
            "where the stay's true LOS is in [0, 10] days; los_mae = mean |pred−true| over "
            "those windows; mortality_auc = sklearn ROC-AUC on the same windows (multi-task "
            "only, valid mortality labels). "
            "subgroup_leq10 / mortality.* stay-level metrics from evaluate_with_detailed_metrics. "
            + _thr_def
        ),
        "val_max_f1_threshold_forced": bool(force_val_max_f1_threshold),
        "preset": label,
        "device": str(device),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": rows,
    }

    out_path = out_dir / f"subgroup_leq10_batch_{ts}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = out_dir / "subgroup_leq10_batch_latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{prefix}Wrote: {out_path}")
    print(f"{prefix}Wrote: {latest}")

    cr = _csv_rows_from_job_entries(rows)
    _write_summary_csv(csv_path_resolved, cr)
    print(f"{prefix}Final CSV ({len(cr)} rows): {csv_path_resolved}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch test metrics: stay-level (train_loop) + ECG windows with true stay LOS≤10d "
            "(MAE + mortality AUC when multi-task)."
        )
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=("h4_baseline", "feature_development", "both"),
        default=None,
        metavar="NAME",
        help=(
            "Fixed SLURM job-ID groups with separate output dirs + CSVs: "
            "h4_baseline → outputs/results/test_10days_split/h4_baseline/, "
            "feature_development → .../feature_development/. "
            "'both' runs h4_baseline then feature_development. "
            "Ignores --job, --jobs-file, --out-dir, --csv when set."
        ),
    )
    parser.add_argument(
        "--job",
        type=int,
        action="append",
        dest="jobs",
        help=(
            "SLURM job ID (repeatable). Combined with --jobs-file (file first, then CLI). "
            "Use with --out-dir/--csv, or use --preset instead. "
            "If omitted together with --jobs-file and without --preset, the script exits with an error."
        ),
    )
    parser.add_argument(
        "--jobs-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Text file of job IDs (one per line; optional commas/spaces; # comments). "
            "IDs are de-duplicated in order; can be combined with --job."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "cpu | cuda | mps (Apple Silicon GPU). "
            "Default: cuda if available, else mps if available, else cpu."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Summary CSV: job_id, status, error, auc, mae, r2, f1, precision; refreshed after "
            "each job. UTF-8 with BOM for Excel. "
            "Default: <out-dir>/baseline_10days_split.csv. "
            "auc/mae from ecg_windows_true_stay_los_leq10d (ECG windows, true stay LOS≤10d). "
            "r2 from stay-level test LOS overall (evaluate_with_detailed_metrics). "
            "f1/precision from stay-level mortality (multi-task; threshold like training: "
            "val max-F1 unless config fixed_0_5)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If <out-dir>/subgroup_leq10_batch_progress.json exists (each --preset group has its "
            "own out-dir), load it and skip jobs with ok=true. Failed or missing jobs are "
            "re-evaluated. Use the same job order as the original run."
        ),
    )
    parser.add_argument(
        "--reuse-partial-csv",
        action="store_true",
        help=(
            "If the summary CSV already exists, load rows where ``mae`` is set and treat those "
            "job IDs as completed (skip full eval). Jobs in the current batch list without such "
            "a row (or with empty ``mae``) are evaluated normally. Works with --preset (each "
            "group's csv_name under that preset's out_dir). Progress JSON from --resume wins "
            "over CSV seeds."
        ),
    )
    parser.add_argument(
        "--val-max-f1-threshold",
        action="store_true",
        help=(
            "Multi-task only: pick mortality threshold by validation max F1 (stay-level) and "
            "apply on test, even if the checkpoint config sets fixed_0_5 / h1_h4 / legacy. "
            "Does not change model YAMLs; output JSON marks forced_val_max_f1_by_cli per job. "
            "Note: training final eval still follows checkpoint config unless you change that too."
        ),
    )
    args = parser.parse_args(argv)

    device = _pick_device(args.device)

    es = _load_evaluate_subgroup_module()
    # Loaded via importlib from scripts/analysis/evaluate_subgroup.py — older copies may omit HuBERT.
    _reg = getattr(es, "_MODEL_REGISTRY", None)
    if isinstance(_reg, dict) and "HuBERT_ECG" not in _reg:
        from src.models import HuBERT_ECG as _HuBERT_ECG

        _reg["HuBERT_ECG"] = _HuBERT_ECG

    batches: List[Tuple[str, Tuple[int, ...], Path, Path]]

    if args.preset:
        if args.jobs or args.jobs_file:
            print(
                "[warn] --preset ignores --job / --jobs-file (using GROUP_PRESETS job lists)."
            )
        if args.out_dir or args.csv:
            print("[warn] --preset ignores --out-dir / --csv (using preset paths).")

        names = (
            ("h4_baseline", "feature_development")
            if args.preset == "both"
            else (args.preset,)
        )
        batches = []
        for name in names:
            spec = GROUP_PRESETS[name]
            od = Path(spec["out_dir"])
            if not od.is_absolute():
                od = REPO_ROOT / od
            csv_p = (od / spec["csv_name"]).resolve()
            batches.append((name, tuple(spec["job_ids"]), od.resolve(), csv_p))
    else:
        job_ids = _resolve_job_ids(args.jobs_file, args.jobs)
        if not job_ids:
            parser.error(
                "No job IDs: pass --preset (h4_baseline | feature_development | both), "
                "or --job / --jobs-file."
            )
        out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR
        out_dir = out_dir.expanduser()
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
        default_csv_path = out_dir / DEFAULT_SUMMARY_CSV
        csv_path_resolved = (
            Path(args.csv).expanduser().resolve()
            if args.csv
            else default_csv_path.resolve()
        )
        batches = [
            (
                "manual",
                job_ids,
                out_dir.resolve(),
                csv_path_resolved.resolve(),
            )
        ]

    for label, job_ids, out_dir, csv_path_resolved in batches:
        print(f"\n=== Batch: {label} ({len(job_ids)} jobs) → {out_dir} ===\n")
        if args.val_max_f1_threshold:
            print(
                "[--val-max-f1-threshold] Using validation max-F1 mortality threshold "
                "(override fixed_0_5 / h1_h4 / legacy in checkpoint config).\n"
            )
        _run_batch(
            label=label,
            job_ids=job_ids,
            out_dir=out_dir,
            csv_path_resolved=csv_path_resolved,
            device=device,
            es=es,
            resume=args.resume,
            reuse_partial_csv=args.reuse_partial_csv,
            force_val_max_f1_threshold=bool(args.val_max_f1_threshold),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
