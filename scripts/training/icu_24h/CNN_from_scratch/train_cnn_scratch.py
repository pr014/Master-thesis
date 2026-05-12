"""Single training entry for CNN from scratch (ICU 24h).

Everything is driven by YAML: ``data.data_dir`` (e.g. P1/P2), ``data.augmentation.*`` toggles,
``data.los_task`` / ``training.loss``, optional ``multi_task``. Optional merges:

* ``--base`` / ``config_merge.base_config_path`` — merged before the primary config (e.g. weighted LOS).
* ``--experiment`` / ``config_merge.experiment_config_path`` — merged after the primary config.

CLI flags override the corresponding ``config_merge`` entries in the primary YAML.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# …/scripts/training/icu_24h/CNN_from_scratch/train_cnn_scratch.py → repo root is parents[4]
_REPO_ROOT = Path(__file__).resolve().parents[4]
# Repo root on path so ``from src....`` resolves to <root>/src/...
sys.path.insert(0, str(_REPO_ROOT))

from src.data.ecg import create_dataloaders
from src.data.labeling import (
    ICUStayMapper,
    get_num_classes_from_config,
    load_icustays,
    load_mortality_mapping,
)
from src.models import CNNScratch, MultiTaskECGModel
from src.training import Trainer, evaluate_and_print_results
from src.training.losses import get_loss, get_multi_task_loss
from src.utils.config_loader import load_config, load_yaml

DEFAULT_MODEL_CONFIG = Path("configs/model/CNN/cnn_scratch.yaml")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNNScratch on ICU 24h ECG (config-driven).")
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Primary model YAML (merged last; overrides --base).",
    )
    p.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Optional base YAML merged before --config (e.g. weighted LOS / binning).",
    )
    p.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help="Optional experiment YAML merged after --config.",
    )
    return p.parse_args()


def _resolve_merge_paths(
    args: argparse.Namespace, model_config_path: Path
) -> Tuple[Optional[Path], Optional[Path]]:
    """CLI overrides ``config_merge`` in the primary YAML."""
    raw = load_yaml(model_config_path)
    cm = raw.get("config_merge") or {}
    base: Optional[Path] = args.base
    if base is None and cm.get("base_config_path"):
        base = Path(cm["base_config_path"])
    experiment: Optional[Path] = args.experiment
    if experiment is None and cm.get("experiment_config_path"):
        experiment = Path(cm["experiment_config_path"])
    return base, experiment


def _normalize_task_type(config: Dict[str, Any]) -> str:
    d = config.get("data", {})
    task = (d.get("task_type") or d.get("los_task") or "regression").strip().lower()
    config.setdefault("data", {})
    config["data"]["task_type"] = task
    return task


def load_merged_config(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Path, Optional[Path], Optional[Path]]:
    model_path = Path(args.config).resolve()
    base_path, experiment_path = _resolve_merge_paths(args, model_path)
    config = load_config(
        base_config_path=base_path,
        model_config_path=model_path,
        experiment_config_path=experiment_path,
    )
    task = _normalize_task_type(config)
    if task == "classification":
        ncls = get_num_classes_from_config(config)
        if ncls is not None:
            config.setdefault("model", {})["num_classes"] = int(ncls)
    return config, model_path, base_path, experiment_path


def main() -> None:
    args = _parse_args()
    config, model_config_path, base_path, experiment_path = load_merged_config(args)

    data_cfg = config.get("data", {})
    aug = data_cfg.get("augmentation", {})
    print("=" * 60)
    print("Training Configuration (CNN from scratch)")
    print("=" * 60)
    print(f"Primary config:  {model_config_path}")
    if base_path is not None:
        print(f"Base config:     {base_path.resolve()}")
    if experiment_path is not None:
        print(f"Experiment config: {experiment_path.resolve()}")
    print(f"Model type:      {config.get('model', {}).get('type', 'unknown')}")
    print(f"Data directory:  {data_cfg.get('data_dir', '')}")
    print(f"LOS task:        {data_cfg.get('task_type', 'regression')}")
    print(f"Loss type:       {config.get('training', {}).get('loss', {}).get('type', 'unknown')}")
    print(f"Augmentation:    {aug.get('enabled', False)}")
    print("=" * 60)

    icustays_env = os.getenv("ICUSTAYS_PATH")
    if icustays_env:
        env_path = Path(icustays_env)
        if env_path.exists():
            icustays_path = env_path
        else:
            print(f"Warning: ICUSTAYS_PATH set but missing: {env_path}. Falling back.")
            icustays_env = None
    if not icustays_env:
        data_dir = data_cfg.get("data_dir", "")
        if data_dir:
            icustays_path = Path(data_dir).parent.parent / "labeling" / "labels_csv" / "icustays.csv"
        else:
            icustays_path = Path("data/labeling/labels_csv/icustays.csv")

    icustays_path = Path(icustays_path)
    if not icustays_path.exists():
        raise FileNotFoundError(
            f"icustays.csv not found at: {icustays_path}\n"
            "Set ICUSTAYS_PATH or place icustays.csv under data/labeling/labels_csv."
        )

    print(f"Loading ICU stays from: {icustays_path}")
    icustays_df = load_icustays(str(icustays_path))
    print(f"Loaded {len(icustays_df)} ICU stays")

    multi_task_config = config.get("multi_task", {})
    is_multi_task = multi_task_config.get("enabled", False)

    mortality_mapping = None
    if is_multi_task:
        admissions_path = Path(
            multi_task_config.get("admissions_path", "data/labeling/labels_csv/admissions.csv")
        )
        if not admissions_path.is_absolute():
            project_root = _REPO_ROOT
            admissions_path = project_root / admissions_path
        if not admissions_path.exists():
            dd = data_cfg.get("data_dir", "")
            if dd:
                admissions_path = Path(dd).parent.parent / "labeling" / "labels_csv" / "admissions.csv"
        if not admissions_path.exists():
            raise FileNotFoundError(
                f"admissions.csv not found for multi-task at: {admissions_path}\n"
                "Set multi_task.admissions_path or disable multi_task."
            )
        print(f"Loading admissions from: {admissions_path}")
        mortality_mapping = load_mortality_mapping(str(admissions_path), icustays_df)
        n_die = sum(mortality_mapping.values())
        print(f"Mortality mapping: {n_die:,} died, {len(mortality_mapping) - n_die:,} survived")

    icu_mapper = ICUStayMapper(icustays_df, mortality_mapping=mortality_mapping)

    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        labels=None,
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,
    )

    base_model = CNNScratch(config)
    if is_multi_task:
        print("Creating Multi-Task model (LOS + Mortality)...")
        model: Union[CNNScratch, MultiTaskECGModel] = MultiTaskECGModel(base_model, config)
        print(f"Multi-Task model parameters: {model.count_parameters():,}")
    else:
        model = base_model

    if is_multi_task:
        criterion = get_multi_task_loss(config)
        print("Using Multi-Task Loss")
    else:
        criterion = get_loss(config)
        print(f"Using single-task loss: {type(criterion).__name__}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
    )

    paths: Dict[str, str] = {"model": str(model_config_path)}
    if base_path is not None:
        paths["base"] = str(base_path.resolve())
    if experiment_path is not None:
        paths["experiment"] = str(experiment_path.resolve())
    trainer.config_paths = paths

    trainer.job_id = os.getenv("SLURM_JOB_ID")
    if trainer.job_id:
        print(f"SLURM Job ID: {trainer.job_id}")

    history = trainer.train()

    print("Training completed!")
    print(f"Best validation loss: {min(history.get('val_loss', [float('inf')])):.4f}")
    if history.get("val_los_mae"):
        print(f"Best validation MAE: {min(history.get('val_los_mae', [float('inf')])):.4f} days")
    if history.get("val_los_r2"):
        print(f"Best validation R²: {max(history.get('val_los_r2', [float('-inf')])):.4f}")

    history = evaluate_and_print_results(trainer, test_loader, history, config)
    return history


if __name__ == "__main__":
    main()
