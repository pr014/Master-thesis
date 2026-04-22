"""Training script for Hybrid CNN-LSTM with LOS regression.

LOS Regression Task: Predicts continuous LOS in days (not binned classes).

Usage:
  python scripts/training/icu_24h/hybrid_cnn_lstm/train_hybrid_cnn_lstm_24h.py
  python scripts/training/icu_24h/hybrid_cnn_lstm/train_hybrid_cnn_lstm_24h.py --resume outputs/checkpoints/HybridCNNLSTM_best_3353650.pt
  python scripts/training/icu_24h/hybrid_cnn_lstm/train_hybrid_cnn_lstm_24h.py --experiment-config outputs/tuning/hybrid_cnn_lstm/demo/trial_configs/trial_001.yaml --trial-id trial_001 --sweep-id demo
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.models import HybridCNNLSTM
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.training import Trainer, setup_icustays_mapper, evaluate_and_print_results
from src.training.losses import get_loss, get_multi_task_loss
from src.utils.config_loader import load_config


def _resolve_optional_path(value: Optional[str]) -> Optional[Path]:
    """Resolve a CLI/env path to a Path object."""
    if not value:
        return None
    return Path(value)


def run_training(
    model_config_path: Path,
    experiment_config_path: Optional[Path] = None,
    trial_id: Optional[str] = None,
    sweep_id: Optional[str] = None,
    resume_from: Optional[Path] = None,
    print_parameter_debug: bool = True,
) -> Dict[str, Any]:
    """Run Hybrid CNN-LSTM training + test eval; return metrics for Optuna/logging."""
    _ = print_parameter_debug  # Interface parity with other train scripts.

    config = load_config(
        model_config_path=model_config_path,
        experiment_config_path=experiment_config_path,
    )
    config.setdefault("runtime", {})
    config["runtime"]["model_config_path"] = str(model_config_path.resolve())
    if experiment_config_path is not None:
        config["runtime"]["experiment_config_path"] = str(experiment_config_path.resolve())
    if trial_id:
        config["runtime"]["trial_id"] = trial_id
    if sweep_id:
        config["runtime"]["sweep_id"] = sweep_id

    print("=" * 60)
    print("Training Hybrid CNN-LSTM for 24h Dataset")
    print("Task: LOS REGRESSION (continuous prediction in days)")
    print("=" * 60)
    print(f"Model config: {model_config_path}")
    print(f"Experiment config: {experiment_config_path if experiment_config_path else 'None'}")
    print(f"Trial ID: {trial_id if trial_id else 'None'}")
    print(f"Sweep ID: {sweep_id if sweep_id else 'None'}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")
    print(f"Loss type: {config.get('training', {}).get('loss', {}).get('type', 'mse')}")

    model_config = config.get("model", {})
    print(
        f"CNN: 12->{model_config.get('conv1_out', 32)}->{model_config.get('conv2_out', 64)}->{model_config.get('conv3_out', 128)}"
    )
    print(f"LSTM hidden_dim: {model_config.get('hidden_dim', 128)} (per direction)")
    print(f"LSTM num_layers: {model_config.get('num_layers', 2)}")
    print(f"LSTM bidirectional: {model_config.get('bidirectional', True)}")
    print(f"LSTM pooling: {model_config.get('pooling', 'last')}")

    demographic_config = config.get("data", {}).get("demographic_features", {})
    if demographic_config.get("enabled", False):
        print("Demographic features: Enabled (Age & Sex)")
        print(f"  Age normalization: {demographic_config.get('age_normalization', 'N/A')}")
        print(f"  Sex encoding: {demographic_config.get('sex_encoding', 'N/A')}")
    else:
        print("Demographic features: Disabled")

    icu_unit_config = config.get("data", {}).get("icu_unit_features", {})
    print(f"ICU unit features: {icu_unit_config.get('enabled', False)}")
    if icu_unit_config.get("enabled", False):
        icu_list = icu_unit_config.get("icu_unit_list", [])
        print(f"  ICU units: {len(icu_list)} + 1 (Other) = {len(icu_list) + 1} features")
    sofa_cfg = config.get("data", {}).get("sofa_features", {})
    if sofa_cfg.get("enabled", False):
        print(
            f"SOFA features: Enabled (columns={sofa_cfg.get('columns', ['sofa_total'])}, "
            f"filter_to_valid_sofa={sofa_cfg.get('filter_to_valid_sofa', True)})"
        )
    else:
        print("SOFA features: Disabled")
    therapy_cfg = config.get("data", {}).get("icu_therapy_support_features", {})
    if therapy_cfg.get("enabled", False):
        print(
            f"ICU therapy support features: Enabled (columns={therapy_cfg.get('columns', [])}, "
            f"filter_to_labeled_only={therapy_cfg.get('filter_to_labeled_only', True)})"
        )
    else:
        print("ICU therapy support features: Disabled")
    print("=" * 60)

    # Load ICU stays and create mapper
    icu_mapper = setup_icustays_mapper(config)

    # Check if multi-task is enabled
    multi_task_config = config.get("multi_task", {})
    is_multi_task = multi_task_config.get("enabled", False)

    # Create DataLoaders (labels will be auto-generated via icu_mapper)
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        labels=None,  # Will be auto-generated
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,  # Will be auto-generated from mortality_mapping
    )

    # Create base model
    base_model = HybridCNNLSTM(config)
    print(f"Model created with {base_model.count_parameters():,} parameters")

    # Wrap in MultiTaskECGModel if multi-task is enabled
    if is_multi_task:
        print("Creating Multi-Task model (LOS Regression + Mortality Classification)...")
        model = MultiTaskECGModel(base_model, config)
        print(f"Multi-Task model created with {model.count_parameters():,} parameters")
    else:
        model = base_model

    # Create loss function
    if is_multi_task:
        criterion = get_multi_task_loss(config)
        print("Using Multi-Task Loss (LOS MSE + Mortality BCE)")
    else:
        criterion = get_loss(config)
        print(f"Using Single-Task Loss (LOS MSE: {type(criterion).__name__})")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,  # Pass custom criterion if multi-task
    )

    # Store config paths for checkpoint saving
    trainer.config_paths = {
        "model": str(model_config_path.resolve()),
    }
    if experiment_config_path is not None:
        trainer.config_paths["experiment"] = str(experiment_config_path.resolve())

    job_id = (
        os.getenv("SLURM_JOB_ID")
        or os.getenv("OPTUNA_TRIAL_JOB_ID")
        or "local"
    )
    trainer.job_id = job_id
    print(f"Job ID (checkpoints / eval): {job_id}")
    if trial_id:
        print(f"Trial metadata - trial_id: {trial_id}")
    if sweep_id:
        print(f"Trial metadata - sweep_id: {sweep_id}")

    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")

    # Train
    history = trainer.train(resume_from=resume_from)

    best_val_loss = min(history.get("val_loss", [float("inf")]))
    best_val_mae = min(history.get("val_los_mae", [float("inf")]))
    best_val_r2 = max(history.get("val_los_r2", [float("-inf")]))
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation MAE: {best_val_mae:.4f} days")
    print(f"Best validation R²: {best_val_r2:.4f}")

    # Test evaluation
    history = evaluate_and_print_results(trainer, test_loader, history, config)

    out: Dict[str, Any] = {
        "best_val_mae": float(best_val_mae),
        "best_val_loss": float(best_val_loss),
        "best_val_r2": float(best_val_r2),
        "job_id": job_id,
        "history": history,
    }
    if history.get("test_los_mae") is not None:
        out["test_los_mae"] = float(history["test_los_mae"])
    if history.get("test_los_r2") is not None:
        out["test_los_r2"] = float(history["test_los_r2"])
    return out


def main() -> Dict[str, Any]:
    """Main training function for 24h dataset.

    LOS Regression Task: Predicts continuous LOS in days.
    """
    parser = argparse.ArgumentParser(description="Train Hybrid CNN-LSTM for LOS regression")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g. outputs/checkpoints/HybridCNNLSTM_best_3353650.pt)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Optional path to the base Hybrid YAML (defaults to configs/model/hybrid_cnn_lstm/hybrid_cnn_lstm.yaml).",
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        default=None,
        help="Optional override YAML merged on top of the base Hybrid config for one tuning trial.",
    )
    parser.add_argument(
        "--trial-id",
        type=str,
        default=None,
        help="Optional random-search trial identifier for traceability.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Optional random-search sweep identifier for traceability.",
    )
    parser.add_argument(
        "--no-parameter-debug",
        action="store_true",
        help="Reserved for API parity with other training scripts.",
    )
    args = parser.parse_args()
    resume_path = args.resume or os.getenv("RESUME_PATH")
    resume_from = Path(resume_path) if resume_path else None

    model_config_path = _resolve_optional_path(args.model_config or os.getenv("MODEL_CONFIG_PATH"))
    if model_config_path is None:
        model_config_path = Path("configs/model/hybrid_cnn_lstm/hybrid_cnn_lstm.yaml")

    experiment_config_path = _resolve_optional_path(
        args.experiment_config or os.getenv("EXPERIMENT_CONFIG_PATH")
    )
    trial_id = args.trial_id or os.getenv("TRIAL_ID")
    sweep_id = args.sweep_id or os.getenv("SWEEP_ID")

    return run_training(
        model_config_path=model_config_path,
        experiment_config_path=experiment_config_path,
        trial_id=trial_id,
        sweep_id=sweep_id,
        resume_from=resume_from,
        print_parameter_debug=not args.no_parameter_debug,
    )


if __name__ == "__main__":
    main()
