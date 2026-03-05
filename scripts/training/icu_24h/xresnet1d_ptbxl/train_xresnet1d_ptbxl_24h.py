"""Training script for XResNet1D-101 (PTB-XL pretrained) with LOS regression + Mortality.

Transfer Learning: Loads pretrained weights from PTB-XL, fine-tunes for LOS + Mortality.

Usage:
  python scripts/training/icu_24h/xresnet1d_ptbxl/train_xresnet1d_ptbxl_24h.py
  python scripts/training/icu_24h/xresnet1d_ptbxl/train_xresnet1d_ptbxl_24h.py --resume outputs/checkpoints/XResNetPTBXL_best_12345.pt
"""

from pathlib import Path
import argparse
import sys
import os
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.models.xresnet1d_ptbxl import XResNetPTBXL
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.training import Trainer, setup_icustays_mapper, evaluate_and_print_results
from src.training.losses import get_loss, get_multi_task_loss
from src.utils.config_loader import load_config


def main():
    """Main training function for 24h dataset with XResNet1D-101 + PTB-XL pretrained weights."""
    parser = argparse.ArgumentParser(
        description="Train XResNet1D-101 (PTB-XL pretrained) for LOS + Mortality"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()
    resume_path = args.resume or os.getenv("RESUME_PATH")
    resume_from = Path(resume_path) if resume_path else None

    model_config_path = Path("configs/model/xresnet1d_ptbxl/xresnet1d_ptbxl.yaml")
    config = load_config(model_config_path=model_config_path)

    print("=" * 60)
    print("Training XResNet1D-101 (PTB-XL Pretrained) for 24h Dataset")
    print("Task: LOS REGRESSION + Mortality Classification (Multi-Task)")
    print("=" * 60)
    print(f"Model config: {model_config_path}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")
    print(f"Loss type: {config.get('training', {}).get('loss', {}).get('type', 'huber')}")

    model_config = config.get("model", {})
    pretrained_config = model_config.get("pretrained", {})
    print(f"Pretrained: {pretrained_config.get('enabled', False)}")
    if pretrained_config.get("enabled"):
        print(f"  Checkpoint: {pretrained_config.get('checkpoint_path', 'N/A')}")
    print(f"kernel_size: {model_config.get('kernel_size', 5)}")
    print(f"lin_ftrs_head: {model_config.get('lin_ftrs_head', [128])}")

    demographic_config = config.get("data", {}).get("demographic_features", {})
    if demographic_config.get("enabled", False):
        print("Demographic features: Enabled (Age & Sex)")
    else:
        print("Demographic features: Disabled")

    diagnosis_config = config.get("data", {}).get("diagnosis_features", {})
    if diagnosis_config.get("enabled", False):
        diagnosis_list = diagnosis_config.get("diagnosis_list", [])
        print(f"Diagnosis features: Enabled ({len(diagnosis_list)} diagnoses)")
    else:
        print("Diagnosis features: Disabled")

    icu_unit_config = config.get("data", {}).get("icu_unit_features", {})
    print(f"ICU unit features: {icu_unit_config.get('enabled', False)}")
    if icu_unit_config.get("enabled", False):
        icu_list = icu_unit_config.get("icu_unit_list", [])
        print(f"  ICU units: {len(icu_list)} features")
    print("=" * 60)

    icu_mapper = setup_icustays_mapper(config)
    multi_task_config = config.get("multi_task", {})
    is_multi_task = multi_task_config.get("enabled", False)

    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        labels=None,
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,
    )

    base_model = XResNetPTBXL(config)
    print(f"Model created with {base_model.count_parameters():,} parameters")

    if is_multi_task:
        print("Creating Multi-Task model (LOS Regression + Mortality Classification)...")
        model = MultiTaskECGModel(base_model, config)
        print(f"Multi-Task model created with {model.count_parameters():,} parameters")
    else:
        model = base_model

    if is_multi_task:
        criterion = get_multi_task_loss(config)
        print("Using Multi-Task Loss (LOS Huber + Mortality BCE)")
    else:
        criterion = get_loss(config)
        print(f"Using Single-Task Loss: {type(criterion).__name__}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
    )

    trainer.config_paths = {"model": str(model_config_path.resolve())}
    trainer.job_id = os.getenv("SLURM_JOB_ID")
    if trainer.job_id:
        print(f"SLURM Job ID: {trainer.job_id}")

    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")

    history = trainer.train(resume_from=resume_from)

    print("Training completed!")
    print(f"Best validation loss: {min(history.get('val_loss', [float('inf')])):.4f}")
    print(f"Best validation MAE: {min(history.get('val_los_mae', [float('inf')])):.4f} days")
    print(f"Best validation R²: {max(history.get('val_los_r2', [float('-inf')])):.4f}")

    history = evaluate_and_print_results(trainer, test_loader, history, config)

    return history


if __name__ == "__main__":
    main()
