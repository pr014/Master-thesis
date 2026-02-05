"""Training script for FastAI xResNet1D-101 with LOS regression.

LOS Regression Task: Predicts continuous LOS in days (not binned classes).
"""

from pathlib import Path
import sys
import os
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models import XResNet1D101
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.training import Trainer, setup_icustays_mapper, evaluate_and_print_results
from src.training.losses import get_loss, get_multi_task_loss
from src.utils.config_loader import load_config


def main():
    """Main training function for 24h dataset.
    
    LOS Regression Task: Predicts continuous LOS in days.
    """
    # Load configs
    base_config_path = Path("configs/icu_24h/output/weighted_exact_days.yaml")
    model_config_path = Path("configs/model/xresnet1d_101/xresnet1d_101_pretrained.yaml")
    
    # Optional: Load demographic features config
    # Set to None to disable demographic features
    feature_config_path = Path("configs/features/demographic_features.yaml")
    if not feature_config_path.exists():
        feature_config_path = None
        print("Note: Demographic features config not found. Training without Age & Sex features.")
    
    config = load_config(
        base_config_path=base_config_path,
        model_config_path=model_config_path,
        experiment_config_path=feature_config_path,  # Optional feature config
    )
    
    print("="*60)
    print("Training FastAI xResNet1D-101 for 24h Dataset")
    print("Task: LOS REGRESSION (continuous prediction in days)")
    print("="*60)
    print(f"Base config: {base_config_path}")
    print(f"Model config: {model_config_path}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")
    print(f"Loss type: {config.get('training', {}).get('loss', {}).get('type', 'mse')}")
    
    pretrained_config = config.get('model', {}).get('pretrained', {})
    if pretrained_config.get('enabled', False):
        print(f"Pretrained weights: {pretrained_config.get('weights_path', 'N/A')}")
    
    demographic_config = config.get('data', {}).get('demographic_features', {})
    if demographic_config.get('enabled', False):
        print(f"Demographic features: Enabled (Age & Sex)")
        print(f"  Age normalization: {demographic_config.get('age_normalization', 'N/A')}")
        print(f"  Sex encoding: {demographic_config.get('sex_encoding', 'N/A')}")
    else:
        print(f"Demographic features: Disabled")
    
    diagnosis_config = config.get('data', {}).get('diagnosis_features', {})
    if diagnosis_config.get('enabled', False):
        diagnosis_list = diagnosis_config.get('diagnosis_list', [])
        print(f"Diagnosis features: Enabled ({len(diagnosis_list)} diagnoses)")
    else:
        print(f"Diagnosis features: Disabled")
    print("="*60)
    
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
    base_model = XResNet1D101(config)
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
        "base": str(base_config_path.resolve()),
        "model": str(model_config_path.resolve()),
    }
    
    # Store job ID if available (from SLURM)
    trainer.job_id = os.getenv("SLURM_JOB_ID")
    if trainer.job_id:
        print(f"SLURM Job ID: {trainer.job_id}")
    
    # Train
    history = trainer.train()
    
    print("Training completed!")
    print(f"Best validation loss: {min(history.get('val_loss', [float('inf')])):.4f}")
    print(f"Best validation MAE: {min(history.get('val_los_mae', [float('inf')])):.4f} days")
    print(f"Best validation R²: {max(history.get('val_los_r2', [float('-inf')])):.4f}")
    
    # Test evaluation
    history = evaluate_and_print_results(trainer, test_loader, history, config)
    
    return history


if __name__ == "__main__":
    main()
