"""Training script for LSTM1D Bidirectional with LOS regression.

LOS Regression Task: Predicts continuous LOS in days (not binned classes).
Uses data augmentation but no demographic/diagnosis features.
"""

from pathlib import Path
import sys
import os
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models.lstm import LSTM1D_Bidirectional
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.training import Trainer, setup_icustays_mapper, evaluate_and_print_results
from src.training.losses import get_loss, get_multi_task_loss
from src.utils.config_loader import load_config


def main():
    """Main training function for 24h dataset.
    
    LOS Regression Task: Predicts continuous LOS in days.
    """
    # Load config (standalone model config with all parameters)
    model_config_path = Path("configs/model/lstm/bidirectional/lstm_bi_2layer.yaml")  # Using optimized 2-layer config
    
    config = load_config(model_config_path=model_config_path)
    
    print("="*60)
    print("Training LSTM1D Bidirectional for 24h Dataset")
    print("Task: LOS REGRESSION (continuous prediction in days)")
    print("="*60)
    print(f"Model config: {model_config_path}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")
    print(f"Loss type: {config.get('training', {}).get('loss', {}).get('type', 'mse')}")
    
    model_config = config.get('model', {})
    print(f"LSTM hidden_dim: {model_config.get('hidden_dim', 'N/A')} (per direction)")
    print(f"LSTM num_layers: {model_config.get('num_layers', 'N/A')}")
    print(f"LSTM pooling: {model_config.get('pooling', 'N/A')}")
    print(f"LSTM bidirectional: {model_config.get('bidirectional', 'N/A')}")
    print(f"Effective feature dim: {model_config.get('hidden_dim', 128) * 2} (128*2 = 256)")
    
    augmentation_config = config.get('data', {}).get('augmentation', {})
    print(f"Augmentation: {augmentation_config.get('enabled', False)}")
    if augmentation_config.get('enabled', False):
        print(f"  Gaussian noise: {augmentation_config.get('gaussian_noise', False)}")
        print(f"  Amplitude scaling: {augmentation_config.get('amplitude_scaling', False)}")
    
    demographic_config = config.get('data', {}).get('demographic_features', {})
    print(f"Demographic features: {demographic_config.get('enabled', False)}")
    
    diagnosis_config = config.get('data', {}).get('diagnosis_features', {})
    print(f"Diagnosis features: {diagnosis_config.get('enabled', False)}")
    
    icu_unit_config = config.get('data', {}).get('icu_unit_features', {})
    print(f"ICU unit features: {icu_unit_config.get('enabled', False)}")
    if icu_unit_config.get('enabled', False):
        icu_list = icu_unit_config.get('icu_unit_list', [])
        print(f"  ICU units: {len(icu_list)} + 1 (Other) = {len(icu_list) + 1} features")
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
    base_model = LSTM1D_Bidirectional(config)
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
