"""Training script for ResNet1D-14 with LOS bin classification (Training from Scratch).
This script uses class weights specifically calculated for the 24h ECG dataset.
Config: configs/icu_24h/24h_weighted/balanced_weights.yaml (balanced method)
"""

from pathlib import Path
import sys
import os
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models import ResNet1D14
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.training import Trainer, setup_icustays_mapper, evaluate_and_print_results
from src.training.losses import get_loss, get_multi_task_loss
from src.utils.config_loader import load_config


def main():
    """Main training function for 24h dataset with class weights (training from scratch)."""
    # Load configs - using 24h weighted config (balanced method)
    base_config_path = Path("configs/icu_24h/24h_weighted/balanced_weights.yaml")
    model_config_path = Path("configs/model/resnet14/resnet1d_14_scratch.yaml")
    
    config = load_config(
        base_config_path=base_config_path,
        model_config_path=model_config_path,
    )
    
    print("="*60)
    print("Training ResNet1D-14 from Scratch with Class Weights for 24h Dataset")
    print("="*60)
    print(f"Base config: {base_config_path}")
    print(f"Model config: {model_config_path}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")
    print(f"Loss type: {config.get('training', {}).get('loss', {}).get('type', 'unknown')}")
    if 'weight' in config.get('training', {}).get('loss', {}):
        weights = config.get('training', {}).get('loss', {}).get('weight', [])
        print(f"Class weights: {weights}")
    pretrained_config = config.get('model', {}).get('pretrained', {})
    print(f"Pretrained weights: Disabled (training from scratch)")
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
    base_model = ResNet1D14(config)
    
    # Wrap in MultiTaskECGModel if multi-task is enabled
    if is_multi_task:
        print("Creating Multi-Task model (LOS + Mortality)...")
        model = MultiTaskECGModel(base_model, config)
        print(f"Multi-Task model created with {model.count_parameters():,} parameters")
    else:
        model = base_model
    
    # Create loss function
    if is_multi_task:
        criterion = get_multi_task_loss(config)
        print("Using Multi-Task Loss (LOS + Mortality)")
    else:
        criterion = get_loss(config)
        print("Using Single-Task Loss (LOS only)")
    
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
    
    # Test evaluation (if test_loader is available)
    history = evaluate_and_print_results(trainer, test_loader, history, config)
    
    return history


if __name__ == "__main__":
    main()

