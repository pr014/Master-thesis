"""Training script for Hybrid CNN-LSTM with LOS regression.

LOS Regression Task: Predicts continuous LOS in days (not binned classes).

Usage:
  python scripts/training/icu_24h/hybrid_cnn_lstm/train_hybrid_cnn_lstm_24h.py
  python scripts/training/icu_24h/hybrid_cnn_lstm/train_hybrid_cnn_lstm_24h.py --resume outputs/checkpoints/HybridCNNLSTM_best_3353650.pt
"""

from pathlib import Path
import argparse
import sys
import os
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.models import HybridCNNLSTM
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.training import Trainer, setup_icustays_mapper, evaluate_and_print_results
from src.training.losses import get_loss, get_multi_task_loss
from src.utils.config_loader import load_config


def main():
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
    args = parser.parse_args()
    resume_path = args.resume or os.getenv("RESUME_PATH")
    resume_from = Path(resume_path) if resume_path else None

    # Load config (standalone model config with all parameters)
    model_config_path = Path("configs/model/hybrid_cnn_lstm/hybrid_cnn_lstm.yaml")
    
    config = load_config(model_config_path=model_config_path)
    
    print("="*60)
    print("Training Hybrid CNN-LSTM for 24h Dataset")
    print("Task: LOS REGRESSION (continuous prediction in days)")
    print("="*60)
    print(f"Model config: {model_config_path}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")
    print(f"Loss type: {config.get('training', {}).get('loss', {}).get('type', 'mse')}")
    
    model_config = config.get('model', {})
    print(f"CNN: 12→{model_config.get('conv1_out', 32)}→{model_config.get('conv2_out', 64)}→{model_config.get('conv3_out', 128)}")
    print(f"LSTM hidden_dim: {model_config.get('hidden_dim', 128)} (per direction)")
    print(f"LSTM num_layers: {model_config.get('num_layers', 2)}")
    print(f"LSTM bidirectional: {model_config.get('bidirectional', True)}")
    print(f"LSTM pooling: {model_config.get('pooling', 'last')}")
    
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
    
    icu_unit_config = config.get('data', {}).get('icu_unit_features', {})
    print(f"ICU unit features: {icu_unit_config.get('enabled', False)}")
    if icu_unit_config.get('enabled', False):
        icu_list = icu_unit_config.get('icu_unit_list', [])
        print(f"  ICU units: {len(icu_list)} + 1 (Other) = {len(icu_list) + 1} features")
    sofa_cfg = config.get("data", {}).get("sofa_features", {})
    if sofa_cfg.get("enabled", False):
        print(
            f"SOFA features: Enabled (columns={sofa_cfg.get('columns', ['sofa_total'])}, "
            f"filter_to_valid_sofa={sofa_cfg.get('filter_to_valid_sofa', True)})"
        )
    else:
        print("SOFA features: Disabled")
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
    
    # Store job ID if available (from SLURM)
    trainer.job_id = os.getenv("SLURM_JOB_ID")
    if trainer.job_id:
        print(f"SLURM Job ID: {trainer.job_id}")

    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")

    # Train
    history = trainer.train(resume_from=resume_from)
    
    print("Training completed!")
    print(f"Best validation loss: {min(history.get('val_loss', [float('inf')])):.4f}")
    print(f"Best validation MAE: {min(history.get('val_los_mae', [float('inf')])):.4f} days")
    print(f"Best validation R²: {max(history.get('val_los_r2', [float('-inf')])):.4f}")
    
    # Test evaluation
    history = evaluate_and_print_results(trainer, test_loader, history, config)
    
    return history


if __name__ == "__main__":
    main()
