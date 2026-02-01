"""Training script for ECG-CPC with two-phase training.

Phase 1: Frozen backbone (S4 encoder frozen, only heads trained)
Phase 2: Fine-tuning with layer-dependent learning rates

Based on Al-Masud et al. (2025): "Benchmarking ECG Foundational Models"
Config: configs/icu_24h/24h_weighted/sqrt_weights.yaml (sqrt method)
"""

from pathlib import Path
import sys
import os
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.models import ECG_S4_CPC
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.training.trainer_ecg_cpc import ECG_CPC_Trainer
from src.training import setup_icustays_mapper, evaluate_and_print_results
from src.training.losses import get_multi_task_loss
from src.utils.config_loader import load_config


def main():
    """Main training function for 24h dataset with two-phase training."""
    # Load configs - using 24h weighted config (sqrt method)
    base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
    model_config_path = Path("configs/model/ecg_cpc/ecg_cpc.yaml")
    
    # Optional: Load demographic features config
    feature_config_path = Path("configs/features/demographic_features.yaml")
    if not feature_config_path.exists():
        feature_config_path = None
        print("Note: Demographic features config not found. Training without Age & Sex features.")
    
    config = load_config(
        base_config_path=base_config_path,
        model_config_path=model_config_path,
        experiment_config_path=feature_config_path,
    )
    
    print("="*60)
    print("Training ECG-CPC (S4 + CPC Pretraining) for 24h Dataset")
    print("="*60)
    print(f"Base config: {base_config_path}")
    print(f"Model config: {model_config_path}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")
    
    model_config = config.get('model', {})
    print(f"S4 Encoder: {model_config.get('n_layers', 4)} layers, d_model={model_config.get('d_model', 256)}, d_state={model_config.get('d_state', 64)}")
    
    pretrained_config = model_config.get('pretrained', {})
    if pretrained_config.get('enabled', False):
        weights_path = pretrained_config.get('weights_path', '')
        if weights_path:
            print(f"Pretrained weights: {weights_path}")
        else:
            print("Pretrained weights: Enabled but path not set (training from scratch)")
    else:
        print("Pretrained weights: Disabled (training from scratch)")
    
    demographic_config = config.get('data', {}).get('demographic_features', {})
    if demographic_config.get('enabled', False):
        print(f"Demographic features: Enabled (Age & Sex)")
    else:
        print(f"Demographic features: Disabled")
    print("="*60)
    
    # Load ICU stays and create mapper
    icu_mapper = setup_icustays_mapper(config)
    
    # Multi-task is required for ECG-CPC
    multi_task_config = config.get("multi_task", {})
    is_multi_task = multi_task_config.get("enabled", False)
    if not is_multi_task:
        print("Warning: Multi-task is disabled. ECG-CPC requires multi-task learning.")
        print("Enabling multi-task in config...")
        config["multi_task"] = {"enabled": True}
        is_multi_task = True
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        labels=None,
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,
    )
    
    # Create base model
    base_model = ECG_S4_CPC(config)
    print(f"Model created with {base_model.count_parameters():,} parameters")
    
    # Wrap in MultiTaskECGModel for multi-task compatibility
    print("Creating Multi-Task model (LOS + Mortality)...")
    model = MultiTaskECGModel(base_model, config)
    print(f"Multi-Task model created with {model.count_parameters():,} parameters")
    
    # Create multi-task loss
    criterion = get_multi_task_loss(config)
    print("Using Multi-Task Loss (LOS weight=1.0, Mortality weight=0.5)")
    
    # Get job ID for checkpoint naming
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id:
        print(f"SLURM Job ID: {job_id}")
    
    # ========== PHASE 1: Frozen Backbone ==========
    print("\n" + "="*60)
    print("PHASE 1: FROZEN BACKBONE TRAINING")
    print("="*60)
    
    # For ECG_CPC_Trainer, we need to pass base_model for freeze/unfreeze methods
    # But we'll use the wrapped model for actual training
    # Create a custom trainer that uses wrapped model but can access base_model
    trainer_phase1 = ECG_CPC_Trainer(
        model=base_model,  # Base model for freeze/unfreeze
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
        phase=1,
        wrapped_model=model,  # Wrapped model for training loop
    )
    
    # Store config paths and job ID
    trainer_phase1.config_paths = {
        "base": str(base_config_path.resolve()),
        "model": str(model_config_path.resolve()),
    }
    trainer_phase1.job_id = job_id
    
    # Train Phase 1
    history_phase1 = trainer_phase1.train_phase1(max_epochs=20, patience=20)
    
    print("\nPhase 1 completed!")
    print(f"Best validation loss: {min(history_phase1.get('val_loss', [float('inf')])):.4f}")
    
    # Save Phase 1 checkpoint
    checkpoint_dir = Path(config.get("checkpoint", {}).get("save_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    phase1_checkpoint_path = checkpoint_dir / f"ecg_cpc_phase1_best_{job_id if job_id else 'local'}.pt"
    
    # Save best model from Phase 1
    if hasattr(trainer_phase1.checkpoint, 'best_model_state'):
        torch.save({
            'model_state_dict': trainer_phase1.checkpoint.best_model_state,
            'epoch': trainer_phase1.checkpoint.best_epoch,
            'val_loss': trainer_phase1.checkpoint.best_score,
            'history': history_phase1,
            'config': config,
        }, phase1_checkpoint_path)
        print(f"Saved Phase 1 checkpoint to: {phase1_checkpoint_path}")
    else:
        # Fallback: save current model state
        torch.save({
            'model_state_dict': base_model.state_dict(),
            'epoch': trainer_phase1.current_epoch,
            'history': history_phase1,
            'config': config,
        }, phase1_checkpoint_path)
        print(f"Saved Phase 1 checkpoint to: {phase1_checkpoint_path}")
    
    # ========== PHASE 2: Fine-tuning ==========
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNING WITH LAYER-DEPENDENT LEARNING RATES")
    print("="*60)
    
    trainer_phase2 = ECG_CPC_Trainer(
        model=base_model,  # Base model for freeze/unfreeze
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
        phase=2,
        wrapped_model=model,  # Wrapped model for training loop
    )
    
    # Store config paths and job ID
    trainer_phase2.config_paths = {
        "base": str(base_config_path.resolve()),
        "model": str(model_config_path.resolve()),
    }
    trainer_phase2.job_id = job_id
    
    # Train Phase 2 (load Phase 1 checkpoint)
    history_phase2 = trainer_phase2.train_phase2(
        phase1_checkpoint_path=phase1_checkpoint_path,
        max_epochs=30,
        patience=15,
    )
    
    print("\nPhase 2 completed!")
    print(f"Best validation loss: {min(history_phase2.get('val_loss', [float('inf')])):.4f}")
    
    # Save Phase 2 checkpoint
    phase2_checkpoint_path = checkpoint_dir / f"ecg_cpc_phase2_best_{job_id if job_id else 'local'}.pt"
    
    if hasattr(trainer_phase2.checkpoint, 'best_model_state'):
        torch.save({
            'model_state_dict': trainer_phase2.checkpoint.best_model_state,
            'epoch': trainer_phase2.checkpoint.best_epoch,
            'val_loss': trainer_phase2.checkpoint.best_score,
            'history': history_phase2,
            'config': config,
        }, phase2_checkpoint_path)
        print(f"Saved Phase 2 checkpoint to: {phase2_checkpoint_path}")
    else:
        torch.save({
            'model_state_dict': base_model.state_dict(),
            'epoch': trainer_phase2.current_epoch,
            'history': history_phase2,
            'config': config,
        }, phase2_checkpoint_path)
        print(f"Saved Phase 2 checkpoint to: {phase2_checkpoint_path}")
    
    # ========== FINAL EVALUATION ==========
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load best Phase 2 model for evaluation
    if phase2_checkpoint_path.exists():
        checkpoint = torch.load(phase2_checkpoint_path, map_location=trainer_phase2.device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        # Ensure model is on the correct device after loading
        base_model.to(trainer_phase2.device)
        print(f"Loaded best Phase 2 model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Wrap model for evaluation
    model = MultiTaskECGModel(base_model, config)
    
    # Create a simple trainer for evaluation
    from src.training import Trainer
    eval_trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
    )
    eval_trainer.job_id = job_id
    
    # Evaluate on test set
    history_final = evaluate_and_print_results(eval_trainer, test_loader, history_phase2, config)
    
    print("\n" + "="*60)
    print("TWO-PHASE TRAINING COMPLETED")
    print("="*60)
    print(f"Phase 1 checkpoint: {phase1_checkpoint_path}")
    print(f"Phase 2 checkpoint: {phase2_checkpoint_path}")
    
    return history_phase2


if __name__ == "__main__":
    main()

