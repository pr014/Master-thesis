"""Training script for DeepECG-SL (WCR) with two-phase training.

Phase 1: Frozen backbone (WCR encoder frozen, only heads trained)
Phase 2: Fine-tuning with layer-dependent learning rates

Based on Transfer Learning strategy for self-supervised pretrained models.
Config: configs/icu_24h/24h_weighted/sqrt_weights.yaml (sqrt method)
"""

from pathlib import Path
import sys
import os
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.models import DeepECG_SL
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.training import Trainer, setup_icustays_mapper, evaluate_and_print_results
from src.training.losses import get_multi_task_loss
from src.utils.config_loader import load_config


class DeepECG_SL_Trainer(Trainer):
    """Specialized trainer for DeepECG-SL with two-phase training."""
    
    def __init__(
        self,
        model: MultiTaskECGModel,
        train_loader,
        val_loader,
        config: dict,
        device=None,
        criterion=None,
        phase: int = 1,
    ):
        """Initialize DeepECG-SL trainer.
        
        Args:
            phase: Training phase (1: frozen, 2: fine-tuning)
        """
        self.phase = phase
        super().__init__(model, train_loader, val_loader, config, device, criterion)
        
        # Override optimizer with layer-dependent learning rates
        self.optimizer = self._create_layer_dependent_optimizer()
        
        # Setup phase-specific settings
        if phase == 1:
            self._setup_phase1()
        elif phase == 2:
            self._setup_phase2()
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 1 (frozen) or 2 (fine-tuning)")
    
    def _setup_phase1(self) -> None:
        """Setup Phase 1: Frozen backbone training."""
        print("="*60)
        print("Phase 1: Frozen Backbone Training")
        print("="*60)
        print("WCR encoder: FROZEN (no gradient updates)")
        print("Input adapter: FROZEN (no gradient updates)")
        print("Trainable: Only shared layers and classification heads")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print("="*60)
        
        # Freeze backbone (should already be frozen from model init, but ensure it)
        base_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model
        if hasattr(base_model, 'freeze_backbone'):
            base_model.freeze_backbone()
    
    def _setup_phase2(self) -> None:
        """Setup Phase 2: Fine-tuning with layer-dependent learning rates."""
        print("="*60)
        print("Phase 2: Fine-tuning with Layer-Dependent Learning Rates")
        print("="*60)
        print("WCR encoder: UNFROZEN (with reduced learning rate)")
        print("Input adapter: UNFROZEN (with reduced learning rate)")
        print("Learning rates:")
        for i, group in enumerate(self.optimizer.param_groups):
            print(f"  Group {i}: lr={group['lr']:.2e}, params={len(group['params'])}")
        print("="*60)
        
        # Unfreeze backbone
        base_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model
        if hasattr(base_model, 'unfreeze_backbone'):
            base_model.unfreeze_backbone()
    
    def _create_layer_dependent_optimizer(self):
        """Create optimizer with layer-dependent learning rates."""
        opt_config = self.config.get("training", {}).get("optimizer", {})
        opt_type = opt_config.get("type", "Adam").lower()
        base_lr = opt_config.get("lr", 5e-4)
        if isinstance(base_lr, str):
            base_lr = float(base_lr)
        weight_decay = opt_config.get("weight_decay", 1e-4)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        
        # Get base model
        base_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model
        
        # Phase-specific learning rates
        if self.phase == 1:
            # Phase 1: Only heads are trainable (WCR + Input Adapter frozen)
            head_lr = base_lr  # 1e-3 typically
            param_groups = [
                {
                    'params': list(base_model.shared_bn.parameters()) +
                             list(base_model.shared_fc.parameters()) +
                             list(base_model.los_head.parameters()) +
                             list(base_model.mortality_head.parameters()),
                    'lr': head_lr,
                }
            ]
        else:  # Phase 2
            # Phase 2: Layer-dependent learning rates
            # WCR encoder: 1e-5 (100× reduction)
            # Input adapter: 1e-5 (100× reduction)
            # Shared layers: 1e-4 (10× reduction)
            # Heads: 1e-4 (10× reduction)
            backbone_lr = 1e-5
            head_lr = 1e-4
            
            param_groups = [
                {
                    'params': base_model.wcr_encoder.parameters(),
                    'lr': backbone_lr,
                },
                {
                    'params': base_model.input_adapter.parameters(),
                    'lr': backbone_lr,
                },
                {
                    'params': list(base_model.shared_bn.parameters()) +
                             list(base_model.shared_fc.parameters()),
                    'lr': head_lr,
                },
                {
                    'params': base_model.los_head.parameters(),
                    'lr': head_lr,
                },
                {
                    'params': base_model.mortality_head.parameters(),
                    'lr': head_lr,
                },
            ]
        
        # Create optimizer with parameter groups
        betas = opt_config.get("betas", [0.9, 0.999])
        if opt_type == "adamw":
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay, betas=betas)
        elif opt_type == "adam":
            return torch.optim.Adam(param_groups, weight_decay=weight_decay, betas=betas)
        elif opt_type == "sgd":
            momentum = opt_config.get("momentum", 0.9)
            return torch.optim.SGD(param_groups, lr=base_lr, weight_decay=weight_decay, momentum=momentum)
        else:
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay, betas=betas)
    
    def train_phase1(self, max_epochs: int = 20, patience: int = 20):
        """Train Phase 1: Frozen backbone."""
        if self.phase != 1:
            raise ValueError("train_phase1() can only be called when phase=1")
        
        self.early_stopping.patience = patience
        original_max_epochs = self.config.get("training", {}).get("num_epochs", 50)
        self.config["training"]["num_epochs"] = max_epochs
        
        history = self.train()
        
        self.config["training"]["num_epochs"] = original_max_epochs
        return history
    
    def train_phase2(self, phase1_checkpoint_path=None, max_epochs: int = 30, patience: int = 15):
        """Train Phase 2: Fine-tuning."""
        if self.phase != 2:
            raise ValueError("train_phase2() can only be called when phase=2")
        
        # Load Phase 1 checkpoint if provided
        if phase1_checkpoint_path is not None and phase1_checkpoint_path.exists():
            print(f"Loading Phase 1 checkpoint from: {phase1_checkpoint_path}")
            checkpoint = torch.load(phase1_checkpoint_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print(f"Loaded Phase 1 model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint, strict=False)
                print("Loaded Phase 1 model state dict")
        
        self.early_stopping.patience = patience
        original_max_epochs = self.config.get("training", {}).get("num_epochs", 50)
        self.config["training"]["num_epochs"] = max_epochs
        
        # Reset training state for Phase 2
        self.current_epoch = 0
        self.history = {
            "train_loss": [],
            "train_los_acc": [],
            "val_loss": [],
            "val_los_acc": [],
            "train_mortality_acc": [],
            "val_mortality_acc": [],
            "train_mortality_auc": [],
            "val_mortality_auc": [],
        }
        
        history = self.train()
        
        self.config["training"]["num_epochs"] = original_max_epochs
        return history


def main():
    """Main training function for 24h dataset with two-phase training."""
    # Load configs - using exact_days with 9 classes (max_days=8)
    base_config_path = Path("configs/icu_24h/output/weighted_exact_days.yaml")
    model_config_path = Path("configs/model/deepecg_sl/deepecg_sl.yaml")
    
    # Load demographic features config (Age + Sex only, NO diagnoses)
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
    print("Training DeepECG-SL (WCR + Self-Supervised Pretraining) for 24h Dataset")
    print("Configuration: exact_days with 8 classes (max_days=7), Age + Sex only (NO diagnoses)")
    print("="*60)
    print(f"Base config: {base_config_path}")
    print(f"Model config: {model_config_path}")
    print(f"Feature config: {feature_config_path if feature_config_path else 'None (no features)'}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")
    
    # Print LOS binning info
    los_binning = config.get('data', {}).get('los_binning', {})
    print(f"LOS binning: strategy={los_binning.get('strategy', 'unknown')}, max_days={los_binning.get('max_days', 'unknown')}")
    from src.data.labeling import get_num_classes_from_config
    num_classes = get_num_classes_from_config(config)
    print(f"Number of classes: {num_classes}")
    
    model_config = config.get('model', {})
    wcr_config = model_config.get('wcr', {})
    print(f"WCR Model: {wcr_config.get('model_name', 'wcr_77_classes')}")
    print(f"WCR d_model: {wcr_config.get('d_model', 512)}")
    
    pretrained_config = model_config.get('pretrained', {})
    if pretrained_config.get('enabled', False):
        cache_dir = pretrained_config.get('cache_dir', 'data/pretrained_weights/deepecg_sl')
        print(f"Pretrained weights: Enabled (cache: {cache_dir})")
        print("  Weights will be downloaded automatically from HuggingFace if not cached")
    else:
        print("Pretrained weights: Disabled (training from scratch)")
    
    demographic_config = config.get('data', {}).get('demographic_features', {})
    if demographic_config.get('enabled', False):
        print(f"Demographic features: Enabled (Age & Sex)")
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
    
    # Multi-task is required for DeepECG-SL
    multi_task_config = config.get("multi_task", {})
    is_multi_task = multi_task_config.get("enabled", False)
    if not is_multi_task:
        print("Warning: Multi-task is disabled. DeepECG-SL requires multi-task learning.")
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
    
    # Create base model (will automatically download weights if needed)
    print("\nCreating DeepECG-SL model...")
    print("This may take a while on first run (downloading pretrained weights)...")
    base_model = DeepECG_SL(config)
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
    
    trainer_phase1 = DeepECG_SL_Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
        phase=1,
    )
    
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
    phase1_checkpoint_path = checkpoint_dir / f"deepecg_sl_phase1_best_{job_id if job_id else 'local'}.pt"
    
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
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': trainer_phase1.current_epoch,
            'history': history_phase1,
            'config': config,
        }, phase1_checkpoint_path)
        print(f"Saved Phase 1 checkpoint to: {phase1_checkpoint_path}")
    
    # ========== PHASE 2: Fine-tuning ==========
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNING WITH LAYER-DEPENDENT LEARNING RATES")
    print("="*60)
    
    trainer_phase2 = DeepECG_SL_Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
        phase=2,
    )
    
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
    phase2_checkpoint_path = checkpoint_dir / f"deepecg_sl_phase2_best_{job_id if job_id else 'local'}.pt"
    
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
            'model_state_dict': model.state_dict(),
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
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(trainer_phase2.device)
        print(f"Loaded best Phase 2 model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create a simple trainer for evaluation
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

