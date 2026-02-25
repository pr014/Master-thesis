"""Training script for DeepECG-SL (WCR) with two-phase training.

Phase 1: Frozen backbone (WCR encoder frozen, only heads trained)
Phase 2: Fine-tuning with layer-dependent learning rates

Based on Transfer Learning strategy for self-supervised pretrained models.

LOS Regression Task: Predicts continuous LOS in days.
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
        print("Trainable: Only shared layers and regression/classification heads")
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
            "train_los_mae": [],
            "train_los_rmse": [],
            "train_los_r2": [],
            "val_loss": [],
            "val_los_mae": [],
            "val_los_rmse": [],
            "val_los_r2": [],
            "train_mortality_acc": [],
            "val_mortality_acc": [],
            "val_mortality_auc": [],
        }
        
        history = self.train()
        
        self.config["training"]["num_epochs"] = original_max_epochs
        return history


def main():
    """Main training function for 24h dataset with two-phase training.
    
    LOS Regression Task: Predicts continuous LOS in days (not binned classes).
    """
    # Load config (standalone model config with all parameters)
    model_config_path = Path("configs/model/deepecg_sl/deepecg_sl.yaml")
    
    config = load_config(model_config_path=model_config_path)
    
    print("="*60)
    print("Training DeepECG-SL (WCR + Self-Supervised Pretraining) for 24h Dataset")
    print("Task: LOS REGRESSION (continuous prediction in days)")
    print("="*60)
    print(f"Model config: {model_config_path}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")
    
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
    
    icu_unit_config = config.get('data', {}).get('icu_unit_features', {})
    if icu_unit_config.get('enabled', False):
        icu_list = icu_unit_config.get('icu_unit_list', [])
        print(f"ICU unit features: Enabled ({len(icu_list)} + 1 Other = {len(icu_list) + 1} features)")
    else:
        print(f"ICU unit features: Disabled")
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
    print("Creating Multi-Task model (LOS Regression + Mortality Classification)...")
    model = MultiTaskECGModel(base_model, config)
    print(f"Multi-Task model created with {model.count_parameters():,} parameters")
    
    # ========== DEBUG: Detailed Parameter Analysis ==========
    print("\n" + "="*60)
    print("DEBUG: Detailed Parameter Analysis")
    print("="*60)
    
    # Count parameters per component
    def count_component_params(component, name):
        total = sum(p.numel() for p in component.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in component.parameters() if not p.requires_grad)
        return total, frozen
    
    # Base model components
    wcr_trainable, wcr_frozen = count_component_params(base_model.wcr_encoder, "WCR Encoder")
    adapter_trainable, adapter_frozen = count_component_params(base_model.input_adapter, "Input Adapter")
    shared_trainable, shared_frozen = count_component_params(base_model.shared_bn, "Shared BN")
    shared_trainable += sum(p.numel() for p in base_model.shared_fc.parameters() if p.requires_grad)
    shared_trainable += sum(p.numel() for p in base_model.shared_dropout1.parameters() if p.requires_grad)
    shared_trainable += sum(p.numel() for p in base_model.shared_dropout2.parameters() if p.requires_grad)
    los_trainable, _ = count_component_params(base_model.los_head, "LOS Head")
    mortality_trainable, _ = count_component_params(base_model.mortality_head, "Mortality Head")
    
    print(f"WCR Encoder:        {wcr_trainable:>15,} trainable, {wcr_frozen:>15,} frozen")
    print(f"Input Adapter:      {adapter_trainable:>15,} trainable, {adapter_frozen:>15,} frozen")
    print(f"Shared Layers:      {shared_trainable:>15,} trainable")
    print(f"LOS Head:           {los_trainable:>15,} trainable (output: 1 = continuous)")
    print(f"Mortality Head:     {mortality_trainable:>15,} trainable (output: 1 = binary)")
    print(f"{'─'*60}")
    total_trainable = wcr_trainable + adapter_trainable + shared_trainable + los_trainable + mortality_trainable
    total_frozen = wcr_frozen + adapter_frozen
    print(f"TOTAL:              {total_trainable:>15,} trainable, {total_frozen:>15,} frozen")
    freeze_backbone = model_config.get("wcr", {}).get("freeze_backbone", True)
    if freeze_backbone:
        print(f"Expected (Phase 1): ~100K–500K trainable (heads only)")
    else:
        print(f"Expected (Phase 2): ~90M trainable (full fine-tuning)")
    
    # Verify feature_dim from config
    feature_dim = config.get("model", {}).get("feature_dim", 768)
    print(f"\nFeature dimension (from config): {feature_dim}")
    print(f"Expected: 768 (WCR encoder output)")
    
    # Check if encoder parameters are trainable
    encoder_trainable_count = sum(1 for p in base_model.wcr_encoder.parameters() if p.requires_grad)
    encoder_total_count = sum(1 for p in base_model.wcr_encoder.parameters())
    print(f"\nWCR Encoder parameter check:")
    print(f"  Trainable parameter tensors: {encoder_trainable_count}/{encoder_total_count}")
    if encoder_trainable_count == 0:
        print("  ✓ Encoder frozen (Phase 1 head-only)")
    elif encoder_trainable_count == encoder_total_count:
        print("  ✓ All encoder parameters trainable (Phase 2 full fine-tuning)")
    else:
        print(f"  ⚠️  WARNING: Only {encoder_trainable_count}/{encoder_total_count} parameter groups are trainable")
    
    print("="*60 + "\n")
    
    # Create multi-task loss (MSE for LOS regression, BCE for mortality)
    criterion = get_multi_task_loss(config)
    print("Using Multi-Task Loss (LOS MSE weight=1.0, Mortality BCE weight=0.5)")
    
    # Get job ID for checkpoint naming
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id:
        print(f"SLURM Job ID: {job_id}")
    
    # ========== PHASE 2: Full fine-tuning (entire model + demographics) ==========
    print("\n" + "="*60)
    print("PHASE 2: FULL FINE-TUNING (Pretrained + Demographics)")
    print("="*60)
    print("WCR encoder: UNFROZEN (layer-dependent LR)")
    print("Input adapter: UNFROZEN")
    print("Trainable: Entire model (backbone + shared layers + heads)")
    print("Demographics: Enabled (Age & Sex)")
    print("="*60)
    
    checkpoint_dir = Path(config.get("checkpoint", {}).get("save_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== PHASE 2: Full fine-tuning ==========
    trainer_phase2 = DeepECG_SL_Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
        phase=2,  # Full fine-tuning (entire model)
    )
    
    trainer_phase2.config_paths = {
        "model": str(model_config_path.resolve()),
    }
    trainer_phase2.job_id = job_id
    
    # Train Phase 2 (no Phase 1 checkpoint - start fresh with unfrozen encoder)
    history_phase2 = trainer_phase2.train_phase2(
        phase1_checkpoint_path=None,
        max_epochs=30,
        patience=15,
    )
    
    print("\nPhase 2 completed!")
    print(f"Best validation loss: {min(history_phase2.get('val_loss', [float('inf')])):.4f}")
    print(f"Best validation MAE: {min(history_phase2.get('val_los_mae', [float('inf')])):.4f} days")
    
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
    print("FULL FINE-TUNING COMPLETED (Pretrained + Demographics)")
    print("="*60)
    print(f"Phase 2 checkpoint: {phase2_checkpoint_path}")
    
    return history_phase2


if __name__ == "__main__":
    main()
