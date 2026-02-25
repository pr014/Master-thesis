"""Training script for HuBERT-ECG with two-phase training.

Phase 1: Frozen backbone (HuBERT encoder frozen, only heads trained)
Phase 2: Fine-tuning with layer-dependent learning rates

Based on Transfer Learning strategy for self-supervised pretrained models.

LOS Regression Task: Predicts continuous LOS in days (not binned classes).
"""

from pathlib import Path
import sys
import os
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.models import HuBERT_ECG
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.training import Trainer, setup_icustays_mapper, evaluate_and_print_results
from src.training.losses import get_multi_task_loss
from src.utils.config_loader import load_config


class HuBERT_ECG_Trainer(Trainer):
    """Specialized trainer for HuBERT-ECG with two-phase training."""
    
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
        """Initialize HuBERT-ECG trainer.
        
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
        print("HuBERT encoder: FROZEN (no gradient updates)")
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
        print("HuBERT encoder: UNFROZEN (with reduced learning rate)")
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
        # Note: HuBERT_ECG architecture has classifier_dropout + heads (no shared_bn/shared_fc)
        if self.phase == 1:
            # Phase 1: Only heads are trainable (HuBERT encoder frozen)
            head_lr = base_lr  # 1e-3 typically
            param_groups = [
                {
                    'params': list(base_model.classifier_dropout.parameters()) +
                             list(base_model.los_head.parameters()) +
                             list(base_model.mortality_head.parameters()),
                    'lr': head_lr,
                }
            ]
        else:  # Phase 2
            # Phase 2: Layer-dependent learning rates
            # HuBERT encoder: 1e-5 (100× reduction)
            # Classifier dropout + Heads: 1e-4 (10× reduction)
            backbone_lr = 1e-5
            head_lr = 1e-4
            
            param_groups = [
                {
                    'params': base_model.hubert_encoder.parameters(),
                    'lr': backbone_lr,
                },
                {
                    'params': list(base_model.classifier_dropout.parameters()) +
                             list(base_model.los_head.parameters()) +
                             list(base_model.mortality_head.parameters()),
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
    """Main training function for HuBERT-ECG with Phase 1 only (frozen backbone / head-only).
    
    Phase 1: Frozen backbone, only task heads trained.
    Phase 2: Fine-tuning (skipped in this baseline run, H1).
    
    LOS Regression Task: Predicts continuous LOS in days.
    """
    # Load config
    model_config_path = Path("configs/model/hubert_ecg/hubert_ecg.yaml")
    config = load_config(model_config_path=model_config_path)
    
    print("="*60)
    print("Training HuBERT-ECG - Phase 1 Only (Frozen Backbone / Head-Only)")
    print("Task: LOS REGRESSION (continuous prediction in days)")
    print("="*60)
    print(f"Model config: {model_config_path}")
    
    model_config = config.get('model', {})
    hubert_config = model_config.get('hubert', {})
    pretrained_config = model_config.get('pretrained', {})
    
    if pretrained_config.get('enabled', False):
        cache_dir = pretrained_config.get('cache_dir', 'data/pretrained_weights/Hubert_ECG/base')
        print(f"Pretrained weights: Enabled ({cache_dir})")
    else:
        print("Pretrained weights: Disabled (random init)")
    
    demographic_config = config.get('data', {}).get('demographic_features', {})
    print(f"Demographic features: {'Enabled' if demographic_config.get('enabled', False) else 'Disabled'}")
    
    diagnosis_config = config.get('data', {}).get('diagnosis_features', {})
    print(f"Diagnosis features: {'Enabled' if diagnosis_config.get('enabled', False) else 'Disabled'}")
    print("="*60)
    
    # Load ICU stays and create mapper
    icu_mapper = setup_icustays_mapper(config)
    
    # Ensure multi-task is enabled
    if not config.get("multi_task", {}).get("enabled", False):
        config["multi_task"] = {"enabled": True}
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        labels=None,
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,
    )
    
    # Create base model with pretrained weights + frozen backbone
    print("\nCreating HuBERT-ECG model (loading pretrained weights)...")
    base_model = HuBERT_ECG(config)
    
    total_params = base_model.count_parameters()
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} (backbone frozen, heads only)")
    
    # Wrap in MultiTaskECGModel
    model = MultiTaskECGModel(base_model, config)
    
    # Loss function
    criterion = get_multi_task_loss(config)
    
    # Get SLURM job ID
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id:
        print(f"SLURM Job ID: {job_id}")
    
    # ========== PHASE 1: Frozen Backbone (Head-Only Training) ==========
    print("\n" + "="*60)
    print("PHASE 1: FROZEN BACKBONE - HEAD-ONLY TRAINING")
    print("="*60)
    
    trainer_phase1 = HuBERT_ECG_Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
        phase=1,
    )
    trainer_phase1.config_paths = {"model": str(model_config_path.resolve())}
    trainer_phase1.job_id = job_id
    
    # Phase 1: max 50 epochs (full num_epochs from config), patience 20
    history_phase1 = trainer_phase1.train_phase1(
        max_epochs=config.get("training", {}).get("num_epochs", 50),
        patience=config.get("early_stopping", {}).get("patience", 20),
    )
    
    print("\nPhase 1 completed!")
    print(f"Best validation loss: {min(history_phase1.get('val_loss', [float('inf')])):.4f}")
    print(f"Best validation MAE:  {min(history_phase1.get('val_los_mae', [float('inf')])):.4f} days")
    
    # Save Phase 1 checkpoint
    checkpoint_dir = Path(config.get("checkpoint", {}).get("save_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    phase1_checkpoint_path = checkpoint_dir / f"hubert_ecg_phase1_best_{job_id if job_id else 'local'}.pt"
    
    save_dict = {
        'model_state_dict': (
            trainer_phase1.checkpoint.best_model_state
            if hasattr(trainer_phase1.checkpoint, 'best_model_state')
            else model.state_dict()
        ),
        'epoch': getattr(trainer_phase1.checkpoint, 'best_epoch', trainer_phase1.current_epoch),
        'val_loss': getattr(trainer_phase1.checkpoint, 'best_score', None),
        'history': history_phase1,
        'config': config,
    }
    torch.save(save_dict, phase1_checkpoint_path)
    print(f"Saved Phase 1 checkpoint to: {phase1_checkpoint_path}")
    
    # ========== FINAL EVALUATION ==========
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load best Phase 1 model
    if phase1_checkpoint_path.exists():
        checkpoint = torch.load(phase1_checkpoint_path, map_location=trainer_phase1.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(trainer_phase1.device)
        print(f"Loaded best Phase 1 model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    eval_trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
    )
    eval_trainer.job_id = job_id
    
    history_final = evaluate_and_print_results(eval_trainer, test_loader, history_phase1, config)
    
    print("\n" + "="*60)
    print("PHASE 1 (HEAD-ONLY) TRAINING COMPLETED")
    print("="*60)
    print(f"Checkpoint: {phase1_checkpoint_path}")
    
    return history_phase1


if __name__ == "__main__":
    main()
