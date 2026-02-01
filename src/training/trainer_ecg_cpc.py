"""Specialized trainer for ECG-CPC with two-phase training and layer-dependent learning rates.

Based on Al-Masud et al. (2025): "Benchmarking ECG Foundational Models"
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer import Trainer
from ..models.ecg_cpc import ECG_S4_CPC


class ECG_CPC_Trainer(Trainer):
    """Specialized trainer for ECG-CPC with two-phase training.
    
    Phase 1: Frozen backbone (S4 encoder frozen, only heads trained)
    Phase 2: Fine-tuning with layer-dependent learning rates
    """
    
    def __init__(
        self,
        model: ECG_S4_CPC,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None,
        phase: int = 1,
        wrapped_model: Optional[nn.Module] = None,
    ):
        """Initialize ECG-CPC trainer.
        
        Args:
            model: ECG_S4_CPC base model instance (for freeze/unfreeze)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on
            criterion: Loss function (MultiTaskLoss)
            phase: Training phase (1: frozen, 2: fine-tuning)
            wrapped_model: Optional wrapped model (MultiTaskECGModel) for training loop
        """
        self.phase = phase
        self.base_model = model  # Store base model for freeze/unfreeze
        self.wrapped_model = wrapped_model  # Store wrapped model for training
        
        # Use wrapped model if provided, otherwise use base model
        training_model = wrapped_model if wrapped_model is not None else model
        
        # Initialize base trainer (will create optimizer, but we'll override it)
        super().__init__(training_model, train_loader, val_loader, config, device, criterion)
        
        # Override optimizer with layer-dependent learning rates (use base_model)
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
        print("RNN encoder: FROZEN (no gradient updates)")
        print("S4 encoder: FROZEN (no gradient updates)")
        print("Trainable: Only classification heads")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print("="*60)
        
        # Freeze RNN and S4 encoders
        self.base_model.freeze_backbone()
    
    def _setup_phase2(self) -> None:
        """Setup Phase 2: Fine-tuning with layer-dependent learning rates."""
        print("="*60)
        print("Phase 2: Fine-tuning with Layer-Dependent Learning Rates")
        print("="*60)
        print("RNN encoder: UNFROZEN (with reduced learning rate)")
        print("S4 encoder: UNFROZEN (with reduced learning rate)")
        print("Learning rates:")
        for i, group in enumerate(self.optimizer.param_groups):
            print(f"  Group {i}: lr={group['lr']:.2e}, params={len(group['params'])}")
        print("="*60)
        
        # Unfreeze RNN and S4 encoders
        self.base_model.unfreeze_backbone()
    
    def _create_layer_dependent_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with layer-dependent learning rates.
        
        Phase 1: All trainable params use base_lr (heads only, S4 frozen)
        Phase 2: S4 encoder uses lr=1e-5, heads use lr=1e-4
        
        Returns:
            Optimizer with parameter groups for different learning rates
        """
        opt_config = self.config.get("training", {}).get("optimizer", {})
        opt_type = opt_config.get("type", "Adam").lower()
        base_lr = opt_config.get("lr", 5e-4)
        if isinstance(base_lr, str):
            base_lr = float(base_lr)
        weight_decay = opt_config.get("weight_decay", 1e-4)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        
        # Phase-specific learning rates (Al-Masud et al. 2025)
        # Use base_model for parameter groups
        if self.phase == 1:
            # Phase 1: Only heads are trainable (RNN + S4 frozen)
            # Use base_lr for all trainable parameters
            head_lr = base_lr  # 1e-3 typically
            param_groups = [
                {
                    'params': list(self.base_model.shared_bn.parameters()) +
                             list(self.base_model.shared_fc.parameters()) +
                             list(self.base_model.los_head.parameters()) +
                             list(self.base_model.mortality_head.parameters()),
                    'lr': head_lr,
                }
            ]
        else:  # Phase 2
            # Phase 2: Layer-dependent learning rates
            # RNN encoder: 1e-5 (100× reduction)
            # S4 encoder: 1e-5 (100× reduction)
            # Heads: 1e-4 (10× reduction from Phase 1)
            backbone_lr = 1e-5
            head_lr = 1e-4
            
            param_groups = [
                {
                    'params': self.base_model.rnn_encoder.parameters(),
                    'lr': backbone_lr,
                },
                {
                    'params': self.base_model.s4_encoder.parameters(),
                    'lr': backbone_lr,
                },
                {
                    'params': list(self.base_model.shared_bn.parameters()) +
                             list(self.base_model.shared_fc.parameters()),
                    'lr': head_lr,
                },
                {
                    'params': self.base_model.los_head.parameters(),
                    'lr': head_lr,
                },
                {
                    'params': self.base_model.mortality_head.parameters(),
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
            # Default to AdamW
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay, betas=betas)
    
    def train_phase1(
        self,
        max_epochs: int = 20,
        patience: int = 20,
    ) -> Dict[str, Any]:
        """Train Phase 1: Frozen backbone.
        
        Args:
            max_epochs: Maximum number of epochs (default: 20)
            patience: Early stopping patience (default: 20)
            
        Returns:
            Training history dictionary
        """
        if self.phase != 1:
            raise ValueError("train_phase1() can only be called when phase=1")
        
        # Update early stopping patience
        self.early_stopping.patience = patience
        
        # Update max epochs in config for this phase
        original_max_epochs = self.config.get("training", {}).get("num_epochs", 50)
        self.config["training"]["num_epochs"] = max_epochs
        
        # Train
        history = self.train()
        
        # Restore original max_epochs
        self.config["training"]["num_epochs"] = original_max_epochs
        
        return history
    
    def train_phase2(
        self,
        phase1_checkpoint_path: Optional[Path] = None,
        max_epochs: int = 30,
        patience: int = 15,
    ) -> Dict[str, Any]:
        """Train Phase 2: Fine-tuning.
        
        Args:
            phase1_checkpoint_path: Path to Phase 1 best checkpoint (optional)
            max_epochs: Maximum number of epochs (default: 30)
            patience: Early stopping patience (default: 15)
            
        Returns:
            Training history dictionary
        """
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
        
        # Update early stopping patience
        self.early_stopping.patience = patience
        
        # Update max epochs in config for this phase
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
        
        # Train
        history = self.train()
        
        # Restore original max_epochs
        self.config["training"]["num_epochs"] = original_max_epochs
        
        return history

