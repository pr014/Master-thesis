"""Base trainer class for model training."""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.device import get_device, set_seed
from ..utils.logger import setup_logger, TensorBoardLogger
from ..utils.config_loader import load_config
from .train_loop import train_epoch, validate_epoch
from .losses import get_loss
from .callbacks import EarlyStopping, ModelCheckpoint


class Trainer:
    """Trainer for ECG models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Configuration dictionary.
            device: Device to train on (auto-detected if None).
            criterion: Optional loss function (if None, will be created from config).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device
        if device is None:
            device_config = config.get("device", {})
            device_str = device_config.get("device")
            self.device = get_device(device_str)
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Set seed for reproducibility
        seed = config.get("seed", 42)
        set_seed(seed)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss (use provided criterion or create from config)
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = get_loss(config)
        
        # Move loss weights to device if they exist
        if hasattr(self.criterion, 'weight') and self.criterion.weight is not None:
            self.criterion.weight = self.criterion.weight.to(self.device)
        
        # Setup callbacks
        self.early_stopping = EarlyStopping(config)
        model_name = config.get("model", {}).get("type", "model")
        self.checkpoint = ModelCheckpoint(config, model_name=model_name)
        
        # Store config paths for checkpoint saving (set by training script)
        self.config_paths = {}
        
        # Store job ID if available (from SLURM environment)
        self.job_id = None
        
        # Setup logging
        log_config = config.get("logging", {})
        log_dir = Path(log_config.get("log_dir", "outputs/logs"))
        self.logger = setup_logger("training", log_dir=log_dir)
        self.tb_logger = TensorBoardLogger(log_dir / "tensorboard") if log_config.get("use_tensorboard", True) else None
        
        # Training state
        self.current_epoch = 0
        self.history = {
            "train_loss": [],
            "train_los_acc": [],
            "val_loss": [],
            "val_los_acc": [],
            # Optional multi-task metrics (will be populated if present)
            "train_mortality_acc": [],
            "val_mortality_acc": [],
            "val_mortality_auc": [],
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_config = self.config.get("training", {}).get("optimizer", {})
        opt_type = opt_config.get("type", "Adam").lower()
        lr = opt_config.get("lr", 5e-4)
        # Ensure lr is a float (YAML might load scientific notation as string)
        if isinstance(lr, str):
            lr = float(lr)
        weight_decay = opt_config.get("weight_decay", 1e-4)
        # Ensure weight_decay is a float
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        
        if opt_type == "adam":
            betas = opt_config.get("betas", [0.9, 0.999])
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        elif opt_type == "sgd":
            momentum = opt_config.get("momentum", 0.9)
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        elif opt_type == "adamw":
            betas = opt_config.get("betas", [0.9, 0.999])
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        else:
            # Default to Adam
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config."""
        sched_config = self.config.get("training", {}).get("scheduler", {})
        sched_type = sched_config.get("type")
        
        if sched_type is None:
            return None
        
        if sched_type == "ReduceLROnPlateau":
            factor = sched_config.get("factor", 0.1)
            # Ensure factor is a float
            if isinstance(factor, str):
                factor = float(factor)
            patience = sched_config.get("patience", 5)
            min_lr = sched_config.get("min_lr", 1e-6)
            # Ensure min_lr is a float (YAML might load scientific notation as string)
            if isinstance(min_lr, str):
                min_lr = float(min_lr)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )
        elif sched_type == "cosine":
            T_max = self.config.get("training", {}).get("num_epochs", 50)
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        else:
            return None
    
    def train(self) -> Dict[str, list]:
        """Train the model.
        
        Returns:
            Training history dictionary.
        """
        num_epochs = self.config.get("training", {}).get("num_epochs", 50)
        val_frequency = self.config.get("validation", {}).get("val_frequency", 1)
        log_frequency = self.config.get("logging", {}).get("log_frequency", 10)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Training
            train_metrics = train_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
                self.config,
            )
            
            # Validation
            if epoch % val_frequency == 0:
                val_metrics = validate_epoch(
                    self.model,
                    self.val_loader,
                    self.criterion,
                    self.device,
                )
            else:
                val_metrics = {}
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_metric = self.early_stopping.monitor
                    metric_value = all_metrics.get(monitor_metric, train_metrics.get("train_loss", 0.0))
                    self.scheduler.step(metric_value)
                else:
                    self.scheduler.step()
            
            # Update history
            for key, value in all_metrics.items():
                if key in self.history:
                    self.history[key].append(value)
                else:
                    self.history[key] = [value]
            
            # Logging
            if epoch % log_frequency == 0 or epoch == 1:
                log_msg = (
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_metrics.get('train_loss', 0.0):.4f}, "
                    f"Train LOS Acc: {train_metrics.get('train_los_acc', 0.0):.4f}, "
                    f"Val Loss: {val_metrics.get('val_loss', 0.0):.4f}, "
                    f"Val LOS Acc: {val_metrics.get('val_los_acc', 0.0):.4f}"
                )
                self.logger.info(log_msg)
                # Also print to stdout (captured by SLURM) for parsing
                print(log_msg)

                # Extra logging if mortality metrics are available
                if "val_mortality_acc" in val_metrics or "val_mortality_auc" in val_metrics:
                    mort_msg = (
                        f"           Mortality - "
                        f"Val Acc: {val_metrics.get('val_mortality_acc', 0.0):.4f}, "
                        f"Val AUC: {val_metrics.get('val_mortality_auc', 0.0):.4f}"
                    )
                    self.logger.info(mort_msg)
                    # Also print to stdout (captured by SLURM) for parsing
                    print(mort_msg)
            
            # TensorBoard logging
            if self.tb_logger is not None:
                for key, value in all_metrics.items():
                    self.tb_logger.log_scalar(key, value, epoch)
            
            # Checkpointing
            is_best = False
            if val_metrics:
                monitor_metric = self.early_stopping.monitor
                current_value = all_metrics.get(monitor_metric)
                if current_value is not None:
                    if self.early_stopping.mode == "min":
                        is_best = current_value < self.checkpoint.best_value
                    else:
                        is_best = current_value > self.checkpoint.best_value
            
            self.checkpoint.save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                all_metrics,
                is_best=is_best,
                config=self.config,
                config_paths=self.config_paths,
                job_id=getattr(self, 'job_id', None),
            )
            
            # Early stopping
            if val_metrics and self.early_stopping(all_metrics, epoch):
                self.logger.info(f"Early stopping at epoch {epoch}")
                self.logger.info(f"Best epoch: {self.early_stopping.get_best_epoch()}")
                break
        
        # Close TensorBoard logger
        if self.tb_logger is not None:
            self.tb_logger.close()
        
        self.logger.info("Training completed")
        return self.history
