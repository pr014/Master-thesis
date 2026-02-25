"""Training callbacks (EarlyStopping, Checkpointing, etc.)."""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
import random


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize early stopping.
        
        Args:
            config: Configuration dictionary with early_stopping settings.
        """
        early_stop_config = config.get("early_stopping", {})
        self.enabled = early_stop_config.get("enabled", True)
        self.patience = early_stop_config.get("patience", 7)
        self.monitor = early_stop_config.get("monitor", "val_loss")
        self.mode = early_stop_config.get("mode", "min")
        
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.best_epoch = 0
    
    def __call__(self, metrics: Dict[str, float], epoch: int) -> bool:
        """Check if training should stop.
        
        Args:
            metrics: Dictionary of metrics from current epoch.
            epoch: Current epoch number.
        
        Returns:
            True if training should stop, False otherwise.
        """
        if not self.enabled:
            return False
        
        current_value = metrics.get(self.monitor, None)
        if current_value is None:
            return False
        
        # Check if improved
        if self.mode == "min":
            improved = current_value < self.best_value
        else:  # mode == "max"
            improved = current_value > self.best_value
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        # Stop if patience exceeded
        return self.counter >= self.patience
    
    def get_best_epoch(self) -> int:
        """Get epoch with best metric value."""
        return self.best_epoch


class ModelCheckpoint:
    """Model checkpointing callback."""
    
    def __init__(self, config: Dict[str, Any], model_name: str = "model"):
        """Initialize model checkpointing.
        
        Args:
            config: Configuration dictionary with checkpoint settings.
            model_name: Name prefix for checkpoint files.
        """
        checkpoint_config = config.get("checkpoint", {})
        self.save_dir = Path(checkpoint_config.get("save_dir", "outputs/checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = checkpoint_config.get("save_frequency", 1)
        self.save_best = checkpoint_config.get("save_best", True)
        self.metric = checkpoint_config.get("metric", "val_loss")
        
        self.model_name = model_name
        self.best_value = float("inf")
        self.best_epoch = 0
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        config: Optional[Dict[str, Any]] = None,
        config_paths: Optional[Dict[str, str]] = None,
        job_id: Optional[str] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        history: Optional[Dict[str, list]] = None,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optimizer state.
            epoch: Current epoch.
            metrics: Current metrics.
            is_best: Whether this is the best model so far.
            config: Full configuration dictionary to save.
            config_paths: Dictionary with config file paths (e.g., {"base": "...", "model": "..."}).
            job_id: SLURM job ID (if available).
            scheduler: Learning rate scheduler (optional, for resume training).
            history: Training history dictionary (optional, for resume training).
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        
        # Add scheduler state (CRITICAL for resume training)
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            # For ReduceLROnPlateau, we also need to track the last metric value
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                checkpoint["scheduler_last_metric"] = getattr(scheduler, 'best', None)
        
        # Add random states for reproducibility
        checkpoint["random_states"] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        
        # Add training history (useful for resume and visualization)
        if history is not None:
            checkpoint["history"] = history
        
        # Add config if provided
        if config is not None:
            checkpoint["config"] = config
        
        # Add config paths if provided
        if config_paths is not None:
            checkpoint["config_paths"] = config_paths
        
        # Add job ID if provided
        if job_id is not None:
            checkpoint["job_id"] = job_id
        
        # Save regular checkpoint
        if epoch % self.save_frequency == 0:
            checkpoint_path = self.save_dir / f"{self.model_name}_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best and self.save_best:
            # Only save with job_id for 100% traceability
            if job_id is not None:
                best_path_with_job = self.save_dir / f"{self.model_name}_best_{job_id}.pt"
                # Check if file already exists
                if best_path_with_job.exists():
                    # Allow overwriting if it's from the same job (e.g., better model found later)
                    try:
                        existing_checkpoint = torch.load(best_path_with_job, map_location="cpu")
                        existing_job_id = existing_checkpoint.get("job_id")
                        if existing_job_id != job_id:
                            # Different job ID - prevent overwriting for traceability
                            raise FileExistsError(
                                f"Checkpoint already exists with different job_id: {best_path_with_job}. "
                                f"Existing job_id: {existing_job_id}, Current job_id: {job_id}. "
                                f"This should not happen. Check for duplicate job IDs."
                            )
                        # Same job ID - allow overwriting (better model found)
                        print(f"Overwriting best checkpoint from same job: {best_path_with_job}")
                    except Exception as e:
                        # If we can't read the checkpoint, be safe and raise error
                        raise FileExistsError(
                            f"Checkpoint already exists but could not verify job_id: {best_path_with_job}. "
                            f"Error: {e}"
                        )
                torch.save(checkpoint, best_path_with_job)
                print(f"Saved best model checkpoint: {best_path_with_job}")
            else:
                # No fallback - job_id is required for traceability
                raise ValueError(
                    "No job_id available. Cannot save checkpoint without job_id for traceability. "
                    "Ensure SLURM_JOB_ID environment variable is set."
                )
            
            self.best_value = metrics.get(self.metric, float("inf"))
            self.best_epoch = epoch
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        
        Returns:
            Checkpoint dictionary.
        """
        return torch.load(checkpoint_path, map_location="cpu")
    
    @staticmethod
    def restore_random_states(checkpoint: Dict[str, Any]) -> None:
        """Restore random states from checkpoint for reproducibility.
        
        Args:
            checkpoint: Checkpoint dictionary containing random_states.
        """
        if "random_states" in checkpoint:
            random_states = checkpoint["random_states"]
            if "torch" in random_states:
                torch.set_rng_state(random_states["torch"])
            if "numpy" in random_states:
                np.random.set_state(random_states["numpy"])
            if "python" in random_states:
                random.setstate(random_states["python"])
