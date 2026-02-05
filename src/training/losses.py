"""Loss functions for training."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(config: Dict[str, Any]) -> nn.Module:
    """Get loss function from config.
    
    Args:
        config: Configuration dictionary with loss settings.
    
    Returns:
        Loss function module.
    """
    loss_config = config.get("training", {}).get("loss", {})
    loss_type = loss_config.get("type", "mse")  # Default to MSE for regression
    
    # Check task type - regression or classification
    data_config = config.get("data", {})
    task_type = data_config.get("task_type", "regression")  # Default to regression
    
    if task_type == "regression" or loss_type == "mse":
        # MSE Loss for regression
        return nn.MSELoss()
    
    elif loss_type == "l1" or loss_type == "mae":
        # L1 Loss (Mean Absolute Error) for regression
        return nn.L1Loss()
    
    elif loss_type == "huber":
        # Huber Loss (smooth L1) for regression - robust to outliers
        delta = loss_config.get("delta", 1.0)
        return nn.HuberLoss(delta=delta)
    
    elif loss_type == "cross_entropy":
        # Standard cross-entropy loss (for backward compatibility with classification)
        use_weighted = loss_config.get("enabled", True)
        weight = None
        if use_weighted:
            weight = loss_config.get("weight", None)
            if weight is not None:
                weight = torch.tensor(weight, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)
    
    elif loss_type == "weighted_ce":
        # Weighted cross-entropy (for backward compatibility with classification)
        use_weighted = loss_config.get("enabled", True)
        weight = None
        if use_weighted:
            weight = loss_config.get("weight", None)
            if weight is not None:
                weight = torch.tensor(weight, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)
    
    else:
        # Default to MSE for regression
        return nn.MSELoss()


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning (LOS Regression + Mortality Classification).
    
    total_loss = alpha * los_loss + beta * mortality_loss
    
    LOS: Regression task using MSE/L1 loss
    Mortality: Binary classification using BCE loss
    """
    
    def __init__(
        self,
        los_loss_weight: float = 1.0,
        mortality_loss_weight: float = 1.0,
        los_loss_type: str = "mse",
        mortality_use_weighted: bool = False,
        mortality_pos_weight: Optional[float] = None,
    ):
        """Initialize multi-task loss.
        
        Args:
            los_loss_weight: Weight for LOS regression loss.
            mortality_loss_weight: Weight for mortality prediction loss.
            los_loss_type: Type of LOS loss ("mse", "l1", "huber").
            mortality_use_weighted: Whether to use weighted BCE for mortality.
            mortality_pos_weight: Positive class weight for mortality (if weighted).
        """
        super().__init__()
        self.los_loss_weight = los_loss_weight
        self.mortality_loss_weight = mortality_loss_weight
        self.los_loss_type = los_loss_type
        
        # LOS loss (regression)
        if los_loss_type == "mse":
            self.los_loss_fn = nn.MSELoss(reduction='none')
        elif los_loss_type == "l1" or los_loss_type == "mae":
            self.los_loss_fn = nn.L1Loss(reduction='none')
        elif los_loss_type == "huber":
            self.los_loss_fn = nn.HuberLoss(reduction='none')
        else:
            self.los_loss_fn = nn.MSELoss(reduction='none')
        
        # Mortality loss (binary classification)
        self.mortality_use_weighted = mortality_use_weighted
        self.mortality_pos_weight = mortality_pos_weight
    
    def forward(
        self,
        los_predictions: torch.Tensor,
        los_labels: torch.Tensor,
        mortality_probs: torch.Tensor,
        mortality_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined multi-task loss.
        
        Args:
            los_predictions: LOS predictions of shape (B, 1) or (B,) - continuous values in days.
            los_labels: LOS labels of shape (B,) - continuous values in days (float32).
            mortality_probs: Mortality probabilities of shape (B, 1) in range [0, 1].
            mortality_labels: Mortality labels of shape (B,) with values 0 or 1.
        
        Returns:
            Dictionary with:
                - 'total': Combined loss (scalar tensor).
                - 'los': LOS loss (scalar tensor).
                - 'mortality': Mortality loss (scalar tensor).
        """
        # Ensure los_predictions is (B,) for loss computation
        if los_predictions.dim() > 1:
            los_predictions = los_predictions.squeeze(-1)
        
        # Filter valid samples for LOS (los_label >= 0 means it's available)
        # For regression, we use a threshold like -1 to indicate invalid
        los_valid_mask = los_labels >= 0
        
        if los_valid_mask.any():
            los_predictions_valid = los_predictions[los_valid_mask]
            los_labels_valid = los_labels[los_valid_mask].float()
            
            # Compute LOS regression loss
            los_loss_per_sample = self.los_loss_fn(los_predictions_valid, los_labels_valid)
            los_loss_value = los_loss_per_sample.mean()
        else:
            los_loss_value = torch.tensor(0.0, device=los_predictions.device)
        
        # Filter valid samples for mortality (mortality_label >= 0 means it's available)
        mortality_valid_mask = mortality_labels >= 0
        
        if mortality_valid_mask.any():
            # Ensure mortality_probs and mortality_labels have compatible shapes
            mortality_probs_flat = mortality_probs.squeeze(-1) if mortality_probs.dim() > 1 else mortality_probs
            mortality_probs_valid = mortality_probs_flat[mortality_valid_mask].float()
            mortality_labels_valid = mortality_labels[mortality_valid_mask].float()
            
            # Use F.binary_cross_entropy with manual pos_weight implementation
            if self.mortality_use_weighted and self.mortality_pos_weight is not None:
                # Manual weighted BCE: loss = -[w_pos * y * log(p) + (1-y) * log(1-p)]
                pos_weight = torch.tensor(self.mortality_pos_weight, dtype=torch.float32, device=mortality_probs_valid.device)
                # Clamp probabilities to avoid log(0)
                mortality_probs_valid = torch.clamp(mortality_probs_valid, min=1e-7, max=1-1e-7)
                loss_pos = pos_weight * mortality_labels_valid * torch.log(mortality_probs_valid)
                loss_neg = (1 - mortality_labels_valid) * torch.log(1 - mortality_probs_valid)
                mortality_loss_value = -(loss_pos + loss_neg).mean()
            else:
                mortality_loss_value = F.binary_cross_entropy(
                    mortality_probs_valid,
                    mortality_labels_valid
                )
        else:
            mortality_loss_value = torch.tensor(0.0, device=mortality_probs.device)
        
        # Combined loss
        total_loss = (
            self.los_loss_weight * los_loss_value +
            self.mortality_loss_weight * mortality_loss_value
        )
        
        return {
            "total": total_loss,
            "los": los_loss_value,
            "mortality": mortality_loss_value,
        }


def get_multi_task_loss(config: Dict[str, Any]) -> MultiTaskLoss:
    """Get multi-task loss function from config.
    
    Args:
        config: Configuration dictionary with multi-task and loss settings.
    
    Returns:
        MultiTaskLoss instance.
    """
    multi_task_config = config.get("multi_task", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    
    # Get loss weights
    los_loss_weight = multi_task_config.get("los_loss_weight", 1.0)
    mortality_loss_weight = multi_task_config.get("mortality_loss_weight", 1.0)
    
    # Get LOS loss type (default to MSE for regression)
    los_loss_config = training_config.get("loss", {})
    los_loss_type = los_loss_config.get("type", "mse")
    
    # Check task type - if classification, use cross_entropy compatible loss
    task_type = data_config.get("task_type", "regression")
    if task_type == "classification":
        # For backward compatibility, but we're now using regression
        los_loss_type = "mse"  # Still use MSE as a fallback
    
    # Get mortality loss settings
    mortality_use_weighted = multi_task_config.get("mortality_use_weighted_loss", False)
    mortality_pos_weight = multi_task_config.get("mortality_pos_weight", None)
    
    return MultiTaskLoss(
        los_loss_weight=los_loss_weight,
        mortality_loss_weight=mortality_loss_weight,
        los_loss_type=los_loss_type,
        mortality_use_weighted=mortality_use_weighted,
        mortality_pos_weight=mortality_pos_weight,
    )
