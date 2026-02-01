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
    loss_type = loss_config.get("type", "cross_entropy")
    
    # Check if weighted classes are enabled
    use_weighted = loss_config.get("enabled", True)  # Default: enabled for backward compatibility
    
    if loss_type == "cross_entropy":
        # Standard cross-entropy loss
        weight = None
        if use_weighted:
            weight = loss_config.get("weight", None)
            if weight is not None:
                weight = torch.tensor(weight, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)
    
    elif loss_type == "focal":
        # Focal loss (if needed later)
        alpha = loss_config.get("alpha", 1.0)
        gamma = loss_config.get("gamma", 2.0)
        # TODO: Implement FocalLoss class if needed
        return nn.CrossEntropyLoss()
    
    elif loss_type == "weighted_ce":
        # Weighted cross-entropy (only if enabled)
        weight = None
        if use_weighted:
            weight = loss_config.get("weight", None)
            if weight is not None:
                weight = torch.tensor(weight, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)
    
    else:
        # Default to cross-entropy
        return nn.CrossEntropyLoss()


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning (LOS + Mortality).
    
    total_loss = alpha * los_loss + beta * mortality_loss
    """
    
    def __init__(
        self,
        los_loss_weight: float = 1.0,
        mortality_loss_weight: float = 1.0,
        los_loss: Optional[nn.Module] = None,
        mortality_use_weighted: bool = False,
        mortality_pos_weight: Optional[float] = None,
    ):
        """Initialize multi-task loss.
        
        Args:
            los_loss_weight: Weight for LOS classification loss.
            mortality_loss_weight: Weight for mortality prediction loss.
            los_loss: LOS loss function (default: CrossEntropyLoss).
            mortality_use_weighted: Whether to use weighted BCE for mortality.
            mortality_pos_weight: Positive class weight for mortality (if weighted).
        """
        super().__init__()
        self.los_loss_weight = los_loss_weight
        self.mortality_loss_weight = mortality_loss_weight
        
        # LOS loss (multi-class classification)
        # NOTE: We intentionally compute LOS loss via F.cross_entropy to safely place
        # class weights on the same device as logits (avoids CPU/GPU mismatch on cluster).
        self.los_loss = los_loss if los_loss is not None else nn.CrossEntropyLoss()
        
        # Mortality loss (binary classification)
        # Store pos_weight as a parameter (will be moved to device in forward)
        self.mortality_use_weighted = mortality_use_weighted
        self.mortality_pos_weight = mortality_pos_weight
        # We'll use F.binary_cross_entropy directly to handle pos_weight on correct device
    
    def forward(
        self,
        los_logits: torch.Tensor,
        los_labels: torch.Tensor,
        mortality_probs: torch.Tensor,
        mortality_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined multi-task loss.
        
        Args:
            los_logits: LOS logits of shape (B, num_classes).
            los_labels: LOS labels of shape (B,).
            mortality_probs: Mortality probabilities of shape (B, 1) in range [0, 1].
            mortality_labels: Mortality labels of shape (B,) with values 0 or 1.
        
        Returns:
            Dictionary with:
                - 'total': Combined loss (scalar tensor).
                - 'los': LOS loss (scalar tensor).
                - 'mortality': Mortality loss (scalar tensor).
        """
        # Filter valid samples (mortality_label >= 0 means it's available)
        valid_mask = mortality_labels >= 0
        
        # LOS loss (always computed if los_labels are valid)
        los_valid_mask = los_labels >= 0
        if los_valid_mask.any():
            los_logits_valid = los_logits[los_valid_mask]
            los_labels_valid = los_labels[los_valid_mask]

            weight = None
            # If los_loss is CrossEntropyLoss, pull its configured weight and move to device.
            if isinstance(self.los_loss, nn.CrossEntropyLoss) and getattr(self.los_loss, "weight", None) is not None:
                weight = self.los_loss.weight.to(device=los_logits_valid.device, dtype=torch.float32)

            los_loss_value = F.cross_entropy(los_logits_valid, los_labels_valid, weight=weight)
        else:
            los_loss_value = torch.tensor(0.0, device=los_logits.device)
        
        # Mortality loss (only computed if mortality labels are available)
        if valid_mask.any():
            # Ensure mortality_probs and mortality_labels have compatible shapes
            mortality_probs_flat = mortality_probs.squeeze(1) if mortality_probs.dim() > 1 else mortality_probs
            mortality_probs_valid = mortality_probs_flat[valid_mask].float()
            mortality_labels_valid = mortality_labels[valid_mask].float()
            
            # Use F.binary_cross_entropy with manual pos_weight implementation
            # F.binary_cross_entropy doesn't support pos_weight, so we implement it manually
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
    
    # Get loss weights
    los_loss_weight = multi_task_config.get("los_loss_weight", 1.0)
    mortality_loss_weight = multi_task_config.get("mortality_loss_weight", 1.0)
    
    # Get LOS loss (can be weighted)
    los_loss_config = training_config.get("loss", {})
    los_loss_type = los_loss_config.get("type", "cross_entropy")
    use_weighted = los_loss_config.get("enabled", True)  # Default: enabled for backward compatibility
    
    los_weight = None
    if use_weighted:
        los_weight = los_loss_config.get("weight", None)
        if los_weight is not None:
            los_weight = torch.tensor(los_weight, dtype=torch.float32)
    
    if los_loss_type in ["cross_entropy", "weighted_ce"]:
        los_loss = nn.CrossEntropyLoss(weight=los_weight)
    else:
        los_loss = nn.CrossEntropyLoss()
    
    # Get mortality loss settings
    mortality_use_weighted = multi_task_config.get("mortality_use_weighted_loss", False)
    mortality_pos_weight = multi_task_config.get("mortality_pos_weight", None)
    
    return MultiTaskLoss(
        los_loss_weight=los_loss_weight,
        mortality_loss_weight=mortality_loss_weight,
        los_loss=los_loss,
        mortality_use_weighted=mortality_use_weighted,
        mortality_pos_weight=mortality_pos_weight,
    )