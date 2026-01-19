"""Loss functions for training."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


def get_loss(config: Dict[str, Any]) -> nn.Module:
    """Get loss function from config.
    
    Args:
        config: Configuration dictionary with loss settings.
    
    Returns:
        Loss function module.
    """
    loss_config = config.get("training", {}).get("loss", {})
    loss_type = loss_config.get("type", "cross_entropy")
    
    if loss_type == "cross_entropy":
        # Standard cross-entropy loss
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
        # Weighted cross-entropy
        weight = loss_config.get("weight", None)
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)
    
    else:
        # Default to cross-entropy
        return nn.CrossEntropyLoss()
