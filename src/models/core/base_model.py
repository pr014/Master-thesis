"""Base model class for all ECG models (supports regression and classification)."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseECGModel(nn.Module, ABC):
    """Abstract base class for all ECG models.
    
    Supports both regression (LOS prediction in days) and classification tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the base model.
        
        Args:
            config: Configuration dictionary containing model hyperparameters.
        """
        super().__init__()
        self.config = config
        
        # Get task type from config (default: regression)
        data_config = config.get("data", {})
        self.task_type = data_config.get("task_type", "regression")
        
        # For classification (backward compatibility), use num_classes
        # For regression, this is not used
        if self.task_type == "classification":
            self.num_classes = config.get("num_classes", 10)
        else:
            self.num_classes = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor, demographic_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor. Shape depends on model type:
               - CNN: (B, C, T) - Batch, Channels, Time
               - LSTM: (B, T, C) - Batch, Time, Channels
               - Transformer: (B, T, C) or (B, C, T) depending on architecture
            demographic_features: Optional tensor of shape (B, 2) or (B, 3) containing Age & Sex.
                                 None if demographic features are disabled.
        
        Returns:
            For regression: Output tensor of shape (B, 1) - continuous LOS prediction in days
            For classification: Output logits tensor of shape (B, num_classes)
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions from input.
        
        For regression: Returns continuous LOS values in days.
        For classification: Returns class indices.
        
        Args:
            x: Input tensor (same format as forward()).
        
        Returns:
            predictions: Predictions of shape (B,)
        """
        with torch.no_grad():
            output = self.forward(x)
            if self.task_type == "regression":
                # For regression, squeeze the output to get (B,)
                predictions = output.squeeze(-1) if output.dim() > 1 else output
            else:
                # For classification, use argmax
                predictions = torch.argmax(output, dim=1)
        return predictions
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters.
        
        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            dict: Dictionary containing model information.
        """
        info = {
            "model_type": self.__class__.__name__,
            "num_parameters": self.count_parameters(),
            "task_type": self.task_type,
            "config": self.config,
        }
        if self.task_type == "classification" and self.num_classes is not None:
            info["num_classes"] = self.num_classes
        return info
    
    def get_features(self, x: torch.Tensor, demographic_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract features from input without final prediction head.
        
        This method is used for multi-task learning where we need features
        before the final FC layer. Subclasses should override this method
        if they want to support multi-task learning.
        
        Args:
            x: Input tensor (same format as forward()).
            demographic_features: Optional tensor of shape (B, 2) or (B, 3) containing Age & Sex.
                                 None if demographic features are disabled.
        
        Returns:
            features: Feature tensor of shape (B, feature_dim) before final FC layer.
                     If demographic features are enabled, this includes the demographic features.
        
        Raises:
            NotImplementedError: If the model doesn't support feature extraction.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_features(). "
            "This method is required for multi-task learning."
        )
