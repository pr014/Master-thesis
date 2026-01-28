"""Base model class for all ECG classification models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseECGModel(nn.Module, ABC):
    """Abstract base class for all ECG classification models.
    
    All model implementations must inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the base model.
        
        Args:
            config: Configuration dictionary containing model hyperparameters.
        """
        super().__init__()
        self.config = config
        self.num_classes = config.get("num_classes", 2)
    
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
            logits: Output logits tensor of shape (B, num_classes)
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions from input.
        
        Args:
            x: Input tensor (same format as forward()).
        
        Returns:
            predictions: Class indices of shape (B,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities from input.
        
        Args:
            x: Input tensor (same format as forward()).
        
        Returns:
            probabilities: Class probabilities of shape (B, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities
    
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
        return {
            "model_type": self.__class__.__name__,
            "num_parameters": self.count_parameters(),
            "num_classes": self.num_classes,
            "config": self.config,
        }
    
    def get_features(self, x: torch.Tensor, demographic_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract features from input without final classification head.
        
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