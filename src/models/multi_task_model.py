"""Multi-Task ECG Model wrapper for LOS classification + Mortality prediction."""

from typing import Dict, Any
import torch
import torch.nn as nn
from .base_model import BaseECGModel


class MultiTaskECGModel(nn.Module):
    """Wrapper for existing ECG models to add mortality prediction head.
    
    This wrapper takes a base model (e.g., CNNScratch, ResNet1D14) and adds
    a second output head for mortality prediction while keeping the original
    LOS classification head.
    
    Architecture:
    - Shared Backbone: Base model without final FC layer
    - LOS Head: Original final FC layer from base model
    - Mortality Head: New head for binary classification (0/1)
    """
    
    def __init__(self, base_model: BaseECGModel, config: Dict[str, Any]):
        """Initialize Multi-Task model.
        
        Args:
            base_model: Base ECG model (must implement get_features() method).
            config: Configuration dictionary. Should contain:
                - training.dropout_rate: Dropout rate for mortality head
                - multi_task: Optional multi-task specific config
        """
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Get dropout rate from config
        training_config = config.get("training", {})
        dropout_rate = training_config.get("dropout_rate", 0.3)
        
        # Get feature dimension by running a dummy forward pass
        # This works for models that implement get_features()
        dummy_input = torch.zeros(1, 12, 5000)
        with torch.no_grad():
            try:
                features = base_model.get_features(dummy_input)
                feature_dim = features.shape[1]
            except NotImplementedError:
                raise NotImplementedError(
                    f"{base_model.__class__.__name__} does not support multi-task learning. "
                    "The model must implement get_features() method."
                )
        
        # Extract LOS head from base model
        # For CNNScratch: fc2 is the LOS head
        # For ResNet1D14: fc is the LOS head
        # We need to identify the final layer dynamically
        self.los_head = self._extract_los_head(base_model, feature_dim)
        
        # Create mortality head
        # Architecture: Dropout -> Linear -> Sigmoid
        self.mortality_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()  # Output probability 0.0-1.0
        )
    
    def _extract_los_head(self, base_model: BaseECGModel, feature_dim: int) -> nn.Module:
        """Extract the LOS classification head from base model.
        
        Args:
            base_model: Base model to extract head from.
            feature_dim: Dimension of features before final head.
        
        Returns:
            LOS head module (final FC layer).
        """
        model_name = base_model.__class__.__name__
        
        if model_name == "CNNScratch":
            # For CNNScratch: fc2 is the final layer (64 -> num_classes)
            return base_model.fc2
        elif model_name == "ResNet1D14":
            # For ResNet1D14: fc is the final layer (512 -> num_classes)
            return base_model.fc
        else:
            # Generic approach: try to find the last Linear layer
            # This is a fallback for other models
            for name, module in reversed(list(base_model.named_modules())):
                if isinstance(module, nn.Linear) and module.out_features == base_model.num_classes:
                    return module
            
            raise ValueError(
                f"Could not extract LOS head from {model_name}. "
                "Please implement _extract_los_head() for this model type."
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-task model.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
        
        Returns:
            Dictionary with:
                - 'los': LOS logits of shape (B, num_classes)
                - 'mortality': Mortality probabilities of shape (B, 1) in range [0, 1]
        """
        # Extract features using base model
        features = self.base_model.get_features(x)  # (B, feature_dim)
        
        # LOS classification head
        los_logits = self.los_head(features)  # (B, num_classes)
        
        # Mortality prediction head
        mortality_probs = self.mortality_head(features)  # (B, 1)
        
        return {
            "los": los_logits,
            "mortality": mortality_probs
        }
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predictions from both tasks.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
        
        Returns:
            Dictionary with:
                - 'los': LOS class predictions of shape (B,)
                - 'mortality': Mortality binary predictions of shape (B,) (0 or 1)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            los_predictions = torch.argmax(outputs["los"], dim=1)
            mortality_predictions = (outputs["mortality"] > 0.5).long().squeeze(1)
        
        return {
            "los": los_predictions,
            "mortality": mortality_predictions
        }
    
    def predict_proba(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get probabilities from both tasks.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
        
        Returns:
            Dictionary with:
                - 'los': LOS class probabilities of shape (B, num_classes)
                - 'mortality': Mortality probabilities of shape (B, 1) in range [0, 1]
        """
        with torch.no_grad():
            outputs = self.forward(x)
            los_probs = torch.softmax(outputs["los"], dim=1)
            mortality_probs = outputs["mortality"]  # Already sigmoid
        
        return {
            "los": los_probs,
            "mortality": mortality_probs
        }
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters.
        
        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

