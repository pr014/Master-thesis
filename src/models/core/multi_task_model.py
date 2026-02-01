"""Multi-Task ECG Model wrapper for LOS classification + Mortality prediction."""

from typing import Dict, Any, Optional
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
        # Set model to eval mode to avoid BatchNorm issues with single sample
        was_training = base_model.training
        base_model.eval()
        
        # Get device from model parameters (to ensure dummy_input is on same device)
        device = next(base_model.parameters()).device
        dummy_input = torch.zeros(1, 12, 5000, device=device)
        
        # Check if demographic features are enabled
        data_config = config.get("data", {})
        demographic_config = data_config.get("demographic_features", {})
        use_demographics = demographic_config.get("enabled", False)
        dummy_demographic_features = None
        if use_demographics:
            sex_encoding = demographic_config.get("sex_encoding", "binary")
            demo_dim = 2 if sex_encoding == "binary" else 3
            dummy_demographic_features = torch.zeros(1, demo_dim, device=device)
        
        with torch.no_grad():
            try:
                features = base_model.get_features(dummy_input, demographic_features=dummy_demographic_features)
                feature_dim = features.shape[1]
            except NotImplementedError:
                raise NotImplementedError(
                    f"{base_model.__class__.__name__} does not support multi-task learning. "
                    "The model must implement get_features() method."
                )
        # Restore original training mode
        if was_training:
            base_model.train()
        
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
        elif model_name == "XResNet1D101":
            # For XResNet1D101: fc is the final layer (512 -> num_classes)
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
    
    def forward(self, x: torch.Tensor, demographic_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-task model.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
            demographic_features: Optional tensor of shape (B, 2) or (B, 3) containing Age & Sex.
                                 None if demographic features are disabled.
        
        Returns:
            Dictionary with:
                - 'los': LOS logits of shape (B, num_classes)
                - 'mortality': Mortality probabilities of shape (B, 1) in range [0, 1]
        """
        # Extract features using base model (includes demographic features if enabled)
        features = self.base_model.get_features(x, demographic_features=demographic_features)  # (B, feature_dim)
        
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

