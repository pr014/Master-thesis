"""HuBERT-ECG model with self-supervised pretraining for multi-task learning.

Architecture follows the ORIGINAL HuBERT-ECG paper repository:
https://github.com/Edoar-do/HuBERT-ECG

Paper: "HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications"

Adapted for multi-task learning (LOS regression + Mortality prediction).
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from ..core.base_model import BaseECGModel
from .encoder import HuBERTEncoder


class HuBERT_ECG(BaseECGModel):
    """HuBERT-ECG model for multi-task learning (LOS regression + Mortality).
    
    Architecture follows HuBERTForECGClassification from original repo:
    - Input: (B, 12, 5000) ECG signals @ 500Hz
    - HuBERT Encoder: Pretrained Transformer with mean pooling -> (B, 768)
    - Classifier dropout (following original repo)
    - Late Fusion: Concat([ECG(768), demographics(2), …]) [optional]
    - LOS Head: FC(feature_dim -> 1) for regression (continuous LOS in days)
    - Mortality Head: FC(feature_dim -> 1) + Sigmoid
    
    Output format for multi-task:
    - 'los': LOS regression prediction (continuous value in days)
    - 'mortality': Mortality probability
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize HuBERT-ECG model.
        
        Args:
            config: Configuration dictionary. Should contain:
                - model.hubert.*: HuBERT encoder configuration
                - model.pretrained.*: Pretrained weights configuration
                - training.dropout_rate: Dropout rate for classifier
                - data.demographic_features.enabled: Whether to use demographics
                - data.task_type: "regression" (default) or "classification"
        """
        super().__init__(config)
        
        # Get training config
        training_config = config.get("training", {})
        # Use classifier_dropout_prob from original repo (default: 0.1)
        dropout_rate = training_config.get("dropout_rate", 0.1)
        
        # Get model config
        model_config = config.get("model", {})
        
        # Get feature dimensions
        # HuBERT hidden_size = 768
        feature_dim = model_config.get("feature_dim", 768)
        
        data_config = config.get("data", {})
        demographic_config = data_config.get("demographic_features", {})
        self.use_demographics = demographic_config.get("enabled", False)

        icu_unit_config = data_config.get("icu_unit_features", {})
        self.use_icu_units = icu_unit_config.get("enabled", False)
        icu_unit_list = icu_unit_config.get("icu_unit_list", [])
        self.icu_unit_dim = len(icu_unit_list) if self.use_icu_units else 0

        sofa_config = data_config.get("sofa_features", {})
        self.use_sofa = sofa_config.get("enabled", False)
        sofa_columns = sofa_config.get("columns", ["sofa_total"])
        if isinstance(sofa_columns, str):
            sofa_columns = [sofa_columns]
        self.sofa_dim = len(sofa_columns) if self.use_sofa else 0

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        
        # HuBERT Encoder: (B, 12, 5000) -> (B, 768) with mean pooling
        self.hubert_encoder = HuBERTEncoder(config, device=device)
        
        # Calculate feature dimension after fusion
        fused_feature_dim = feature_dim
        if self.use_demographics:
            sex_encoding = demographic_config.get("sex_encoding", "binary")
            demo_dim = 2 if sex_encoding == "binary" else 3
            fused_feature_dim += demo_dim
        if self.use_icu_units:
            fused_feature_dim += self.icu_unit_dim
        if self.use_sofa:
            fused_feature_dim += self.sofa_dim

        # Classifier dropout (following HuBERTForECGClassification)
        self.classifier_dropout = nn.Dropout(dropout_rate)
        
        # Task-specific heads (following original: simple linear classifier without hidden layer)
        # LOS Head: Output 1 value for regression (continuous LOS in days)
        self.los_head = nn.Linear(fused_feature_dim, 1)
        self.mortality_head = nn.Linear(fused_feature_dim, 1)
        
        # Move all layers to the correct device
        self.classifier_dropout = self.classifier_dropout.to(device)
        self.los_head = self.los_head.to(device)
        self.mortality_head = self.mortality_head.to(device)
        
        # Initialize freeze state based on config
        hubert_config = model_config.get("hubert", {})
        freeze_backbone = hubert_config.get("freeze_backbone", True)
        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()
    
    def freeze_backbone(self) -> None:
        """Freeze HuBERT encoder (for transfer learning Phase 1)."""
        self.hubert_encoder.freeze()
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze HuBERT encoder (for fine-tuning Phase 2)."""
        self.hubert_encoder.unfreeze()
    
    def _forward_features(
        self,
        x: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
        icu_unit_features: Optional[torch.Tensor] = None,
        sofa_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features before task heads.
        
        Args:
            x: ECG input tensor of shape (B, 12, 5000)
            demographic_features: Optional demographic features of shape (B, 2) or (B, 3)
            icu_unit_features: Optional ICU one-hot of shape (B, icu_unit_dim)
            sofa_features: Optional SOFA vector of shape (B, sofa_dim)

        Returns:
            Features tensor of shape (B, fused_feature_dim)
        """
        # HuBERT Encoder: (B, 12, 5000) -> (B, hidden_size) with mean pooling
        x = self.hubert_encoder(x)

        # Late fusion (order aligned with Hybrid/DeepECG)
        if self.use_demographics and demographic_features is not None:
            demographic_features = demographic_features.to(x.device)
            x = torch.cat([x, demographic_features], dim=1)
        if self.use_icu_units:
            if icu_unit_features is None:
                icu_unit_features = torch.zeros(
                    x.shape[0], self.icu_unit_dim, device=x.device, dtype=x.dtype
                )
            else:
                icu_unit_features = icu_unit_features.to(x.device)
            x = torch.cat([x, icu_unit_features], dim=1)
        if self.use_sofa:
            if sofa_features is None:
                sofa_features = torch.zeros(
                    x.shape[0], self.sofa_dim, device=x.device, dtype=x.dtype
                )
            else:
                sofa_features = sofa_features.to(x.device)
            x = torch.cat([x, sofa_features], dim=1)

        # Classifier dropout (following HuBERTForECGClassification line 117)
        x = self.classifier_dropout(x)

        return x
    
    def forward(
        self,
        x: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
        icu_unit_features: Optional[torch.Tensor] = None,
        sofa_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning LOS prediction and mortality probabilities.
        
        Args:
            x: ECG input tensor of shape (B, 12, 5000)
            demographic_features: Optional demographic features of shape (B, 2) or (B, 3)
            icu_unit_features: Optional ICU unit one-hot
            sofa_features: Optional SOFA features

        Returns:
            Dictionary with:
                - 'los': LOS regression output of shape (B, 1) - continuous LOS in days
                - 'mortality': Mortality probabilities of shape (B, 1)
        """
        # Extract features
        features = self._forward_features(
            x,
            demographic_features=demographic_features,
            icu_unit_features=icu_unit_features,
            sofa_features=sofa_features,
        )
        
        # Task-specific heads (simple linear without activation for regression)
        los_predictions = self.los_head(features)  # (B, 1) - continuous LOS in days
        mortality_logits = self.mortality_head(features)  # (B, 1)
        mortality_probs = torch.sigmoid(mortality_logits)  # (B, 1)
        
        return {
            "los": los_predictions,
            "mortality": mortality_probs
        }
    
    def get_features(
        self,
        x: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
        icu_unit_features: Optional[torch.Tensor] = None,
        sofa_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features before task heads (for MultiTaskECGModel compatibility)."""
        return self._forward_features(
            x,
            demographic_features=demographic_features,
            icu_unit_features=icu_unit_features,
            sofa_features=sofa_features,
        )
