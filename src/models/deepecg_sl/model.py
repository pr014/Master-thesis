"""DeepECG-SL (WCR) model with self-supervised pretraining for multi-task learning."""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from ..core.base_model import BaseECGModel
from .input_adapter import InputAdapter
from .wcr_encoder import WCREncoder


class DeepECG_SL(BaseECGModel):
    """DeepECG-SL model with WCR Transformer Encoder for multi-task learning.
    
    Architecture:
    - Input: (B, 12, 5000) ECG signals @ 500Hz
    - Input Adapter: Conv1D (5000 → 2500)
    - WCR Encoder: Pretrained Transformer (2500 → seq_len, 512)
    - Global Pooling: (seq_len, 512) → (512)
    - Late Fusion: Concat([ECG(512), demographics(2), diagnoses(N)])
    - Shared Layers: BN → Dropout → FC(514+N → 128) → ReLU → Dropout
    - LOS Head: FC(128 → 10)
    - Mortality Head: FC(128 → 1) + Sigmoid
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DeepECG-SL model.
        
        Args:
            config: Configuration dictionary. Should contain:
                - model.num_classes: Number of LOS classes (default: 10)
                - model.wcr.*: WCR encoder configuration
                - model.input_adapter.*: Input adapter configuration
                - model.pretrained.*: Pretrained weights configuration
                - training.dropout_rate: Dropout rate for shared layers
                - data.demographic_features.enabled: Whether to use demographics
                - data.diagnosis_features.enabled: Whether to use diagnoses
        """
        super().__init__(config)
        
        # Get training config
        training_config = config.get("training", {})
        dropout_rate = training_config.get("dropout_rate", 0.3)
        
        # Get model config
        model_config = config.get("model", {})
        num_classes = model_config.get("num_classes", self.num_classes)
        self.num_classes = num_classes
        
        # Get feature dimensions
        feature_dim = model_config.get("feature_dim", 512)
        shared_dim = model_config.get("shared_dim", 128)
        
        # Check demographic and diagnosis features
        data_config = config.get("data", {})
        demographic_config = data_config.get("demographic_features", {})
        self.use_demographics = demographic_config.get("enabled", False)
        
        diagnosis_config = data_config.get("diagnosis_features", {})
        self.use_diagnoses = diagnosis_config.get("enabled", False)
        diagnosis_list = diagnosis_config.get("diagnosis_list", [])
        diagnosis_dim = len(diagnosis_list) if self.use_diagnoses else 0
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Input Adapter: (B, 12, 5000) → (B, 12, 2500)
        self.input_adapter = InputAdapter(config)
        
        # WCR Encoder: (B, 12, 2500) → (B, seq_len, 512)
        self.wcr_encoder = WCREncoder(config, device=device)
        
        # Global Average Pooling: (B, seq_len, 512) → (B, 512)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Calculate feature dimension after fusion
        fused_feature_dim = feature_dim
        if self.use_demographics:
            sex_encoding = demographic_config.get("sex_encoding", "binary")
            demo_dim = 2 if sex_encoding == "binary" else 3
            fused_feature_dim += demo_dim
        if self.use_diagnoses:
            fused_feature_dim += diagnosis_dim
        
        # Shared layers
        self.shared_bn = nn.BatchNorm1d(fused_feature_dim)
        self.shared_dropout1 = nn.Dropout(dropout_rate)
        self.shared_fc = nn.Linear(fused_feature_dim, shared_dim)
        self.shared_relu = nn.ReLU()
        self.shared_dropout2 = nn.Dropout(dropout_rate)
        
        # Task heads
        self.los_head = nn.Linear(shared_dim, num_classes)
        self.mortality_head = nn.Linear(shared_dim, 1)
        
        # Initialize freeze state based on config
        wcr_config = model_config.get("wcr", {})
        freeze_backbone = wcr_config.get("freeze_backbone", True)
        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()
    
    def freeze_backbone(self) -> None:
        """Freeze WCR encoder and input adapter (for transfer learning Phase 1)."""
        self.wcr_encoder.freeze()
        for param in self.input_adapter.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze WCR encoder and input adapter (for fine-tuning Phase 2)."""
        self.wcr_encoder.unfreeze()
        for param in self.input_adapter.parameters():
            param.requires_grad = True
    
    def _forward_features(
        self,
        x: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
        diagnosis_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features before task heads.
        
        Args:
            x: ECG input tensor of shape (B, 12, 5000)
            demographic_features: Optional demographic features of shape (B, 2) or (B, 3)
            diagnosis_features: Optional diagnosis features of shape (B, diagnosis_dim)
            
        Returns:
            Features tensor of shape (B, shared_dim)
        """
        # Input Adapter: (B, 12, 5000) → (B, 12, 2500)
        x = self.input_adapter(x)
        
        # WCR Encoder: (B, 12, 2500) → (B, seq_len, 512)
        # WCR encoder expects (B, seq_len, features) format
        x = x.transpose(1, 2)  # (B, 2500, 12)
        encoder_out = self.wcr_encoder(x)  # (B, seq_len, 512)
        
        # Global Average Pooling: (B, seq_len, 512) → (B, 512, 1)
        # Transpose for pooling: (B, seq_len, 512) → (B, 512, seq_len)
        encoder_out = encoder_out.transpose(1, 2)  # (B, 512, seq_len)
        x = self.global_pool(encoder_out)  # (B, 512, 1)
        
        # Squeeze: (B, 512, 1) → (B, 512)
        x = x.squeeze(-1)  # (B, 512)
        
        # Late fusion with demographics and diagnoses
        if self.use_demographics and demographic_features is not None:
            x = torch.cat([x, demographic_features], dim=1)
        if self.use_diagnoses and diagnosis_features is not None:
            x = torch.cat([x, diagnosis_features], dim=1)
        
        # Shared layer
        # BatchNorm expects (B, C) for 1D
        if x.dim() == 2:
            x = self.shared_bn(x)  # (B, 512+2+diagnosis_dim) or (B, 512)
        x = self.shared_dropout1(x)
        x = self.shared_fc(x)  # (B, 128)
        x = self.shared_relu(x)
        x = self.shared_dropout2(x)
        
        return x  # (B, 128)
    
    def forward(
        self,
        x: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
        diagnosis_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning LOS logits and mortality probabilities.
        
        Args:
            x: ECG input tensor of shape (B, 12, 5000)
            demographic_features: Optional demographic features of shape (B, 2) or (B, 3)
            diagnosis_features: Optional diagnosis features of shape (B, diagnosis_dim)
            
        Returns:
            Tuple of:
                - los_logits: LOS classification logits of shape (B, num_classes)
                - mortality_probs: Mortality probabilities of shape (B, 1)
        """
        # Extract features
        features = self._forward_features(
            x,
            demographic_features=demographic_features,
            diagnosis_features=diagnosis_features
        )
        
        # Task-specific heads
        los_logits = self.los_head(features)  # (B, 10)
        mortality_logits = self.mortality_head(features)  # (B, 1)
        mortality_probs = torch.sigmoid(mortality_logits)  # (B, 1)
        
        return los_logits, mortality_probs
    
    def get_features(
        self,
        x: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
        diagnosis_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features before task heads (for MultiTaskECGModel compatibility).
        
        Args:
            x: ECG input tensor of shape (B, 12, 5000)
            demographic_features: Optional demographic features
            diagnosis_features: Optional diagnosis features
            
        Returns:
            Features tensor of shape (B, shared_dim)
        """
        return self._forward_features(
            x,
            demographic_features=demographic_features,
            diagnosis_features=diagnosis_features
        )

