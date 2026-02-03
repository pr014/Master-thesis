"""CNN model for ECG classification - simple architecture from scratch."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from ..core.base_model import BaseECGModel


class CNNScratch(BaseECGModel):
    """Simple CNN architecture for ECG classification.
    
    Architecture:
    - 3 Conv1D blocks with increasing filters (32, 64, 128)
    - Global Average Pooling
    - 2 Dense layers for classification
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize CNN model.
        
        Args:
            config: Configuration dictionary. Should contain:
                - num_classes: Number of output classes
                - dropout_rate: Dropout rate (from baseline.yaml)
                - model: CNN-specific parameters (optional)
        """
        super().__init__(config)
        
        # Get dropout rate from config (from baseline.yaml)
        # Config structure: config["training"]["dropout_rate"]
        training_config = config.get("training", {})
        dropout_rate = training_config.get("dropout_rate", 0.3)
        
        # Get num_classes from model config or use from base class
        model_config = config.get("model", {})
        num_classes = model_config.get("num_classes", self.num_classes)
        self.num_classes = num_classes
        
        # Input: (B, 12, 5000) - 12 leads, 5000 time steps
        
        # Conv Block 1: 12 -> 32 filters
        self.conv1 = nn.Conv1d(
            in_channels=12,
            out_channels=32,
            kernel_size=7,
            padding=3  # Same padding
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # Output: (B, 32, 2500)
        
        # Conv Block 2: 32 -> 64 filters
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2  # Same padding
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # Output: (B, 64, 1250)
        
        # Conv Block 3: 64 -> 128 filters
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1  # Same padding
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        # Output: (B, 128, 625)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Output: (B, 128, 1) -> (B, 128)
        
        # Check if demographic features are enabled
        data_config = config.get("data", {})
        demographic_config = data_config.get("demographic_features", {})
        self.use_demographics = demographic_config.get("enabled", False)
        
        # Check if diagnosis features are enabled
        diagnosis_config = data_config.get("diagnosis_features", {})
        self.use_diagnoses = diagnosis_config.get("enabled", False)
        diagnosis_list = diagnosis_config.get("diagnosis_list", [])
        diagnosis_dim = len(diagnosis_list) if self.use_diagnoses else 0
        
        # Calculate feature dimension (ECG features + optional demographic + diagnosis features)
        feature_dim = 128  # ECG features after global pooling
        if self.use_demographics:
            sex_encoding = demographic_config.get("sex_encoding", "binary")
            demo_dim = 2 if sex_encoding == "binary" else 3  # binary: [age, sex], onehot: [age, sex_0, sex_1]
            feature_dim += demo_dim
        if self.use_diagnoses:
            feature_dim += diagnosis_dim
        
        # Classification Head
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(feature_dim, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, self.num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(
        self, 
        x: torch.Tensor, 
        demographic_features: Optional[torch.Tensor] = None,
        diagnosis_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
            demographic_features: Optional tensor of shape (B, 2) or (B, 3) containing Age & Sex.
                                 None if demographic features are disabled.
            diagnosis_features: Optional tensor of shape (B, diagnosis_dim) containing binary diagnosis features.
                               None if diagnosis features are disabled.
        
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # Global Average Pooling
        x = self.global_pool(x)  # (B, 128, 1)
        x = x.squeeze(-1)  # (B, 128)
        
        # Late fusion: Concatenate ECG features with demographic and diagnosis features
        fused = x
        if self.use_demographics and demographic_features is not None:
            fused = torch.cat([fused, demographic_features], dim=1)
        if self.use_diagnoses and diagnosis_features is not None:
            fused = torch.cat([fused, diagnosis_features], dim=1)
        
        # Classification Head
        x = self.dropout1(fused)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(
        self, 
        x: torch.Tensor, 
        demographic_features: Optional[torch.Tensor] = None,
        diagnosis_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract features before final classification head.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
            demographic_features: Optional tensor of shape (B, 2) or (B, 3) containing Age & Sex.
                                 None if demographic features are disabled.
            diagnosis_features: Optional tensor of shape (B, diagnosis_dim) containing binary diagnosis features.
                               None if diagnosis features are disabled.
        
        Returns:
            features: Feature tensor of shape (B, 64) after fc1 and before fc2.
                     If demographic/diagnosis features are enabled, this includes them.
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # Global Average Pooling
        x = self.global_pool(x)  # (B, 128, 1)
        x = x.squeeze(-1)  # (B, 128)
        
        # Late fusion: Concatenate ECG features with demographic and diagnosis features
        fused = x
        if self.use_demographics and demographic_features is not None:
            fused = torch.cat([fused, demographic_features], dim=1)
        if self.use_diagnoses and diagnosis_features is not None:
            fused = torch.cat([fused, diagnosis_features], dim=1)
        
        # Classification Head (up to fc1)
        x = self.dropout1(fused)
        x = self.fc1(x)
        x = self.relu(x)
        
        return x  # (B, 64) - features before final fc2 layer

