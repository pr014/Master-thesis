"""Unidirectional LSTM model for ECG classification.

LSTM architecture for processing 12-lead ECG signals as time series.
Designed for LOS classification and mortality prediction.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from ...core.base_model import BaseECGModel


class LSTM1D_Unidirectional(BaseECGModel):
    """Unidirectional LSTM architecture for ECG classification.
    
    Architecture:
    - Input Transformation: (B, 12, 5000) → (B, 5000, 12)
    - Optional: Embedding Layer (12 → hidden_dim)
    - LSTM Stack:
      * 1-Layer: 128 units (based on Kim et al. 2020)
      * 2-Layer: Layer 1 (128 units), Layer 2 (configurable; default: 128 units)
    - Pooling: Last hidden state, Mean, or Max pooling
    - Classification Head: BatchNorm → Dropout → Linear
    
    Input: (B, 12, 5000) - 12 ECG leads, 5000 time steps
    Output: (B, num_classes) - Classification logits
    
    Based on: Kim et al. 2020 - recommends 128 units for raw ECG signals
    For 2-layer: Layer 2 hidden size is configurable via `hidden_dim_layer2`.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LSTM1D_Unidirectional model.
        
        Args:
            config: Configuration dictionary. Should contain:
                - num_classes: Number of output classes
                - dropout_rate: Dropout rate (from baseline.yaml)
                - model: Model-specific parameters:
                    - hidden_dim: Hidden dimension for LSTM Layer 1 (default: 128)
                    - hidden_dim_layer2: Hidden dimension for LSTM Layer 2 (only if num_layers=2; default: hidden_dim)
                    - num_layers: Number of LSTM layers (default: 1)
                    - pooling: Pooling strategy ("last", "mean", "max") (default: "last")
                    - use_embedding: Whether to use embedding layer (default: False)
        """
        super().__init__(config)
        
        # Get dropout rate from config
        training_config = config.get("training", {})
        dropout_rate = training_config.get("dropout_rate", 0.3)
        
        # Get num_classes from model config or use from base class
        model_config = config.get("model", {})
        num_classes = model_config.get("num_classes", self.num_classes)
        self.num_classes = num_classes
        
        # Get LSTM-specific parameters
        hidden_dim = model_config.get("hidden_dim", 128)
        num_layers = model_config.get("num_layers", 1)
        pooling = model_config.get("pooling", "last")
        use_embedding = model_config.get("use_embedding", False)
        bidirectional = model_config.get("bidirectional", False)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling
        self.bidirectional = bidirectional
        
        # Input: (B, 12, 5000) - 12 leads, 5000 time steps
        input_size = 12  # Number of ECG leads
        
        # Optional embedding layer
        if use_embedding:
            self.embedding = nn.Linear(input_size, hidden_dim)
            lstm_input_size = hidden_dim
        else:
            self.embedding = None
            lstm_input_size = input_size
        
        # For 2-layer LSTM: Layer 1 uses hidden_dim, Layer 2 uses hidden_dim_layer2 (configurable; default: hidden_dim)
        # PyTorch's nn.LSTM doesn't support different hidden_dims per layer, so we need separate layers
        if num_layers == 1:
            # Single layer LSTM
            self.lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=hidden_dim,  # 128
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0
            )
            lstm_output_dim = hidden_dim * (2 if bidirectional else 1)  # 128
        else:
            # Two-layer LSTM: Layer 1 (hidden_dim), Layer 2 (hidden_dim_layer2)
            hidden_dim_layer2 = model_config.get("hidden_dim_layer2", hidden_dim)
            self.lstm_layer1 = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=hidden_dim,  # 128
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0
            )
            self.lstm_layer2 = nn.LSTM(
                input_size=hidden_dim * (2 if bidirectional else 1),  # 128 (from Layer 1 output)
                hidden_size=hidden_dim_layer2,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0
            )
            lstm_output_dim = hidden_dim_layer2 * (2 if bidirectional else 1)
        
        # Calculate feature dimension after LSTM
        
        # Check if demographic features are enabled
        data_config = config.get("data", {})
        demographic_config = data_config.get("demographic_features", {})
        self.use_demographics = demographic_config.get("enabled", False)
        
        # Calculate feature dimension (LSTM features + optional demographic features)
        feature_dim = lstm_output_dim
        if self.use_demographics:
            sex_encoding = demographic_config.get("sex_encoding", "binary")
            demo_dim = 2 if sex_encoding == "binary" else 3  # binary: [age, sex], onehot: [age, sex_0, sex_1]
            feature_dim += demo_dim
        
        # Classification Head
        self.bn_final = nn.BatchNorm1d(feature_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(feature_dim, self.num_classes)
    
    def _pool_lstm_output(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """Pool LSTM output based on configured strategy.
        
        Args:
            lstm_output: LSTM output tensor of shape (B, T, hidden_dim) or (B, T, hidden_dim*2) if bidirectional
        
        Returns:
            Pooled features of shape (B, hidden_dim) or (B, hidden_dim*2)
        """
        if self.pooling == "last":
            # Last hidden state: (B, T, hidden_dim) → (B, hidden_dim)
            return lstm_output[:, -1, :]
        elif self.pooling == "mean":
            # Mean pooling: average over time dimension
            return lstm_output.mean(dim=1)
        elif self.pooling == "max":
            # Max pooling: maximum over time dimension
            return lstm_output.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}. Use 'last', 'mean', or 'max'.")
    
    def forward(self, x: torch.Tensor, demographic_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
            demographic_features: Optional tensor of shape (B, 2) or (B, 3) containing Age & Sex.
                                 None if demographic features are disabled.
        
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        # Transform input: (B, 12, 5000) → (B, 5000, 12)
        x = x.transpose(1, 2)  # (B, 5000, 12)
        
        # Optional embedding
        if self.embedding is not None:
            x = self.embedding(x)  # (B, 5000, hidden_dim)
        
        # LSTM forward pass
        if self.num_layers == 1:
            lstm_output, (hidden, cell) = self.lstm(x)  # lstm_output: (B, 5000, hidden_dim) or (B, 5000, hidden_dim*2)
        else:
            # Two-layer LSTM
            lstm_output1, _ = self.lstm_layer1(x)  # (B, 5000, 128)
            lstm_output, _ = self.lstm_layer2(lstm_output1)  # (B, 5000, hidden_dim_layer2)
        
        # Pool LSTM output
        ecg_features = self._pool_lstm_output(lstm_output)  # (B, hidden_dim) or (B, hidden_dim_layer2)
        
        # Late fusion: Concatenate ECG features with demographic features
        if self.use_demographics and demographic_features is not None:
            # Concatenate: [ECG features, demographic features]
            fused_features = torch.cat([ecg_features, demographic_features], dim=1)
        else:
            fused_features = ecg_features
        
        # Classification Head
        x = self.bn_final(fused_features)
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)
        
        return x
    
    def get_features(self, x: torch.Tensor, demographic_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract features before final classification head.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
            demographic_features: Optional tensor of shape (B, 2) or (B, 3) containing Age & Sex.
                                 None if demographic features are disabled.
        
        Returns:
            features: Feature tensor of shape (B, feature_dim) after pooling and before fc.
                     If demographic features are enabled, this includes the demographic features.
        """
        # Transform input: (B, 12, 5000) → (B, 5000, 12)
        x = x.transpose(1, 2)  # (B, 5000, 12)
        
        # Optional embedding
        if self.embedding is not None:
            x = self.embedding(x)  # (B, 5000, hidden_dim)
        
        # LSTM forward pass
        if self.num_layers == 1:
            lstm_output, (hidden, cell) = self.lstm(x)  # lstm_output: (B, 5000, hidden_dim) or (B, 5000, hidden_dim*2)
        else:
            # Two-layer LSTM
            lstm_output1, _ = self.lstm_layer1(x)  # (B, 5000, 128)
            lstm_output, _ = self.lstm_layer2(lstm_output1)  # (B, 5000, hidden_dim_layer2)
        
        # Pool LSTM output
        ecg_features = self._pool_lstm_output(lstm_output)  # (B, hidden_dim) or (B, hidden_dim_layer2)
        
        # Late fusion: Concatenate ECG features with demographic features
        if self.use_demographics and demographic_features is not None:
            # Concatenate: [ECG features, demographic features]
            fused_features = torch.cat([ecg_features, demographic_features], dim=1)
        else:
            fused_features = ecg_features
        
        # Apply BatchNorm and Dropout but not final FC layer
        x = self.bn_final(fused_features)
        x = self.dropout(x)
        
        return x  # (B, feature_dim) - features before final fc layer

