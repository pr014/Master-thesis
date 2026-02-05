"""Bidirectional LSTM model for ECG regression/classification.

Bidirectional LSTM architecture for processing 12-lead ECG signals as time series.
Designed for LOS regression and mortality prediction.
Uses bidirectional processing to capture both forward and backward temporal dependencies.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from ...core.base_model import BaseECGModel


class LSTM1D_Bidirectional(BaseECGModel):
    """Bidirectional LSTM architecture for ECG regression/classification.
    
    Architecture (Improved):
    - Input Transformation: (B, 12, 5000) → (B, 5000, 12)
    - Optional: Embedding Layer (12 → 64) for better feature representation
    - BiLSTM Stack:
      * 1-Layer: 128 units per direction → 256 total output
      * 2-Layer: Layer 1 (128 per direction → 256), Layer 2 (configurable; default: 128 per direction → 256)
      * Dropout between layers (default: 0.2)
    - Pooling: Last hidden state, Mean, or Max pooling (default: "mean")
    - Classification Head: BatchNorm → Dropout → Linear
    
    Input: (B, 12, 5000) - 12 ECG leads, 5000 time steps
    Output: (B, 1) for regression or (B, num_classes) for classification
    
    Scientific Justifications:
    - Bidirectional LSTM: Graves & Schmidhuber (2005) demonstrate that bidirectional LSTMs
      capture both forward and backward temporal dependencies, ideal for retrospective
      analysis of medical time series.
    - Mean Pooling: Lin et al. (2013) show that mean pooling aggregates information
      across all timesteps, while "last" pooling discards 99.98% of sequence information.
    - Embedding Layer: Mikolov et al. (2013) demonstrate that embeddings increase
      representation capacity from raw signal values.
    - Multi-Layer LSTM: Sutskever et al. (2014) show that multi-layer LSTMs learn
      hierarchical temporal dependencies better (Layer 1: local patterns, Layer 2: global patterns).
    - Dropout: Srivastava et al. (2014) demonstrate that dropout between layers
      reduces overfitting, especially important for multi-layer architectures.
    
    Note: Bidirectional LSTMs process sequences in both directions, making them
    suitable for retrospective analysis. For real-time ICU deployment, unidirectional
    LSTMs are more appropriate as they only use past information.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LSTM1D_Bidirectional model.
        
        Args:
            config: Configuration dictionary. Should contain:
                - num_classes: Number of output classes (ignored for regression)
                - dropout_rate: Dropout rate (from baseline.yaml)
                - model: Model-specific parameters:
                    - hidden_dim: Hidden dimension per direction for Layer 1 (default: 128)
                    - hidden_dim_layer2: Hidden dimension per direction for Layer 2 (only if num_layers=2; default: hidden_dim)
                    - num_layers: Number of LSTM layers (default: 2)
                    - lstm_dropout: Dropout between LSTM layers (default: 0.2)
                    - pooling: Pooling strategy ("last", "mean", "max") (default: "mean")
                    - use_embedding: Whether to use embedding layer (default: True)
                    - embedding_dim: Embedding dimension (default: 64)
                    - bidirectional: Should be True (enforced)
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
        hidden_dim = model_config.get("hidden_dim", 128)  # Per direction for Layer 1 (default: 128, same as unidirectional)
        hidden_dim_layer2 = model_config.get("hidden_dim_layer2", hidden_dim)  # Per direction for Layer 2 (default: same as Layer 1)
        num_layers = model_config.get("num_layers", 2)  # Default: 2 layers for hierarchical dependencies
        pooling = model_config.get("pooling", "mean")  # Default: mean pooling uses all timesteps
        use_embedding = model_config.get("use_embedding", True)  # Default: embedding improves features
        lstm_dropout = model_config.get("lstm_dropout", 0.2)  # Dropout between LSTM layers
        bidirectional = model_config.get("bidirectional", True)  # Should be True for BiLSTM
        
        # Enforce bidirectional=True
        if not bidirectional:
            raise ValueError("LSTM1D_Bidirectional requires bidirectional=True. Use LSTM1D_Unidirectional for unidirectional processing.")
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling
        self.bidirectional = True  # Always True for this model
        self.lstm_dropout = lstm_dropout
        
        # Store num_layers for use in forward methods
        self._num_layers = num_layers
        
        # Input: (B, 12, 5000) - 12 leads, 5000 time steps
        input_size = 12  # Number of ECG leads
        
        # Optional embedding layer (improves feature representation)
        embedding_dim = model_config.get("embedding_dim", 64)
        if use_embedding:
            self.embedding = nn.Linear(input_size, embedding_dim)
            lstm_input_size = embedding_dim
        else:
            self.embedding = None
            lstm_input_size = input_size
        
        # For 2-layer BiLSTM: Layer 1 uses hidden_dim (per direction), Layer 2 uses hidden_dim_layer2 (per direction).
        # With bidirectional=True, the output feature dim is automatically doubled (forward||backward).
        # PyTorch's nn.LSTM doesn't support different hidden_dims per layer, so we need separate layers
        if num_layers == 1:
            # Single layer BiLSTM
            self.lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=hidden_dim,  # 128 per direction
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0
            )
            lstm_output_dim = hidden_dim * 2  # 128 * 2 = 256 total
            self.lstm_dropout_layer = None
        else:
            # Two-layer BiLSTM: Layer 1 (hidden_dim per direction), Layer 2 (hidden_dim_layer2 per direction)
            self.lstm_layer1 = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=hidden_dim,  # 128 per direction
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0
            )
            # Dropout between LSTM layers (prevents overfitting)
            self.lstm_dropout_layer = nn.Dropout(lstm_dropout)
            self.lstm_layer2 = nn.LSTM(
                input_size=hidden_dim * 2,  # 256 (from Layer 1 output)
                hidden_size=hidden_dim_layer2,  # per direction
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0
            )
            lstm_output_dim = hidden_dim_layer2 * 2  # total output dim (forward||backward)
        
        # Calculate feature dimension after BiLSTM
        # For 1-layer: 128*2 = 256, for 2-layer: 64*2 = 128
        
        # Check if demographic features are enabled
        data_config = config.get("data", {})
        demographic_config = data_config.get("demographic_features", {})
        self.use_demographics = demographic_config.get("enabled", False)
        
        # Check if diagnosis features are enabled
        diagnosis_config = data_config.get("diagnosis_features", {})
        self.use_diagnoses = diagnosis_config.get("enabled", False)
        diagnosis_list = diagnosis_config.get("diagnosis_list", [])
        diagnosis_dim = len(diagnosis_list) if self.use_diagnoses else 0
        
        # Calculate feature dimension (LSTM features + optional demographic + diagnosis features)
        feature_dim = lstm_output_dim
        if self.use_demographics:
            sex_encoding = demographic_config.get("sex_encoding", "binary")
            demo_dim = 2 if sex_encoding == "binary" else 3  # binary: [age, sex], onehot: [age, sex_0, sex_1]
            feature_dim += demo_dim
        if self.use_diagnoses:
            feature_dim += diagnosis_dim
        
        # Prediction Head
        self.bn_final = nn.BatchNorm1d(feature_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer: 1 neuron for regression, num_classes for classification
        output_dim = 1 if self.task_type == "regression" else (self.num_classes or 10)
        self.fc = nn.Linear(feature_dim, output_dim)
    
    def _pool_lstm_output(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """Pool LSTM output based on configured strategy.
        
        Args:
            lstm_output: BiLSTM output tensor of shape (B, T, hidden_dim*2)
        
        Returns:
            Pooled features of shape (B, hidden_dim*2)
        """
        if self.pooling == "last":
            # Last hidden state: (B, T, hidden_dim*2) → (B, hidden_dim*2)
            return lstm_output[:, -1, :]
        elif self.pooling == "mean":
            # Mean pooling: average over time dimension
            return lstm_output.mean(dim=1)
        elif self.pooling == "max":
            # Max pooling: maximum over time dimension
            return lstm_output.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}. Use 'last', 'mean', or 'max'.")
    
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
            For regression: Output tensor of shape (B, 1) - continuous LOS in days
            For classification: Output logits of shape (B, num_classes)
        """
        # Transform input: (B, 12, 5000) → (B, 5000, 12)
        x = x.transpose(1, 2)  # (B, 5000, 12)
        
        # Optional embedding
        if self.embedding is not None:
            x = self.embedding(x)  # (B, 5000, hidden_dim)
        
        # BiLSTM forward pass
        if self._num_layers == 1:
            lstm_output, (hidden, cell) = self.lstm(x)  # lstm_output: (B, 5000, hidden_dim*2)
        else:
            # Two-layer BiLSTM
            lstm_output1, _ = self.lstm_layer1(x)  # (B, 5000, 256)
            lstm_output, _ = self.lstm_layer2(lstm_output1)  # (B, 5000, 128)
        
        # Pool LSTM output
        ecg_features = self._pool_lstm_output(lstm_output)  # (B, hidden_dim*2 or hidden_dim_layer2*2)
        
        # Late fusion: Concatenate ECG features with demographic and diagnosis features
        fused_features = ecg_features
        if self.use_demographics and demographic_features is not None:
            fused_features = torch.cat([fused_features, demographic_features], dim=1)
        if self.use_diagnoses and diagnosis_features is not None:
            fused_features = torch.cat([fused_features, diagnosis_features], dim=1)
        
        # Classification Head
        x = self.bn_final(fused_features)
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)
        
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
            features: Feature tensor of shape (B, feature_dim) after pooling and before fc.
                     If demographic/diagnosis features are enabled, this includes them.
        """
        # Transform input: (B, 12, 5000) → (B, 5000, 12)
        x = x.transpose(1, 2)  # (B, 5000, 12)
        
        # Optional embedding
        if self.embedding is not None:
            x = self.embedding(x)  # (B, 5000, hidden_dim)
        
        # BiLSTM forward pass
        if self._num_layers == 1:
            lstm_output, (hidden, cell) = self.lstm(x)  # lstm_output: (B, 5000, hidden_dim*2)
        else:
            # Two-layer BiLSTM with dropout between layers
            lstm_output1, _ = self.lstm_layer1(x)  # (B, 5000, 256)
            lstm_output1 = self.lstm_dropout_layer(lstm_output1)  # Apply dropout between layers
            lstm_output, _ = self.lstm_layer2(lstm_output1)  # (B, 5000, hidden_dim_layer2*2)
        
        # Pool LSTM output
        ecg_features = self._pool_lstm_output(lstm_output)  # (B, hidden_dim*2 or hidden_dim_layer2*2)
        
        # Late fusion: Concatenate ECG features with demographic and diagnosis features
        fused_features = ecg_features
        if self.use_demographics and demographic_features is not None:
            fused_features = torch.cat([fused_features, demographic_features], dim=1)
        if self.use_diagnoses and diagnosis_features is not None:
            fused_features = torch.cat([fused_features, diagnosis_features], dim=1)
        
        # Apply BatchNorm and Dropout but not final FC layer
        x = self.bn_final(fused_features)
        x = self.dropout(x)
        
        return x  # (B, feature_dim) - features before final fc layer

