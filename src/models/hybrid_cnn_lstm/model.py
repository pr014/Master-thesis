"""Hybrid CNN-LSTM model for ECG regression/classification."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from ..core.base_model import BaseECGModel


class HybridCNNLSTM(BaseECGModel):
    """Hybrid CNN-LSTM architecture for ECG regression/classification.

    Architecture:
    - Input: (B, 12, 5000)
    - 3 Conv1D blocks: 12→32→64→128 with MaxPool(2)
    - BiLSTM stack: 2 layers, 128 units per direction (consistent with Bidirectional LSTM)
      * Layer 1: 128 per direction → 256 output
      * Layer 2: 128 per direction → 256 output (input is 256 from Layer 1)
      * Dropout between layers (default: 0.2)
    - Pooling: Mean pooling (uses all timesteps, recommended) or last/max
    - Late fusion with demographics and diagnoses (optional)
    - Shared layer: BN → Dropout → Dense(64) → ReLU → Dropout
    - Output: LOS prediction (B, 1) for regression or (B, num_classes) for classification
    
    Scientific Justifications:
    - Bidirectional LSTM: Graves & Schmidhuber (2005) demonstrate that bidirectional LSTMs
      capture both forward and backward temporal dependencies, ideal for retrospective
      analysis of medical time series.
    - Mean Pooling: Lin et al. (2013) show that mean pooling aggregates information
      across all timesteps, while "last" pooling discards 99.98% of sequence information.
    - Multi-Layer LSTM: Sutskever et al. (2014) show that multi-layer LSTMs learn
      hierarchical temporal dependencies better (Layer 1: local patterns, Layer 2: global patterns).
    - Dropout: Srivastava et al. (2014) demonstrate that dropout between layers
      reduces overfitting, especially important for multi-layer architectures.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        training_config = config.get("training", {})
        dropout_rate = training_config.get("dropout_rate", 0.3)

        # Get model config
        model_config = config.get("model", {})

        # CNN hyperparameters (defaults follow the requested architecture)
        conv1_out = model_config.get("conv1_out", 32)
        conv2_out = model_config.get("conv2_out", 64)
        conv3_out = model_config.get("conv3_out", 128)
        conv1_kernel = model_config.get("conv1_kernel", 5)
        conv2_kernel = model_config.get("conv2_kernel", 5)
        conv3_kernel = model_config.get("conv3_kernel", 3)
        conv1_padding = model_config.get("conv1_padding", conv1_kernel // 2)
        conv2_padding = model_config.get("conv2_padding", conv2_kernel // 2)
        conv3_padding = model_config.get("conv3_padding", conv3_kernel // 2)

        # LSTM hyperparameters (consistent with Bidirectional LSTM model)
        hidden_dim = model_config.get("hidden_dim", 128)  # Per direction for Layer 1
        hidden_dim_layer2 = model_config.get("hidden_dim_layer2", hidden_dim)  # Per direction for Layer 2
        num_layers = model_config.get("num_layers", 2)  # Default: 2 layers for hierarchical dependencies
        pooling = model_config.get("pooling", "mean")  # Default: mean pooling uses all timesteps
        bidirectional = model_config.get("bidirectional", True)  # Should be True for BiLSTM
        lstm_dropout = model_config.get("lstm_dropout", 0.2)  # Dropout between LSTM layers

        # Enforce bidirectional=True for consistency with Bidirectional LSTM
        if not bidirectional:
            raise ValueError("HybridCNNLSTM requires bidirectional=True. Use unidirectional LSTM model for unidirectional processing.")
        
        self.hidden_dim = hidden_dim
        self.hidden_dim_layer2 = hidden_dim_layer2
        self.num_layers = num_layers
        self.pooling = pooling
        self.bidirectional = True  # Always True for this model
        self.lstm_dropout = lstm_dropout
        
        # Store num_layers for use in forward methods
        self._num_layers = num_layers

        # Convolutional blocks
        self.conv1 = nn.Conv1d(
            in_channels=12,
            out_channels=conv1_out,
            kernel_size=conv1_kernel,
            padding=conv1_padding,
        )
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=conv2_kernel,
            padding=conv2_padding,
        )
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv1d(
            in_channels=conv2_out,
            out_channels=conv3_out,
            kernel_size=conv3_kernel,
            padding=conv3_padding,
        )
        self.bn3 = nn.BatchNorm1d(conv3_out)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.relu = nn.ReLU()

        # LSTM stack (two separate layers for flexibility, consistent with Bidirectional LSTM)
        if num_layers == 1:
            # Single layer BiLSTM
            self.lstm = nn.LSTM(
                input_size=conv3_out,
                hidden_size=hidden_dim,  # 128 per direction
                num_layers=1,
                batch_first=True,
                bidirectional=True,  # Always bidirectional
                dropout=0.0,
            )
            lstm_output_dim = hidden_dim * 2  # 128 * 2 = 256 total
            self.lstm_dropout_layer = None
        else:
            # Two-layer BiLSTM: Layer 1 (hidden_dim per direction), Layer 2 (hidden_dim_layer2 per direction)
            self.lstm_layer1 = nn.LSTM(
                input_size=conv3_out,
                hidden_size=hidden_dim,  # 128 per direction
                num_layers=1,
                batch_first=True,
                bidirectional=True,  # Always bidirectional
                dropout=0.0,
            )
            # Dropout between LSTM layers (prevents overfitting)
            self.lstm_dropout_layer = nn.Dropout(lstm_dropout)
            self.lstm_layer2 = nn.LSTM(
                input_size=hidden_dim * 2,  # 256 (from Layer 1 output)
                hidden_size=hidden_dim_layer2,  # per direction
                num_layers=1,
                batch_first=True,
                bidirectional=True,  # Always bidirectional
                dropout=0.0,
            )
            lstm_output_dim = hidden_dim_layer2 * 2  # total output dim (forward||backward)

        # Demographic features (late fusion)
        data_config = config.get("data", {})
        demographic_config = data_config.get("demographic_features", {})
        self.use_demographics = demographic_config.get("enabled", False)
        
        # Check if diagnosis features are enabled
        diagnosis_config = data_config.get("diagnosis_features", {})
        self.use_diagnoses = diagnosis_config.get("enabled", False)
        diagnosis_list = diagnosis_config.get("diagnosis_list", [])
        diagnosis_dim = len(diagnosis_list) if self.use_diagnoses else 0

        # Check if ICU unit features are enabled
        icu_unit_config = data_config.get("icu_unit_features", {})
        self.use_icu_units = icu_unit_config.get("enabled", False)
        icu_unit_list = icu_unit_config.get("icu_unit_list", [])
        icu_unit_dim = len(icu_unit_list) if self.use_icu_units else 0

        feature_dim = lstm_output_dim
        if self.use_demographics:
            sex_encoding = demographic_config.get("sex_encoding", "binary")
            demo_dim = 2 if sex_encoding == "binary" else 3
            feature_dim += demo_dim
        if self.use_diagnoses:
            feature_dim += diagnosis_dim
        if self.use_icu_units:
            feature_dim += icu_unit_dim

        # Shared layer
        self.bn_final = nn.BatchNorm1d(feature_dim)
        self.dropout_shared = nn.Dropout(dropout_rate)
        self.fc_shared = nn.Linear(feature_dim, 64)
        self.dropout_out = nn.Dropout(dropout_rate)

        # LOS head: 1 neuron for regression, num_classes for classification
        output_dim = 1 if self.task_type == "regression" else (self.num_classes or 10)
        self.fc_los = nn.Linear(64, output_dim)

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

    def _forward_lstm(self, x: torch.Tensor, apply_dropout: bool = False) -> torch.Tensor:
        """Forward pass through LSTM stack.
        
        Args:
            x: Input tensor of shape (B, T, C) after CNN processing
            apply_dropout: Whether to apply dropout between LSTM layers (only in get_features)
        
        Returns:
            LSTM output tensor of shape (B, T, hidden_dim*2)
        """
        # LSTM stack (consistent with Bidirectional LSTM implementation)
        if self._num_layers == 1:
            lstm_output, _ = self.lstm(x)  # lstm_output: (B, T, hidden_dim*2)
        else:
            # Two-layer BiLSTM
            lstm_output1, _ = self.lstm_layer1(x)  # (B, T, 256)
            if apply_dropout and self.lstm_dropout_layer is not None:
                # Apply dropout between layers (only in get_features, not in forward)
                lstm_output1 = self.lstm_dropout_layer(lstm_output1)
            lstm_output, _ = self.lstm_layer2(lstm_output1)  # (B, T, hidden_dim_layer2*2)
        
        return lstm_output

    def _forward_features(
        self, 
        x: torch.Tensor, 
        demographic_features: Optional[torch.Tensor] = None,
        diagnosis_features: Optional[torch.Tensor] = None,
        icu_unit_features: Optional[torch.Tensor] = None,
        apply_lstm_dropout: bool = False
    ) -> torch.Tensor:
        # CNN blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Prepare for LSTM: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        # LSTM stack (consistent with Bidirectional LSTM: no dropout in forward, dropout in get_features)
        lstm_output = self._forward_lstm(x, apply_dropout=apply_lstm_dropout)

        # Pool LSTM output
        ecg_features = self._pool_lstm_output(lstm_output)

        # Late fusion with demographics, diagnoses, and ICU units
        fused_features = ecg_features
        if self.use_demographics and demographic_features is not None:
            fused_features = torch.cat([fused_features, demographic_features], dim=1)
        if self.use_diagnoses and diagnosis_features is not None:
            fused_features = torch.cat([fused_features, diagnosis_features], dim=1)
        if self.use_icu_units and icu_unit_features is not None:
            fused_features = torch.cat([fused_features, icu_unit_features], dim=1)

        # Shared layer
        x = self.bn_final(fused_features)
        x = self.dropout_shared(x)
        x = self.fc_shared(x)
        x = self.relu(x)
        x = self.dropout_out(x)

        return x

    def forward(
        self, 
        x: torch.Tensor, 
        demographic_features: Optional[torch.Tensor] = None,
        diagnosis_features: Optional[torch.Tensor] = None,
        icu_unit_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass returning LOS prediction.
        
        Consistent with Bidirectional LSTM: NO dropout between LSTM layers in forward().
        
        Returns:
            For regression: Output tensor of shape (B, 1) - continuous LOS in days
            For classification: Output logits of shape (B, num_classes)
        """
        features = self._forward_features(
            x, 
            demographic_features=demographic_features, 
            diagnosis_features=diagnosis_features,
            icu_unit_features=icu_unit_features,
            apply_lstm_dropout=False  # No dropout between LSTM layers in forward (like Bidirectional LSTM)
        )
        return self.fc_los(features)

    def get_features(
        self, 
        x: torch.Tensor, 
        demographic_features: Optional[torch.Tensor] = None,
        diagnosis_features: Optional[torch.Tensor] = None,
        icu_unit_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract features before final LOS head.
        
        Consistent with Bidirectional LSTM: dropout between LSTM layers is applied in get_features().
        
        Returns:
            Feature tensor of shape (B, feature_dim) after pooling and before fc.
            If demographic/diagnosis/ICU unit features are enabled, this includes them.
        """
        return self._forward_features(
            x, 
            demographic_features=demographic_features, 
            diagnosis_features=diagnosis_features,
            icu_unit_features=icu_unit_features,
            apply_lstm_dropout=True  # Apply dropout between LSTM layers in get_features (like Bidirectional LSTM)
        )
