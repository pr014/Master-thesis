"""Hybrid CNN-LSTM model for ECG classification."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from ..core.base_model import BaseECGModel


class HybridCNNLSTM(BaseECGModel):
    """Hybrid CNN-LSTM architecture for ECG classification.

    Architecture:
    - Input: (B, 12, 5000)
    - 3 Conv1D blocks: 12→32→64→128 with MaxPool(2)
    - BiLSTM stack: 2 layers, 128 units per direction
    - Pooling: last timestep
    - Late fusion with demographics (optional)
    - Shared layer: BN → Dropout → Dense(64) → ReLU → Dropout
    - Output: LOS logits (B, num_classes)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        training_config = config.get("training", {})
        dropout_rate = training_config.get("dropout_rate", 0.3)

        model_config = config.get("model", {})
        num_classes = model_config.get("num_classes", self.num_classes)
        self.num_classes = num_classes

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

        # LSTM hyperparameters
        hidden_dim = model_config.get("hidden_dim", 128)
        hidden_dim_layer2 = model_config.get("hidden_dim_layer2", hidden_dim)
        num_layers = model_config.get("num_layers", 2)
        pooling = model_config.get("pooling", "last")
        bidirectional = model_config.get("bidirectional", True)

        self.hidden_dim = hidden_dim
        self.hidden_dim_layer2 = hidden_dim_layer2
        self.num_layers = num_layers
        self.pooling = pooling
        self.bidirectional = bidirectional

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

        # LSTM stack (two separate layers for flexibility)
        if num_layers == 1:
            self.lstm = nn.LSTM(
                input_size=conv3_out,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0,
            )
            lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        else:
            self.lstm_layer1 = nn.LSTM(
                input_size=conv3_out,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0,
            )
            self.lstm_layer2 = nn.LSTM(
                input_size=hidden_dim * (2 if bidirectional else 1),
                hidden_size=hidden_dim_layer2,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0,
            )
            lstm_output_dim = hidden_dim_layer2 * (2 if bidirectional else 1)

        # Demographic features (late fusion)
        data_config = config.get("data", {})
        demographic_config = data_config.get("demographic_features", {})
        self.use_demographics = demographic_config.get("enabled", False)

        feature_dim = lstm_output_dim
        if self.use_demographics:
            sex_encoding = demographic_config.get("sex_encoding", "binary")
            demo_dim = 2 if sex_encoding == "binary" else 3
            feature_dim += demo_dim

        # Shared layer
        self.bn_final = nn.BatchNorm1d(feature_dim)
        self.dropout_shared = nn.Dropout(dropout_rate)
        self.fc_shared = nn.Linear(feature_dim, 64)
        self.dropout_out = nn.Dropout(dropout_rate)

        # LOS head
        self.fc_los = nn.Linear(64, self.num_classes)

    def _pool_lstm_output(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """Pool LSTM output based on configured strategy."""
        if self.pooling == "last":
            return lstm_output[:, -1, :]
        if self.pooling == "mean":
            return lstm_output.mean(dim=1)
        if self.pooling == "max":
            return lstm_output.max(dim=1)[0]
        raise ValueError(
            f"Unknown pooling strategy: {self.pooling}. Use 'last', 'mean', or 'max'."
        )

    def _forward_features(
        self, x: torch.Tensor, demographic_features: Optional[torch.Tensor] = None
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

        # LSTM stack
        if self.num_layers == 1:
            lstm_output, _ = self.lstm(x)
        else:
            lstm_output1, _ = self.lstm_layer1(x)
            lstm_output, _ = self.lstm_layer2(lstm_output1)

        # Pool LSTM output
        ecg_features = self._pool_lstm_output(lstm_output)

        # Late fusion with demographics
        if self.use_demographics and demographic_features is not None:
            fused_features = torch.cat([ecg_features, demographic_features], dim=1)
        else:
            fused_features = ecg_features

        # Shared layer
        x = self.bn_final(fused_features)
        x = self.dropout_shared(x)
        x = self.fc_shared(x)
        x = self.relu(x)
        x = self.dropout_out(x)

        return x

    def forward(
        self, x: torch.Tensor, demographic_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass returning LOS logits."""
        features = self._forward_features(x, demographic_features=demographic_features)
        return self.fc_los(features)

    def get_features(
        self, x: torch.Tensor, demographic_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract features before final LOS head."""
        return self._forward_features(x, demographic_features=demographic_features)
