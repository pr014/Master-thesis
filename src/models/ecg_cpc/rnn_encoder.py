"""RNN Encoder for ECG-CPC model (before S4).

Based on checkpoint architecture: 4 RNN layers with features [512, 512, 512, 512]
"""

from typing import List
import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    """RNN Encoder with multiple RNN layers.
    
    Architecture (from checkpoint):
    - 4 RNN layers with features [512, 512, 512, 512]
    - Kernel sizes [3, 1, 1, 1]
    - Strides [2, 1, 1, 1]
    
    Input: (B, seq_len, d_input) - e.g., (B, 5000, 12)
    Output: (B, seq_len, d_model) - e.g., (B, 2500, 512) after first stride=2
    """
    
    def __init__(
        self,
        d_input: int = 12,
        features: List[int] = [512, 512, 512, 512],
        kernel_sizes: List[int] = [3, 1, 1, 1],
        strides: List[int] = [2, 1, 1, 1],
        dropout: float = 0.1,
    ):
        """Initialize RNN Encoder.
        
        Args:
            d_input: Input dimension (number of ECG leads, default: 12)
            features: List of feature dimensions for each layer (default: [512, 512, 512, 512])
            kernel_sizes: List of kernel sizes for each layer (default: [3, 1, 1, 1])
            strides: List of stride sizes for each layer (default: [2, 1, 1, 1])
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        
        self.d_input = d_input
        self.features = features
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.dropout = dropout
        
        if len(features) != len(kernel_sizes) or len(features) != len(strides):
            raise ValueError("features, kernel_sizes, and strides must have the same length")
        
        self.n_layers = len(features)
        
        # Build RNN layers
        self.layers = nn.ModuleList()
        
        # Layer 0: d_input -> features[0]
        in_dim = d_input
        for i in range(self.n_layers):
            out_dim = features[i]
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            
            # Conv1D layer (treating sequence as 1D convolution)
            # For RNN-like behavior, we use Conv1D with kernel_size and stride
            # This is similar to how some implementations handle temporal encoding
            conv = nn.Conv1d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2 if stride == 1 else 0,
            )
            self.layers.append(conv)
            
            # BatchNorm
            self.layers.append(nn.BatchNorm1d(out_dim))
            
            # ReLU
            self.layers.append(nn.ReLU())
            
            # Dropout
            self.layers.append(nn.Dropout(dropout))
            
            in_dim = out_dim
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(features[-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RNN encoder.
        
        Args:
            x: Input tensor of shape (B, seq_len, d_input)
            
        Returns:
            Output tensor of shape (B, seq_len_after_strides, d_model)
        """
        # Transpose for Conv1D: (B, seq_len, d_input) -> (B, d_input, seq_len)
        x = x.transpose(1, 2)  # (B, d_input, seq_len)
        
        # Apply RNN layers (Conv1D blocks)
        layer_idx = 0
        for i in range(self.n_layers):
            # Conv1D
            x = self.layers[layer_idx](x)  # (B, features[i], seq_len')
            layer_idx += 1
            
            # BatchNorm
            x = self.layers[layer_idx](x)
            layer_idx += 1
            
            # ReLU
            x = self.layers[layer_idx](x)
            layer_idx += 1
            
            # Dropout
            x = self.layers[layer_idx](x)
            layer_idx += 1
        
        # Transpose back: (B, d_model, seq_len) -> (B, seq_len, d_model)
        x = x.transpose(1, 2)  # (B, seq_len', features[-1])
        
        # Final layer norm
        x = self.final_norm(x)
        
        return x

