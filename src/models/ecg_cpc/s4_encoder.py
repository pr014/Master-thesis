"""S4 Encoder for ECG-CPC model.

Based on Gu et al. (2022): "Efficiently modeling long sequences with S4"
"""

from typing import Dict, Any
import torch
import torch.nn as nn

try:
    from s4_torch import S4
    S4_AVAILABLE = True
except ImportError:
    try:
        # Use S4 implementation from ecg-fm-benchmarking repo
        from .s4_impl import S4
        S4_AVAILABLE = True
        print("Using S4 from local implementation (ecg-fm-benchmarking repo)")
    except ImportError as e:
        S4_AVAILABLE = False
        print(f"Warning: S4 not available. Error: {e}")


class S4Encoder(nn.Module):
    """S4 Encoder with multiple S4 blocks.
    
    Architecture:
    - Input: (B, seq_len, d_input) - e.g., (B, 5000, 12)
    - Output: (B, seq_len, d_model) - e.g., (B, 5000, 256)
    
    Based on Gu et al. (2022) S4 paper.
    """
    
    def __init__(
        self,
        d_input: int = 12,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.1,
        prenorm: bool = True,
    ):
        """Initialize S4 Encoder.
        
        Args:
            d_input: Input dimension (number of ECG leads, default: 12)
            d_model: Hidden dimension (default: 256)
            d_state: State dimension for S4 (default: 64)
            n_layers: Number of S4 layers (default: 4)
            dropout: Dropout rate (default: 0.1)
            prenorm: Whether to use pre-normalization (default: True)
        """
        super().__init__()
        
        if not S4_AVAILABLE:
            raise ImportError(
                "s4-torch is required for S4Encoder. "
                "Install with: pip install s4-torch"
            )
        
        self.d_input = d_input
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.dropout = dropout
        self.prenorm = prenorm
        
        # Input projection: d_input -> d_model
        self.input_proj = nn.Linear(d_input, d_model)
        
        # S4 layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Layer normalization (prenorm: before S4, postnorm: after S4)
            if prenorm:
                self.layers.append(nn.LayerNorm(d_model))
            
            # S4 layer
            self.layers.append(
                S4(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                    transposed=False,  # Input is (B, L, D), not (B, D, L)
                )
            )
            
            # Dropout after S4
            self.layers.append(nn.Dropout(dropout))
            
            # Post-norm (if not using prenorm)
            if not prenorm:
                self.layers.append(nn.LayerNorm(d_model))
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through S4 encoder.
        
        Args:
            x: Input tensor of shape (B, seq_len, d_input)
            
        Returns:
            Output tensor of shape (B, seq_len, d_model)
        """
        # Input projection: (B, seq_len, d_input) -> (B, seq_len, d_model)
        x = self.input_proj(x)
        
        # Apply S4 layers
        layer_idx = 0
        for i in range(self.n_layers):
            # Pre-norm (if enabled)
            if self.prenorm:
                x = self.layers[layer_idx](x)  # LayerNorm
                layer_idx += 1
            
            # S4 layer
            residual = x
            s4_output = self.layers[layer_idx](x)  # S4 returns (output, state)
            # S4 returns (output, state) tuple, we only need output
            if isinstance(s4_output, tuple):
                x = s4_output[0]
            else:
                x = s4_output
            layer_idx += 1
            
            # Residual connection
            x = x + residual
            
            # Dropout
            x = self.layers[layer_idx](x)  # Dropout
            layer_idx += 1
            
            # Post-norm (if enabled)
            if not self.prenorm:
                x = self.layers[layer_idx](x)  # LayerNorm
                layer_idx += 1
        
        # Final normalization
        x = self.final_norm(x)
        
        return x

