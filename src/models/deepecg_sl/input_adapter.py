"""Input Adapter for DeepECG-SL to adapt 5000 samples to 2500 samples."""

from typing import Dict, Any
import torch
import torch.nn as nn


class InputAdapter(nn.Module):
    """Adapter layer to convert input from (B, 12, 5000) to (B, 12, 2500).
    
    Uses Conv1D with stride=2 to downsample while preserving local features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Input Adapter.
        
        Args:
            config: Configuration dictionary. Should contain:
                - model.input_adapter.type: "conv1d", "linear", or "pooling"
                - model.input_adapter.in_length: Input length (default: 5000)
                - model.input_adapter.out_length: Output length (default: 2500)
                - model.input_adapter.kernel_size: Kernel size for conv1d (default: 3)
                - model.input_adapter.stride: Stride for conv1d (default: 2)
        """
        super().__init__()
        
        model_config = config.get("model", {})
        adapter_config = model_config.get("input_adapter", {})
        
        adapter_type = adapter_config.get("type", "conv1d")
        in_length = adapter_config.get("in_length", 5000)
        out_length = adapter_config.get("out_length", 2500)
        kernel_size = adapter_config.get("kernel_size", 3)
        stride = adapter_config.get("stride", 2)
        
        self.adapter_type = adapter_type
        self.in_length = in_length
        self.out_length = out_length
        
        if adapter_type == "conv1d":
            # Conv1D with stride=2: 5000 -> 2500
            # Input: (B, 12, 5000) -> Output: (B, 12, 2500)
            padding = (kernel_size - stride) // 2 if stride > 1 else kernel_size // 2
            self.adapter = nn.Conv1d(
                in_channels=12,
                out_channels=12,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
        elif adapter_type == "linear":
            # Linear projection: (B, 12, 5000) -> (B, 12, 2500)
            self.adapter = nn.Linear(in_length, out_length)
        elif adapter_type == "pooling":
            # Average pooling: (B, 12, 5000) -> (B, 12, 2500)
            kernel_size_pool = in_length // out_length
            self.adapter = nn.AvgPool1d(kernel_size=kernel_size_pool, stride=kernel_size_pool)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through input adapter.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
            
        Returns:
            Output tensor of shape (B, 12, 2500)
        """
        if self.adapter_type == "linear":
            # Linear: (B, 12, 5000) -> (B, 12, 2500)
            x = x.transpose(1, 2)  # (B, 5000, 12)
            x = self.adapter(x)  # (B, 2500, 12)
            x = x.transpose(1, 2)  # (B, 12, 2500)
        else:
            # Conv1D or Pooling: (B, 12, 5000) -> (B, 12, 2500)
            x = self.adapter(x)
        
        return x

