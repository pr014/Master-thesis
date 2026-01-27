"""Building blocks for FastAI xResNet1D architecture."""

import torch
import torch.nn as nn


class XResNetBlock1D(nn.Module):
    """FastAI xResNet Residual Block for 1D signals.
    
    Structure:
    - convs: Main path with 3 convolutions (1x1, 5x1, 1x1)
    - convpath: Parallel path with same structure (dual path)
    - Both paths are added together
    - BatchNorm after each conv
    - ReLU activation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 4,
        kernel_size: int = 5,
        stride: int = 1
    ):
        """Initialize xResNet block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (base channels)
            expansion: Expansion factor (out_channels * expansion = expanded channels)
            kernel_size: Kernel size for middle convolution
            stride: Stride for middle convolution
        """
        super().__init__()
        
        expanded_channels = out_channels * expansion
        
        # Main path: convs
        # Structure: 1x1 → 5x1 → 1x1 (expand → depthwise → project)
        self.convs = nn.Sequential(
            # convs.0: 1x1
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # convs.1: kernel_size x 1
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # convs.2: 1x1 project
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=expanded_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm1d(expanded_channels)
        )
        
        # Parallel path: convpath (same structure as convs)
        self.convpath = nn.Sequential(
            # convpath.0.0: 1x1
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # convpath.0.1: kernel_size x 1
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # convpath.0.2: 1x1 project
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=expanded_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm1d(expanded_channels)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, in_channels, T)
        
        Returns:
            Output tensor of shape (B, expanded_channels, T')
        """
        # Dual path: both paths process input, then add
        out_convs = self.convs(x)
        out_convpath = self.convpath(x)
        
        out = out_convs + out_convpath  # Add both paths
        out = self.relu(out)
        
        return out

