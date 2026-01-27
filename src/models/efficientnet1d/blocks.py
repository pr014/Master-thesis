"""Building blocks for EfficientNet1D architecture."""

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for 1D convolutions.
    
    Applies channel attention mechanism:
    1. Global Average Pooling
    2. FC (channels → channels//reduction) → Swish
    3. FC (channels//reduction → channels) → Sigmoid
    4. Channel-wise multiplication
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """Initialize SE block.
        
        Args:
            channels: Number of input/output channels
            reduction: Reduction ratio for the first FC layer
        """
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            Swish(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, channels, T)
        
        Returns:
            Output tensor of shape (B, channels, T) with channel attention applied
        """
        se_weights = self.se(x)  # (B, channels)
        # Reshape for channel-wise multiplication: (B, channels, 1)
        se_weights = se_weights.unsqueeze(-1)
        return x * se_weights


class MBConvBlock1D(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block for 1D signals.
    
    Structure:
    1. Expand-Conv (optional): 1x1 Conv1d (Cin → Cin*t) → BN → Swish
    2. Depthwise-Conv: Grouped Conv1d (k, groups=Cin*t) → BN → Swish
    3. SE Block: Channel attention
    4. Project-Conv: 1x1 Conv1d (Cin*t → Cout) → BN
    5. Skip-Connection: If stride=1 AND Cin=Cout
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expansion_factor: int,
        se_ratio: float = 0.25
    ):
        """Initialize MBConv block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for depthwise convolution
            stride: Stride for depthwise convolution
            expansion_factor: Expansion factor t (1 for MBConv1, 6 for MBConv6)
            se_ratio: Squeeze-and-Excitation reduction ratio
        """
        super().__init__()
        
        expanded_channels = in_channels * expansion_factor
        self.use_expand = expansion_factor > 1
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # 1. Expand-Conv (only if expansion_factor > 1)
        if self.use_expand:
            self.expand_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=expanded_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.expand_bn = nn.BatchNorm1d(expanded_channels)
        else:
            expanded_channels = in_channels
        
        # 2. Depthwise-Conv
        padding = kernel_size // 2  # Same padding
        self.depthwise_conv = nn.Conv1d(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=expanded_channels,  # Depthwise: each channel convolved separately
            bias=False
        )
        self.depthwise_bn = nn.BatchNorm1d(expanded_channels)
        
        # 3. SE Block
        self.use_se = se_ratio > 0
        if self.use_se:
            # Calculate reduction ratio: if se_ratio=0.25, then reduction=4
            reduction = int(1.0 / se_ratio) if se_ratio > 0 else 4
            self.se = SEBlock1D(expanded_channels, reduction=reduction)
        
        # 4. Project-Conv
        self.project_conv = nn.Conv1d(
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.project_bn = nn.BatchNorm1d(out_channels)
        
        # 5. Skip-Connection (if applicable)
        if self.use_residual:
            # Identity connection (no additional layers needed)
            pass
        
        self.swish = Swish()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, in_channels, T)
        
        Returns:
            Output tensor of shape (B, out_channels, T')
        """
        identity = x
        
        # 1. Expand-Conv
        if self.use_expand:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.swish(x)
        
        # 2. Depthwise-Conv
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.swish(x)
        
        # 3. SE Block
        if self.use_se:
            x = self.se(x)
        
        # 4. Project-Conv
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # 5. Skip-Connection
        if self.use_residual:
            x = x + identity
        
        return x

