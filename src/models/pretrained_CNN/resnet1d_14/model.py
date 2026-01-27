"""ResNet1D-14 model for ECG classification - shallow ResNet architecture."""

from typing import Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
from ...base_model import BaseECGModel


class ResidualBlock1D(nn.Module):
    """1D Residual Block with two convolutional layers."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        """Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolutions
            stride: Stride for first convolution
        """
        super().__init__()
        
        # First conv layer
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,  # Same padding
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second conv layer
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,  # Same padding
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.
        
        Args:
            x: Input tensor of shape (B, in_channels, T)
        
        Returns:
            Output tensor of shape (B, out_channels, T')
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Residual connection
        out = self.relu(out)
        
        return out


class ResNet1D14(BaseECGModel):
    """ResNet1D-14 architecture for ECG classification.
    
    Architecture:
    - Initial Conv1D layer (12 → 64 channels)
    - Layer 1: 3 Residual Blocks (64 channels)
    - Layer 2: 3 Residual Blocks (128 channels, downsample)
    - Layer 3: 3 Residual Blocks (256 channels, downsample)
    - Layer 4: 2 Residual Blocks (512 channels, downsample)
    - Global Average Pooling
    - Classification Head with Dropout
    
    Total: ~3-4M parameters
    
    Note: Currently trained from scratch (no pretrained weights).
    Pretrained weights from PTB-XL can be loaded later if available.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ResNet1D-14 model.
        
        Args:
            config: Configuration dictionary. Should contain:
                - num_classes: Number of output classes
                - dropout_rate: Dropout rate (from baseline.yaml)
                - model: Model-specific parameters (optional)
        """
        super().__init__(config)
        
        # Get dropout rate from config (from baseline.yaml)
        training_config = config.get("training", {})
        dropout_rate = training_config.get("dropout_rate", 0.3)
        
        # Get num_classes from model config or use from base class
        model_config = config.get("model", {})
        num_classes = model_config.get("num_classes", self.num_classes)
        self.num_classes = num_classes
        
        # Input: (B, 12, 5000) - 12 leads, 5000 time steps
        
        # Initial convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=12,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,  # Same padding
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # Output: (B, 64, 1250)
        
        # Layer 1: 3 Residual Blocks with 64 channels
        self.layer1 = nn.Sequential(
            ResidualBlock1D(64, 64, kernel_size=3, stride=1),
            ResidualBlock1D(64, 64, kernel_size=3, stride=1),
            ResidualBlock1D(64, 64, kernel_size=3, stride=1)
        )
        # Output: (B, 64, 1250)
        
        # Layer 2: 3 Residual Blocks with 128 channels (downsample)
        self.layer2 = nn.Sequential(
            ResidualBlock1D(64, 128, kernel_size=3, stride=2),  # Downsample
            ResidualBlock1D(128, 128, kernel_size=3, stride=1),
            ResidualBlock1D(128, 128, kernel_size=3, stride=1)
        )
        # Output: (B, 128, 625)
        
        # Layer 3: 3 Residual Blocks with 256 channels (downsample)
        self.layer3 = nn.Sequential(
            ResidualBlock1D(128, 256, kernel_size=3, stride=2),  # Downsample
            ResidualBlock1D(256, 256, kernel_size=3, stride=1),
            ResidualBlock1D(256, 256, kernel_size=3, stride=1)
        )
        # Output: (B, 256, 313)
        
        # Layer 4: 2 Residual Blocks with 512 channels (downsample)
        self.layer4 = nn.Sequential(
            ResidualBlock1D(256, 512, kernel_size=3, stride=2),  # Downsample
            ResidualBlock1D(512, 512, kernel_size=3, stride=1)
        )
        # Output: (B, 512, 157)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Output: (B, 512, 1) -> (B, 512)
        
        # Classification Head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, self.num_classes)
        
        # Load pretrained weights if enabled
        self._load_pretrained_weights(config)
    
    def _load_pretrained_weights(self, config: Dict[str, Any]) -> None:
        """Load pretrained weights from PTB-XL if enabled in config.
        
        Args:
            config: Configuration dictionary. Should contain:
                - model.pretrained.enabled: Whether to load pretrained weights
                - model.pretrained.weights_path: Path to pretrained weights file
                - model.pretrained.freeze_backbone: Whether to freeze backbone layers
        """
        pretrained_config = config.get("model", {}).get("pretrained", {})
        enabled = pretrained_config.get("enabled", False)
        
        if not enabled:
            return
        
        weights_path = pretrained_config.get("weights_path", "")
        if not weights_path:
            print("Warning: pretrained.enabled is True but weights_path is empty. Training from scratch.")
            return
        
        # Resolve path (relative to project root or absolute)
        weights_path = Path(weights_path)
        if not weights_path.is_absolute():
            # Try to find project root (go up from src/models/pretrained_CNN/resnet1d_14/)
            project_root = Path(__file__).parent.parent.parent.parent.parent
            weights_path = project_root / weights_path
        
        if not weights_path.exists():
            print(f"Warning: Pretrained weights file not found at {weights_path}. Training from scratch.")
            return
        
        try:
            print(f"Loading pretrained weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Check if it's a full checkpoint with 'model_state_dict' or 'state_dict'
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    # Assume the dict itself is the state_dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove prefix if present (e.g., "model.", "backbone.")
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                # Remove common prefixes
                new_key = key
                for prefix in ["model.", "backbone.", "module."]:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        break
                cleaned_state_dict[new_key] = value
            
            # Filter state dict to only include backbone layers (exclude fc and dropout)
            model_state_dict = self.state_dict()
            pretrained_state_dict = {}
            skipped_keys = []
            
            for key, value in cleaned_state_dict.items():
                # Skip classification head (fc, dropout)
                if key.startswith("fc") or key.startswith("dropout"):
                    skipped_keys.append(key)
                    continue
                
                # Check if key exists in current model
                if key in model_state_dict:
                    # Check shape compatibility
                    if model_state_dict[key].shape == value.shape:
                        pretrained_state_dict[key] = value
                    else:
                        print(f"Warning: Shape mismatch for {key}: model {model_state_dict[key].shape} vs pretrained {value.shape}. Skipping.")
                        skipped_keys.append(key)
                else:
                    skipped_keys.append(key)
            
            # Load pretrained weights
            missing_keys, unexpected_keys = self.load_state_dict(pretrained_state_dict, strict=False)
            
            if pretrained_state_dict:
                print(f"Successfully loaded {len(pretrained_state_dict)} pretrained layers.")
                if skipped_keys:
                    print(f"Skipped {len(skipped_keys)} layers (classification head or incompatible shapes).")
                if missing_keys:
                    print(f"Warning: {len(missing_keys)} model layers were not found in pretrained weights (will use random initialization).")
            else:
                print("Warning: No compatible pretrained weights found. Training from scratch.")
            
            # Freeze backbone if requested
            freeze_backbone = pretrained_config.get("freeze_backbone", False)
            if freeze_backbone:
                print("Freezing backbone layers (only classification head will be trained).")
                for name, param in self.named_parameters():
                    # Freeze all except fc layer
                    if not name.startswith("fc"):
                        param.requires_grad = False
                        
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Training from scratch.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
        
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        # Initial conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)  # (B, 64, 1250)
        x = self.layer2(x)  # (B, 128, 625)
        x = self.layer3(x)  # (B, 256, 313)
        x = self.layer4(x)  # (B, 512, 157)
        
        # Global Average Pooling
        x = self.global_pool(x)  # (B, 512, 1)
        x = x.squeeze(-1)  # (B, 512)
        
        # Classification Head
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification head.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
        
        Returns:
            features: Feature tensor of shape (B, 512) after global pooling and before fc.
        """
        # Initial conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)  # (B, 64, 1250)
        x = self.layer2(x)  # (B, 128, 625)
        x = self.layer3(x)  # (B, 256, 313)
        x = self.layer4(x)  # (B, 512, 157)
        
        # Global Average Pooling
        x = self.global_pool(x)  # (B, 512, 1)
        x = x.squeeze(-1)  # (B, 512)
        
        # Apply dropout but not final FC layer
        x = self.dropout(x)
        
        return x  # (B, 512) - features before final fc layer

