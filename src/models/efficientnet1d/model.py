"""EfficientNet1D-B1 model for ECG classification.

Based on Tan & Le (2019) EfficientNet paper, adapted for 1D ECG signals.
Architecture follows EfficientNet-B1 specifications with ~7.8M parameters.
"""

from typing import Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
from ..base_model import BaseECGModel
from .blocks import MBConvBlock1D, Swish


class EfficientNet1D_B1(BaseECGModel):
    """EfficientNet1D-B1 architecture for ECG classification.
    
    Architecture (from EfficientNet-B1 paper):
    - Stem: Conv1d 12→32, kernel=3, stride=2, padding=1
    - Stage 1: 2x MBConv1 (t=1, k=3, Cout=16, s=1)
    - Stage 2: 3x MBConv6 (t=6, k=3, Cout=24, s=2)
    - Stage 3: 3x MBConv6 (t=6, k=5, Cout=40, s=2)
    - Stage 4: 4x MBConv6 (t=6, k=3, Cout=80, s=2)
    - Stage 5: 4x MBConv6 (t=6, k=5, Cout=112, s=1)
    - Stage 6: 5x MBConv6 (t=6, k=5, Cout=192, s=2)
    - Stage 7: 1x MBConv6 (t=6, k=3, Cout=320, s=1)
    - Head: Conv1d 320→1280 → AdaptiveAvgPool1d(1) → Dropout → FC 1280→num_classes
    
    Total: ~7.8M parameters
    
    Input: (B, 12, 5000) - 12 ECG leads, 5000 time steps
    Output: (B, num_classes) - Classification logits
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize EfficientNet1D-B1 model.
        
        Args:
            config: Configuration dictionary. Should contain:
                - num_classes: Number of output classes
                - dropout_rate: Dropout rate (from baseline.yaml)
                - model: Model-specific parameters (optional)
        """
        super().__init__(config)
        
        # Get dropout rate from config
        training_config = config.get("training", {})
        dropout_rate = training_config.get("dropout_rate", 0.3)
        
        # Get num_classes from model config or use from base class
        model_config = config.get("model", {})
        num_classes = model_config.get("num_classes", self.num_classes)
        self.num_classes = num_classes
        
        # Input: (B, 12, 5000) - 12 leads, 5000 time steps
        
        # Stem: Initial convolution
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=12,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,  # Same padding
                bias=False
            ),
            nn.BatchNorm1d(32),
            Swish()
        )
        # Output: (B, 32, 2500)
        
        # Stage 1: 2x MBConv1 (t=1, k=3, Cout=16, s=1)
        self.stage1 = nn.Sequential(
            MBConvBlock1D(in_channels=32, out_channels=16, kernel_size=3, stride=1, expansion_factor=1),
            MBConvBlock1D(in_channels=16, out_channels=16, kernel_size=3, stride=1, expansion_factor=1)
        )
        # Output: (B, 16, 2500)
        
        # Stage 2: 3x MBConv6 (t=6, k=3, Cout=24, s=2)
        self.stage2 = nn.Sequential(
            MBConvBlock1D(in_channels=16, out_channels=24, kernel_size=3, stride=2, expansion_factor=6),
            MBConvBlock1D(in_channels=24, out_channels=24, kernel_size=3, stride=1, expansion_factor=6),
            MBConvBlock1D(in_channels=24, out_channels=24, kernel_size=3, stride=1, expansion_factor=6)
        )
        # Output: (B, 24, 1250)
        
        # Stage 3: 3x MBConv6 (t=6, k=5, Cout=40, s=2)
        self.stage3 = nn.Sequential(
            MBConvBlock1D(in_channels=24, out_channels=40, kernel_size=5, stride=2, expansion_factor=6),
            MBConvBlock1D(in_channels=40, out_channels=40, kernel_size=5, stride=1, expansion_factor=6),
            MBConvBlock1D(in_channels=40, out_channels=40, kernel_size=5, stride=1, expansion_factor=6)
        )
        # Output: (B, 40, 625)
        
        # Stage 4: 4x MBConv6 (t=6, k=3, Cout=80, s=2)
        self.stage4 = nn.Sequential(
            MBConvBlock1D(in_channels=40, out_channels=80, kernel_size=3, stride=2, expansion_factor=6),
            MBConvBlock1D(in_channels=80, out_channels=80, kernel_size=3, stride=1, expansion_factor=6),
            MBConvBlock1D(in_channels=80, out_channels=80, kernel_size=3, stride=1, expansion_factor=6),
            MBConvBlock1D(in_channels=80, out_channels=80, kernel_size=3, stride=1, expansion_factor=6)
        )
        # Output: (B, 80, 313)
        
        # Stage 5: 4x MBConv6 (t=6, k=5, Cout=112, s=1)
        self.stage5 = nn.Sequential(
            MBConvBlock1D(in_channels=80, out_channels=112, kernel_size=5, stride=1, expansion_factor=6),
            MBConvBlock1D(in_channels=112, out_channels=112, kernel_size=5, stride=1, expansion_factor=6),
            MBConvBlock1D(in_channels=112, out_channels=112, kernel_size=5, stride=1, expansion_factor=6),
            MBConvBlock1D(in_channels=112, out_channels=112, kernel_size=5, stride=1, expansion_factor=6)
        )
        # Output: (B, 112, 313)
        
        # Stage 6: 5x MBConv6 (t=6, k=5, Cout=192, s=2)
        self.stage6 = nn.Sequential(
            MBConvBlock1D(in_channels=112, out_channels=192, kernel_size=5, stride=2, expansion_factor=6),
            MBConvBlock1D(in_channels=192, out_channels=192, kernel_size=5, stride=1, expansion_factor=6),
            MBConvBlock1D(in_channels=192, out_channels=192, kernel_size=5, stride=1, expansion_factor=6),
            MBConvBlock1D(in_channels=192, out_channels=192, kernel_size=5, stride=1, expansion_factor=6),
            MBConvBlock1D(in_channels=192, out_channels=192, kernel_size=5, stride=1, expansion_factor=6)
        )
        # Output: (B, 192, 157)
        
        # Stage 7: 1x MBConv6 (t=6, k=3, Cout=320, s=1)
        self.stage7 = nn.Sequential(
            MBConvBlock1D(in_channels=192, out_channels=320, kernel_size=3, stride=1, expansion_factor=6)
        )
        # Output: (B, 320, 157)
        
        # Head: Final layers
        self.head = nn.Sequential(
            nn.Conv1d(
                in_channels=320,
                out_channels=1280,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm1d(1280),
            Swish()
        )
        # Output: (B, 1280, 157)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Output: (B, 1280, 1) -> (B, 1280)
        
        # Classification Head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1280, self.num_classes)
        
        # Load pretrained weights if enabled
        self._load_pretrained_weights(config)
    
    def _load_pretrained_weights(self, config: Dict[str, Any]) -> None:
        """Load pretrained weights if enabled in config.
        
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
            # Try to find project root (go up from src/models/efficientnet1d/)
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
        # Stem
        x = self.stem(x)  # (B, 32, 2500)
        
        # Stages
        x = self.stage1(x)  # (B, 16, 2500)
        x = self.stage2(x)  # (B, 24, 1250)
        x = self.stage3(x)  # (B, 40, 625)
        x = self.stage4(x)  # (B, 80, 313)
        x = self.stage5(x)  # (B, 112, 313)
        x = self.stage6(x)  # (B, 192, 157)
        x = self.stage7(x)  # (B, 320, 157)
        
        # Head
        x = self.head(x)  # (B, 1280, 157)
        
        # Global Average Pooling
        x = self.global_pool(x)  # (B, 1280, 1)
        x = x.squeeze(-1)  # (B, 1280)
        
        # Classification Head
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification head.
        
        Args:
            x: Input tensor of shape (B, 12, 5000)
        
        Returns:
            features: Feature tensor of shape (B, 1280) after global pooling and before fc.
        """
        # Stem
        x = self.stem(x)  # (B, 32, 2500)
        
        # Stages
        x = self.stage1(x)  # (B, 16, 2500)
        x = self.stage2(x)  # (B, 24, 1250)
        x = self.stage3(x)  # (B, 40, 625)
        x = self.stage4(x)  # (B, 80, 313)
        x = self.stage5(x)  # (B, 112, 313)
        x = self.stage6(x)  # (B, 192, 157)
        x = self.stage7(x)  # (B, 320, 157)
        
        # Head
        x = self.head(x)  # (B, 1280, 157)
        
        # Global Average Pooling
        x = self.global_pool(x)  # (B, 1280, 1)
        x = x.squeeze(-1)  # (B, 1280)
        
        # Apply dropout but not final FC layer
        x = self.dropout(x)
        
        return x  # (B, 1280) - features before final fc layer

