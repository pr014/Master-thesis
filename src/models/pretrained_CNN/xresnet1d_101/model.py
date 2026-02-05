"""FastAI xResNet1D-101 model for ECG regression/classification.

Based on FastAI xResNet architecture, adapted for 1D ECG signals.
Pretrained weights from PTB-XL dataset.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
from ...core.base_model import BaseECGModel
from .blocks import XResNetBlock1D


class XResNet1D101(BaseECGModel):
    """FastAI xResNet1D-101 architecture for ECG regression/classification.
    
    Architecture:
    - Stem: 3 Convolutional Layers (12 → 32 → 32 → 64)
    - Layer 4: 3 xResNet Blocks (64 → 256)
    - Layer 5: 4 xResNet Blocks (256 → 256)
    - Layer 6: 23 xResNet Blocks (256 → 256) - Hauptteil
    - Layer 7: 3 xResNet Blocks (256 → 256)
    - Head: BatchNorm(512) → Linear(512 → num_classes)
    
    Total: ~3.7M parameters
    
    Input: (B, 12, 5000) - 12 ECG leads, 5000 time steps
    Output: (B, num_classes) - Classification logits
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize XResNet1D-101 model.
        
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
        
        # Stem: Initial convolutions (Layer 0-2)
        self.stem = nn.Sequential(
            # Layer 0: 12 → 32
            nn.Conv1d(
                in_channels=12,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,  # Same padding
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # Layer 1: 32 → 32
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,  # Same padding
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # Layer 2: 32 → 64
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,  # Same padding
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        # Output: (B, 64, 5000)
        
        # Layer 4: 3 xResNet Blocks (64 → 256)
        self.layer4 = nn.Sequential(
            XResNetBlock1D(in_channels=64, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1)
        )
        # Output: (B, 256, 5000)
        
        # Layer 5: 4 xResNet Blocks (256 → 256)
        self.layer5 = nn.Sequential(
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1)
        )
        # Output: (B, 256, 5000)
        
        # Layer 6: 23 xResNet Blocks (256 → 256) - Hauptteil
        self.layer6 = nn.Sequential(
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1)
        )
        # Output: (B, 256, 5000)
        
        # Layer 7: 3 xResNet Blocks (256 → 256)
        self.layer7 = nn.Sequential(
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1),
            XResNetBlock1D(in_channels=256, out_channels=64, expansion=4, kernel_size=5, stride=1)
        )
        # Output: (B, 256, 5000)
        
        # Transition: 256 → 512 (to match pretrained Layer 8 input)
        self.transition = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        # Output: (B, 512, 5000)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Output: (B, 512, 1) -> (B, 512)
        
        # Head: Final layers (replaces Layer 8 from pretrained model)
        # Original: BatchNorm(512) → Linear(512 → 128) → ... → Linear(128 → 71)
        # Ours: BatchNorm(512) → Dropout → Linear(512 → num_classes)
        self.bn_final = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Check if demographic features are enabled
        data_config = config.get("data", {})
        demographic_config = data_config.get("demographic_features", {})
        self.use_demographics = demographic_config.get("enabled", False)
        
        # Check if diagnosis features are enabled
        diagnosis_config = data_config.get("diagnosis_features", {})
        self.use_diagnoses = diagnosis_config.get("enabled", False)
        diagnosis_list = diagnosis_config.get("diagnosis_list", [])
        diagnosis_dim = len(diagnosis_list) if self.use_diagnoses else 0
        
        # Calculate feature dimension (ECG features + optional demographic + diagnosis features)
        feature_dim = 512  # ECG features after global pooling
        if self.use_demographics:
            sex_encoding = demographic_config.get("sex_encoding", "binary")
            demo_dim = 2 if sex_encoding == "binary" else 3  # binary: [age, sex], onehot: [age, sex_0, sex_1]
            feature_dim += demo_dim
        if self.use_diagnoses:
            feature_dim += diagnosis_dim
        
        # Output layer: 1 neuron for regression, num_classes for classification
        output_dim = 1 if self.task_type == "regression" else (self.num_classes or 10)
        self.fc = nn.Linear(feature_dim, output_dim)
        
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
            # Try to find project root (go up from src/models/pretrained_CNN/xresnet1d_101/)
            project_root = Path(__file__).parent.parent.parent.parent.parent
            weights_path = project_root / weights_path
        
        if not weights_path.exists():
            print(f"Warning: Pretrained weights file not found at {weights_path}. Training from scratch.")
            return
        
        try:
            print(f"Loading pretrained weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            
            # FastAI checkpoint format: {'model': state_dict, 'opt': optimizer_state}
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                pretrained_state_dict = checkpoint["model"]
            else:
                print("Warning: Checkpoint format not recognized. Training from scratch.")
                return
            
            # Map FastAI layer names to our model structure
            model_state_dict = self.state_dict()
            mapped_state_dict = {}
            skipped_keys = []
            
            # Mapping rules:
            # FastAI: {layer}.{block}.{path}.{conv_idx}.{param}
            # Ours: {layer_name}.{block_idx}.{path}.{conv_idx}.{param}
            
            for fastai_key, value in pretrained_state_dict.items():
                if not isinstance(value, torch.Tensor):
                    continue
                
                # Skip Layer 8 (head) - we use our own head
                if fastai_key.startswith("8."):
                    skipped_keys.append(fastai_key)
                    continue
                
                # Parse FastAI key
                parts = fastai_key.split(".")
                if len(parts) < 2:
                    skipped_keys.append(fastai_key)
                    continue
                
                layer_num = parts[0]
                our_key = None
                
                # Map stem layers (0, 1, 2)
                # Our stem is Sequential: [Conv0, BN0, ReLU, Conv1, BN1, ReLU, Conv2, BN2, ReLU]
                # FastAI: Layer 0 → stem[0,1], Layer 1 → stem[3,4], Layer 2 → stem[6,7]
                if layer_num == "0":
                    # FastAI: "0.0.weight" → "stem.0.weight" (Conv)
                    # FastAI: "0.1.*" → "stem.1.*" (BN)
                    if parts[1] == "0" and len(parts) == 2:
                        our_key = "stem.0.weight"
                    elif parts[1] == "1":
                        our_key = f"stem.1.{'.'.join(parts[2:])}"
                elif layer_num == "1":
                    # FastAI: "1.0.weight" → "stem.3.weight" (Conv)
                    # FastAI: "1.1.*" → "stem.4.*" (BN)
                    if parts[1] == "0" and len(parts) == 2:
                        our_key = "stem.3.weight"
                    elif parts[1] == "1":
                        our_key = f"stem.4.{'.'.join(parts[2:])}"
                elif layer_num == "2":
                    # FastAI: "2.0.weight" → "stem.6.weight" (Conv)
                    # FastAI: "2.1.*" → "stem.7.*" (BN)
                    if parts[1] == "0" and len(parts) == 2:
                        our_key = "stem.6.weight"
                    elif parts[1] == "1":
                        our_key = f"stem.7.{'.'.join(parts[2:])}"
                # Map residual block layers (4, 5, 6, 7)
                elif layer_num in ["4", "5", "6", "7"]:
                    # FastAI: "4.0.convs.0.0.weight" → "layer4.0.convs.0.weight"
                    # Our Sequential: convs[0]=Conv, convs[1]=BN, convs[3]=Conv, convs[4]=BN, convs[6]=Conv, convs[7]=BN
                    # Mapping: conv_idx * 3 for Conv, conv_idx * 3 + 1 for BN
                    layer_name = f"layer{layer_num}"
                    if len(parts) >= 2:
                        block_idx = parts[1]
                        # Handle convs and convpath structure
                        if len(parts) >= 4 and parts[2] in ["convs", "convpath"]:
                            path_type = parts[2]
                            conv_idx_str = parts[3]
                            if conv_idx_str.isdigit():
                                conv_idx = int(conv_idx_str)
                                # FastAI structure: convs.0.0.weight (Conv), convs.0.1.weight (BN)
                                if len(parts) >= 5:
                                    param_idx = parts[4]
                                    if param_idx == "0" and len(parts) == 5:
                                        # Conv weight: convs.0.0.weight → convs.0.weight
                                        seq_idx = conv_idx * 3
                                        our_key = f"{layer_name}.{block_idx}.{path_type}.{seq_idx}.weight"
                                    elif param_idx == "1":
                                        # BN params: convs.0.1.weight → convs.1.weight
                                        seq_idx = conv_idx * 3 + 1
                                        if len(parts) > 5:
                                            our_key = f"{layer_name}.{block_idx}.{path_type}.{seq_idx}.{'.'.join(parts[5:])}"
                                        else:
                                            # Just weight or bias
                                            if "weight" in fastai_key:
                                                our_key = f"{layer_name}.{block_idx}.{path_type}.{seq_idx}.weight"
                                            elif "bias" in fastai_key:
                                                our_key = f"{layer_name}.{block_idx}.{path_type}.{seq_idx}.bias"
                                            else:
                                                our_key = f"{layer_name}.{block_idx}.{path_type}.{seq_idx}.{'.'.join(parts[5:])}"
                                    else:
                                        # Other params (running_mean, etc.)
                                        seq_idx = conv_idx * 3 + 1  # Usually BN params
                                        our_key = f"{layer_name}.{block_idx}.{path_type}.{seq_idx}.{'.'.join(parts[4:])}"
                                else:
                                    # Fallback
                                    rest = ".".join(parts[3:])
                                    our_key = f"{layer_name}.{block_idx}.{path_type}.{rest}"
                            else:
                                rest = ".".join(parts[3:])
                                our_key = f"{layer_name}.{block_idx}.{path_type}.{rest}"
                        else:
                            rest = ".".join(parts[2:])
                            our_key = f"{layer_name}.{block_idx}.{rest}"
                
                # Check if key exists and shapes match
                if our_key and our_key in model_state_dict:
                    if model_state_dict[our_key].shape == value.shape:
                        mapped_state_dict[our_key] = value
                    else:
                        # Don't print too many warnings
                        if len(skipped_keys) < 10:
                            print(f"Warning: Shape mismatch for {fastai_key} → {our_key}: "
                                  f"model {model_state_dict[our_key].shape} vs pretrained {value.shape}. Skipping.")
                        skipped_keys.append(fastai_key)
                elif our_key:
                    # Key mapping failed - try to find similar key
                    # This is a fallback for debugging
                    skipped_keys.append(fastai_key)
                else:
                    skipped_keys.append(fastai_key)
            
            # Load mapped weights
            if mapped_state_dict:
                missing_keys, unexpected_keys = self.load_state_dict(mapped_state_dict, strict=False)
                print(f"Successfully loaded {len(mapped_state_dict)} pretrained layers.")
                if skipped_keys:
                    print(f"Skipped {len(skipped_keys)} layers (head or incompatible).")
                if missing_keys:
                    print(f"Warning: {len(missing_keys)} model layers were not found in pretrained weights.")
            else:
                print("Warning: No compatible pretrained weights found. Training from scratch.")
            
            # Freeze backbone if requested
            freeze_backbone = pretrained_config.get("freeze_backbone", False)
            if freeze_backbone:
                print("Freezing backbone layers (only classification head will be trained).")
                for name, param in self.named_parameters():
                    # Freeze all except fc and bn_final
                    if not name.startswith("fc") and not name.startswith("bn_final"):
                        param.requires_grad = False
                        
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            import traceback
            traceback.print_exc()
            print("Training from scratch.")
    
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
        # Stem
        x = self.stem(x)  # (B, 64, 5000)
        
        # Residual Blocks
        x = self.layer4(x)  # (B, 256, 5000)
        x = self.layer5(x)  # (B, 256, 5000)
        x = self.layer6(x)  # (B, 256, 5000)
        x = self.layer7(x)  # (B, 256, 5000)
        
        # Transition: 256 → 512
        x = self.transition(x)  # (B, 512, 5000)
        
        # Global Average Pooling
        x = self.global_pool(x)  # (B, 512, 1)
        x = x.squeeze(-1)  # (B, 512)
        
        # Head
        ecg_features = self.bn_final(x)  # (B, 512)
        
        # Late fusion: Concatenate ECG features with demographic and diagnosis features
        fused_features = ecg_features
        if self.use_demographics and demographic_features is not None:
            fused_features = torch.cat([fused_features, demographic_features], dim=1)
        if self.use_diagnoses and diagnosis_features is not None:
            fused_features = torch.cat([fused_features, diagnosis_features], dim=1)
        
        x = self.dropout(fused_features)
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
            features: Feature tensor of shape (B, 512) or (B, 512+2/3+diagnosis_dim) after fusion.
                     If demographic/diagnosis features are enabled, includes them.
        """
        # Stem
        x = self.stem(x)  # (B, 64, 5000)
        
        # Residual Blocks
        x = self.layer4(x)  # (B, 256, 5000)
        x = self.layer5(x)  # (B, 256, 5000)
        x = self.layer6(x)  # (B, 256, 5000)
        x = self.layer7(x)  # (B, 256, 5000)
        
        # Transition: 256 → 512
        x = self.transition(x)  # (B, 512, 5000)
        
        # Global Average Pooling
        x = self.global_pool(x)  # (B, 512, 1)
        x = x.squeeze(-1)  # (B, 512)
        
        # Apply BN and dropout but not final FC layer
        ecg_features = self.bn_final(x)  # (B, 512)
        
        # Late fusion: Concatenate ECG features with demographic and diagnosis features
        fused_features = ecg_features
        if self.use_demographics and demographic_features is not None:
            fused_features = torch.cat([fused_features, demographic_features], dim=1)
        if self.use_diagnoses and diagnosis_features is not None:
            fused_features = torch.cat([fused_features, diagnosis_features], dim=1)
        
        # Apply dropout but not final FC layer
        fused_features = self.dropout(fused_features)
        
        return fused_features  # (B, 512) or (B, 512+2/3+diagnosis_dim) - features before final fc layer

