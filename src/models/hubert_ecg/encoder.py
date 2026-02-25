"""HuBERT-ECG Encoder Wrapper for Multi-Task Learning.

Architecture follows the ORIGINAL HuBERT-ECG paper repository:
https://github.com/Edoar-do/HuBERT-ECG

Paper: "HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications"

Key components:
- HuBERTECG: Base transformer model extending HubertModel from transformers
- HuBERTForECGClassification: Classification wrapper with mean pooling
- Input: (B, 12, 5000) @ 500Hz flattened to (B, 60000)
- Output: (B, hidden_size=768)
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add clones directory to path for HuBERT-ECG imports (ORIGINAL Paper repository)
project_root = Path(__file__).parent.parent.parent.parent
clones_path = project_root / "clones" / "HuBERT-ECG" / "code"
if str(clones_path) not in sys.path:
    sys.path.insert(0, str(clones_path))

from hubert_ecg import HuBERTECG, HuBERTECGConfig


class HuBERTEncoder(nn.Module):
    """HuBERT-ECG Encoder wrapper following original ecg-fm-benchmarking architecture.
    
    Loads pretrained HuBERT-ECG model and extracts encoder features.
    Removes pretraining heads (final_proj, label_embedding) and disables masking.
    
    Input: (B, 12, 5000) @ 500Hz
    Output: (B, hidden_size=768) after mean pooling
    
    Architecture matches HuBERTForECGClassification from original repo:
    1. Flatten input: (B, 12, 5000) -> (B, 60000)
    2. HuBERT forward: (B, 60000) -> (B, seq_len, 768)
    3. Mean pooling: (B, seq_len, 768) -> (B, 768)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """Initialize HuBERT Encoder.
        
        Args:
            config: Configuration dictionary. Should contain:
                - model.hubert.checkpoint_path: Path to checkpoint file
                - model.hubert.config_path: Optional path to config.json
                - model.hubert.hidden_size: Hidden dimension (default: 768)
                - model.pretrained.cache_dir: Cache directory for models
            device: Device to load model on (optional)
        """
        super().__init__()
        
        model_config = config.get("model", {})
        hubert_config = model_config.get("hubert", {})
        pretrained_config = model_config.get("pretrained", {})
        
        checkpoint_path = hubert_config.get("checkpoint_path")
        config_path = hubert_config.get("config_path", None)
        self.hidden_size = hubert_config.get("hidden_size", 768)
        cache_dir = pretrained_config.get("cache_dir", "data/pretrained_weights/Hubert_ECG/base")
        
        # Resolve cache directory path (relative to project root)
        # Path: src/models/hubert_ecg/encoder.py -> parent.parent.parent.parent = MA-thesis-1/
        cache_dir_path = Path(cache_dir)
        if not cache_dir_path.is_absolute():
            project_root_path = Path(__file__).parent.parent.parent.parent  # src/../../../.. = MA-thesis-1/
            cache_dir_path = project_root_path / cache_dir
        
        # Resolve checkpoint path
        checkpoint_path_obj = Path(checkpoint_path) if checkpoint_path else None
        if checkpoint_path_obj:
            if checkpoint_path_obj.is_absolute():
                if not checkpoint_path_obj.exists():
                    checkpoint_filename = checkpoint_path_obj.name
                    checkpoint_path_obj = cache_dir_path / checkpoint_filename
            else:
                checkpoint_path_obj = cache_dir_path / checkpoint_path
        
        # Determine device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Load HuBERT config from JSON file (required for original HuBERT-ECG)
        config_path_obj = None
        if config_path:
            config_path_obj = Path(config_path)
            if not config_path_obj.is_absolute():
                config_path_obj = cache_dir_path / config_path
            elif not config_path_obj.exists():
                config_path_obj = cache_dir_path / config_path_obj.name
        else:
            # Default: look for config.json in cache directory
            config_path_obj = cache_dir_path / "config.json"
        
        if config_path_obj and config_path_obj.exists():
            import json
            with open(config_path_obj, 'r') as f:
                config_dict = json.load(f)
            hubert_model_config = HuBERTECGConfig(**config_dict)
            print(f"Loaded HuBERT config from: {config_path_obj}")
        else:
            # Fallback: create default HuBERT-ECG config matching paper specifications
            print(f"Warning: Config file not found, using default HuBERT-ECG paper config")
            hubert_model_config = HuBERTECGConfig(
                # HuBERT-ECG base model parameters from paper
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                # Feature extractor for ECG @ 500Hz
                conv_dim=[512, 512, 512, 512, 512, 512, 512],
                conv_stride=[5, 2, 2, 2, 2, 2, 2],
                conv_kernel=[10, 3, 3, 3, 3, 2, 2],
                # Pretraining heads (will be removed for fine-tuning)
                ensemble_length=1,
                vocab_sizes=[100],
            )
        
        # Initialize HuBERT-ECG model (following HuBERTForECGClassification pattern)
        print(f"Initializing HuBERT-ECG model with hidden_size={self.hidden_size}")
        self.hubert_model = HuBERTECG(hubert_model_config)
        
        # Disable masking for fine-tuning (as in original HuBERTForECGClassification)
        self.hubert_model.config.mask_time_prob = 0.0
        self.hubert_model.config.mask_feature_prob = 0.0
        
        # Load pretrained weights if checkpoint path is provided
        if checkpoint_path_obj and checkpoint_path_obj.exists():
            print(f"Loading HuBERT-ECG checkpoint from: {checkpoint_path_obj}")
            self._load_checkpoint(str(checkpoint_path_obj))
        elif checkpoint_path:
            print(f"Warning: Checkpoint not found at {checkpoint_path_obj}. Model will use random weights.")
        
        # Remove pretraining heads (following HuBERTForECGClassification pattern)
        if hasattr(self.hubert_model, 'final_proj'):
            del self.hubert_model.final_proj
        if hasattr(self.hubert_model, 'label_embedding'):
            del self.hubert_model.label_embedding
        
        # Move to device
        self.hubert_model = self.hubert_model.to(device)
        self.hubert_model.eval()  # Set to eval mode initially
        
        print(f"HuBERT Encoder loaded on {device}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint weights.
        
        Args:
            checkpoint_path: Path to checkpoint file (.pt or .safetensors)
        """
        try:
            if checkpoint_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(state_dict, dict):
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "hubert_ecg" in state_dict:
                    state_dict = state_dict["hubert_ecg"]
            
            # Remove prefix if present
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                for prefix in ["model.", "backbone.", "module.", "hubert_ecg."]:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        break
                cleaned_state_dict[new_key] = value
            
            # Filter out pretraining heads
            filtered_state_dict = {}
            for key, value in cleaned_state_dict.items():
                if not key.startswith("final_proj") and not key.startswith("label_embedding"):
                    filtered_state_dict[key] = value
            
            # Load state dict
            missing_keys, unexpected_keys = self.hubert_model.load_state_dict(filtered_state_dict, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}... (showing first 5)")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}... (showing first 5)")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Model will use random weights.")
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """Forward pass through HuBERT encoder.
        
        Architecture follows HuBERTForECGClassification from original repo:
        1. Flatten input: (B, 12, 5000) -> (B, 60000)
        2. HuBERT forward: (B, 60000) -> (B, seq_len, 768)
        3. Mean pooling: (B, seq_len, 768) -> (B, 768)
        
        Args:
            x: Input tensor of shape (B, 12, 5000) @ 500Hz
            attention_mask: Optional attention mask tensor
            return_sequence: If True, return (B, seq_len, 768) before pooling
        
        Returns:
            Encoder output of shape (B, 768) after mean pooling
            Or (B, seq_len, 768) if return_sequence=True
        """
        # Ensure input is on correct device and float
        x = x.to(self.device).float()
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        batch_size = x.shape[0]
        
        # Flatten input: (B, 12, 5000) -> (B, 60000)
        # This matches fm_ecg.py line 970: x = x.reshape(x.shape[0], -1)
        x = x.reshape(batch_size, -1)  # (B, 12*5000) = (B, 60000)
        
        # Forward through HuBERT model
        output = self.hubert_model(
            input_values=x,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        
        # Extract encoder output: (B, seq_len, 768)
        sequence_output = output.last_hidden_state
        
        if return_sequence:
            return sequence_output
        
        # Mean pooling over sequence dimension (following HuBERTForECGClassification)
        # From hubert_ecg_classification.py line 110-111:
        # if attention_mask is None: x = x.mean(dim=1)
        if attention_mask is None:
            pooled_output = sequence_output.mean(dim=1)  # (B, 768)
        else:
            # Weighted mean based on attention mask
            padding_mask = self.hubert_model._get_feature_vector_attention_mask(
                sequence_output.shape[1], attention_mask
            )
            sequence_output[~padding_mask] = 0.0
            pooled_output = sequence_output.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        
        return pooled_output  # (B, 768)
    
    def freeze(self) -> None:
        """Freeze encoder parameters (no gradient updates)."""
        for param in self.hubert_model.parameters():
            param.requires_grad = False
        self.hubert_model.eval()
    
    def unfreeze(self) -> None:
        """Unfreeze encoder parameters (enable gradient updates)."""
        for param in self.hubert_model.parameters():
            param.requires_grad = True
        self.hubert_model.train()
