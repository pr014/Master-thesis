"""Checkpoint utilities for HuBERT-ECG model loading."""

from typing import Dict, Any, Optional, Tuple
import torch
from pathlib import Path
import json


def load_hubert_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    map_location: str = "cpu"
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
    """Load HuBERT-ECG checkpoint and config.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt or .safetensors)
        config_path: Optional path to config.json file
        map_location: Device to load checkpoint on (default: "cpu")
    
    Returns:
        Tuple of (state_dict, config_dict)
        - state_dict: Model state dictionary
        - config_dict: Config dictionary (None if config_path not provided or not found)
    """
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    if checkpoint_path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        except ImportError:
            raise ImportError("safetensors library not installed. Install with: pip install safetensors")
    else:
        # Use torch.load
        state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(state_dict, dict):
        # Check if it's a full checkpoint with 'model_state_dict' or 'state_dict'
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "hubert_ecg" in state_dict:
            # Nested structure
            state_dict = state_dict["hubert_ecg"]
    
    # Remove prefix if present (e.g., "model.", "backbone.", "module.")
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ["model.", "backbone.", "module.", "hubert_ecg."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        cleaned_state_dict[new_key] = value
    
    # Filter out pretraining head components (final_proj, label_embedding)
    filtered_state_dict = {}
    for key, value in cleaned_state_dict.items():
        if not key.startswith("final_proj") and not key.startswith("label_embedding"):
            filtered_state_dict[key] = value
    
    # Load config if provided
    config_dict = None
    if config_path:
        config_path_obj = Path(config_path)
        if config_path_obj.exists():
            with open(config_path_obj, 'r') as f:
                config_dict = json.load(f)
        else:
            print(f"Warning: Config file not found at {config_path_obj}")
    
    return filtered_state_dict, config_dict

