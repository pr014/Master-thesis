#!/usr/bin/env python3
"""Analyze ECG-CPC pretrained checkpoint structure.

This script analyzes the ECG-CPC checkpoint file to understand:
- Checkpoint structure and format
- Layer names and shapes
- Compatibility with our ECG_S4_CPC model
- Mapping suggestions for loading weights
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.ecg_cpc import ECG_S4_CPC
from src.utils.config_loader import load_config


def analyze_checkpoint_structure(checkpoint: Any, max_keys: int = 50) -> Dict[str, Any]:
    """Analyze the structure of a checkpoint file.
    
    Args:
        checkpoint: Loaded checkpoint (dict, state_dict, or other)
        max_keys: Maximum number of keys to display in sample
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "type": type(checkpoint).__name__,
        "is_dict": isinstance(checkpoint, dict),
        "top_level_keys": [],
        "has_state_dict": False,
        "has_model_state_dict": False,
        "state_dict_keys": [],
        "num_layers": 0,
        "layer_types": {},
        "sample_keys": [],
        "state_dict": None,
    }
    
    if isinstance(checkpoint, dict):
        analysis["top_level_keys"] = list(checkpoint.keys())
        
        # Check for PyTorch Lightning format
        # Lightning checkpoints typically have: 'state_dict', 'epoch', 'global_step', etc.
        state_dict = None
        
        if "state_dict" in checkpoint:
            analysis["has_state_dict"] = True
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            analysis["has_model_state_dict"] = True
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            # Some checkpoints have nested 'model' key
            model_obj = checkpoint["model"]
            if isinstance(model_obj, dict):
                if all(isinstance(v, torch.Tensor) for v in model_obj.values() if v is not None):
                    state_dict = model_obj
                elif "state_dict" in model_obj:
                    state_dict = model_obj["state_dict"]
        else:
            # Check if top-level dict itself is a state_dict
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values() if v is not None):
                state_dict = checkpoint
        
        if isinstance(state_dict, dict):
            analysis["state_dict"] = state_dict
            analysis["state_dict_keys"] = list(state_dict.keys())
            analysis["num_layers"] = len(state_dict)
            
            # Analyze layer types
            for key in state_dict.keys():
                layer_type = key.split('.')[0] if '.' in key else key
                if layer_type not in analysis["layer_types"]:
                    analysis["layer_types"][layer_type] = []
                analysis["layer_types"][layer_type].append(key)
            
            # Sample keys (first max_keys)
            analysis["sample_keys"] = analysis["state_dict_keys"][:max_keys]
    
    return analysis


def get_shape_info(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple[int, ...]]:
    """Get shape information for all tensors in state_dict.
    
    Args:
        state_dict: State dictionary with tensors
        
    Returns:
        Dictionary mapping keys to shapes
    """
    shapes = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            shapes[key] = tuple(value.shape)
    return shapes


def find_s4_layers(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Find all S4-related layers in the checkpoint.
    
    Args:
        state_dict: State dictionary
        
    Returns:
        List of keys that appear to be S4 layers
    """
    s4_keys = []
    for key in state_dict.keys():
        key_lower = key.lower()
        if any(term in key_lower for term in ['s4', 'state_space', 'ssm', 'pred']):
            s4_keys.append(key)
    return s4_keys


def find_encoder_layers(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Find encoder-related layers in the checkpoint.
    
    Args:
        state_dict: State dictionary
        
    Returns:
        List of keys that appear to be encoder layers
    """
    encoder_keys = []
    for key in state_dict.keys():
        key_lower = key.lower()
        if any(term in key_lower for term in ['enc', 'encoder', 'rnn', 'lstm']):
            encoder_keys.append(key)
    return encoder_keys


def compare_with_our_model(
    checkpoint_state_dict: Dict[str, torch.Tensor],
    our_model: ECG_S4_CPC
) -> Dict[str, Any]:
    """Compare checkpoint structure with our model.
    
    Args:
        checkpoint_state_dict: State dict from checkpoint
        our_model: Our ECG_S4_CPC model instance
        
    Returns:
        Comparison results
    """
    our_state_dict = our_model.s4_encoder.state_dict()
    
    comparison = {
        "checkpoint_keys": list(checkpoint_state_dict.keys()),
        "our_keys": list(our_state_dict.keys()),
        "matching_keys": [],
        "shape_matches": [],
        "shape_mismatches": [],
        "missing_in_checkpoint": [],
        "extra_in_checkpoint": [],
    }
    
    # Find matching keys
    for our_key in our_state_dict.keys():
        # Try exact match
        if our_key in checkpoint_state_dict:
            comparison["matching_keys"].append(our_key)
            our_shape = our_state_dict[our_key].shape
            checkpoint_shape = checkpoint_state_dict[our_key].shape
            if our_shape == checkpoint_shape:
                comparison["shape_matches"].append((our_key, our_shape))
            else:
                comparison["shape_mismatches"].append((our_key, our_shape, checkpoint_shape))
        else:
            # Try to find with prefix removal
            found = False
            for checkpoint_key in checkpoint_state_dict.keys():
                # Remove common prefixes
                clean_checkpoint_key = checkpoint_key
                for prefix in ["model.", "backbone.", "module.", "s4_encoder.", "ts.pred.", "ts.enc."]:
                    if clean_checkpoint_key.startswith(prefix):
                        clean_checkpoint_key = clean_checkpoint_key[len(prefix):]
                        break
                
                if clean_checkpoint_key == our_key:
                    comparison["matching_keys"].append(f"{checkpoint_key} -> {our_key}")
                    our_shape = our_state_dict[our_key].shape
                    checkpoint_shape = checkpoint_state_dict[checkpoint_key].shape
                    if our_shape == checkpoint_shape:
                        comparison["shape_matches"].append((our_key, our_shape, checkpoint_key))
                    else:
                        comparison["shape_mismatches"].append((our_key, our_shape, checkpoint_shape, checkpoint_key))
                    found = True
                    break
            
            if not found:
                comparison["missing_in_checkpoint"].append(our_key)
    
    # Find extra keys in checkpoint
    for checkpoint_key in checkpoint_state_dict.keys():
        if checkpoint_key not in our_state_dict:
            # Check if it's an S4-related layer we might want
            if any(term in checkpoint_key.lower() for term in ['s4', 'state_space', 'ssm']):
                comparison["extra_in_checkpoint"].append(checkpoint_key)
    
    return comparison


def print_analysis(
    checkpoint_path: Path,
    analysis: Dict[str, Any],
    shapes: Dict[str, Tuple[int, ...]],
    s4_layers: List[str],
    encoder_layers: List[str],
    comparison: Optional[Dict[str, Any]] = None
):
    """Print analysis results.
    
    Args:
        checkpoint_path: Path to checkpoint file
        analysis: Analysis dictionary
        shapes: Shape information
        s4_layers: List of S4 layer keys
        encoder_layers: List of encoder layer keys
        comparison: Optional comparison with our model
    """
    print("=" * 80)
    print(f"ECG-CPC Checkpoint Analysis: {checkpoint_path.name}")
    print("=" * 80)
    
    print(f"\n[1] Checkpoint Type: {analysis['type']}")
    print(f"    Is Dict: {analysis['is_dict']}")
    
    if analysis['top_level_keys']:
        print(f"\n[2] Top-Level Keys ({len(analysis['top_level_keys'])}):")
        for key in analysis['top_level_keys'][:20]:
            print(f"    - {key}")
        if len(analysis['top_level_keys']) > 20:
            print(f"    ... and {len(analysis['top_level_keys']) - 20} more")
    
    if analysis['state_dict']:
        print(f"\n[3] State Dict Info:")
        print(f"    Total Layers: {analysis['num_layers']}")
        print(f"    Has 'state_dict' key: {analysis['has_state_dict']}")
        print(f"    Has 'model_state_dict' key: {analysis['has_model_state_dict']}")
        
        print(f"\n[4] Layer Types Found:")
        for layer_type, keys in analysis['layer_types'].items():
            print(f"    {layer_type}: {len(keys)} layers")
            if len(keys) <= 5:
                for key in keys:
                    print(f"      - {key}")
            else:
                for key in keys[:3]:
                    print(f"      - {key}")
                print(f"      ... and {len(keys) - 3} more")
        
        print(f"\n[5] S4-Related Layers ({len(s4_layers)}):")
        for key in s4_layers[:20]:
            shape = shapes.get(key, "N/A")
            print(f"    - {key}: {shape}")
        if len(s4_layers) > 20:
            print(f"    ... and {len(s4_layers) - 20} more")
        
        print(f"\n[6] Encoder-Related Layers ({len(encoder_layers)}):")
        for key in encoder_layers[:20]:
            shape = shapes.get(key, "N/A")
            print(f"    - {key}: {shape}")
        if len(encoder_layers) > 20:
            print(f"    ... and {len(encoder_layers) - 20} more")
        
        print(f"\n[7] Sample Keys (first 20):")
        for key in analysis['sample_keys'][:20]:
            shape = shapes.get(key, "N/A")
            print(f"    - {key}: {shape}")
    
    if comparison:
        print(f"\n[8] Comparison with Our Model:")
        print(f"    Checkpoint Keys: {len(comparison['checkpoint_keys'])}")
        print(f"    Our Model Keys: {len(comparison['our_keys'])}")
        print(f"    Matching Keys: {len(comparison['matching_keys'])}")
        print(f"    Shape Matches: {len(comparison['shape_matches'])}")
        print(f"    Shape Mismatches: {len(comparison['shape_mismatches'])}")
        print(f"    Missing in Checkpoint: {len(comparison['missing_in_checkpoint'])}")
        print(f"    Extra in Checkpoint: {len(comparison['extra_in_checkpoint'])}")
        
        if comparison['shape_matches']:
            print(f"\n    ✓ Shape Matches ({len(comparison['shape_matches'])}):")
            for match in comparison['shape_matches'][:10]:
                if len(match) == 2:
                    print(f"      - {match[0]}: {match[1]}")
                else:
                    print(f"      - {match[0]}: {match[1]} (from {match[2]})")
            if len(comparison['shape_matches']) > 10:
                print(f"      ... and {len(comparison['shape_matches']) - 10} more")
        
        if comparison['shape_mismatches']:
            print(f"\n    ✗ Shape Mismatches ({len(comparison['shape_mismatches'])}):")
            for mismatch in comparison['shape_mismatches'][:10]:
                if len(mismatch) == 3:
                    print(f"      - {mismatch[0]}: our {mismatch[1]} vs checkpoint {mismatch[2]}")
                else:
                    print(f"      - {mismatch[0]}: our {mismatch[1]} vs checkpoint {mismatch[2]} (from {mismatch[3]})")
            if len(comparison['shape_mismatches']) > 10:
                print(f"      ... and {len(comparison['shape_mismatches']) - 10} more")
        
        if comparison['missing_in_checkpoint']:
            print(f"\n    ⚠ Missing in Checkpoint ({len(comparison['missing_in_checkpoint'])}):")
            for key in comparison['missing_in_checkpoint'][:10]:
                print(f"      - {key}")
            if len(comparison['missing_in_checkpoint']) > 10:
                print(f"      ... and {len(comparison['missing_in_checkpoint']) - 10} more")
        
        if comparison['extra_in_checkpoint']:
            print(f"\n    ℹ Extra in Checkpoint (S4-related, {len(comparison['extra_in_checkpoint'])}):")
            for key in comparison['extra_in_checkpoint'][:10]:
                shape = shapes.get(key, "N/A")
                print(f"      - {key}: {shape}")
            if len(comparison['extra_in_checkpoint']) > 10:
                print(f"      ... and {len(comparison['extra_in_checkpoint']) - 10} more")
    
    print("\n" + "=" * 80)


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ECG-CPC checkpoint structure")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/pretrained_weights/ECG-CPC/last_11597276.ckpt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/pretrained_weights/ECG-CPC/config_last_11597276_ckpt.yaml",
        help="Path to checkpoint config YAML"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with our model architecture"
    )
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else None
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        return 1
    
    print("Loading checkpoint...")
    checkpoint = None
    
    # Use custom unpickler to skip missing modules
    try:
        import pickle
        import io
        import torch.serialization
        
        class SkipModuleUnpickler(pickle.Unpickler):
            """Custom unpickler that skips missing modules and extracts state_dict."""
            def __init__(self, file, *args, **kwargs):
                super().__init__(file, *args, **kwargs)
                self.state_dict_found = None
                # Handle persistent IDs (PyTorch Lightning uses these)
                self.persistent_load = self._persistent_load
            
            def _persistent_load(self, pid):
                # For persistent IDs, return None or a dummy object
                # PyTorch Lightning uses these for tensor storage
                return None
            
            def find_class(self, module, name):
                # Skip problematic modules by returning a dummy class
                if any(skip in module for skip in ['clinical_ts', 'omegaconf', 'hydra']):
                    # Return a dummy class that can be instantiated
                    class DummyClass:
                        def __init__(self, *args, **kwargs):
                            pass
                        def __getattr__(self, name):
                            return None
                        def __setattr__(self, name, value):
                            object.__setattr__(self, name, value)
                    return DummyClass
                try:
                    return super().find_class(module, name)
                except (ModuleNotFoundError, AttributeError):
                    # Return dummy for any missing module
                    class DummyClass:
                        def __init__(self, *args, **kwargs):
                            pass
                        def __getattr__(self, name):
                            return None
                    return DummyClass
        
        print("  Using custom unpickler to skip missing modules...")
        with open(checkpoint_path, 'rb') as f:
            unpickler = SkipModuleUnpickler(f)
            try:
                checkpoint = unpickler.load()
                print("  ✓ Successfully loaded checkpoint structure")
            except Exception as e:
                # If full load fails, try to extract just state_dict
                print(f"  Warning: Full load failed ({str(e)[:100]}), trying to extract state_dict...")
                # Reset file
                f.seek(0)
                # Try torch's internal loading with custom find_class
                try:
                    # Use torch's load but with custom unpickler
                    import torch._utils
                    checkpoint = torch._utils._load(checkpoint_path, map_location="cpu")
                except:
                    # Last resort: try zipfile extraction (PyTorch Lightning uses zip format)
                    f.seek(0)
                    try:
                        import zipfile
                        import tempfile
                        import os
                        
                        # Check if it's a zip file
                        with zipfile.ZipFile(checkpoint_path, 'r') as zf:
                            file_list = zf.namelist()
                            print(f"  Detected zip-based checkpoint with {len(file_list)} files")
                            
                            # Look for data.pkl (PyTorch Lightning format)
                            data_pkl = 'archive/data.pkl'
                            if data_pkl in file_list:
                                print(f"  Found PyTorch Lightning data.pkl")
                                # Extract to temp and try to load
                                with tempfile.TemporaryDirectory() as tmpdir:
                                    zf.extract(data_pkl, tmpdir)
                                    full_path = os.path.join(tmpdir, data_pkl)
                                    try:
                                        # Try loading with custom unpickler
                                        with open(full_path, 'rb') as df:
                                            unpickler = SkipModuleUnpickler(df)
                                            data = unpickler.load()
                                            if isinstance(data, dict):
                                                checkpoint = data
                                                print(f"  ✓ Extracted checkpoint from {data_pkl}")
                                            else:
                                                # Wrap in dict if needed
                                                checkpoint = {"data": data}
                                                print(f"  ✓ Extracted data from {data_pkl} (wrapped in dict)")
                                    except Exception as e:
                                        print(f"  ✗ Failed to load {data_pkl}: {str(e)[:100]}")
                            else:
                                # Look for other data files
                                data_files = [f for f in file_list if 'data.pkl' in f or ('data' in f.lower() and f.endswith('.pkl'))]
                                if data_files:
                                    print(f"  Found data files: {data_files}")
                                    with tempfile.TemporaryDirectory() as tmpdir:
                                        zf.extractall(tmpdir)
                                        for data_file in data_files:
                                            full_path = os.path.join(tmpdir, data_file)
                                            try:
                                                with open(full_path, 'rb') as df:
                                                    unpickler = SkipModuleUnpickler(df)
                                                    data = unpickler.load()
                                                    if isinstance(data, dict):
                                                        checkpoint = data
                                                        print(f"  ✓ Extracted from {data_file}")
                                                        break
                                            except Exception as e:
                                                continue
                    except zipfile.BadZipFile:
                        # Not a zip file, continue with other methods
                        pass
                    except Exception as e:
                        print(f"  Zip extraction failed: {str(e)[:100]}")
    except Exception as e:
        print(f"  ✗ Custom unpickler failed: {str(e)[:200]}")
    
    # Fallback: try standard torch.load (will fail but gives better error)
    if checkpoint is None or not isinstance(checkpoint, dict):
        print("  Trying standard torch.load (will likely fail but provides better error info)...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            print("  ✓ Standard load succeeded")
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Standard load failed: {error_msg[:200]}")
            if checkpoint is None:
                print("\nERROR: Could not load checkpoint with any method.")
                print("The checkpoint requires 'clinical_ts' module which is not available.")
                print("\nPossible solutions:")
                print("1. Extract state_dict manually using the original project's environment")
                print("2. Contact the checkpoint authors for a state_dict-only version")
                print("3. Use a different checkpoint format if available")
                return 1
    
    # Analyze checkpoint structure
    print("Analyzing checkpoint structure...")
    analysis = analyze_checkpoint_structure(checkpoint)
    
    if not analysis['state_dict']:
        print("ERROR: Could not find state_dict in checkpoint")
        print(f"Top-level keys: {analysis['top_level_keys']}")
        return 1
    
    # Get shape information
    shapes = get_shape_info(analysis['state_dict'])
    
    # Find S4 and encoder layers
    s4_layers = find_s4_layers(analysis['state_dict'])
    encoder_layers = find_encoder_layers(analysis['state_dict'])
    
    # Load config if available
    if config_path and config_path.exists():
        print(f"\nLoading config from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            print("Config loaded successfully")
            if 'ts' in config_data:
                ts_config = config_data.get('ts', {})
                print(f"\nConfig Info:")
                if 'enc' in ts_config:
                    print(f"  Encoder: {ts_config['enc']}")
                if 'pred' in ts_config:
                    print(f"  Predictor: {ts_config['pred']}")
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")
    
    # Compare with our model if requested
    comparison = None
    if args.compare:
        print("\nComparing with our model architecture...")
        try:
            # Load our model config
            base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
            model_config_path = Path("configs/model/ecg_cpc/ecg_cpc.yaml")
            
            config = load_config(
                base_config_path=base_config_path,
                model_config_path=model_config_path,
            )
            
            # Create our model
            our_model = ECG_S4_CPC(config)
            print(f"Our model created with {our_model.count_parameters():,} parameters")
            
            # Compare
            comparison = compare_with_our_model(analysis['state_dict'], our_model)
        except Exception as e:
            print(f"Warning: Failed to compare with our model: {e}")
            import traceback
            traceback.print_exc()
    
    # Print results
    print_analysis(
        checkpoint_path=checkpoint_path,
        analysis=analysis,
        shapes=shapes,
        s4_layers=s4_layers,
        encoder_layers=encoder_layers,
        comparison=comparison
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

