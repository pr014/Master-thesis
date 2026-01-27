#!/usr/bin/env python3
"""Analyze pretrained weights files and YAML configs in PTB-Xl-analysis directory.

This script analyzes all pretrained weights files (.pt) and their corresponding
YAML configuration files in the data/pretrained_weights/PTB-Xl-analysis directory.
It determines architecture compatibility with ResNet1D14 and provides
information about weight extraction possibilities.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.pretrained_CNN.resnet1d_14.model import ResNet1D14


def analyze_checkpoint_structure(checkpoint: Any) -> Dict[str, Any]:
    """Analyze the structure of a checkpoint file.
    
    Args:
        checkpoint: Loaded checkpoint (dict, state_dict, or other)
        
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
    }
    
    if isinstance(checkpoint, dict):
        analysis["top_level_keys"] = list(checkpoint.keys())
        
        # Check for common checkpoint formats (fairseq-style checkpoints)
        # These often have "model" key containing the actual state_dict
        state_dict = None
        
        if "model" in checkpoint:
            # Fairseq checkpoint format - model contains the state_dict
            model_obj = checkpoint["model"]
            if isinstance(model_obj, dict):
                # If model is a dict, it might be the state_dict directly
                # or it might have nested structure
                if all(isinstance(v, torch.Tensor) for v in model_obj.values() if v is not None):
                    state_dict = model_obj
                elif "state_dict" in model_obj:
                    state_dict = model_obj["state_dict"]
                elif "model_state_dict" in model_obj:
                    state_dict = model_obj["model_state_dict"]
        elif "state_dict" in checkpoint:
            analysis["has_state_dict"] = True
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            analysis["has_model_state_dict"] = True
            state_dict = checkpoint["model_state_dict"]
        else:
            # Check if top-level dict itself is a state_dict (all values are tensors)
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values() if v is not None):
                state_dict = checkpoint
        
        if state_dict is None:
            # Try to find state_dict recursively
            def find_state_dict(d, depth=0, max_depth=3):
                if depth > max_depth:
                    return None
                if isinstance(d, dict):
                    # Check if this dict looks like a state_dict
                    if all(isinstance(v, torch.Tensor) for v in d.values() if v is not None and not isinstance(v, dict)):
                        return d
                    # Recursively search
                    for v in d.values():
                        if isinstance(v, dict):
                            result = find_state_dict(v, depth + 1, max_depth)
                            if result is not None:
                                return result
                return None
            
            state_dict = find_state_dict(checkpoint)
        
        if isinstance(state_dict, dict):
            analysis["state_dict_keys"] = list(state_dict.keys())
            analysis["num_layers"] = len(state_dict)
            
            # Analyze layer types
            for key in state_dict.keys():
                if isinstance(state_dict[key], torch.Tensor):
                    # Extract layer type from key (e.g., "conv1.weight" -> "conv1")
                    layer_name = key.split('.')[0]
                    if layer_name not in analysis["layer_types"]:
                        analysis["layer_types"][layer_name] = {
                            "count": 0,
                            "shapes": [],
                            "keys": []
                        }
                    analysis["layer_types"][layer_name]["count"] += 1
                    analysis["layer_types"][layer_name]["shapes"].append(
                        tuple(state_dict[key].shape)
                    )
                    analysis["layer_types"][layer_name]["keys"].append(key)
            
            # Get sample keys (first 20)
            analysis["sample_keys"] = list(state_dict.keys())[:20]
    
    return analysis


def get_resnet1d14_expected_layers() -> List[str]:
    """Get expected layer names for ResNet1D14 architecture.
    
    Returns:
        List of expected layer names
    """
    # Create a dummy config to instantiate the model
    config = {
        "model": {
            "type": "ResNet1D14",
            "num_classes": 10,
            "pretrained": {"enabled": False}
        },
        "training": {"dropout_rate": 0.3}
    }
    
    model = ResNet1D14(config)
    expected_keys = list(model.state_dict().keys())
    return expected_keys


def check_compatibility(
    checkpoint_keys: List[str],
    resnet_keys: List[str]
) -> Dict[str, Any]:
    """Check compatibility between checkpoint and ResNet1D14.
    
    Args:
        checkpoint_keys: Keys from the checkpoint
        resnet_keys: Expected keys from ResNet1D14
        
    Returns:
        Compatibility analysis
    """
    # Remove prefixes from checkpoint keys
    cleaned_checkpoint_keys = []
    for key in checkpoint_keys:
        new_key = key
        for prefix in ["model.", "backbone.", "module."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        cleaned_checkpoint_keys.append(new_key)
    
    # Filter out classification head (fc, dropout) from both
    checkpoint_backbone_keys = [
        k for k in cleaned_checkpoint_keys
        if not k.startswith("fc") and not k.startswith("dropout")
    ]
    resnet_backbone_keys = [
        k for k in resnet_keys
        if not k.startswith("fc") and not k.startswith("dropout")
    ]
    
    # Find matching keys
    matching_keys = set(checkpoint_backbone_keys) & set(resnet_backbone_keys)
    missing_in_checkpoint = set(resnet_backbone_keys) - set(checkpoint_backbone_keys)
    extra_in_checkpoint = set(checkpoint_backbone_keys) - set(resnet_backbone_keys)
    
    # Check for transformer/attention indicators
    transformer_indicators = [
        "attention", "attn", "transformer", "encoder", "decoder",
        "embed", "pos_embed", "patch_embed", "norm", "mlp"
    ]
    has_transformer_layers = any(
        any(indicator in key.lower() for indicator in transformer_indicators)
        for key in checkpoint_keys
    )
    
    compatibility = {
        "is_resnet": len(matching_keys) > 0,
        "matching_keys_count": len(matching_keys),
        "matching_keys": list(matching_keys)[:10],  # First 10
        "missing_in_checkpoint_count": len(missing_in_checkpoint),
        "missing_in_checkpoint": list(missing_in_checkpoint)[:10],
        "extra_in_checkpoint_count": len(extra_in_checkpoint),
        "extra_in_checkpoint": list(extra_in_checkpoint)[:10],
        "has_transformer_layers": has_transformer_layers,
        "compatibility_score": len(matching_keys) / len(resnet_backbone_keys) if resnet_backbone_keys else 0.0,
    }
    
    return compatibility


def load_yaml_config(yaml_path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse YAML configuration file.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        Parsed YAML dictionary or None if error
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load YAML file {yaml_path}: {e}")
        return None


def extract_model_info_from_yaml(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model architecture information from YAML config.
    
    Args:
        config: Parsed YAML configuration
        
    Returns:
        Dictionary with extracted model information
    """
    info = {
        "model_name": None,
        "model_type": None,
        "num_labels": None,
        "encoder_layers": None,
        "encoder_embed_dim": None,
        "encoder_attention_heads": None,
        "task": None,
        "has_pretrained_weights": False,
    }
    
    model_config = config.get("model", {})
    if model_config:
        info["model_name"] = model_config.get("_name")
        info["num_labels"] = model_config.get("num_labels")
        
        # Check for transformer/wav2vec2 architecture
        if "encoder_layers" in model_config:
            info["encoder_layers"] = model_config.get("encoder_layers")
            info["encoder_embed_dim"] = model_config.get("encoder_embed_dim")
            info["encoder_attention_heads"] = model_config.get("encoder_attention_heads")
            info["model_type"] = "transformer/wav2vec2"
        elif info["model_name"]:
            info["model_type"] = info["model_name"]
    
    task_config = config.get("task", {})
    if task_config:
        info["task"] = task_config.get("_name")
    
    # Check if this is a pretrained model
    if "no_pretrained_weights" in model_config:
        info["has_pretrained_weights"] = not model_config.get("no_pretrained_weights", True)
    
    return info


def analyze_weight_extraction_possibility(
    analysis: Dict[str, Any],
    compatibility: Dict[str, Any],
    yaml_info: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze if weights can be extracted and applied to ResNet1D14.
    
    Args:
        analysis: Checkpoint structure analysis
        compatibility: Compatibility analysis with ResNet1D14
        yaml_info: Model information from YAML
        
    Returns:
        Dictionary with extraction possibility analysis
    """
    extraction_info = {
        "can_extract": False,
        "extraction_method": None,
        "compatible_layers": [],
        "incompatible_reason": None,
        "recommendations": [],
    }
    
    # If it's a transformer/wav2vec2 model, weights cannot be directly applied
    if compatibility["has_transformer_layers"]:
        extraction_info["can_extract"] = False
        extraction_info["incompatible_reason"] = "Transformer/Wav2Vec2 architecture is fundamentally different from ResNet1D14"
        extraction_info["recommendations"] = [
            "Cannot directly transfer weights due to architectural mismatch",
            "Consider using the transformer model as a feature extractor",
            "Or train a ResNet1D14 model from scratch or with ResNet-compatible pretrained weights"
        ]
        return extraction_info
    
    # If it's ResNet-compatible
    if compatibility["is_resnet"]:
        if compatibility["compatibility_score"] > 0.5:
            extraction_info["can_extract"] = True
            extraction_info["extraction_method"] = "Direct transfer (backbone layers only)"
            extraction_info["compatible_layers"] = compatibility["matching_keys"]
            extraction_info["recommendations"] = [
                "Extract backbone weights (exclude classification head)",
                "Load using ResNet1D14._load_pretrained_weights() method",
                "Classification head will be randomly initialized"
            ]
        elif compatibility["compatibility_score"] > 0.0:
            extraction_info["can_extract"] = True
            extraction_info["extraction_method"] = "Partial transfer (some layers compatible)"
            extraction_info["compatible_layers"] = compatibility["matching_keys"]
            extraction_info["recommendations"] = [
                "Only compatible layers can be transferred",
                "Many layers will need random initialization",
                "May require fine-tuning from early epochs"
            ]
        else:
            extraction_info["can_extract"] = False
            extraction_info["incompatible_reason"] = "Architecture mismatch - no compatible layers found"
    else:
        extraction_info["can_extract"] = False
        extraction_info["incompatible_reason"] = "Unknown or incompatible architecture"
    
    return extraction_info


def print_analysis(
    weights_path: Path,
    analysis: Dict[str, Any],
    compatibility: Dict[str, Any],
    yaml_info: Optional[Dict[str, Any]] = None,
    extraction_info: Optional[Dict[str, Any]] = None
):
    """Print formatted analysis results.
    
    Args:
        weights_path: Path to the weights file
        analysis: Checkpoint structure analysis
        compatibility: Compatibility analysis with ResNet1D14
        yaml_info: Model information from YAML (optional)
        extraction_info: Weight extraction analysis (optional)
    """
    print("=" * 80)
    print(f"PRETRAINED WEIGHTS ANALYSIS: {weights_path.name}")
    print("=" * 80)
    
    # Print YAML info if available
    if yaml_info:
        print("\n0. MODEL CONFIGURATION (from YAML)")
        print("-" * 80)
        print(f"Model Name: {yaml_info.get('model_name', 'N/A')}")
        print(f"Model Type: {yaml_info.get('model_type', 'N/A')}")
        print(f"Task: {yaml_info.get('task', 'N/A')}")
        print(f"Number of Labels: {yaml_info.get('num_labels', 'N/A')}")
        if yaml_info.get('encoder_layers'):
            print(f"Encoder Layers: {yaml_info['encoder_layers']}")
            print(f"Encoder Embed Dim: {yaml_info['encoder_embed_dim']}")
            print(f"Encoder Attention Heads: {yaml_info['encoder_attention_heads']}")
    
    print("\n1. CHECKPOINT STRUCTURE")
    print("-" * 80)
    print(f"File: {weights_path}")
    print(f"File size: {weights_path.stat().st_size / (1024**3):.2f} GB")
    print(f"Type: {analysis['type']}")
    print(f"Is Dict: {analysis['is_dict']}")
    print(f"Number of layers: {analysis['num_layers']}")
    
    if analysis['top_level_keys']:
        print(f"\nTop-level keys ({len(analysis['top_level_keys'])}):")
        for key in analysis['top_level_keys'][:10]:
            print(f"  - {key}")
        if len(analysis['top_level_keys']) > 10:
            print(f"  ... and {len(analysis['top_level_keys']) - 10} more")
    
    if analysis['has_state_dict']:
        print("\n✓ Contains 'state_dict' key")
    if analysis['has_model_state_dict']:
        print("✓ Contains 'model_state_dict' key")
    
    print("\n2. LAYER TYPES")
    print("-" * 80)
    if analysis['layer_types']:
        # Sort by count
        sorted_types = sorted(
            analysis['layer_types'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        for layer_name, info in sorted_types[:15]:  # Top 15
            print(f"\n{layer_name}:")
            print(f"  Count: {info['count']}")
            print(f"  Sample shapes: {info['shapes'][:3]}")
            print(f"  Sample keys:")
            for key in info['keys'][:3]:
                print(f"    - {key}")
    
    print("\n3. SAMPLE KEYS (first 20)")
    print("-" * 80)
    for key in analysis['sample_keys']:
        print(f"  - {key}")
    
    print("\n4. COMPATIBILITY WITH RESNET1D14")
    print("-" * 80)
    print(f"Is ResNet architecture: {compatibility['is_resnet']}")
    print(f"Compatibility score: {compatibility['compatibility_score']:.2%}")
    print(f"Matching backbone layers: {compatibility['matching_keys_count']}")
    print(f"Missing layers in checkpoint: {compatibility['missing_in_checkpoint_count']}")
    print(f"Extra layers in checkpoint: {compatibility['extra_in_checkpoint_count']}")
    print(f"Has Transformer/Attention layers: {compatibility['has_transformer_layers']}")
    
    if compatibility['matching_keys']:
        print(f"\nMatching keys (sample):")
        for key in compatibility['matching_keys']:
            print(f"  ✓ {key}")
    
    if compatibility['missing_in_checkpoint']:
        print(f"\nMissing keys in checkpoint (sample):")
        for key in compatibility['missing_in_checkpoint']:
            print(f"  ✗ {key}")
    
    if compatibility['extra_in_checkpoint']:
        print(f"\nExtra keys in checkpoint (sample):")
        for key in compatibility['extra_in_checkpoint']:
            print(f"  + {key}")
    
    # Print extraction analysis
    if extraction_info:
        print("\n5. WEIGHT EXTRACTION POSSIBILITY")
        print("-" * 80)
        if extraction_info['can_extract']:
            print("✓ YES - Weights CAN be extracted and applied to ResNet1D14")
            print(f"  Method: {extraction_info['extraction_method']}")
            if extraction_info['compatible_layers']:
                print(f"  Compatible layers: {len(extraction_info['compatible_layers'])}")
        else:
            print("✗ NO - Weights CANNOT be directly applied to ResNet1D14")
            if extraction_info['incompatible_reason']:
                print(f"  Reason: {extraction_info['incompatible_reason']}")
        
        if extraction_info['recommendations']:
            print("\n  Recommendations:")
            for rec in extraction_info['recommendations']:
                print(f"    - {rec}")
    
    print("\n6. CONCLUSION")
    print("-" * 80)
    if compatibility['has_transformer_layers']:
        print("⚠️  This is a TRANSFORMER/WAV2VEC2 model, not ResNet1D14.")
        print("   The weights are NOT directly compatible with ResNet1D14.")
        print("   You cannot extract and apply these weights to your ResNet architecture.")
    elif compatibility['is_resnet'] and compatibility['compatibility_score'] > 0.5:
        print("✓ This appears to be a ResNet architecture.")
        print(f"  Compatibility: {compatibility['compatibility_score']:.2%}")
        print("  The weights MAY be compatible with ResNet1D14.")
        print("  You CAN extract backbone weights and apply them to your ResNet1D14 model.")
    elif compatibility['is_resnet'] and compatibility['compatibility_score'] > 0.0:
        print("⚠️  This appears to be a ResNet architecture, but with significant differences.")
        print(f"  Compatibility: {compatibility['compatibility_score']:.2%}")
        print("  Partial weight extraction may be possible.")
    else:
        print("✗ This does NOT appear to be a ResNet architecture.")
        print("  The weights are NOT compatible with ResNet1D14.")
        print("  You cannot extract and apply these weights to your ResNet architecture.")
    
    print("\n" + "=" * 80)


def analyze_single_file(
    weights_path: Path,
    yaml_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Analyze a single weights file.
    
    Args:
        weights_path: Path to weights file
        yaml_path: Optional path to corresponding YAML file
        
    Returns:
        Dictionary with all analysis results
    """
    results = {
        "weights_path": weights_path,
        "yaml_path": yaml_path,
        "yaml_info": None,
        "analysis": None,
        "compatibility": None,
        "extraction_info": None,
        "error": None,
    }
    
    try:
        # Load YAML if available
        yaml_info = None
        if yaml_path and yaml_path.exists():
            config = load_yaml_config(yaml_path)
            if config:
                yaml_info = extract_model_info_from_yaml(config)
                results["yaml_info"] = yaml_info
        
        # Load checkpoint
        # Use weights_only=False for compatibility with older checkpoints that may contain numpy objects
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        
        # Analyze structure
        analysis = analyze_checkpoint_structure(checkpoint)
        results["analysis"] = analysis
        
        # Get ResNet1D14 expected layers
        resnet_keys = get_resnet1d14_expected_layers()
        
        # Check compatibility
        checkpoint_keys = analysis['state_dict_keys'] if analysis['state_dict_keys'] else []
        compatibility = check_compatibility(checkpoint_keys, resnet_keys)
        results["compatibility"] = compatibility
        
        # Analyze extraction possibility
        extraction_info = analyze_weight_extraction_possibility(
            analysis, compatibility, yaml_info
        )
        results["extraction_info"] = extraction_info
        
        # Print results
        print_analysis(weights_path, analysis, compatibility, yaml_info, extraction_info)
        
    except Exception as e:
        error_msg = f"Error analyzing {weights_path.name}: {e}"
        results["error"] = error_msg
        print(f"\n{error_msg}")
        import traceback
        traceback.print_exc()
    
    return results


def find_matching_yaml(weights_path: Path) -> Optional[Path]:
    """Find matching YAML file for a weights file.
    
    Args:
        weights_path: Path to weights file
        
    Returns:
        Path to matching YAML file or None
    """
    weights_stem = weights_path.stem  # filename without extension
    yaml_path = weights_path.parent / f"{weights_stem}.yaml"
    
    if yaml_path.exists():
        return yaml_path
    return None


def analyze_directory(analysis_dir: Path):
    """Analyze all weights files in a directory.
    
    Args:
        analysis_dir: Directory containing weights and YAML files
    """
    # Find all .pt and .pth files
    weights_files = sorted(list(analysis_dir.glob("*.pt")) + list(analysis_dir.glob("*.pth")))
    
    if not weights_files:
        print(f"No .pt files found in {analysis_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYZING ALL FILES IN: {analysis_dir}")
    print(f"Found {len(weights_files)} weights file(s)")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for weights_path in weights_files:
        yaml_path = find_matching_yaml(weights_path)
        results = analyze_single_file(weights_path, yaml_path)
        all_results.append(results)
        print("\n\n")  # Add spacing between files
    
    # Print summary
    print_summary(all_results)


def print_summary(all_results: List[Dict[str, Any]]):
    """Print summary of all analyses.
    
    Args:
        all_results: List of analysis results for all files
    """
    print("\n" + "=" * 80)
    print("SUMMARY - ALL FILES")
    print("=" * 80)
    
    for i, results in enumerate(all_results, 1):
        weights_path = results["weights_path"]
        yaml_info = results.get("yaml_info")
        extraction_info = results.get("extraction_info")
        error = results.get("error")
        
        print(f"\n{i}. {weights_path.name}")
        print("-" * 80)
        
        if error:
            print(f"  ❌ Error: {error}")
            continue
        
        if yaml_info:
            print(f"  Model: {yaml_info.get('model_name', 'N/A')} ({yaml_info.get('model_type', 'N/A')})")
        
        compatibility = results.get("compatibility", {})
        if compatibility:
            print(f"  Architecture: {'Transformer/Wav2Vec2' if compatibility.get('has_transformer_layers') else 'ResNet-like'}")
            print(f"  Compatibility Score: {compatibility.get('compatibility_score', 0):.2%}")
        
        if extraction_info:
            can_extract = extraction_info.get("can_extract", False)
            status = "✓ CAN extract" if can_extract else "✗ CANNOT extract"
            print(f"  Weight Extraction: {status}")
            if not can_extract and extraction_info.get("incompatible_reason"):
                print(f"    Reason: {extraction_info['incompatible_reason']}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pretrained weights files and YAML configs in analysis directory"
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to weights file (.pt) or analysis directory. If not provided, analyzes data/pretrained_weights/PTB-Xl-analysis"
    )
    
    args = parser.parse_args()
    
    # Determine path
    if args.path:
        path = Path(args.path)
    else:
        # Default to PTB-Xl-analysis directory
        project_root = Path(__file__).parent.parent
        path = project_root / "data" / "pretrained_weights" / "PTB-Xl-analysis"
    
    if not path.exists():
        print(f"Error: Path not found: {path}")
        sys.exit(1)
    
    # Check if it's a file or directory
    if path.is_file():
        # Single file analysis
        yaml_path = find_matching_yaml(path)
        analyze_single_file(path, yaml_path)
    elif path.is_dir():
        # Directory analysis
        analyze_directory(path)
    else:
        print(f"Error: Path is neither a file nor directory: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

