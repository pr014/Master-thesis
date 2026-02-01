#!/usr/bin/env python3
"""Simple analysis of ECG-CPC checkpoint structure.

This script provides basic information about the checkpoint structure
without trying to fully load it (which requires the original project's modules).
"""

import sys
from pathlib import Path
import zipfile
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def analyze_checkpoint_structure(checkpoint_path: Path):
    """Analyze checkpoint structure without fully loading it."""
    print("=" * 80)
    print(f"ECG-CPC Checkpoint Analysis: {checkpoint_path.name}")
    print("=" * 80)
    
    # Check if it's a zip file (PyTorch Lightning format)
    try:
        with zipfile.ZipFile(checkpoint_path, 'r') as zf:
            file_list = zf.namelist()
            print(f"\n[1] Checkpoint Format: PyTorch Lightning (ZIP-based)")
            print(f"    Total files in archive: {len(file_list)}")
            
            # Find key files
            data_pkl = 'archive/data.pkl'
            format_version = 'archive/.format_version'
            
            print(f"\n[2] Key Files:")
            if data_pkl in file_list:
                info = zf.getinfo(data_pkl)
                print(f"    ✓ {data_pkl}: {info.file_size:,} bytes")
            if format_version in file_list:
                with zf.open(format_version) as f:
                    version = f.read().decode('utf-8').strip()
                    print(f"    ✓ {format_version}: {version}")
            
            # List data files (tensors)
            data_files = [f for f in file_list if f.startswith('archive/data/') and f != 'archive/data.pkl']
            print(f"\n[3] Tensor Storage Files: {len(data_files)} files")
            if data_files:
                total_size = sum(zf.getinfo(f).file_size for f in data_files)
                print(f"    Total tensor storage: {total_size / (1024**2):.2f} MB")
                print(f"    Sample files: {data_files[:5]}")
            
            # Try to read some metadata
            print(f"\n[4] Archive Structure:")
            dirs = set()
            for f in file_list:
                if '/' in f:
                    dirs.add(f.split('/')[0])
            print(f"    Top-level directories: {sorted(dirs)}")
            
    except zipfile.BadZipFile:
        print(f"\n[1] Checkpoint Format: Standard PyTorch (.pt/.pth)")
        print("    Note: This appears to be a standard PyTorch checkpoint")
        print("    It requires the original project's modules to load fully.")
    
    # Analyze config if available
    config_path = checkpoint_path.parent / f"config_{checkpoint_path.stem}.yaml"
    if not config_path.exists():
        # Try alternative naming
        config_path = checkpoint_path.parent / "config_last_11597276_ckpt.yaml"
    
    if config_path.exists():
        print(f"\n[5] Config File Found: {config_path.name}")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"\n[6] Config Analysis:")
            if 'ts' in config:
                ts_config = config.get('ts', {})
                if 'enc' in ts_config:
                    enc = ts_config['enc']
                    print(f"    Encoder Type: {enc}")
                    if isinstance(enc, dict):
                        if 'features' in enc:
                            print(f"      Features: {enc['features']}")
                        if 'kss' in enc:
                            print(f"      Kernel Sizes: {enc['kss']}")
                        if 'strides' in enc:
                            print(f"      Strides: {enc['strides']}")
                
                if 'pred' in ts_config:
                    pred = ts_config['pred']
                    print(f"    Predictor Type: {pred}")
                    if isinstance(pred, dict):
                        if 'model_dim' in pred:
                            print(f"      Model Dim: {pred['model_dim']}")
                        if 'state_dim' in pred:
                            print(f"      State Dim: {pred['state_dim']}")
                        if 'backbone' in pred:
                            print(f"      Backbone: {pred['backbone']}")
                        if 'causal' in pred:
                            print(f"      Causal: {pred['causal']}")
            
            if 'base' in config:
                base = config.get('base', {})
                print(f"\n[7] Base Config:")
                if 'input_channels' in base:
                    print(f"    Input Channels: {base['input_channels']}")
                if 'input_size' in base:
                    print(f"    Input Size: {base['input_size']}s")
                if 'fs' in base:
                    print(f"    Sampling Rate: {base['fs']} Hz")
                if 'batch_size' in base:
                    print(f"    Batch Size: {base['batch_size']}")
        
        except Exception as e:
            print(f"    Error reading config: {e}")
    
    # Compatibility analysis
    print(f"\n[8] Compatibility with Our Model:")
    print(f"    Our Model Architecture:")
    print(f"      - S4 Encoder: d_model=256, d_state=64, n_layers=4")
    print(f"      - Input: 12 channels, 5000 timesteps (10s @ 500Hz)")
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if 'ts' in config and 'pred' in config.get('ts', {}):
                pred = config['ts']['pred']
                if isinstance(pred, dict):
                    checkpoint_d_model = pred.get('model_dim', 'N/A')
                    checkpoint_d_state = pred.get('state_dim', 'N/A')
                    print(f"\n    Checkpoint Architecture (from config):")
                    print(f"      - S4 Predictor: model_dim={checkpoint_d_model}, state_dim={checkpoint_d_state}")
                    
                    if checkpoint_d_model != 'N/A' and checkpoint_d_state != 'N/A':
                        print(f"\n    ⚠ ARCHITECTURE MISMATCH:")
                        print(f"      - Model Dim: Checkpoint={checkpoint_d_model} vs Our=256")
                        print(f"      - State Dim: Checkpoint={checkpoint_d_state} vs Our=64")
                        print(f"\n    Recommendations:")
                        print(f"      1. Partial loading: Only load compatible layers")
                        print(f"      2. Adapt our model: Change d_model to {checkpoint_d_model}, d_state to {checkpoint_d_state}")
                        print(f"      3. Train from scratch: Use our architecture without pretrained weights")
        except:
            pass
    
    print(f"\n[9] Next Steps:")
    print(f"    1. To extract state_dict, you need access to the original project environment")
    print(f"    2. Or use this script to create an extraction helper:")
    print(f"       python scripts/analysis/extract_ecg_cpc_state_dict.py")
    print(f"    3. Check if checkpoint authors provide a state_dict-only version")
    
    print("\n" + "=" * 80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple ECG-CPC checkpoint analysis")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/pretrained_weights/ECG-CPC/last_11597276.ckpt",
        help="Path to checkpoint file"
    )
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        return 1
    
    analyze_checkpoint_structure(checkpoint_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

