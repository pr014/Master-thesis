#!/usr/bin/env python3
"""Export training overview from all checkpoints and logs.
Run this on the server to generate a CSV with all training information.
"""

import sys
from pathlib import Path
import torch
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional
import re


def extract_job_id_from_log(log_file: Path) -> Optional[str]:
    """Extract job ID from SLURM log filename."""
    match = re.search(r'slurm_(\d+)\.out', str(log_file))
    return match.group(1) if match else None


def load_checkpoint_info(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Load information from checkpoint file.
    
    Returns:
        Dictionary with checkpoint info or None if failed.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        info = {
            "checkpoint_file": checkpoint_path.name,
            "epoch": checkpoint.get("epoch", "unknown"),
            "job_id": checkpoint.get("job_id", ""),
        }
        
        # Metrics
        metrics = checkpoint.get("metrics", {})
        info["best_val_loss"] = metrics.get("val_loss", "")
        info["best_val_acc"] = metrics.get("val_acc", "")
        info["train_loss"] = metrics.get("train_loss", "")
        info["train_acc"] = metrics.get("train_acc", "")
        
        # Config info
        config = checkpoint.get("config", {})
        if config:
            info["model_type"] = config.get("model", {}).get("type", "")
            info["num_classes"] = config.get("model", {}).get("num_classes", "")
            info["batch_size"] = config.get("training", {}).get("batch_size", "")
            info["learning_rate"] = config.get("training", {}).get("optimizer", {}).get("lr", "")
            info["loss_type"] = config.get("training", {}).get("loss", {}).get("type", "")
            
            # Class weights
            weights = config.get("training", {}).get("loss", {}).get("weight", [])
            info["class_weights"] = str(weights) if weights else ""
            
            # Augmentation
            aug = config.get("data", {}).get("augmentation", {})
            info["augmentation"] = "enabled" if aug.get("enabled", False) else "disabled"
            
            # Data
            info["data_dir"] = config.get("data", {}).get("data_dir", "")
        
        # Config paths
        config_paths = checkpoint.get("config_paths", {})
        info["config_base"] = Path(config_paths.get("base", "")).name if config_paths.get("base") else ""
        info["config_model"] = Path(config_paths.get("model", "")).name if config_paths.get("model") else ""
        
        return info
    except Exception as e:
        print(f"Warning: Could not load {checkpoint_path}: {e}", file=sys.stderr)
        return None


def extract_test_metrics_from_log(log_file: Path) -> Dict[str, Any]:
    """Extract test metrics from SLURM log file."""
    metrics = {}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Test accuracy
        match = re.search(r'Test Accuracy:\s+([\d.]+)', content)
        if match:
            metrics["test_accuracy"] = float(match.group(1))
        
        # Test loss
        match = re.search(r'Test Loss:\s+([\d.]+)', content)
        if match:
            metrics["test_loss"] = float(match.group(1))
        
        # Balanced accuracy
        match = re.search(r'Balanced Accuracy:\s+([\d.]+)', content)
        if match:
            metrics["test_balanced_acc"] = float(match.group(1))
        
        # Macro F1
        match = re.search(r'Macro F1-Score:\s+([\d.]+)', content)
        if match:
            metrics["test_macro_f1"] = float(match.group(1))
        
    except Exception:
        pass
    
    return metrics


def main():
    """Main function."""
    checkpoint_dir = Path("outputs/checkpoints")
    log_dir = Path("outputs/logs")
    output_file = Path("training_overview.csv")
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Scanning checkpoints in: {checkpoint_dir}")
    print(f"Scanning logs in: {log_dir}")
    
    # Collect all checkpoint info
    all_info = []
    
    for checkpoint_file in sorted(checkpoint_dir.glob("*.pt")):
        info = load_checkpoint_info(checkpoint_file)
        if not info:
            continue
        
        # Try to find matching log file
        job_id = info.get("job_id")
        if job_id and log_dir.exists():
            log_file = log_dir / f"slurm_{job_id}.out"
            if log_file.exists():
                test_metrics = extract_test_metrics_from_log(log_file)
                info.update(test_metrics)
        
        # If no job_id in checkpoint, try to find from log filename
        if not job_id and log_dir.exists():
            # Try to match checkpoint timestamp or name with log
            # This is a fallback - not perfect but better than nothing
            pass
        
        all_info.append(info)
    
    if not all_info:
        print("No checkpoints found.", file=sys.stderr)
        sys.exit(1)
    
    # Write CSV
    fieldnames = [
        "job_id",
        "checkpoint_file",
        "epoch",
        "config_base",
        "config_model",
        "model_type",
        "num_classes",
        "batch_size",
        "learning_rate",
        "loss_type",
        "class_weights",
        "augmentation",
        "best_val_loss",
        "best_val_acc",
        "train_loss",
        "train_acc",
        "test_accuracy",
        "test_loss",
        "test_balanced_acc",
        "test_macro_f1",
        "data_dir",
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_info)
    
    print(f"\n✓ Exported {len(all_info)} training runs to: {output_file}")
    print(f"\nYou can now download this file and open it in Excel:")
    print(f"  scp user@server:{output_file.absolute()} ./")


if __name__ == "__main__":
    main()

