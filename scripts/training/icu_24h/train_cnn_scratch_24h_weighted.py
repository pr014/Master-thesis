"""Training script for CNN from scratch with LOS bin classification.
This script uses class weights specifically calculated for the 24h ECG dataset.
Config: configs/24h_weighted/balanced_weights.yaml (balanced method)
"""

from pathlib import Path
import sys
import os
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models import CNNScratch
from src.data.ecg import create_dataloaders
from src.data.labeling import load_icustays, ICUStayMapper
from src.training import Trainer
from src.utils.config_loader import load_config


def main():
    """Main training function for 24h dataset with class weights."""
    # Load configs - using 24h weighted config (balanced method)
    base_config_path = Path("configs/24h_weighted/balanced_weights.yaml")
    model_config_path = Path("configs/model/cnn_scratch.yaml")
    
    config = load_config(
        base_config_path=base_config_path,
        model_config_path=model_config_path,
    )
    
    print("="*60)
    print("Training CNN with Class Weights for 24h Dataset")
    print("="*60)
    print(f"Base config: {base_config_path}")
    print(f"Model config: {model_config_path}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")
    print(f"Loss type: {config.get('training', {}).get('loss', {}).get('type', 'unknown')}")
    if 'weight' in config.get('training', {}).get('loss', {}):
        weights = config.get('training', {}).get('loss', {}).get('weight', [])
        print(f"Class weights: {weights}")
    print("="*60)
    
    # Load ICU stays and create mapper
    # Path can be set via environment variable or config
    icustays_path = os.getenv("ICUSTAYS_PATH")
    if icustays_path is None:
        # Try relative to data_dir
        data_dir = config.get("data", {}).get("data_dir", "")
        if data_dir:
            icustays_path = Path(data_dir).parent / "icustays.csv"
        else:
            # Default fallback (relative to project root)
            icustays_path = Path("data/icustays.csv")
    
    icustays_path = Path(icustays_path)
    if not icustays_path.exists():
        raise FileNotFoundError(
            f"icustays.csv not found at: {icustays_path}\n"
            f"Set ICUSTAYS_PATH environment variable or place icustays.csv in data directory."
        )
    
    print(f"Loading ICU stays from: {icustays_path}")
    icustays_df = load_icustays(str(icustays_path))
    icu_mapper = ICUStayMapper(icustays_df)
    print(f"Loaded {len(icustays_df)} ICU stays")
    
    # Create DataLoaders (labels will be auto-generated via icu_mapper)
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        labels=None,  # Will be auto-generated
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
    )
    
    # Create model
    model = CNNScratch(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    # Store config paths for checkpoint saving
    trainer.config_paths = {
        "base": str(base_config_path.resolve()),
        "model": str(model_config_path.resolve()),
    }
    
    # Store job ID if available (from SLURM)
    trainer.job_id = os.getenv("SLURM_JOB_ID")
    if trainer.job_id:
        print(f"SLURM Job ID: {trainer.job_id}")
    
    # Train
    history = trainer.train()
    
    print("Training completed!")
    print(f"Best validation loss: {min(history.get('val_loss', [float('inf')])):.4f}")
    
    # Test evaluation (if test_loader is available)
    if test_loader is not None:
        print("\n" + "="*60)
        print("Evaluating on test set...")
        print("="*60)
        
        from src.training.train_loop import evaluate_with_detailed_metrics
        from src.training.losses import get_loss
        import numpy as np
        
        # Load best model checkpoint (use job_id version if available)
        checkpoint_dir = Path(config.get("checkpoint", {}).get("save_dir", "outputs/checkpoints"))
        model_name = config.get("model", {}).get("type", "model")
        
        # Load checkpoint with job_id (required for traceability)
        if trainer.job_id:
            best_model_path = checkpoint_dir / f"{model_name}_best_{trainer.job_id}.pt"
            if best_model_path.exists():
                print(f"Loading best model from: {best_model_path}")
                checkpoint = torch.load(best_model_path, map_location=trainer.device)
                trainer.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                print(f"Warning: Best model checkpoint not found at {best_model_path}. Using current model state.")
        else:
            raise ValueError(
                "No job_id available. Cannot load checkpoint without job_id for traceability. "
                "Ensure SLURM_JOB_ID environment variable is set."
            )
        
        # Get number of classes from config
        num_classes = config.get("model", {}).get("num_classes", 10)
        
        # Evaluate on test set with detailed metrics
        # Use the criterion from trainer (already has weights on correct device)
        test_metrics = evaluate_with_detailed_metrics(
            model=trainer.model,
            val_loader=test_loader,
            criterion=trainer.criterion,
            device=trainer.device,
            num_classes=num_classes,
        )
        
        # Print test results
        print("\nTest Set Results:")
        print(f"  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
        print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f} ({test_metrics['balanced_accuracy']*100:.2f}%)")
        print(f"  Macro Precision: {test_metrics['macro_precision']:.4f}")
        print(f"  Macro Recall: {test_metrics['macro_recall']:.4f}")
        print(f"  Macro F1-Score: {test_metrics['macro_f1']:.4f}")
        print(f"  Number of ICU stays evaluated: {test_metrics['num_stays']}")
        
        # Per-class metrics
        print("\n  Per-Class Metrics:")
        print(f"    {'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("    " + "-" * 44)
        for cls in range(num_classes):
            print(f"    {cls:<8} {test_metrics['per_class_precision'][cls]:<12.4f} "
                  f"{test_metrics['per_class_recall'][cls]:<12.4f} "
                  f"{test_metrics['per_class_f1'][cls]:<12.4f}")
        
        # Confusion Matrix
        if test_metrics['confusion_matrix'] is not None:
            cm = np.array(test_metrics['confusion_matrix'])
            print("\n  Confusion Matrix:")
            print("    " + " ".join([f"{i:>6}" for i in range(num_classes)]))
            for i in range(num_classes):
                row_str = f"  {i} " + " ".join([f"{cm[i,j]:>6}" for j in range(num_classes)])
                print(row_str)
        
        # Add test metrics to history
        history["test_loss"] = test_metrics["loss"]
        history["test_acc"] = test_metrics["accuracy"]
        history["test_balanced_acc"] = test_metrics["balanced_accuracy"]
        history["test_macro_precision"] = test_metrics["macro_precision"]
        history["test_macro_recall"] = test_metrics["macro_recall"]
        history["test_macro_f1"] = test_metrics["macro_f1"]
        history["test_num_stays"] = test_metrics["num_stays"]
    else:
        print("\nWarning: No test loader available. Skipping test evaluation.")
    
    return history


if __name__ == "__main__":
    main()
