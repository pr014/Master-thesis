"""Shared utilities for training scripts."""

from pathlib import Path
from typing import Dict, Any, Optional, Callable
import os
import torch
import torch.nn as nn
import numpy as np

from ..data.ecg import create_dataloaders
from ..data.labeling import load_icustays, ICUStayMapper, load_mortality_mapping, get_num_classes_from_config
from .train_loop import evaluate_with_detailed_metrics


def setup_icustays_mapper(config: Dict[str, Any]) -> ICUStayMapper:
    """Load ICU stays and create mapper with optional mortality mapping.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        ICUStayMapper instance.
    """
    # Load ICU stays
    icustays_env = os.getenv("ICUSTAYS_PATH")
    if icustays_env:
        env_path = Path(icustays_env)
        if env_path.exists():
            icustays_path = env_path
        else:
            print(f"Warning: ICUSTAYS_PATH is set but file does not exist: {env_path}. Falling back to default lookup.")
            icustays_env = None

    if not icustays_env:
        # Try relative to data_dir
        data_dir = config.get("data", {}).get("data_dir", "")
        if data_dir:
            icustays_path = Path(data_dir).parent.parent / "labeling" / "labels_csv" / "icustays.csv"
        else:
            # Default fallback (relative to project root)
            icustays_path = Path("data/labeling/labels_csv/icustays.csv")

    icustays_path = Path(icustays_path)
    if not icustays_path.exists():
        raise FileNotFoundError(
            f"icustays.csv not found at: {icustays_path}\n"
            f"Set ICUSTAYS_PATH environment variable or place icustays.csv in data/labeling/labels_csv."
        )
    
    print(f"Loading ICU stays from: {icustays_path}")
    icustays_df = load_icustays(str(icustays_path))
    print(f"Loaded {len(icustays_df)} ICU stays")
    
    # Check if multi-task is enabled
    multi_task_config = config.get("multi_task", {})
    is_multi_task = multi_task_config.get("enabled", False)
    
    # Load mortality mapping if multi-task is enabled
    mortality_mapping = None
    if is_multi_task:
        admissions_path = multi_task_config.get("admissions_path", "data/labeling/labels_csv/admissions.csv")
        admissions_path = Path(admissions_path)
        
        # Try to resolve relative path
        if not admissions_path.is_absolute():
            # Try to find project root (go up from src/training/)
            project_root = Path(__file__).parent.parent.parent
            admissions_path = project_root / admissions_path
        
        if not admissions_path.exists():
            data_dir = config.get("data", {}).get("data_dir", "")
            if data_dir:
                admissions_path = Path(data_dir).parent.parent / "labeling" / "labels_csv" / "admissions.csv"
        
        if not admissions_path.exists():
            raise FileNotFoundError(
                f"admissions.csv not found for multi-task learning at: {admissions_path}\n"
                f"Set multi_task.admissions_path in config or place admissions.csv in data/labeling directory."
            )
        
        print(f"Loading admissions from: {admissions_path}")
        mortality_mapping = load_mortality_mapping(str(admissions_path), icustays_df)
        print(f"Loaded mortality mapping: {sum(mortality_mapping.values())} died, {len(mortality_mapping) - sum(mortality_mapping.values())} survived")
    
    # Create ICU mapper with mortality mapping
    return ICUStayMapper(icustays_df, mortality_mapping=mortality_mapping)


def evaluate_and_print_results(
    trainer,
    test_loader,
    history: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate model on test set and print formatted results.
    
    Args:
        trainer: Trainer instance with model and criterion.
        test_loader: Test data loader.
        history: Training history dictionary.
        config: Configuration dictionary.
        
    Returns:
        Updated history dictionary with test metrics.
    """
    if test_loader is None:
        print("\nWarning: No test loader available. Skipping test evaluation.")
        return history
    
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
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
    
    # Get number of classes from config (use los_binning if available, else model.num_classes, else default 10)
    try:
        num_classes = get_num_classes_from_config(config)
    except (KeyError, ValueError):
        # Fallback to model config or default
        num_classes = config.get("model", {}).get("num_classes", 10)
    
    # Evaluate on test set with detailed metrics
    test_metrics = evaluate_with_detailed_metrics(
        model=trainer.model,
        val_loader=test_loader,
        criterion=trainer.criterion,
        device=trainer.device,
        num_classes=num_classes,
    )
    
    # Print formatted test results summary
    print("\n" + "=" * 80)
    print("📊 TRAINING RESULTS SUMMARY")
    print("=" * 80)

    # Normalize metric keys (evaluate_with_detailed_metrics returns 'los_*')
    los_loss = test_metrics.get("los_loss", test_metrics.get("loss", 0.0))
    los_acc = test_metrics.get("los_accuracy", test_metrics.get("accuracy", 0.0))
    los_bal_acc = test_metrics.get("los_balanced_accuracy", test_metrics.get("balanced_accuracy", 0.0))
    los_macro_precision = test_metrics.get("los_macro_precision", test_metrics.get("macro_precision", 0.0))
    los_macro_recall = test_metrics.get("los_macro_recall", test_metrics.get("macro_recall", 0.0))
    los_macro_f1 = test_metrics.get("los_macro_f1", test_metrics.get("macro_f1", 0.0))
    los_per_class_precision = test_metrics.get("los_per_class_precision", test_metrics.get("per_class_precision"))
    los_per_class_recall = test_metrics.get("los_per_class_recall", test_metrics.get("per_class_recall"))
    los_per_class_f1 = test_metrics.get("los_per_class_f1", test_metrics.get("per_class_f1"))
    los_cm = test_metrics.get("los_confusion_matrix", test_metrics.get("confusion_matrix"))
    
    # Model Performance
    best_val_loss = min(history.get('val_loss', [float('inf')]))
    print("\n🔹 Model Performance:")
    print(f"   Best Validation Loss: {best_val_loss:.4f}")
    print(f"   Test LOS Loss:        {los_loss:.4f}")
    print(f"   Test LOS Accuracy:    {los_acc:.4f} ({los_acc*100:.2f}%)")
    print(f"   LOS Balanced Acc:     {los_bal_acc:.4f} ({los_bal_acc*100:.2f}%)")
    print(f"   LOS Macro Precision:  {los_macro_precision:.4f}")
    print(f"   LOS Macro Recall:     {los_macro_recall:.4f}")
    print(f"   LOS Macro F1-Score:   {los_macro_f1:.4f}")
    print(f"   Test ICU Stays:       {test_metrics.get('num_stays', 0):,}")
    
    # Per-class metrics
    print("\n🔹 Per-Class Metrics:")
    print(f"   {'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("   " + "-" * 44)
    for cls in range(num_classes):
        print(f"   {cls:<8} {los_per_class_precision[cls]:<12.4f} "
              f"{los_per_class_recall[cls]:<12.4f} "
              f"{los_per_class_f1[cls]:<12.4f}")

    # Mortality summary (multi-task)
    if "mortality_auc" in test_metrics:
        print("\n🔹 Mortality (overall):")
        print(f"   Accuracy:  {test_metrics.get('mortality_accuracy', 0.0):.4f}")
        print(f"   Precision: {test_metrics.get('mortality_precision', 0.0):.4f}")
        print(f"   Recall:    {test_metrics.get('mortality_recall', 0.0):.4f}")
        print(f"   F1:        {test_metrics.get('mortality_f1', 0.0):.4f}")
        print(f"   AUC:       {test_metrics.get('mortality_auc', 0.0):.4f}")

        if "mortality_per_los_class" in test_metrics:
            print("\n🔹 Mortality per LOS class:")
            print(f"   {'LOS':<6} {'AUC':<8} {'F1':<8} {'Support':<10}")
            print("   " + "-" * 34)
            for los_cls in range(num_classes):
                m = test_metrics["mortality_per_los_class"].get(los_cls, {})
                print(f"   {los_cls:<6} {m.get('auc', 0.0):<8.4f} {m.get('f1', 0.0):<8.4f} {m.get('support', 0):<10d}")
    
    # Confusion Matrix
    if los_cm is not None:
        cm = np.array(los_cm)
        print("\n🔹 Confusion Matrix:")
        print("   " + " ".join([f"{i:>6}" for i in range(num_classes)]))
        for i in range(num_classes):
            row_str = f"   {i} " + " ".join([f"{cm[i,j]:>6}" for j in range(num_classes)])
            print(row_str)
    
    # Checkpoint info
    print("\n" + "=" * 80)
    if trainer.job_id:
        checkpoint_path = f"outputs/checkpoints/{model_name}_best_{trainer.job_id}.pt"
        print(f"✅ Checkpoints: {checkpoint_path}")
    else:
        print("✅ Checkpoints: outputs/checkpoints/<MODEL_NAME>_best_<JOB_ID>.pt")
    print("=" * 80)
    
    # Add test metrics to history
    history["test_los_loss"] = los_loss
    history["test_los_acc"] = los_acc
    history["test_los_balanced_acc"] = los_bal_acc
    history["test_los_macro_precision"] = los_macro_precision
    history["test_los_macro_recall"] = los_macro_recall
    history["test_los_macro_f1"] = los_macro_f1
    history["test_num_stays"] = test_metrics.get("num_stays", 0)

    if "mortality_auc" in test_metrics:
        history["test_mortality_auc"] = test_metrics.get("mortality_auc", 0.0)
        history["test_mortality_acc"] = test_metrics.get("mortality_accuracy", 0.0)
    
    return history

