"""Common training loop implementation."""

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import get_loss, MultiTaskLoss
from .callbacks import EarlyStopping, ModelCheckpoint


def _is_multi_task_model(model: nn.Module) -> bool:
    """Check if model is a multi-task model (returns Dict from forward)."""
    # Test with dummy input to see if forward returns Dict
    try:
        dummy_input = torch.zeros(1, 12, 5000)
        with torch.no_grad():
            output = model(dummy_input)
            return isinstance(output, dict) and "los" in output and "mortality" in output
    except Exception:
        return False


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Train for one epoch.
    
    Supports both single-task (LOS only) and multi-task (LOS + Mortality) models.
    
    Args:
        model: Model to train.
        train_loader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function (can be MultiTaskLoss for multi-task).
        device: Device to train on.
        config: Configuration dictionary.
    
    Returns:
        Dictionary of training metrics.
    """
    model.train()
    is_multi_task = isinstance(criterion, MultiTaskLoss) or _is_multi_task_model(model)
    
    total_loss = 0.0
    total_los_loss = 0.0
    total_mortality_loss = 0.0
    los_correct = 0
    los_total = 0
    mortality_correct = 0
    mortality_total = 0
    
    gradient_clip_norm = config.get("training", {}).get("gradient_clip_norm", None)
    
    for batch in train_loader:
        signals = batch["signal"].to(device)  # (B, C, T)
        labels = batch["label"].to(device)  # (B,)
        
        # Note: Unmatched samples (label == -1) should already be filtered
        # in dataset initialization, but double-check for safety
        valid_mask = labels >= 0
        if not valid_mask.any():
            continue
        
        signals = signals[valid_mask]
        labels = labels[valid_mask]
        
        # Get mortality labels if available
        mortality_labels = None
        if is_multi_task and "mortality_label" in batch:
            mortality_labels = batch["mortality_label"].to(device)[valid_mask]
        
        # Get demographic features if available
        demographic_features = None
        if "demographic_features" in batch and batch["demographic_features"] is not None:
            demographic_features = batch["demographic_features"].to(device)
            if valid_mask.any():
                # Filter demographic features to match valid_mask
                demographic_features = demographic_features[valid_mask]
        
        # Get diagnosis features if available
        diagnosis_features = None
        if "diagnosis_features" in batch and batch["diagnosis_features"] is not None:
            diagnosis_features = batch["diagnosis_features"].to(device)
            if valid_mask.any():
                # Filter diagnosis features to match valid_mask
                diagnosis_features = diagnosis_features[valid_mask]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(signals, demographic_features=demographic_features, diagnosis_features=diagnosis_features)
        
        # Handle multi-task vs single-task
        if is_multi_task and isinstance(outputs, dict):
            # Multi-task model
            los_logits = outputs["los"]
            mortality_probs = outputs["mortality"]
            
            if isinstance(criterion, MultiTaskLoss):
                loss_dict = criterion(
                    los_logits, labels,
                    mortality_probs, mortality_labels
                )
                loss = loss_dict["total"]
                los_loss = loss_dict["los"]
                mortality_loss = loss_dict["mortality"]
            else:
                # Fallback: use LOS loss only if MultiTaskLoss not provided
                loss = criterion(los_logits, labels)
                los_loss = loss
                mortality_loss = torch.tensor(0.0, device=device)
        else:
            # Single-task model
            logits = outputs if not isinstance(outputs, dict) else outputs.get("los", outputs)
            loss = criterion(logits, labels)
            los_loss = loss
            mortality_loss = torch.tensor(0.0, device=device)
            los_logits = logits
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected! Skipping batch.")
            print(f"  Signal stats: min={signals.min():.4f}, max={signals.max():.4f}, mean={signals.mean():.4f}, std={signals.std():.4f}")
            print(f"  Labels: {labels.unique()}")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_los_loss += los_loss.item()
        total_mortality_loss += mortality_loss.item()
        
        # LOS metrics
        los_predictions = los_logits.argmax(dim=1)
        los_correct += (los_predictions == labels).sum().item()
        los_total += labels.size(0)
        
        # Mortality metrics (if available)
        if is_multi_task and mortality_labels is not None:
            valid_mortality_mask = mortality_labels >= 0
            if valid_mortality_mask.any():
                mortality_preds = (mortality_probs.squeeze(1) > 0.5).long()
                mortality_correct += (mortality_preds[valid_mortality_mask] == mortality_labels[valid_mortality_mask]).sum().item()
                mortality_total += valid_mortality_mask.sum().item()
    
    metrics = {
        "train_loss": total_loss / len(train_loader) if len(train_loader) > 0 else 0.0,
        "train_los_acc": los_correct / los_total if los_total > 0 else 0.0,
    }
    
    if is_multi_task:
        metrics["train_los_loss"] = total_los_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        metrics["train_mortality_loss"] = total_mortality_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        if mortality_total > 0:
            metrics["train_mortality_acc"] = mortality_correct / mortality_total
    
    return metrics


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate for one epoch with stay-level aggregation.
    
    Policy: Stay-level aggregation (Option A)
    - Group predictions by stay_id
    - Aggregate logits per stay by taking mean across ECGs
    - Compute metrics on stays (one prediction per stay)
    - This avoids inflating performance by having many ECGs per ICU stay
    
    Args:
        model: Model to validate.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device to validate on.
    
    Returns:
        Dictionary of validation metrics computed at stay-level.
    """
    model.eval()
    is_multi_task = isinstance(criterion, MultiTaskLoss) or _is_multi_task_model(model)
    
    # Collect all predictions and metadata for stay-level aggregation
    stay_los_logits = {}  # stay_id -> list of LOS logits
    stay_mortality_probs = {}  # stay_id -> list of mortality probs
    stay_los_labels = {}  # stay_id -> LOS label
    stay_mortality_labels = {}  # stay_id -> mortality label
    stay_losses = []  # For computing average loss
    
    with torch.no_grad():
        for batch in val_loader:
            signals = batch["signal"].to(device)  # (B, C, T)
            labels = batch["label"].to(device)  # (B,)
            meta = batch["meta"]
            
            # Get mortality labels if available
            mortality_labels_batch = None
            if is_multi_task and "mortality_label" in batch:
                mortality_labels_batch = batch["mortality_label"].to(device)
            
            # Note: Unmatched samples (label == -1) should already be filtered
            # in dataset, but double-check for safety
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue
            
            signals = signals[valid_mask]
            labels = labels[valid_mask]
            meta = [meta[i] for i in range(len(meta)) if valid_mask[i]]
            if mortality_labels_batch is not None:
                mortality_labels_batch = mortality_labels_batch[valid_mask]
            
            # Get demographic features if available
            demographic_features = None
            if "demographic_features" in batch and batch["demographic_features"] is not None:
                demographic_features = batch["demographic_features"].to(device)
                demographic_features = demographic_features[valid_mask]
            
            # Get diagnosis features if available
            diagnosis_features = None
            if "diagnosis_features" in batch and batch["diagnosis_features"] is not None:
                diagnosis_features = batch["diagnosis_features"].to(device)
                diagnosis_features = diagnosis_features[valid_mask]
            
            # Forward pass
            outputs = model(signals, demographic_features=demographic_features, diagnosis_features=diagnosis_features)
            
            # Handle multi-task vs single-task
            if is_multi_task and isinstance(outputs, dict):
                los_logits = outputs["los"]
                mortality_probs = outputs["mortality"]
                
                # Compute loss if MultiTaskLoss
                if isinstance(criterion, MultiTaskLoss):
                    loss_dict = criterion(
                        los_logits, labels,
                        mortality_probs, mortality_labels_batch
                    )
                    loss = loss_dict["total"]
                else:
                    # Fallback: use LOS loss only
                    loss = criterion(los_logits, labels)
            else:
                # Single-task model
                los_logits = outputs if not isinstance(outputs, dict) else outputs.get("los", outputs)
                loss = criterion(los_logits, labels)
                mortality_probs = None
            
            stay_losses.append(loss.item())
            
            # Group by stay_id for aggregation
            for i in range(len(labels)):
                stay_id = meta[i].get("stay_id")
                if stay_id is None:
                    # Skip if stay_id not available (should not happen after filtering)
                    continue
                
                # LOS aggregation
                if stay_id not in stay_los_logits:
                    stay_los_logits[stay_id] = []
                    stay_los_labels[stay_id] = labels[i].item()
                
                stay_los_logits[stay_id].append(los_logits[i].cpu())
                
                # Mortality aggregation (if available)
                if is_multi_task and mortality_probs is not None:
                    if stay_id not in stay_mortality_probs:
                        if mortality_labels_batch is not None:
                            stay_mortality_labels[stay_id] = mortality_labels_batch[i].item()
                        stay_mortality_probs[stay_id] = []
                    stay_mortality_probs[stay_id].append(mortality_probs[i].cpu())
    
    # Aggregate logits/probs per stay (mean aggregation)
    stay_aggregated_los_logits = []
    stay_aggregated_los_labels = []
    
    for stay_id in stay_los_logits:
        # Mean aggregation of LOS logits across ECGs in the same stay
        aggregated = torch.stack(stay_los_logits[stay_id]).mean(dim=0)  # (num_classes,)
        stay_aggregated_los_logits.append(aggregated)
        stay_aggregated_los_labels.append(stay_los_labels[stay_id])
    
    if len(stay_aggregated_los_logits) == 0:
        return {
            "val_loss": 0.0,
            "val_los_acc": 0.0,
        }
    
    # Stack aggregated predictions
    stay_los_logits_tensor = torch.stack(stay_aggregated_los_logits)  # (num_stays, num_classes)
    stay_los_labels_tensor = torch.tensor(stay_aggregated_los_labels, dtype=torch.long)  # (num_stays,)
    
    # Compute LOS metrics on stay-level
    stay_los_predictions = stay_los_logits_tensor.argmax(dim=1)
    los_correct = (stay_los_predictions == stay_los_labels_tensor).sum().item()
    total_stays = len(stay_los_labels_tensor)
    
    # Average loss across stays
    avg_loss = sum(stay_losses) / len(stay_losses) if stay_losses else 0.0
    
    metrics = {
        "val_loss": avg_loss,
        "val_los_acc": los_correct / total_stays if total_stays > 0 else 0.0,
        "val_num_stays": total_stays,
    }
    
    # Mortality metrics (if available)
    if is_multi_task and stay_mortality_probs:
        # Aggregate mortality probs per stay
        stay_aggregated_mortality_probs = []
        stay_aggregated_mortality_labels = []
        
        for stay_id in stay_mortality_probs:
            if stay_id in stay_mortality_labels:
                aggregated = torch.stack(stay_mortality_probs[stay_id]).mean(dim=0)  # (1,)
                stay_aggregated_mortality_probs.append(aggregated)
                stay_aggregated_mortality_labels.append(stay_mortality_labels[stay_id])
        
        if stay_aggregated_mortality_probs:
            mortality_probs_tensor = torch.stack(stay_aggregated_mortality_probs).squeeze(1)  # (num_stays,)
            mortality_labels_tensor = torch.tensor(stay_aggregated_mortality_labels, dtype=torch.long)  # (num_stays,)
            
            # Filter valid mortality labels
            valid_mortality_mask = mortality_labels_tensor >= 0
            if valid_mortality_mask.any():
                mortality_preds = (mortality_probs_tensor[valid_mortality_mask] > 0.5).long()
                mortality_correct = (mortality_preds == mortality_labels_tensor[valid_mortality_mask]).sum().item()
                mortality_total = valid_mortality_mask.sum().item()
                
                metrics["val_mortality_acc"] = mortality_correct / mortality_total if mortality_total > 0 else 0.0
                
                # Calculate AUC
                try:
                    from sklearn.metrics import roc_auc_score
                    mortality_auc = roc_auc_score(
                        mortality_labels_tensor[valid_mortality_mask].numpy(),
                        mortality_probs_tensor[valid_mortality_mask].numpy()
                    )
                    metrics["val_mortality_auc"] = float(mortality_auc)
                except ImportError:
                    pass
    
    return metrics


def evaluate_with_detailed_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """Evaluate model with detailed metrics including confusion matrix, per-class metrics.
    
    Supports both single-task (LOS only) and multi-task (LOS + Mortality) models.
    
    Policy: Stay-level aggregation (same as validate_epoch)
    - Group predictions by stay_id
    - Aggregate logits/probs per stay by taking mean across ECGs
    - Compute metrics on stays (one prediction per stay)
    
    For multi-task models, computes:
    - LOS metrics: Overall + per-class (as before, unchanged)
    - Mortality metrics: Overall + per LOS-class
    
    Args:
        model: Model to evaluate.
        val_loader: Data loader.
        criterion: Loss function (can be MultiTaskLoss for multi-task).
        device: Device to evaluate on.
        num_classes: Number of LOS classes.
    
    Returns:
        Dictionary with detailed metrics including:
        LOS metrics (unchanged):
        - los_loss: Average loss
        - los_accuracy: Overall accuracy
        - los_balanced_accuracy: Balanced accuracy
        - los_macro_precision: Macro-averaged precision
        - los_macro_recall: Macro-averaged recall
        - los_macro_f1: Macro-averaged F1-score
        - los_per_class_precision: List of precision per class
        - los_per_class_recall: List of recall per class
        - los_per_class_f1: List of F1-score per class
        - los_confusion_matrix: Confusion matrix (numpy array)
        
        Mortality metrics (if multi-task):
        - mortality_accuracy: Overall accuracy
        - mortality_precision: Overall precision
        - mortality_recall: Overall recall
        - mortality_f1: Overall F1-score
        - mortality_auc: Overall AUC
        - mortality_per_los_class: Dict mapping LOS class -> {precision, recall, f1, auc, support}
    """
    try:
        from sklearn.metrics import (
            confusion_matrix,
            precision_score,
            recall_score,
            f1_score,
            balanced_accuracy_score,
            classification_report,
            roc_auc_score,
        )
    except ImportError:
        raise ImportError("scikit-learn is required for detailed metrics. Install with: pip install scikit-learn")
    
    model.eval()
    is_multi_task = isinstance(criterion, MultiTaskLoss) or _is_multi_task_model(model)
    
    # Collect all predictions and metadata for stay-level aggregation
    stay_los_logits = {}  # stay_id -> list of LOS logits
    stay_mortality_probs = {}  # stay_id -> list of mortality probs
    stay_los_labels = {}  # stay_id -> LOS label
    stay_mortality_labels = {}  # stay_id -> mortality label
    stay_losses = []  # For computing average loss
    
    with torch.no_grad():
        for batch in val_loader:
            signals = batch["signal"].to(device)  # (B, C, T)
            labels = batch["label"].to(device)  # (B,)
            meta = batch["meta"]
            
            # Get mortality labels if available
            mortality_labels_batch = None
            if is_multi_task and "mortality_label" in batch:
                mortality_labels_batch = batch["mortality_label"].to(device)
            
            # Filter valid samples
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue
            
            signals = signals[valid_mask]
            labels = labels[valid_mask]
            meta = [meta[i] for i in range(len(meta)) if valid_mask[i]]
            if mortality_labels_batch is not None:
                mortality_labels_batch = mortality_labels_batch[valid_mask]
            
            # Get demographic features if available
            demographic_features = None
            if "demographic_features" in batch and batch["demographic_features"] is not None:
                demographic_features = batch["demographic_features"].to(device)
                demographic_features = demographic_features[valid_mask]
            
            # Get diagnosis features if available
            diagnosis_features = None
            if "diagnosis_features" in batch and batch["diagnosis_features"] is not None:
                diagnosis_features = batch["diagnosis_features"].to(device)
                diagnosis_features = diagnosis_features[valid_mask]
            
            # Forward pass
            outputs = model(signals, demographic_features=demographic_features, diagnosis_features=diagnosis_features)
            
            # Handle multi-task vs single-task
            if is_multi_task and isinstance(outputs, dict):
                los_logits = outputs["los"]
                mortality_probs = outputs["mortality"]
                
                # Compute loss if MultiTaskLoss
                if isinstance(criterion, MultiTaskLoss):
                    loss_dict = criterion(
                        los_logits, labels,
                        mortality_probs, mortality_labels_batch
                    )
                    loss = loss_dict["total"]
                else:
                    # Fallback: use LOS loss only
                    loss = criterion(los_logits, labels)
            else:
                # Single-task model
                los_logits = outputs if not isinstance(outputs, dict) else outputs.get("los", outputs)
                loss = criterion(los_logits, labels)
                mortality_probs = None
            
            stay_losses.append(loss.item())
            
            # Group by stay_id for aggregation
            for i in range(len(labels)):
                stay_id = meta[i].get("stay_id")
                if stay_id is None:
                    continue
                
                # LOS aggregation
                if stay_id not in stay_los_logits:
                    stay_los_logits[stay_id] = []
                    stay_los_labels[stay_id] = labels[i].item()
                
                stay_los_logits[stay_id].append(los_logits[i].cpu())
                
                # Mortality aggregation (if available)
                if is_multi_task and mortality_probs is not None:
                    if stay_id not in stay_mortality_probs:
                        if mortality_labels_batch is not None:
                            stay_mortality_labels[stay_id] = mortality_labels_batch[i].item()
                        stay_mortality_probs[stay_id] = []
                    stay_mortality_probs[stay_id].append(mortality_probs[i].cpu())
    
    # Aggregate logits/probs per stay (mean aggregation)
    stay_aggregated_los_logits = []
    stay_aggregated_los_labels = []
    
    for stay_id in stay_los_logits:
        # Mean aggregation of LOS logits across ECGs in the same stay
        aggregated = torch.stack(stay_los_logits[stay_id]).mean(dim=0)  # (num_classes,)
        stay_aggregated_los_logits.append(aggregated)
        stay_aggregated_los_labels.append(stay_los_labels[stay_id])
    
    if len(stay_aggregated_los_logits) == 0:
        result = {
            "los_loss": 0.0,
            "los_accuracy": 0.0,
            "los_balanced_accuracy": 0.0,
            "los_macro_precision": 0.0,
            "los_macro_recall": 0.0,
            "los_macro_f1": 0.0,
            "los_per_class_precision": [0.0] * num_classes,
            "los_per_class_recall": [0.0] * num_classes,
            "los_per_class_f1": [0.0] * num_classes,
            "los_confusion_matrix": None,
            "num_stays": 0,
        }
        if is_multi_task:
            result.update({
                "mortality_accuracy": 0.0,
                "mortality_precision": 0.0,
                "mortality_recall": 0.0,
                "mortality_f1": 0.0,
                "mortality_auc": 0.0,
                "mortality_per_los_class": {},
            })
        return result
    
    # Stack aggregated predictions
    stay_los_logits_tensor = torch.stack(stay_aggregated_los_logits)  # (num_stays, num_classes)
    stay_los_labels_tensor = torch.tensor(stay_aggregated_los_labels, dtype=torch.long)  # (num_stays,)
    
    # Get LOS predictions
    stay_los_predictions = stay_los_logits_tensor.argmax(dim=1).numpy()
    stay_los_labels_np = stay_los_labels_tensor.numpy()
    
    # Average loss
    avg_loss = sum(stay_losses) / len(stay_losses) if stay_losses else 0.0
    
    # ===== LOS METRICS (unchanged, as before) =====
    los_accuracy = (stay_los_predictions == stay_los_labels_np).mean()
    los_balanced_acc = balanced_accuracy_score(stay_los_labels_np, stay_los_predictions)
    
    # Per-class metrics (with zero_division=0 to handle classes with no predictions)
    los_precision_per_class = precision_score(
        stay_los_labels_np, stay_los_predictions, 
        labels=list(range(num_classes)), 
        average=None, 
        zero_division=0
    )
    los_recall_per_class = recall_score(
        stay_los_labels_np, stay_los_predictions,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )
    los_f1_per_class = f1_score(
        stay_los_labels_np, stay_los_predictions,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )
    
    # Macro-averaged metrics
    los_macro_precision = precision_score(
        stay_los_labels_np, stay_los_predictions,
        labels=list(range(num_classes)),
        average='macro',
        zero_division=0
    )
    los_macro_recall = recall_score(
        stay_los_labels_np, stay_los_predictions,
        labels=list(range(num_classes)),
        average='macro',
        zero_division=0
    )
    los_macro_f1 = f1_score(
        stay_los_labels_np, stay_los_predictions,
        labels=list(range(num_classes)),
        average='macro',
        zero_division=0
    )
    
    # Confusion matrix
    los_cm = confusion_matrix(
        stay_los_labels_np, stay_los_predictions,
        labels=list(range(num_classes))
    )
    
    # Build result dictionary with LOS metrics
    result = {
        "los_loss": avg_loss,
        "los_accuracy": float(los_accuracy),
        "los_balanced_accuracy": float(los_balanced_acc),
        "los_macro_precision": float(los_macro_precision),
        "los_macro_recall": float(los_macro_recall),
        "los_macro_f1": float(los_macro_f1),
        "los_per_class_precision": los_precision_per_class.tolist(),
        "los_per_class_recall": los_recall_per_class.tolist(),
        "los_per_class_f1": los_f1_per_class.tolist(),
        "los_confusion_matrix": los_cm.tolist(),
        "num_stays": len(stay_los_labels_np),
    }
    
    # ===== MORTALITY METRICS (if multi-task) =====
    if is_multi_task and stay_mortality_probs:
        # Aggregate mortality probs per stay
        stay_aggregated_mortality_probs = []
        stay_aggregated_mortality_labels = []
        stay_aggregated_los_labels_for_mortality = []  # LOS class for each stay (for per-class metrics)
        
        for stay_id in stay_mortality_probs:
            if stay_id in stay_mortality_labels:
                aggregated = torch.stack(stay_mortality_probs[stay_id]).mean(dim=0)  # (1,)
                stay_aggregated_mortality_probs.append(aggregated)
                stay_aggregated_mortality_labels.append(stay_mortality_labels[stay_id])
                # Get LOS class for this stay
                if stay_id in stay_los_labels:
                    stay_aggregated_los_labels_for_mortality.append(stay_los_labels[stay_id])
        
        if stay_aggregated_mortality_probs:
            mortality_probs_tensor = torch.stack(stay_aggregated_mortality_probs).squeeze(1)  # (num_stays,)
            mortality_labels_tensor = torch.tensor(stay_aggregated_mortality_labels, dtype=torch.long)  # (num_stays,)
            los_labels_for_mortality = torch.tensor(stay_aggregated_los_labels_for_mortality, dtype=torch.long)  # (num_stays,)
            
            # Filter valid mortality labels
            valid_mortality_mask = mortality_labels_tensor >= 0
            if valid_mortality_mask.any():
                mortality_probs_valid = mortality_probs_tensor[valid_mortality_mask].numpy()
                mortality_labels_valid = mortality_labels_tensor[valid_mortality_mask].numpy()
                los_labels_valid = los_labels_for_mortality[valid_mortality_mask].numpy()
                
                # Overall mortality metrics
                mortality_preds = (mortality_probs_valid > 0.5).astype(int)
                mortality_accuracy = (mortality_preds == mortality_labels_valid).mean()
                mortality_precision = precision_score(mortality_labels_valid, mortality_preds, zero_division=0)
                mortality_recall = recall_score(mortality_labels_valid, mortality_preds, zero_division=0)
                mortality_f1 = f1_score(mortality_labels_valid, mortality_preds, zero_division=0)
                
                try:
                    mortality_auc = roc_auc_score(mortality_labels_valid, mortality_probs_valid)
                except ValueError:
                    # Can happen if only one class present
                    mortality_auc = 0.0
                
                result.update({
                    "mortality_accuracy": float(mortality_accuracy),
                    "mortality_precision": float(mortality_precision),
                    "mortality_recall": float(mortality_recall),
                    "mortality_f1": float(mortality_f1),
                    "mortality_auc": float(mortality_auc),
                })
                
                # Mortality metrics per LOS class
                mortality_per_los_class = {}
                for los_class in range(num_classes):
                    mask = los_labels_valid == los_class
                    if mask.sum() > 0:
                        mortality_probs_class = mortality_probs_valid[mask]
                        mortality_labels_class = mortality_labels_valid[mask]
                        
                        if len(np.unique(mortality_labels_class)) > 1:  # Both classes present
                            mortality_preds_class = (mortality_probs_class > 0.5).astype(int)
                            precision_class = precision_score(mortality_labels_class, mortality_preds_class, zero_division=0)
                            recall_class = recall_score(mortality_labels_class, mortality_preds_class, zero_division=0)
                            f1_class = f1_score(mortality_labels_class, mortality_preds_class, zero_division=0)
                            
                            try:
                                auc_class = roc_auc_score(mortality_labels_class, mortality_probs_class)
                            except ValueError:
                                auc_class = 0.0
                        else:
                            # Only one class present, set metrics to 0
                            precision_class = 0.0
                            recall_class = 0.0
                            f1_class = 0.0
                            auc_class = 0.0
                        
                        mortality_per_los_class[los_class] = {
                            "precision": float(precision_class),
                            "recall": float(recall_class),
                            "f1": float(f1_class),
                            "auc": float(auc_class),
                            "support": int(mask.sum()),
                        }
                    else:
                        # No samples for this LOS class
                        mortality_per_los_class[los_class] = {
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                            "auc": 0.0,
                            "support": 0,
                        }
                
                result["mortality_per_los_class"] = mortality_per_los_class
    
    return result
