"""Common training loop implementation."""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import get_loss
from .callbacks import EarlyStopping, ModelCheckpoint


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        model: Model to train.
        train_loader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to train on.
        config: Configuration dictionary.
    
    Returns:
        Dictionary of training metrics.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
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
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected! Skipping batch.")
            print(f"  Signal stats: min={signals.min():.4f}, max={signals.max():.4f}, mean={signals.mean():.4f}, std={signals.std():.4f}")
            print(f"  Labels: {labels.unique()}")
            print(f"  Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return {
        "train_loss": total_loss / len(train_loader) if len(train_loader) > 0 else 0.0,
        "train_acc": correct / total if total > 0 else 0.0,
    }


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
    
    # Collect all predictions and metadata for stay-level aggregation
    stay_logits = {}  # stay_id -> list of logits
    stay_labels = {}  # stay_id -> label (should be same for all ECGs in stay)
    stay_losses = []  # For computing average loss
    
    with torch.no_grad():
        for batch in val_loader:
            signals = batch["signal"].to(device)  # (B, C, T)
            labels = batch["label"].to(device)  # (B,)
            meta = batch["meta"]
            
            # Note: Unmatched samples (label == -1) should already be filtered
            # in dataset, but double-check for safety
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue
            
            signals = signals[valid_mask]
            labels = labels[valid_mask]
            meta = [meta[i] for i in range(len(meta)) if valid_mask[i]]
            
            # Forward pass
            logits = model(signals)  # (B, num_classes)
            loss = criterion(logits, labels)
            stay_losses.append(loss.item())
            
            # Group by stay_id for aggregation
            for i in range(len(labels)):
                stay_id = meta[i].get("stay_id")
                if stay_id is None:
                    # Skip if stay_id not available (should not happen after filtering)
                    continue
                
                if stay_id not in stay_logits:
                    stay_logits[stay_id] = []
                    stay_labels[stay_id] = labels[i].item()
                
                stay_logits[stay_id].append(logits[i].cpu())
    
    # Aggregate logits per stay (mean aggregation)
    stay_aggregated_logits = []
    stay_aggregated_labels = []
    
    for stay_id in stay_logits:
        # Mean aggregation of logits across ECGs in the same stay
        aggregated = torch.stack(stay_logits[stay_id]).mean(dim=0)  # (num_classes,)
        stay_aggregated_logits.append(aggregated)
        stay_aggregated_labels.append(stay_labels[stay_id])
    
    if len(stay_aggregated_logits) == 0:
        return {
            "val_loss": 0.0,
            "val_acc": 0.0,
        }
    
    # Stack aggregated predictions
    stay_logits_tensor = torch.stack(stay_aggregated_logits)  # (num_stays, num_classes)
    stay_labels_tensor = torch.tensor(stay_aggregated_labels, dtype=torch.long)  # (num_stays,)
    
    # Compute metrics on stay-level
    stay_predictions = stay_logits_tensor.argmax(dim=1)
    correct = (stay_predictions == stay_labels_tensor).sum().item()
    total_stays = len(stay_labels_tensor)
    
    # Average loss across stays
    avg_loss = sum(stay_losses) / len(stay_losses) if stay_losses else 0.0
    
    # Calculate AUC if binary classification
    val_auc = None
    if stay_logits_tensor.shape[1] == 2:  # Binary classification
        try:
            from sklearn.metrics import roc_auc_score
            stay_probs = torch.softmax(stay_logits_tensor, dim=1)
            val_auc = roc_auc_score(stay_labels_tensor.numpy(), stay_probs[:, 1].numpy())
        except ImportError:
            pass
    
    metrics = {
        "val_loss": avg_loss,
        "val_acc": correct / total_stays if total_stays > 0 else 0.0,
        "val_num_stays": total_stays,  # Log number of stays evaluated
    }
    
    if val_auc is not None:
        metrics["val_auc"] = val_auc
    
    return metrics


def evaluate_with_detailed_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """Evaluate model with detailed metrics including confusion matrix, per-class metrics.
    
    Policy: Stay-level aggregation (same as validate_epoch)
    - Group predictions by stay_id
    - Aggregate logits per stay by taking mean across ECGs
    - Compute metrics on stays (one prediction per stay)
    
    Args:
        model: Model to evaluate.
        val_loader: Data loader.
        criterion: Loss function.
        device: Device to evaluate on.
        num_classes: Number of classes.
    
    Returns:
        Dictionary with detailed metrics including:
        - loss: Average loss
        - accuracy: Overall accuracy
        - balanced_accuracy: Balanced accuracy (macro-averaged recall)
        - macro_precision: Macro-averaged precision
        - macro_recall: Macro-averaged recall
        - macro_f1: Macro-averaged F1-score
        - per_class_precision: List of precision per class
        - per_class_recall: List of recall per class
        - per_class_f1: List of F1-score per class
        - confusion_matrix: Confusion matrix (numpy array)
        - num_stays: Number of stays evaluated
    """
    try:
        from sklearn.metrics import (
            confusion_matrix,
            precision_score,
            recall_score,
            f1_score,
            balanced_accuracy_score,
            classification_report,
        )
    except ImportError:
        raise ImportError("scikit-learn is required for detailed metrics. Install with: pip install scikit-learn")
    
    model.eval()
    
    # Collect all predictions and metadata for stay-level aggregation
    stay_logits = {}  # stay_id -> list of logits
    stay_labels = {}  # stay_id -> label (should be same for all ECGs in stay)
    stay_losses = []  # For computing average loss
    
    with torch.no_grad():
        for batch in val_loader:
            signals = batch["signal"].to(device)  # (B, C, T)
            labels = batch["label"].to(device)  # (B,)
            meta = batch["meta"]
            
            # Filter valid samples
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue
            
            signals = signals[valid_mask]
            labels = labels[valid_mask]
            meta = [meta[i] for i in range(len(meta)) if valid_mask[i]]
            
            # Forward pass
            logits = model(signals)  # (B, num_classes)
            loss = criterion(logits, labels)
            stay_losses.append(loss.item())
            
            # Group by stay_id for aggregation
            for i in range(len(labels)):
                stay_id = meta[i].get("stay_id")
                if stay_id is None:
                    continue
                
                if stay_id not in stay_logits:
                    stay_logits[stay_id] = []
                    stay_labels[stay_id] = labels[i].item()
                
                stay_logits[stay_id].append(logits[i].cpu())
    
    # Aggregate logits per stay (mean aggregation)
    stay_aggregated_logits = []
    stay_aggregated_labels = []
    
    for stay_id in stay_logits:
        # Mean aggregation of logits across ECGs in the same stay
        aggregated = torch.stack(stay_logits[stay_id]).mean(dim=0)  # (num_classes,)
        stay_aggregated_logits.append(aggregated)
        stay_aggregated_labels.append(stay_labels[stay_id])
    
    if len(stay_aggregated_logits) == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class_precision": [0.0] * num_classes,
            "per_class_recall": [0.0] * num_classes,
            "per_class_f1": [0.0] * num_classes,
            "confusion_matrix": None,
            "num_stays": 0,
        }
    
    # Stack aggregated predictions
    stay_logits_tensor = torch.stack(stay_aggregated_logits)  # (num_stays, num_classes)
    stay_labels_tensor = torch.tensor(stay_aggregated_labels, dtype=torch.long)  # (num_stays,)
    
    # Get predictions
    stay_predictions = stay_logits_tensor.argmax(dim=1).numpy()
    stay_labels_np = stay_labels_tensor.numpy()
    
    # Average loss
    avg_loss = sum(stay_losses) / len(stay_losses) if stay_losses else 0.0
    
    # Compute metrics
    accuracy = (stay_predictions == stay_labels_np).mean()
    balanced_acc = balanced_accuracy_score(stay_labels_np, stay_predictions)
    
    # Per-class metrics (with zero_division=0 to handle classes with no predictions)
    precision_per_class = precision_score(
        stay_labels_np, stay_predictions, 
        labels=list(range(num_classes)), 
        average=None, 
        zero_division=0
    )
    recall_per_class = recall_score(
        stay_labels_np, stay_predictions,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )
    f1_per_class = f1_score(
        stay_labels_np, stay_predictions,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )
    
    # Macro-averaged metrics
    macro_precision = precision_score(
        stay_labels_np, stay_predictions,
        labels=list(range(num_classes)),
        average='macro',
        zero_division=0
    )
    macro_recall = recall_score(
        stay_labels_np, stay_predictions,
        labels=list(range(num_classes)),
        average='macro',
        zero_division=0
    )
    macro_f1 = f1_score(
        stay_labels_np, stay_predictions,
        labels=list(range(num_classes)),
        average='macro',
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(
        stay_labels_np, stay_predictions,
        labels=list(range(num_classes))
    )
    
    return {
        "loss": avg_loss,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class_precision": precision_per_class.tolist(),
        "per_class_recall": recall_per_class.tolist(),
        "per_class_f1": f1_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
        "num_stays": len(stay_labels_np),
    }
