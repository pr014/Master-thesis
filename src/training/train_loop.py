"""Common training loop implementation for regression and classification."""

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


def _compute_regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics.
    
    Args:
        predictions: Predicted values (continuous)
        targets: Ground truth values (continuous)
    
    Returns:
        Dictionary with MAE, MSE, RMSE, R² metrics
    """
    mae = np.abs(predictions - targets).mean()
    mse = ((predictions - targets) ** 2).mean()
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }


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
    Now supports regression for LOS (continuous prediction in days).
    
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
    
    # Regression metrics for LOS
    all_los_predictions = []
    all_los_targets = []
    
    # Mortality metrics
    mortality_correct = 0
    mortality_total = 0
    
    gradient_clip_norm = config.get("training", {}).get("gradient_clip_norm", None)
    
    for batch in train_loader:
        signals = batch["signal"].to(device)  # (B, C, T)
        labels = batch["label"].to(device)  # (B,) - float for regression
        
        # Note: Unmatched samples (label < 0) should already be filtered
        # in dataset initialization, but double-check for safety
        valid_mask = labels >= 0
        if not valid_mask.any():
            continue
        
        signals = signals[valid_mask]
        labels = labels[valid_mask]
        
        # Sample weights for regression (optional)
        sample_weights = None
        if "sample_weight" in batch:
            sample_weights = batch["sample_weight"].to(device)[valid_mask]
        
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
        
        # Get ICU unit features if available
        icu_unit_features = None
        if "icu_unit_features" in batch and batch["icu_unit_features"] is not None:
            icu_unit_features = batch["icu_unit_features"].to(device)
            if valid_mask.any():
                icu_unit_features = icu_unit_features[valid_mask]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(signals, demographic_features=demographic_features, diagnosis_features=diagnosis_features, icu_unit_features=icu_unit_features)
        
        # Handle multi-task vs single-task
        if is_multi_task and isinstance(outputs, dict):
            # Multi-task model
            los_predictions = outputs["los"]  # (B, 1) for regression
            mortality_probs = outputs["mortality"]
            
            if isinstance(criterion, MultiTaskLoss):
                loss_dict = criterion(
                    los_predictions, labels,
                    mortality_probs, mortality_labels,
                    los_sample_weights=sample_weights,
                )
                loss = loss_dict["total"]
                los_loss = loss_dict["los"]
                mortality_loss = loss_dict["mortality"]
            else:
                # Fallback: use LOS loss only if MultiTaskLoss not provided
                los_preds_flat = los_predictions.squeeze(-1) if los_predictions.dim() > 1 else los_predictions
                if sample_weights is not None:
                    loss = criterion(los_preds_flat, labels.float(), weight=sample_weights)
                else:
                    loss = criterion(los_preds_flat, labels.float())
                los_loss = loss
                mortality_loss = torch.tensor(0.0, device=device)
        elif is_multi_task and isinstance(outputs, tuple):
            # Tuple output (los_predictions, mortality_probs)
            los_predictions, mortality_probs = outputs
            
            if isinstance(criterion, MultiTaskLoss):
                loss_dict = criterion(
                    los_predictions, labels,
                    mortality_probs, mortality_labels,
                    los_sample_weights=sample_weights,
                )
                loss = loss_dict["total"]
                los_loss = loss_dict["los"]
                mortality_loss = loss_dict["mortality"]
            else:
                los_preds_flat = los_predictions.squeeze(-1) if los_predictions.dim() > 1 else los_predictions
                if sample_weights is not None and hasattr(criterion, "forward"):
                    loss = criterion(los_preds_flat, labels.float(), weight=sample_weights)
                else:
                    loss = criterion(los_preds_flat, labels.float())
                los_loss = loss
                mortality_loss = torch.tensor(0.0, device=device)
        else:
            # Single-task model
            los_predictions = outputs if not isinstance(outputs, dict) else outputs.get("los", outputs)
            los_preds_flat = los_predictions.squeeze(-1) if los_predictions.dim() > 1 else los_predictions
            if sample_weights is not None:
                loss = criterion(los_preds_flat, labels.float(), weight=sample_weights)
            else:
                loss = criterion(los_preds_flat, labels.float())
            los_loss = loss
            mortality_loss = torch.tensor(0.0, device=device)
            mortality_probs = None
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected! Skipping batch.")
            print(f"  Signal stats: min={signals.min():.4f}, max={signals.max():.4f}, mean={signals.mean():.4f}, std={signals.std():.4f}")
            print(f"  Labels: min={labels.min():.2f}, max={labels.max():.2f}")
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
        
        # LOS regression metrics - collect predictions and targets
        los_preds_flat = los_predictions.squeeze(-1) if los_predictions.dim() > 1 else los_predictions
        all_los_predictions.extend(los_preds_flat.detach().cpu().numpy().tolist())
        all_los_targets.extend(labels.cpu().numpy().tolist())
        
        # Mortality metrics (if available)
        if is_multi_task and mortality_labels is not None and mortality_probs is not None:
            valid_mortality_mask = mortality_labels >= 0
            if valid_mortality_mask.any():
                mortality_preds = (mortality_probs.squeeze(-1) > 0.5).long()
                mortality_correct += (mortality_preds[valid_mortality_mask] == mortality_labels[valid_mortality_mask]).sum().item()
                mortality_total += valid_mortality_mask.sum().item()
    
    # Compute regression metrics
    if len(all_los_predictions) > 0:
        regression_metrics = _compute_regression_metrics(
            np.array(all_los_predictions),
            np.array(all_los_targets)
        )
    else:
        regression_metrics = {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2": 0.0}
    
    metrics = {
        "train_loss": total_loss / len(train_loader) if len(train_loader) > 0 else 0.0,
        "train_los_mae": regression_metrics["mae"],
        "train_los_mse": regression_metrics["mse"],
        "train_los_rmse": regression_metrics["rmse"],
        "train_los_r2": regression_metrics["r2"],
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
    - Aggregate predictions per stay by taking mean across ECGs
    - Compute metrics on stays (one prediction per stay)
    - This avoids inflating performance by having many ECGs per ICU stay
    
    For regression: LOS predictions are continuous values in days.
    
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
    stay_los_predictions = {}  # stay_id -> list of LOS predictions
    stay_mortality_probs = {}  # stay_id -> list of mortality probs
    stay_los_labels = {}  # stay_id -> LOS label (continuous)
    stay_mortality_labels = {}  # stay_id -> mortality label
    stay_losses = []  # For computing average loss
    
    with torch.no_grad():
        for batch in val_loader:
            signals = batch["signal"].to(device)  # (B, C, T)
            labels = batch["label"].to(device)  # (B,) - float for regression
            meta = batch["meta"]
            
            # Get mortality labels if available
            mortality_labels_batch = None
            if is_multi_task and "mortality_label" in batch:
                mortality_labels_batch = batch["mortality_label"].to(device)
            
            # Note: Unmatched samples (label < 0) should already be filtered
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
            
            # Get ICU unit features if available
            icu_unit_features = None
            if "icu_unit_features" in batch and batch["icu_unit_features"] is not None:
                icu_unit_features = batch["icu_unit_features"].to(device)
                icu_unit_features = icu_unit_features[valid_mask]
            
            # Forward pass
            outputs = model(signals, demographic_features=demographic_features, diagnosis_features=diagnosis_features, icu_unit_features=icu_unit_features)
            
            # Handle multi-task vs single-task
            if is_multi_task and isinstance(outputs, dict):
                los_predictions = outputs["los"]
                mortality_probs = outputs["mortality"]
                
                # Compute loss if MultiTaskLoss
                if isinstance(criterion, MultiTaskLoss):
                    loss_dict = criterion(
                        los_predictions, labels,
                        mortality_probs, mortality_labels_batch
                    )
                    loss = loss_dict["total"]
                else:
                    los_preds_flat = los_predictions.squeeze(-1) if los_predictions.dim() > 1 else los_predictions
                    loss = criterion(los_preds_flat, labels.float())
            elif is_multi_task and isinstance(outputs, tuple):
                los_predictions, mortality_probs = outputs
                
                if isinstance(criterion, MultiTaskLoss):
                    loss_dict = criterion(
                        los_predictions, labels,
                        mortality_probs, mortality_labels_batch
                    )
                    loss = loss_dict["total"]
                else:
                    los_preds_flat = los_predictions.squeeze(-1) if los_predictions.dim() > 1 else los_predictions
                    loss = criterion(los_preds_flat, labels.float())
            else:
                # Single-task model
                los_predictions = outputs if not isinstance(outputs, dict) else outputs.get("los", outputs)
                los_preds_flat = los_predictions.squeeze(-1) if los_predictions.dim() > 1 else los_predictions
                loss = criterion(los_preds_flat, labels.float())
                mortality_probs = None
            
            stay_losses.append(loss.item())
            
            # Group by stay_id for aggregation
            for i in range(len(labels)):
                stay_id = meta[i].get("stay_id")
                if stay_id is None:
                    # Skip if stay_id not available (should not happen after filtering)
                    continue
                
                # LOS aggregation - store continuous predictions
                if stay_id not in stay_los_predictions:
                    stay_los_predictions[stay_id] = []
                    stay_los_labels[stay_id] = labels[i].item()  # Ground truth LOS in days
                
                los_pred = los_predictions[i].squeeze(-1) if los_predictions.dim() > 1 else los_predictions[i]
                stay_los_predictions[stay_id].append(los_pred.cpu().item())
                
                # Mortality aggregation (if available)
                if is_multi_task and mortality_probs is not None:
                    if stay_id not in stay_mortality_probs:
                        if mortality_labels_batch is not None:
                            stay_mortality_labels[stay_id] = mortality_labels_batch[i].item()
                        stay_mortality_probs[stay_id] = []
                    stay_mortality_probs[stay_id].append(mortality_probs[i].cpu().item())
    
    # Aggregate predictions per stay (mean aggregation)
    stay_aggregated_los_predictions = []
    stay_aggregated_los_labels = []
    
    for stay_id in stay_los_predictions:
        # Mean aggregation of LOS predictions across ECGs in the same stay
        aggregated = np.mean(stay_los_predictions[stay_id])
        stay_aggregated_los_predictions.append(aggregated)
        stay_aggregated_los_labels.append(stay_los_labels[stay_id])
    
    if len(stay_aggregated_los_predictions) == 0:
        return {
            "val_loss": 0.0,
            "val_los_mae": 0.0,
            "val_los_mse": 0.0,
            "val_los_rmse": 0.0,
            "val_los_r2": 0.0,
        }
    
    # Compute regression metrics on stay-level
    predictions_np = np.array(stay_aggregated_los_predictions)
    labels_np = np.array(stay_aggregated_los_labels)
    
    regression_metrics = _compute_regression_metrics(predictions_np, labels_np)
    
    # Average loss across stays
    avg_loss = sum(stay_losses) / len(stay_losses) if stay_losses else 0.0
    
    metrics = {
        "val_loss": avg_loss,
        "val_los_mae": regression_metrics["mae"],
        "val_los_mse": regression_metrics["mse"],
        "val_los_rmse": regression_metrics["rmse"],
        "val_los_r2": regression_metrics["r2"],
        "val_num_stays": len(stay_aggregated_los_predictions),
    }
    
    # Mortality metrics (if available)
    if is_multi_task and stay_mortality_probs:
        # Aggregate mortality probs per stay
        stay_aggregated_mortality_probs = []
        stay_aggregated_mortality_labels = []
        
        for stay_id in stay_mortality_probs:
            if stay_id in stay_mortality_labels:
                aggregated = np.mean(stay_mortality_probs[stay_id])
                stay_aggregated_mortality_probs.append(aggregated)
                stay_aggregated_mortality_labels.append(stay_mortality_labels[stay_id])
        
        if stay_aggregated_mortality_probs:
            mortality_probs_np = np.array(stay_aggregated_mortality_probs)
            mortality_labels_np = np.array(stay_aggregated_mortality_labels)
            
            # Filter valid mortality labels
            valid_mortality_mask = mortality_labels_np >= 0
            if valid_mortality_mask.any():
                mortality_probs_valid = mortality_probs_np[valid_mortality_mask]
                mortality_labels_valid = mortality_labels_np[valid_mortality_mask]
                
                mortality_preds = (mortality_probs_valid > 0.5).astype(int)
                mortality_correct = (mortality_preds == mortality_labels_valid).sum()
                mortality_total = len(mortality_labels_valid)
                
                metrics["val_mortality_acc"] = mortality_correct / mortality_total if mortality_total > 0 else 0.0
                
                # Calculate AUC
                try:
                    from sklearn.metrics import roc_auc_score
                    if len(np.unique(mortality_labels_valid)) > 1:
                        mortality_auc = roc_auc_score(mortality_labels_valid, mortality_probs_valid)
                        metrics["val_mortality_auc"] = float(mortality_auc)
                except (ImportError, ValueError):
                    pass
    
    return metrics


def evaluate_with_detailed_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model with detailed regression metrics.
    
    Supports both single-task (LOS only) and multi-task (LOS + Mortality) models.
    
    Policy: Stay-level aggregation (same as validate_epoch)
    - Group predictions by stay_id
    - Aggregate predictions per stay by taking mean across ECGs
    - Compute metrics on stays (one prediction per stay)
    
    For regression, computes:
    - LOS metrics: MAE, MSE, RMSE, R², Median Absolute Error, Percentile Errors
    - Mortality metrics: Accuracy, Precision, Recall, F1, AUC
    
    Args:
        model: Model to evaluate.
        val_loader: Data loader.
        criterion: Loss function (can be MultiTaskLoss for multi-task).
        device: Device to evaluate on.
    
    Returns:
        Dictionary with detailed metrics including:
        LOS regression metrics:
        - los_loss: Average loss
        - los_mae: Mean Absolute Error
        - los_mse: Mean Squared Error
        - los_rmse: Root Mean Squared Error
        - los_r2: R² Score
        - los_median_ae: Median Absolute Error
        - los_p25_error: 25th percentile of absolute errors
        - los_p50_error: 50th percentile (median) of absolute errors
        - los_p75_error: 75th percentile of absolute errors
        - los_p90_error: 90th percentile of absolute errors
        
        Mortality metrics (if multi-task):
        - mortality_accuracy: Overall accuracy
        - mortality_precision: Overall precision
        - mortality_recall: Overall recall
        - mortality_f1: Overall F1-score
        - mortality_auc: Overall AUC
    """
    try:
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )
    except ImportError:
        raise ImportError("scikit-learn is required for detailed metrics. Install with: pip install scikit-learn")
    
    model.eval()
    is_multi_task = isinstance(criterion, MultiTaskLoss) or _is_multi_task_model(model)
    
    # Collect all predictions and metadata for stay-level aggregation
    stay_los_predictions = {}  # stay_id -> list of LOS predictions
    stay_mortality_probs = {}  # stay_id -> list of mortality probs
    stay_los_labels = {}  # stay_id -> LOS label (continuous)
    stay_mortality_labels = {}  # stay_id -> mortality label
    stay_losses = []  # For computing average loss
    
    with torch.no_grad():
        for batch in val_loader:
            signals = batch["signal"].to(device)  # (B, C, T)
            labels = batch["label"].to(device)  # (B,) - float for regression
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
            
            # Get ICU unit features if available
            icu_unit_features = None
            if "icu_unit_features" in batch and batch["icu_unit_features"] is not None:
                icu_unit_features = batch["icu_unit_features"].to(device)
                icu_unit_features = icu_unit_features[valid_mask]
            
            # Forward pass
            outputs = model(signals, demographic_features=demographic_features, diagnosis_features=diagnosis_features, icu_unit_features=icu_unit_features)
            
            # Handle multi-task vs single-task
            if is_multi_task and isinstance(outputs, dict):
                los_predictions = outputs["los"]
                mortality_probs = outputs["mortality"]
                
                if isinstance(criterion, MultiTaskLoss):
                    loss_dict = criterion(
                        los_predictions, labels,
                        mortality_probs, mortality_labels_batch
                    )
                    loss = loss_dict["total"]
                else:
                    los_preds_flat = los_predictions.squeeze(-1) if los_predictions.dim() > 1 else los_predictions
                    loss = criterion(los_preds_flat, labels.float())
            elif is_multi_task and isinstance(outputs, tuple):
                los_predictions, mortality_probs = outputs
                
                if isinstance(criterion, MultiTaskLoss):
                    loss_dict = criterion(
                        los_predictions, labels,
                        mortality_probs, mortality_labels_batch
                    )
                    loss = loss_dict["total"]
                else:
                    los_preds_flat = los_predictions.squeeze(-1) if los_predictions.dim() > 1 else los_predictions
                    loss = criterion(los_preds_flat, labels.float())
            else:
                # Single-task model
                los_predictions = outputs if not isinstance(outputs, dict) else outputs.get("los", outputs)
                los_preds_flat = los_predictions.squeeze(-1) if los_predictions.dim() > 1 else los_predictions
                loss = criterion(los_preds_flat, labels.float())
                mortality_probs = None
            
            stay_losses.append(loss.item())
            
            # Group by stay_id for aggregation
            for i in range(len(labels)):
                stay_id = meta[i].get("stay_id")
                if stay_id is None:
                    continue
                
                # LOS aggregation
                if stay_id not in stay_los_predictions:
                    stay_los_predictions[stay_id] = []
                    stay_los_labels[stay_id] = labels[i].item()
                
                los_pred = los_predictions[i].squeeze(-1) if los_predictions.dim() > 1 else los_predictions[i]
                stay_los_predictions[stay_id].append(los_pred.cpu().item())
                
                # Mortality aggregation (if available)
                if is_multi_task and mortality_probs is not None:
                    if stay_id not in stay_mortality_probs:
                        if mortality_labels_batch is not None:
                            stay_mortality_labels[stay_id] = mortality_labels_batch[i].item()
                        stay_mortality_probs[stay_id] = []
                    stay_mortality_probs[stay_id].append(mortality_probs[i].cpu().item())
    
    # Aggregate predictions per stay (mean aggregation)
    stay_aggregated_los_predictions = []
    stay_aggregated_los_labels = []
    
    for stay_id in stay_los_predictions:
        aggregated = np.mean(stay_los_predictions[stay_id])
        stay_aggregated_los_predictions.append(aggregated)
        stay_aggregated_los_labels.append(stay_los_labels[stay_id])
    
    if len(stay_aggregated_los_predictions) == 0:
        result = {
            "los_loss": 0.0,
            "los_mae": 0.0,
            "los_mse": 0.0,
            "los_rmse": 0.0,
            "los_r2": 0.0,
            "los_median_ae": 0.0,
            "los_p25_error": 0.0,
            "los_p50_error": 0.0,
            "los_p75_error": 0.0,
            "los_p90_error": 0.0,
            "num_stays": 0,
        }
        if is_multi_task:
            result.update({
                "mortality_accuracy": 0.0,
                "mortality_precision": 0.0,
                "mortality_recall": 0.0,
                "mortality_f1": 0.0,
                "mortality_auc": 0.0,
            })
        return result
    
    # Convert to numpy arrays
    predictions_np = np.array(stay_aggregated_los_predictions)
    labels_np = np.array(stay_aggregated_los_labels)
    
    # Average loss
    avg_loss = sum(stay_losses) / len(stay_losses) if stay_losses else 0.0
    
    # ===== LOS REGRESSION METRICS =====
    absolute_errors = np.abs(predictions_np - labels_np)
    
    # Basic metrics
    mae = absolute_errors.mean()
    mse = ((predictions_np - labels_np) ** 2).mean()
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = ((labels_np - predictions_np) ** 2).sum()
    ss_tot = ((labels_np - labels_np.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Median Absolute Error
    median_ae = np.median(absolute_errors)
    
    # Percentile errors
    p25_error = np.percentile(absolute_errors, 25)
    p50_error = np.percentile(absolute_errors, 50)  # Same as median
    p75_error = np.percentile(absolute_errors, 75)
    p90_error = np.percentile(absolute_errors, 90)
    
    # Build result dictionary with LOS metrics
    result = {
        "los_loss": avg_loss,
        "los_mae": float(mae),
        "los_mse": float(mse),
        "los_rmse": float(rmse),
        "los_r2": float(r2),
        "los_median_ae": float(median_ae),
        "los_p25_error": float(p25_error),
        "los_p50_error": float(p50_error),
        "los_p75_error": float(p75_error),
        "los_p90_error": float(p90_error),
        "num_stays": len(predictions_np),
    }
    
    # ===== MORTALITY METRICS (if multi-task) =====
    if is_multi_task and stay_mortality_probs:
        # Aggregate mortality probs per stay
        stay_aggregated_mortality_probs = []
        stay_aggregated_mortality_labels = []
        
        for stay_id in stay_mortality_probs:
            if stay_id in stay_mortality_labels:
                aggregated = np.mean(stay_mortality_probs[stay_id])
                stay_aggregated_mortality_probs.append(aggregated)
                stay_aggregated_mortality_labels.append(stay_mortality_labels[stay_id])
        
        if stay_aggregated_mortality_probs:
            mortality_probs_np = np.array(stay_aggregated_mortality_probs)
            mortality_labels_np = np.array(stay_aggregated_mortality_labels)
            
            # Filter valid mortality labels
            valid_mortality_mask = mortality_labels_np >= 0
            if valid_mortality_mask.any():
                mortality_probs_valid = mortality_probs_np[valid_mortality_mask]
                mortality_labels_valid = mortality_labels_np[valid_mortality_mask]
                
                # Overall mortality metrics
                mortality_preds = (mortality_probs_valid > 0.5).astype(int)
                mortality_accuracy = (mortality_preds == mortality_labels_valid).mean()
                mortality_precision = precision_score(mortality_labels_valid, mortality_preds, zero_division=0)
                mortality_recall = recall_score(mortality_labels_valid, mortality_preds, zero_division=0)
                mortality_f1 = f1_score(mortality_labels_valid, mortality_preds, zero_division=0)
                
                try:
                    if len(np.unique(mortality_labels_valid)) > 1:
                        mortality_auc = roc_auc_score(mortality_labels_valid, mortality_probs_valid)
                    else:
                        mortality_auc = 0.0
                except ValueError:
                    mortality_auc = 0.0
                
                result.update({
                    "mortality_accuracy": float(mortality_accuracy),
                    "mortality_precision": float(mortality_precision),
                    "mortality_recall": float(mortality_recall),
                    "mortality_f1": float(mortality_f1),
                    "mortality_auc": float(mortality_auc),
                })
    
    return result
