"""Deep Learning feature extraction for ECG signals.

Extracts features from trained DL models using get_features() method
for use with classical ML models (e.g., XGBoost).
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader

from ..utils.config_loader import load_config
from src.models import (
    CNNScratch,
    ResNet1D14,
    EfficientNet1D_B1,
    XResNet1D101,
    HybridCNNLSTM,
)
from src.models.lstm import LSTM1D_Unidirectional, LSTM1D_Bidirectional
from src.models.core.multi_task_model import MultiTaskECGModel


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file).
        config: Configuration dictionary (must contain model type).
        device: Device to load model on (default: "cpu").
    
    Returns:
        model: Loaded model in eval mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Get model type from config
    model_type = config.get("model", {}).get("type", "").lower()
    
    # Create model based on type
    if model_type == "cnnscratch":
        base_model = CNNScratch(config)
    elif model_type == "resnet1d14":
        base_model = ResNet1D14(config)
    elif model_type == "efficientnet1d_b1":
        base_model = EfficientNet1D_B1(config)
    elif model_type == "xresnet1d101":
        base_model = XResNet1D101(config)
    elif model_type == "lstm1d":
        base_model = LSTM1D_Unidirectional(config)
    elif model_type == "lstm1d_bidirectional":
        base_model = LSTM1D_Bidirectional(config)
    elif model_type == "hybridcnnlstm":
        base_model = HybridCNNLSTM(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Check if this is a MultiTaskECGModel by checking the state_dict keys
    # MultiTaskECGModel has keys like "base_model.conv1.weight", "los_head.0.weight", etc.
    is_multitask = any(key.startswith("base_model.") or key.startswith("los_head.") or key.startswith("mortality_head.") for key in state_dict.keys())
    
    if is_multitask:
        # Wrap base model in MultiTaskECGModel
        from src.models.core.multi_task_model import MultiTaskECGModel
        model = MultiTaskECGModel(base_model, config)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        # For feature extraction, we need the base_model
        return model.base_model
    else:
        # Load state dict directly into base model
        base_model.load_state_dict(state_dict)
        base_model.eval()
        base_model.to(device)
        return base_model


def extract_dl_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    use_demographics: bool = False,
    use_diagnoses: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract features from trained DL model for all samples in DataLoader.
    
    Args:
        model: Trained PyTorch model (must implement get_features() method).
        dataloader: DataLoader with ECG signals.
        device: Device to run model on (default: "cpu").
        use_demographics: Whether to include demographic features in output.
        use_diagnoses: Whether to include diagnosis features in output.
    
    Returns:
        Tuple of (X_features, y_los, y_mortality):
        - X_features: Feature matrix of shape (N, feature_dim)
        - y_los: LOS labels of shape (N,)
        - y_mortality: Mortality labels of shape (N,) or None if not available
    """
    model.eval()
    model.to(device)
    
    all_features = []
    all_labels = []
    all_mortality_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            signals = batch["signal"].to(device)  # (B, 12, 5000) or (B, 5000, 12)
            labels = batch["label"].cpu().numpy()  # (B,)
            mortality_labels = batch.get("mortality_label")
            if mortality_labels is not None:
                mortality_labels = mortality_labels.cpu().numpy()  # (B,)
            
            # Get demographic and diagnosis features if available
            demographic_features = None
            if use_demographics and "demographic_features" in batch:
                demographic_features = batch["demographic_features"].to(device)
            
            diagnosis_features = None
            if use_diagnoses and "diagnosis_features" in batch:
                diagnosis_features = batch["diagnosis_features"].to(device)
            
            # Extract features using get_features() method
            features = model.get_features(
                signals,
                demographic_features=demographic_features,
                diagnosis_features=diagnosis_features,
            )  # (B, feature_dim)
            
            # Convert to numpy
            features_np = features.cpu().numpy()
            
            all_features.append(features_np)
            all_labels.append(labels)
            if mortality_labels is not None:
                all_mortality_labels.append(mortality_labels)
    
    # Concatenate all batches
    X = np.vstack(all_features)  # (N, feature_dim)
    y_los = np.concatenate(all_labels)  # (N,)
    
    # Filter out invalid labels (label < 0)
    valid_mask = y_los >= 0
    X = X[valid_mask]
    y_los = y_los[valid_mask]
    
    if len(all_mortality_labels) > 0:
        y_mortality = np.concatenate(all_mortality_labels)
        y_mortality = y_mortality[valid_mask]
    else:
        y_mortality = None
    
    return X, y_los, y_mortality


def extract_dl_features_from_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load model from checkpoint and extract features.
    
    Convenience function that combines load_model_from_checkpoint()
    and extract_dl_features().
    
    Args:
        checkpoint_path: Path to model checkpoint.
        config: Configuration dictionary.
        dataloader: DataLoader with ECG signals.
        device: Device to run model on (default: "cpu").
    
    Returns:
        Tuple of (X_features, y_los, y_mortality).
    """
    # Get feature config
    feature_config = config.get("features", {})
    use_demographics = feature_config.get("use_demographics", False)
    use_diagnoses = feature_config.get("use_diagnoses", False)
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, config, device)
    
    # Extract features
    return extract_dl_features(
        model=model,
        dataloader=dataloader,
        device=device,
        use_demographics=use_demographics,
        use_diagnoses=use_diagnoses,
    )

