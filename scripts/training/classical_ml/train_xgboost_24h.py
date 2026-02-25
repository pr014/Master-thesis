"""Training script for XGBoost LOS regression with hand-crafted or DL features."""

from pathlib import Path
from typing import Optional
import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from src.models.classical_ml import XGBoostECGModel
from src.features import extract_handcrafted_features, extract_dl_features_from_checkpoint
from src.data.ecg import create_dataloaders
from src.training.training_utils import setup_icustays_mapper
from src.utils.config_loader import load_config


def extract_features_from_dataloader(
    dataloader,
    feature_type: str,
    config: dict,
    device: str = "cpu",
) -> tuple:
    """
    Extract features from DataLoader.
    
    Args:
        dataloader: PyTorch DataLoader with ECG signals.
        feature_type: "handcrafted" or "dl_features".
        config: Configuration dictionary.
        device: Device for DL feature extraction (if needed).
    
    Returns:
        Tuple of (X_features, y_los, demographic_features, diagnosis_features):
        - X_features: Feature matrix (N, feature_dim)
        - y_los: LOS labels (N,)
        - demographic_features: Optional demographic features (N, demo_dim) or None
        - diagnosis_features: Optional diagnosis features (N, diag_dim) or None
    """
    feature_config = config.get("features", {})
    use_demographics = feature_config.get("use_demographics", False)
    use_diagnoses = feature_config.get("use_diagnoses", False)
    
    all_features = []
    all_labels = []
    all_demographic_features = []
    all_diagnosis_features = []
    
    if feature_type == "handcrafted":
        # Extract hand-crafted features
        fs = config.get("data", {}).get("sampling_rate", 500.0)
        
        for batch in dataloader:
            signals = batch["signal"].numpy()  # (B, 12, 5000) or (B, 5000, 12)
            labels = batch["label"].numpy()  # (B,)
            
            # Filter valid labels
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue
            
            signals = signals[valid_mask]
            labels = labels[valid_mask]
            
            # Extract features for each sample
            batch_features = []
            for i in range(len(signals)):
                signal = signals[i]  # (12, 5000) or (5000, 12)
                features = extract_handcrafted_features(signal, fs=fs)
                batch_features.append(features)
            
            all_features.append(np.vstack(batch_features))
            all_labels.append(labels)
            
            # Collect demographic and diagnosis features if available
            if use_demographics and "demographic_features" in batch:
                demo_features = batch["demographic_features"]
                if demo_features is not None:
                    demo_features = demo_features.numpy()[valid_mask]
                    all_demographic_features.append(demo_features)
            
            if use_diagnoses and "diagnosis_features" in batch:
                diag_features = batch["diagnosis_features"]
                if diag_features is not None:
                    diag_features = diag_features.numpy()[valid_mask]
                    all_diagnosis_features.append(diag_features)
    
    elif feature_type == "dl_features":
        # Extract DL features using trained model
        checkpoint_path = feature_config.get("dl_model_checkpoint")
        if not checkpoint_path:
            raise ValueError("dl_model_checkpoint must be specified in config for dl_features")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"DL model checkpoint not found: {checkpoint_path}")
        
        # Extract features
        X, y_los, _ = extract_dl_features_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            config=config,
            dataloader=dataloader,
            device=device,
        )
        
        # Collect demographic and diagnosis features if available
        if use_demographics or use_diagnoses:
            demo_features_list = []
            diag_features_list = []
            
            for batch in dataloader:
                labels = batch["label"].numpy()
                valid_mask = labels >= 0
                
                if use_demographics and "demographic_features" in batch:
                    demo_features = batch["demographic_features"]
                    if demo_features is not None:
                        demo_features = demo_features.numpy()[valid_mask]
                        demo_features_list.append(demo_features)
                
                if use_diagnoses and "diagnosis_features" in batch:
                    diag_features = batch["diagnosis_features"]
                    if diag_features is not None:
                        diag_features = diag_features.numpy()[valid_mask]
                        diag_features_list.append(diag_features)
            
            if demo_features_list:
                all_demographic_features = np.vstack(demo_features_list)
            if diag_features_list:
                all_diagnosis_features = np.vstack(diag_features_list)
        
        return X, y_los, all_demographic_features if all_demographic_features else None, all_diagnosis_features if all_diagnosis_features else None
    
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}. Must be 'handcrafted' or 'dl_features'")
    
    # Concatenate all batches
    X = np.vstack(all_features) if all_features else np.array([])
    y_los = np.concatenate(all_labels) if all_labels else np.array([])
    
    # Combine demographic and diagnosis features if available
    demographic_features = None
    if all_demographic_features:
        demographic_features = np.vstack(all_demographic_features)
    
    diagnosis_features = None
    if all_diagnosis_features:
        diagnosis_features = np.vstack(all_diagnosis_features)
    
    return X, y_los, demographic_features, diagnosis_features


def combine_features(
    X_ecg: np.ndarray,
    demographic_features: Optional[np.ndarray] = None,
    diagnosis_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine ECG features with demographic and diagnosis features.
    
    Args:
        X_ecg: ECG features (N, ecg_feature_dim).
        demographic_features: Optional demographic features (N, demo_dim).
        diagnosis_features: Optional diagnosis features (N, diag_dim).
    
    Returns:
        Combined features (N, total_feature_dim).
    """
    features_list = [X_ecg]
    
    if demographic_features is not None:
        features_list.append(demographic_features)
    
    if diagnosis_features is not None:
        features_list.append(diagnosis_features)
    
    return np.hstack(features_list)


def main():
    """Main training function."""
    # Load config
    # Find project root (directory containing configs/)
    project_root = Path(__file__).parent.parent.parent.parent
    default_config = project_root / "configs" / "classical_ml" / "xgboost_handcrafted.yaml"
    
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        # If relative path, resolve relative to project root
        if not config_path.is_absolute():
            # If path doesn't start with "configs/", add it
            if not str(config_path).startswith("configs/"):
                config_path = project_root / "configs" / config_path
            else:
                config_path = project_root / config_path
    else:
        config_path = default_config
    
    config = load_config(model_config_path=config_path)
    
    # Log config paths
    print("="*60)
    print("XGBoost Training Configuration")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Feature type: {config.get('features', {}).get('feature_type', 'unknown')}")
    print(f"Use demographics: {config.get('features', {}).get('use_demographics', False)}")
    print(f"Use diagnoses: {config.get('features', {}).get('use_diagnoses', False)}")
    print("="*60)
    
    # Setup ICU mapper
    icu_mapper = setup_icustays_mapper(config)
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        labels=None,  # Will be auto-generated
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    
    # Get feature type
    feature_type = config.get("features", {}).get("feature_type", "handcrafted")
    device = config.get("device", {}).get("device", "cpu")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nExtracting {feature_type} features...")
    
    # Extract features from train set
    print("Extracting train features...")
    X_train, y_train, demo_train, diag_train = extract_features_from_dataloader(
        train_loader,
        feature_type=feature_type,
        config=config,
        device=device,
    )
    
    # Extract features from validation set
    print("Extracting validation features...")
    X_val, y_val, demo_val, diag_val = extract_features_from_dataloader(
        val_loader,
        feature_type=feature_type,
        config=config,
        device=device,
    )
    
    # Combine with demographic and diagnosis features if available
    X_train_combined = combine_features(X_train, demo_train, diag_train)
    X_val_combined = combine_features(X_val, demo_val, diag_val)
    
    print(f"Train features shape: {X_train_combined.shape}")
    print(f"Val features shape: {X_val_combined.shape}")
    print(f"Train samples: {len(y_train)}")
    print(f"Val samples: {len(y_val)}")
    
    # Create XGBoost model
    print("\nInitializing XGBoost model...")
    model = XGBoostECGModel(config, random_state=config.get("seed", 42))
    
    # Train model
    print("\nTraining XGBoost model...")
    model.fit(
        X_train=X_train_combined,
        y_train=y_train,
        X_val=X_val_combined,
        y_val=y_val,
        verbose=True,
    )
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = model.evaluate(X_val_combined, y_val)
    
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)
    print(f"MAE:  {val_metrics['mae']:.4f} days")
    print(f"MSE:  {val_metrics['mse']:.4f} days²")
    print(f"RMSE: {val_metrics['rmse']:.4f} days")
    print(f"R²:   {val_metrics['r2']:.4f}")
    print(f"Median AE: {val_metrics['median_ae']:.4f} days")
    print(f"P25 Error: {val_metrics['p25_error']:.4f} days")
    print(f"P75 Error: {val_metrics['p75_error']:.4f} days")
    print(f"P90 Error: {val_metrics['p90_error']:.4f} days")
    
    # Evaluate on test set if available
    if test_loader is not None:
        print("\nExtracting test features...")
        X_test, y_test, demo_test, diag_test = extract_features_from_dataloader(
            test_loader,
            feature_type=feature_type,
            config=config,
            device=device,
        )
        X_test_combined = combine_features(X_test, demo_test, diag_test)
        
        print(f"Test features shape: {X_test_combined.shape}")
        print(f"Test samples: {len(y_test)}")
        
        print("\nEvaluating on test set...")
        test_metrics = model.evaluate(X_test_combined, y_test)
        
        print("\n" + "="*60)
        print("Test Results")
        print("="*60)
        print(f"MAE:  {test_metrics['mae']:.4f} days")
        print(f"MSE:  {test_metrics['mse']:.4f} days²")
        print(f"RMSE: {test_metrics['rmse']:.4f} days")
        print(f"R²:   {test_metrics['r2']:.4f}")
        print(f"Median AE: {test_metrics['median_ae']:.4f} days")
        print(f"P25 Error: {test_metrics['p25_error']:.4f} days")
        print(f"P75 Error: {test_metrics['p75_error']:.4f} days")
        print(f"P90 Error: {test_metrics['p90_error']:.4f} days")
    else:
        test_metrics = None
        print("\nWarning: No test loader available. Skipping test evaluation.")
    
    # Save model
    job_id = os.getenv("SLURM_JOB_ID", "local")
    checkpoint_dir = Path(config.get("checkpoint", {}).get("save_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = checkpoint_dir / f"xgboost_los_best_{job_id}.pkl"
    print(f"\nSaving model to: {model_path}")
    model.save_model(str(model_path))
    
    # Print feature importance (top 20)
    feature_importance = model.get_feature_importance()
    top_indices = np.argsort(feature_importance)[::-1][:20]
    print("\n" + "="*60)
    print("Top 20 Feature Importances")
    print("="*60)
    for idx in top_indices:
        print(f"Feature {idx}: {feature_importance[idx]:.6f}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Model saved to: {model_path}")
    if job_id:
        print(f"Job ID: {job_id}")
    print("="*60)
    
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
        "job_id": job_id,
    }


if __name__ == "__main__":
    main()

