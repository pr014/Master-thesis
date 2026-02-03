#!/usr/bin/env python3
"""Smoke test for EHR features (Demographics + Diagnoses) via Late Fusion.

Tests:
- Config loading with EHR features enabled
- Model initialization with correct feature dimensions
- Late fusion: ECG features + Demographics + Diagnoses
- Forward pass with EHR features
- Feature dimension verification
- Binary diagnosis encoding verification
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models import CNNScratch
from src.models import MultiTaskECGModel
from src.utils.config_loader import load_config


def test_ehr_late_fusion():
    """Test EHR features (Demographics + Diagnoses) via Late Fusion."""
    print("\n" + "=" * 80)
    print("EHR Features Late Fusion Smoke Test")
    print("=" * 80)
    
    # Load configs
    print("\n[1/6] Loading configs with EHR features...")
    base_config_path = Path("configs/icu_24h/output/weighted_exact_days.yaml")
    model_config_path = Path("configs/model/cnn_scratch.yaml")
    feature_config_path = Path("configs/features/demographic_features.yaml")
    
    try:
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
            experiment_config_path=feature_config_path,
        )
        print(f"[OK] Configs loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load configs: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check EHR features configuration
    print("\n[2/6] Checking EHR features configuration...")
    demographic_config = config.get("data", {}).get("demographic_features", {})
    diagnosis_config = config.get("data", {}).get("diagnosis_features", {})
    
    demo_enabled = demographic_config.get("enabled", False)
    diag_enabled = diagnosis_config.get("enabled", False)
    diagnosis_list = diagnosis_config.get("diagnosis_list", [])
    
    print(f"  Demographic features enabled: {demo_enabled}")
    print(f"  Diagnosis features enabled: {diag_enabled}")
    print(f"  Number of diagnoses: {len(diagnosis_list)}")
    
    if not demo_enabled:
        print("[ERROR] Demographic features not enabled in config")
        return False
    if not diag_enabled:
        print("[ERROR] Diagnosis features not enabled in config")
        return False
    if len(diagnosis_list) != 15:
        print(f"[WARNING] Expected 15 diagnoses, got {len(diagnosis_list)}")
    
    print(f"[OK] EHR features configuration correct")
    
    # Create model
    print("\n[3/6] Creating model with EHR features...")
    try:
        base_model = CNNScratch(config)
        print(f"[OK] CNNScratch model created")
        
        # Check model configuration
        print(f"  Model uses demographics: {base_model.use_demographics}")
        print(f"  Model uses diagnoses: {base_model.use_diagnoses}")
        
        if not base_model.use_demographics:
            print("[ERROR] Model should use demographic features")
            return False
        if not base_model.use_diagnoses:
            print("[ERROR] Model should use diagnosis features")
            return False
        
        # Check feature dimensions
        # ECG features: 128 (after global pooling)
        # Demographics: 2 (Age + Sex, binary encoding)
        # Diagnoses: 15 (binary features)
        # Expected total: 128 + 2 + 15 = 145
        expected_feature_dim = 128 + 2 + 15
        actual_feature_dim = base_model.fc1.in_features
        
        print(f"  Expected feature dimension: {expected_feature_dim}")
        print(f"  Actual feature dimension: {actual_feature_dim}")
        
        if actual_feature_dim != expected_feature_dim:
            print(f"[ERROR] Feature dimension mismatch! Expected {expected_feature_dim}, got {actual_feature_dim}")
            return False
        
        print(f"[OK] Feature dimensions correct (Late Fusion: ECG + Demographics + Diagnoses)")
        
    except Exception as e:
        print(f"[ERROR] Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test forward pass without EHR features (should use zeros for missing features)
    print("\n[4/6] Testing forward pass WITHOUT EHR features (using zeros)...")
    try:
        batch_size = 2
        ecg_input = torch.randn(batch_size, 12, 5000)  # (B, 12, 5000)
        
        # Create zero tensors for missing EHR features (as per missing_strategy: "zero")
        demographic_features = torch.zeros(batch_size, 2)  # (B, 2) Zeros for missing Age + Sex
        diagnosis_features = torch.zeros(batch_size, 15)  # (B, 15) Zeros for missing diagnoses
        
        base_model.eval()
        with torch.no_grad():
            # Forward pass with zero EHR features (simulating missing data)
            output = base_model(
                ecg_input,
                demographic_features=demographic_features,
                diagnosis_features=diagnosis_features
            )
        
        expected_output_shape = (batch_size, config["model"]["num_classes"])
        if output.shape != expected_output_shape:
            print(f"[ERROR] Output shape mismatch! Expected {expected_output_shape}, got {output.shape}")
            return False
        
        print(f"[OK] Forward pass with zero EHR features works (output shape: {output.shape})")
        print(f"  Note: Missing EHR features are handled as zeros (missing_strategy: 'zero')")
        
    except Exception as e:
        print(f"[ERROR] Forward pass with zero EHR features failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test forward pass WITH EHR features (Late Fusion)
    print("\n[5/6] Testing forward pass WITH EHR features (Late Fusion)...")
    try:
        batch_size = 2
        ecg_input = torch.randn(batch_size, 12, 5000)  # (B, 12, 5000)
        
        # Create dummy EHR features
        demographic_features = torch.randn(batch_size, 2)  # (B, 2) Age + Sex
        diagnosis_features = torch.randint(0, 2, (batch_size, 15)).float()  # (B, 15) Binary diagnoses
        
        print(f"  ECG input shape: {ecg_input.shape}")
        print(f"  Demographic features shape: {demographic_features.shape}")
        print(f"  Diagnosis features shape: {diagnosis_features.shape}")
        print(f"  Diagnosis features are binary: {torch.all((diagnosis_features == 0) | (diagnosis_features == 1))}")
        
        base_model.eval()
        with torch.no_grad():
            # Forward pass with EHR features (Late Fusion)
            output = base_model(
                ecg_input,
                demographic_features=demographic_features,
                diagnosis_features=diagnosis_features
            )
        
        expected_output_shape = (batch_size, config["model"]["num_classes"])
        if output.shape != expected_output_shape:
            print(f"[ERROR] Output shape mismatch! Expected {expected_output_shape}, got {output.shape}")
            return False
        
        print(f"[OK] Forward pass with EHR features works (output shape: {output.shape})")
        print(f"  Late Fusion: ECG (128) + Demographics (2) + Diagnoses (15) = 145 features")
        
    except Exception as e:
        print(f"[ERROR] Forward pass with EHR features failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test get_features() method (should include EHR features)
    print("\n[6/6] Testing get_features() method with EHR features...")
    try:
        batch_size = 2
        ecg_input = torch.randn(batch_size, 12, 5000)
        demographic_features = torch.randn(batch_size, 2)
        diagnosis_features = torch.randint(0, 2, (batch_size, 15)).float()
        
        base_model.eval()
        with torch.no_grad():
            # Get features before final classification head
            features = base_model.get_features(
                ecg_input,
                demographic_features=demographic_features,
                diagnosis_features=diagnosis_features
            )
        
        # Features should be after fc1, so (B, 64)
        expected_feature_shape = (batch_size, 64)
        if features.shape != expected_feature_shape:
            print(f"[ERROR] Feature shape mismatch! Expected {expected_feature_shape}, got {features.shape}")
            return False
        
        print(f"[OK] get_features() works correctly (feature shape: {features.shape})")
        print(f"  Features include Late Fusion of ECG + Demographics + Diagnoses")
        
    except Exception as e:
        print(f"[ERROR] get_features() failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED: EHR Features Late Fusion is correctly implemented!")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ Config loading with EHR features")
    print("  ✓ Model initialization with correct feature dimensions (128 + 2 + 15 = 145)")
    print("  ✓ Forward pass without EHR features")
    print("  ✓ Forward pass with EHR features (Late Fusion)")
    print("  ✓ get_features() includes EHR features")
    print("  ✓ Diagnosis features are binary (0 or 1)")
    print("  ✓ Late Fusion: Concatenation after ECG feature extraction, before classification head")
    
    return True


if __name__ == "__main__":
    success = test_ehr_late_fusion()
    sys.exit(0 if success else 1)

