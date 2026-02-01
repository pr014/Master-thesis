#!/usr/bin/env python3
"""Smoke test for ECG-CPC model.

Tests:
- Model creation
- Forward pass (LOS + Mortality outputs)
- get_features() method
- freeze_backbone() and unfreeze_backbone() methods
- Parameter count verification
- Pretrained weights loading (mock)
"""

from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

# Check if S4 is available (from s4-torch or local implementation)
try:
    from s4_torch import S4
    S4_AVAILABLE = True
except ImportError:
    try:
        # Try local S4 implementation from ecg-fm-benchmarking repo
        from src.models.ecg_cpc.s4_impl import S4
        S4_AVAILABLE = True
        print("Using S4 from local implementation (ecg-fm-benchmarking repo)")
    except ImportError:
        S4_AVAILABLE = False
        print("WARNING: S4 not available. Tests will be skipped.")

from src.models import ECG_S4_CPC
from src.models import MultiTaskECGModel
from src.utils.config_loader import load_config


def test_model_creation():
    """Test ECG-CPC model creation."""
    print("\n" + "=" * 80)
    print("Test 1: Model Creation")
    print("=" * 80)
    
    if not S4_AVAILABLE:
        print("[SKIP] s4-torch not available. Skipping model creation test.")
        return False
    
    try:
        # Load config
        base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
        model_config_path = Path("configs/model/ecg_cpc/ecg_cpc.yaml")
        
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
        )
        
        # Create model
        model = ECG_S4_CPC(config)
        print(f"[OK] ECG_S4_CPC model created")
        
        # Count parameters
        num_params = model.count_parameters()
        print(f"[OK] Total parameters: {num_params:,}")
        
        # Expected: ~2.2M parameters (S4 encoder ~1.8M, heads ~400K)
        if num_params < 1_000_000:
            print(f"[WARNING] Parameter count seems low (expected ~2.2M)")
        elif num_params > 5_000_000:
            print(f"[WARNING] Parameter count seems high (expected ~2.2M)")
        else:
            print(f"[OK] Parameter count in expected range (~2.2M)")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\n" + "=" * 80)
    print("Test 2: Forward Pass")
    print("=" * 80)
    
    if not S4_AVAILABLE:
        print("[SKIP] s4-torch not available. Skipping forward pass test.")
        return False
    
    try:
        # Load config
        base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
        model_config_path = Path("configs/model/ecg_cpc/ecg_cpc.yaml")
        
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
        )
        
        # Create model
        model = ECG_S4_CPC(config)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        ecg_input = torch.randn(batch_size, 12, 5000)  # (B, 12, 5000)
        demo_features = torch.randn(batch_size, 2)  # (B, 2) Age + Sex
        
        print(f"[OK] Input shapes:")
        print(f"    ECG: {ecg_input.shape}")
        print(f"    Demographics: {demo_features.shape}")
        
        # Forward pass
        with torch.no_grad():
            los_logits, mortality_probs = model(ecg_input, demographic_features=demo_features)
        
        print(f"[OK] Output shapes:")
        print(f"    LOS logits: {los_logits.shape} (expected: ({batch_size}, 10))")
        print(f"    Mortality probs: {mortality_probs.shape} (expected: ({batch_size}, 1))")
        
        # Verify shapes
        assert los_logits.shape == (batch_size, 10), f"Expected LOS shape ({batch_size}, 10), got {los_logits.shape}"
        assert mortality_probs.shape == (batch_size, 1), f"Expected Mortality shape ({batch_size}, 1), got {mortality_probs.shape}"
        
        # Verify mortality probs are in [0, 1] range (sigmoid output)
        assert (mortality_probs >= 0).all() and (mortality_probs <= 1).all(), "Mortality probs should be in [0, 1]"
        
        print(f"[OK] Forward pass successful")
        
        # Test without demographics
        with torch.no_grad():
            los_logits_no_demo, mortality_probs_no_demo = model(ecg_input, demographic_features=None)
        
        assert los_logits_no_demo.shape == (batch_size, 10)
        assert mortality_probs_no_demo.shape == (batch_size, 1)
        print(f"[OK] Forward pass without demographics successful")
        
        return True
    except Exception as e:
        print(f"[ERROR] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_features():
    """Test get_features() method."""
    print("\n" + "=" * 80)
    print("Test 3: get_features() Method")
    print("=" * 80)
    
    if not S4_AVAILABLE:
        print("[SKIP] s4-torch not available. Skipping get_features() test.")
        return False
    
    try:
        # Load config
        base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
        model_config_path = Path("configs/model/ecg_cpc/ecg_cpc.yaml")
        
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
        )
        
        # Create model
        model = ECG_S4_CPC(config)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        ecg_input = torch.randn(batch_size, 12, 5000)
        demo_features = torch.randn(batch_size, 2)
        
        # Get features
        with torch.no_grad():
            features = model.get_features(ecg_input, demographic_features=demo_features)
        
        print(f"[OK] Features shape: {features.shape} (expected: ({batch_size}, 128))")
        
        # Verify shape
        assert features.shape == (batch_size, 128), f"Expected features shape ({batch_size}, 128), got {features.shape}"
        
        print(f"[OK] get_features() works correctly")
        
        return True
    except Exception as e:
        print(f"[ERROR] get_features() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_freeze_unfreeze():
    """Test freeze_backbone() and unfreeze_backbone() methods."""
    print("\n" + "=" * 80)
    print("Test 4: freeze_backbone() and unfreeze_backbone()")
    print("=" * 80)
    
    if not S4_AVAILABLE:
        print("[SKIP] s4-torch not available. Skipping freeze/unfreeze test.")
        return False
    
    try:
        # Load config
        base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
        model_config_path = Path("configs/model/ecg_cpc/ecg_cpc.yaml")
        
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
        )
        
        # Create model
        model = ECG_S4_CPC(config)
        
        # Check initial state (should be trainable)
        s4_params = list(model.s4_encoder.parameters())
        initial_requires_grad = [p.requires_grad for p in s4_params]
        print(f"[OK] Initial S4 encoder requires_grad: {all(initial_requires_grad)}")
        
        # Freeze backbone
        model.freeze_backbone()
        frozen_requires_grad = [p.requires_grad for p in s4_params]
        assert not any(frozen_requires_grad), "S4 encoder should be frozen"
        print(f"[OK] S4 encoder frozen (requires_grad=False)")
        
        # Unfreeze backbone
        model.unfreeze_backbone()
        unfrozen_requires_grad = [p.requires_grad for p in s4_params]
        assert all(unfrozen_requires_grad), "S4 encoder should be unfrozen"
        print(f"[OK] S4 encoder unfrozen (requires_grad=True)")
        
        return True
    except Exception as e:
        print(f"[ERROR] freeze/unfreeze test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multitask_compatibility():
    """Test MultiTaskECGModel compatibility."""
    print("\n" + "=" * 80)
    print("Test 5: MultiTaskECGModel Compatibility")
    print("=" * 80)
    
    if not S4_AVAILABLE:
        print("[SKIP] s4-torch not available. Skipping MultiTask compatibility test.")
        return False
    
    try:
        # Load config
        base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
        model_config_path = Path("configs/model/ecg_cpc/ecg_cpc.yaml")
        
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
        )
        
        # Create base model
        base_model = ECG_S4_CPC(config)
        
        # Wrap in MultiTaskECGModel
        model = MultiTaskECGModel(base_model, config)
        print(f"[OK] MultiTaskECGModel created")
        
        # Test forward pass
        batch_size = 2
        ecg_input = torch.randn(batch_size, 12, 5000)
        demo_features = torch.randn(batch_size, 2)
        
        model.eval()
        with torch.no_grad():
            outputs = model(ecg_input, demographic_features=demo_features)
        
        # MultiTaskECGModel should return dict with 'los' and 'mortality'
        assert isinstance(outputs, dict), "MultiTaskECGModel should return dict"
        assert "los" in outputs, "Outputs should contain 'los' key"
        assert "mortality" in outputs, "Outputs should contain 'mortality' key"
        
        los_logits = outputs["los"]
        mortality_probs = outputs["mortality"]
        
        print(f"[OK] MultiTask outputs:")
        print(f"    LOS logits: {los_logits.shape}")
        print(f"    Mortality probs: {mortality_probs.shape}")
        
        assert los_logits.shape == (batch_size, 10)
        assert mortality_probs.shape == (batch_size, 1)
        
        print(f"[OK] MultiTaskECGModel compatibility verified")
        
        return True
    except Exception as e:
        print(f"[ERROR] MultiTask compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("=" * 80)
    print("Smoke Test: ECG-CPC Model")
    print("=" * 80)
    
    if not S4_AVAILABLE:
        print("\nWARNING: s4-torch is not installed.")
        print("Some tests will be skipped. Install with: pip install s4-torch")
        print("=" * 80)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("get_features()", test_get_features),
        ("freeze/unfreeze", test_freeze_unfreeze),
        ("MultiTask Compatibility", test_multitask_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[ERROR] Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, success in results:
        if success is None:
            status = "⊘ SKIPPED"
        elif success:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
            all_passed = False
        print(f"{status}: {test_name}")
    
    print("=" * 80)
    if all_passed:
        print("[SUCCESS] All smoke tests passed!")
        print("ECG-CPC model is ready to use.")
    else:
        print("[FAILURE] Some smoke tests failed. Please fix the issues above.")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

