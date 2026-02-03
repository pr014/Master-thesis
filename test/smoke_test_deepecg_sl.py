#!/usr/bin/env python3
"""Smoke test for DeepECG-SL (WCR) model.

Tests:
- Model creation (with automatic weight download if API key is set)
- Forward pass (LOS + Mortality outputs)
- get_features() method
- freeze_backbone() and unfreeze_backbone() methods
- Parameter count verification
- Input adapter (5000 → 2500)
- Multi-Task compatibility
"""

from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

# Check if fairseq-signals is available
try:
    from fairseq_signals.utils import checkpoint_utils
    FAIRSEQ_AVAILABLE = True
except ImportError:
    FAIRSEQ_AVAILABLE = False
    print("WARNING: fairseq-signals not available. Some tests will be skipped.")
    print("Install with: pip install git+https://github.com/HeartWise-AI/fairseq-signals.git")

# Check if HuggingFace API key is set
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    print("WARNING: HUGGINGFACE_API_KEY not set. Model download tests will be skipped.")
    print("Set it with: export HUGGINGFACE_API_KEY='your_key'")

from src.models import DeepECG_SL
from src.models import MultiTaskECGModel
from src.utils.config_loader import load_config


def test_model_creation():
    """Test DeepECG-SL model creation."""
    print("\n" + "=" * 80)
    print("Test 1: Model Creation")
    print("=" * 80)
    
    if not FAIRSEQ_AVAILABLE:
        print("[SKIP] fairseq-signals not available. Skipping model creation test.")
        return False
    
    if not HUGGINGFACE_API_KEY:
        print("[SKIP] HUGGINGFACE_API_KEY not set. Skipping model creation test.")
        return False
    
    try:
        # Load config
        base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
        model_config_path = Path("configs/model/deepecg_sl/deepecg_sl.yaml")
        
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
        )
        
        # Create model (will download weights automatically if not cached)
        print("Creating DeepECG-SL model...")
        print("This may take a while on first run (downloading pretrained weights)...")
        model = DeepECG_SL(config)
        print(f"[OK] DeepECG_SL model created")
        
        # Count parameters
        num_params = model.count_parameters()
        print(f"[OK] Total parameters: {num_params:,}")
        
        # Expected: Large model with WCR encoder
        if num_params < 1_000_000:
            print(f"[WARNING] Parameter count seems low (expected >1M)")
        else:
            print(f"[OK] Parameter count in expected range (>1M)")
        
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
    
    if not FAIRSEQ_AVAILABLE or not HUGGINGFACE_API_KEY:
        print("[SKIP] Prerequisites not met. Skipping forward pass test.")
        return False
    
    try:
        # Load config
        base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
        model_config_path = Path("configs/model/deepecg_sl/deepecg_sl.yaml")
        
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
        )
        
        # Create model
        model = DeepECG_SL(config)
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
    
    if not FAIRSEQ_AVAILABLE or not HUGGINGFACE_API_KEY:
        print("[SKIP] Prerequisites not met. Skipping get_features() test.")
        return False
    
    try:
        # Load config
        base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
        model_config_path = Path("configs/model/deepecg_sl/deepecg_sl.yaml")
        
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
        )
        
        # Create model
        model = DeepECG_SL(config)
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
    
    if not FAIRSEQ_AVAILABLE or not HUGGINGFACE_API_KEY:
        print("[SKIP] Prerequisites not met. Skipping freeze/unfreeze test.")
        return False
    
    try:
        # Load config
        base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
        model_config_path = Path("configs/model/deepecg_sl/deepecg_sl.yaml")
        
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
        )
        
        # Create model
        model = DeepECG_SL(config)
        
        # Check initial state (should be frozen by default)
        wcr_params = list(model.wcr_encoder.parameters())
        adapter_params = list(model.input_adapter.parameters())
        
        initial_wcr_grad = [p.requires_grad for p in wcr_params]
        initial_adapter_grad = [p.requires_grad for p in adapter_params]
        
        print(f"[OK] Initial WCR encoder requires_grad: {any(initial_wcr_grad)}")
        print(f"[OK] Initial Input adapter requires_grad: {any(initial_adapter_grad)}")
        
        # Freeze backbone
        model.freeze_backbone()
        frozen_wcr_grad = [p.requires_grad for p in wcr_params]
        frozen_adapter_grad = [p.requires_grad for p in adapter_params]
        assert not any(frozen_wcr_grad), "WCR encoder should be frozen"
        assert not any(frozen_adapter_grad), "Input adapter should be frozen"
        print(f"[OK] Backbone frozen (requires_grad=False)")
        
        # Unfreeze backbone
        model.unfreeze_backbone()
        unfrozen_wcr_grad = [p.requires_grad for p in wcr_params]
        unfrozen_adapter_grad = [p.requires_grad for p in adapter_params]
        assert all(unfrozen_wcr_grad), "WCR encoder should be unfrozen"
        assert all(unfrozen_adapter_grad), "Input adapter should be unfrozen"
        print(f"[OK] Backbone unfrozen (requires_grad=True)")
        
        return True
    except Exception as e:
        print(f"[ERROR] freeze/unfreeze test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_input_adapter():
    """Test input adapter (5000 → 2500)."""
    print("\n" + "=" * 80)
    print("Test 5: Input Adapter (5000 → 2500)")
    print("=" * 80)
    
    try:
        # Load config
        base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
        model_config_path = Path("configs/model/deepecg_sl/deepecg_sl.yaml")
        
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
        )
        
        from src.models.deepecg_sl.input_adapter import InputAdapter
        
        # Create input adapter
        adapter = InputAdapter(config)
        
        # Test input
        batch_size = 2
        x = torch.randn(batch_size, 12, 5000)
        
        # Forward pass
        with torch.no_grad():
            y = adapter(x)
        
        print(f"[OK] Input shape: {x.shape}")
        print(f"[OK] Output shape: {y.shape} (expected: ({batch_size}, 12, 2500))")
        
        # Verify shape
        assert y.shape == (batch_size, 12, 2500), f"Expected output shape ({batch_size}, 12, 2500), got {y.shape}"
        
        print(f"[OK] Input adapter works correctly")
        
        return True
    except Exception as e:
        print(f"[ERROR] Input adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multitask_compatibility():
    """Test MultiTaskECGModel compatibility."""
    print("\n" + "=" * 80)
    print("Test 6: MultiTaskECGModel Compatibility")
    print("=" * 80)
    
    if not FAIRSEQ_AVAILABLE or not HUGGINGFACE_API_KEY:
        print("[SKIP] Prerequisites not met. Skipping MultiTask compatibility test.")
        return False
    
    try:
        # Load config
        base_config_path = Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml")
        model_config_path = Path("configs/model/deepecg_sl/deepecg_sl.yaml")
        
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
        )
        
        # Create base model
        base_model = DeepECG_SL(config)
        
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
    print("Smoke Test: DeepECG-SL (WCR) Model")
    print("=" * 80)
    
    if not FAIRSEQ_AVAILABLE:
        print("\nWARNING: fairseq-signals is not installed.")
        print("Some tests will be skipped. Install with:")
        print("  pip install git+https://github.com/HeartWise-AI/fairseq-signals.git")
        print("=" * 80)
    
    if not HUGGINGFACE_API_KEY:
        print("\nWARNING: HUGGINGFACE_API_KEY is not set.")
        print("Model download tests will be skipped. Set it with:")
        print("  export HUGGINGFACE_API_KEY='your_key'")
        print("=" * 80)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("get_features()", test_get_features),
        ("freeze/unfreeze", test_freeze_unfreeze),
        ("Input Adapter", test_input_adapter),
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
        print("DeepECG-SL model is ready to use.")
    else:
        print("[FAILURE] Some smoke tests failed. Please fix the issues above.")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

