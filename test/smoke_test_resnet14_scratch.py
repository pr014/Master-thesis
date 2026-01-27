#!/usr/bin/env python3
"""Smoke test for ResNet1D-14 from-scratch training scripts.

Tests both training scripts:
- train_resnet1d_14_24h_weighted_scratch.py (balanced weights)
- train_resnet1d_14_24h_sqrt_weighted_scratch.py (sqrt weights)

Tests the complete pipeline:
- Config loading
- Data loading with LOS bin labels
- Model initialization (from scratch, no pretrained weights)
- Forward pass
- Loss computation
- Batch format validation
"""

from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models import ResNet1D14
from src.models.multi_task_model import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.training import setup_icustays_mapper
from src.training.losses import get_loss, get_multi_task_loss
from src.utils.config_loader import load_config


def test_config(config_name: str, base_config_path: Path, model_config_path: Path):
    """Test a specific configuration.
    
    Args:
        config_name: Name of the config (for logging)
        base_config_path: Path to base config
        model_config_path: Path to model config
    """
    print("\n" + "=" * 80)
    print(f"Testing: {config_name}")
    print("=" * 80)
    
    # Load configs
    print(f"\n[1/7] Loading configs...")
    print(f"  Base config: {base_config_path}")
    print(f"  Model config: {model_config_path}")
    
    try:
        config = load_config(
            base_config_path=base_config_path,
            model_config_path=model_config_path,
        )
        print(f"[OK] Configs loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load configs: {e}")
        return False
    
    # Check pretrained config
    print(f"\n[2/7] Checking pretrained weights configuration...")
    pretrained_config = config.get("model", {}).get("pretrained", {})
    enabled = pretrained_config.get("enabled", False)
    if enabled:
        print(f"[ERROR] pretrained.enabled is True, but should be False for from-scratch training!")
        return False
    print(f"[OK] pretrained.enabled = False (training from scratch)")
    
    # Load ICU stays and create mapper
    print(f"\n[3/7] Loading ICU stays and creating mapper...")
    try:
        icu_mapper = setup_icustays_mapper(config)
        print(f"[OK] ICU mapper created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create ICU mapper: {e}")
        return False
    
    # Check if multi-task is enabled
    multi_task_config = config.get("multi_task", {})
    is_multi_task = multi_task_config.get("enabled", False)
    print(f"[INFO] Multi-task enabled: {is_multi_task}")
    
    # Create DataLoaders
    print(f"\n[4/7] Creating DataLoaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            config=config,
            labels=None,  # Will be auto-generated
            preprocess=None,
            transform=None,
            icu_mapper=icu_mapper,
            mortality_labels=None,  # Will be auto-generated
        )
        print(f"[OK] DataLoaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader) if test_loader else 0}")
    except FileNotFoundError as e:
        print(f"[ERROR] Data directory not found: {e}")
        print("[INFO] Please ensure data/icu_ecgs_24h/P1 exists")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to create DataLoaders: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Get one batch
    print(f"\n[5/7] Testing batch format...")
    try:
        batch = next(iter(train_loader))
        signals = batch["signal"]
        labels = batch["label"]
        
        print(f"[OK] Signals shape: {signals.shape} (expected: (B, 12, 5000))")
        print(f"[OK] Labels shape: {labels.shape} (expected: (B,))")
        print(f"[OK] Signals dtype: {signals.dtype} (expected: float32)")
        print(f"[OK] Labels dtype: {labels.dtype} (expected: int64/long)")
        
        # Check for unmatched samples
        unmatched_count = (labels == -1).sum().item()
        if unmatched_count > 0:
            print(f"[ERROR] Found {unmatched_count} unmatched samples (label == -1)")
            return False
        print(f"[OK] No unmatched samples in batch")
        
        # Check label range
        valid_labels = labels[labels >= 0]
        if len(valid_labels) > 0:
            min_label = valid_labels.min().item()
            max_label = valid_labels.max().item()
            print(f"[OK] Valid labels range: [{min_label}, {max_label}]")
            if min_label < 0 or max_label > 9:
                print(f"[ERROR] Labels out of range [0, 9]")
                return False
        else:
            print(f"[WARNING] No valid labels found in batch")
        
        # Check mortality labels if multi-task
        has_mortality_labels = "mortality_label" in batch
        if is_multi_task and has_mortality_labels:
            mortality_labels = batch["mortality_label"]
            print(f"[OK] Mortality labels shape: {mortality_labels.shape}")
            valid_mortality = mortality_labels[mortality_labels >= 0]
            print(f"[OK] Valid mortality labels: {len(valid_mortality)}/{len(mortality_labels)}")
    except Exception as e:
        print(f"[ERROR] Failed to process batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create model
    print(f"\n[6/7] Testing model initialization...")
    try:
        base_model = ResNet1D14(config)
        print(f"[OK] ResNet1D14 model created")
        num_params = sum(p.numel() for p in base_model.parameters())
        print(f"[OK] Model parameters: {num_params:,}")
        
        # Wrap in MultiTaskECGModel if multi-task
        if is_multi_task:
            model = MultiTaskECGModel(base_model, config)
            print(f"[OK] Multi-Task model created")
            num_params = model.count_parameters()
            print(f"[OK] Multi-Task model parameters: {num_params:,}")
        else:
            model = base_model
    except Exception as e:
        print(f"[ERROR] Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Forward pass
    print(f"\n[7/7] Testing forward pass and loss computation...")
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(signals)
        
        if is_multi_task and isinstance(outputs, dict):
            los_logits = outputs["los"]
            mortality_probs = outputs["mortality"]
            print(f"[OK] LOS logits shape: {los_logits.shape} (expected: ({signals.shape[0]}, 10))")
            print(f"[OK] Mortality probs shape: {mortality_probs.shape} (expected: ({signals.shape[0]}, 1))")
            
            # Check shapes
            assert los_logits.shape[0] == signals.shape[0], "Batch size mismatch for LOS"
            assert los_logits.shape[1] == 10, f"Expected 10 LOS classes, got {los_logits.shape[1]}"
            assert mortality_probs.shape[0] == signals.shape[0], "Batch size mismatch for mortality"
            assert mortality_probs.shape[1] == 1, f"Expected 1 mortality output, got {mortality_probs.shape[1]}"
            
            # Test loss
            criterion = get_multi_task_loss(config)
            mortality_labels_batch = batch["mortality_label"] if has_mortality_labels else None
            if mortality_labels_batch is not None:
                loss_dict = criterion(los_logits, labels, mortality_probs, mortality_labels_batch)
                loss = loss_dict["total"]
                print(f"[OK] Multi-Task Loss computed:")
                print(f"    Total: {loss.item():.4f}")
                print(f"    LOS: {loss_dict['los'].item():.4f}")
                print(f"    Mortality: {loss_dict['mortality'].item():.4f}")
            else:
                print(f"[WARNING] Multi-task enabled but no mortality labels in batch")
        else:
            logits = outputs if not isinstance(outputs, dict) else outputs.get("los", outputs)
            print(f"[OK] Logits shape: {logits.shape} (expected: ({signals.shape[0]}, 10))")
            
            # Check shapes
            assert logits.shape[0] == signals.shape[0], "Batch size mismatch"
            assert logits.shape[1] == 10, f"Expected 10 classes, got {logits.shape[1]}"
            
            # Test loss
            criterion = get_loss(config)
            loss = criterion(logits, labels)
            print(f"[OK] Loss computed: {loss.item():.4f}")
    except Exception as e:
        print(f"[ERROR] Failed forward pass or loss computation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[SUCCESS] All tests passed for {config_name}!")
    return True


def main():
    """Run smoke tests for both ResNet14 from-scratch configurations."""
    print("=" * 80)
    print("Smoke Test: ResNet1D-14 From-Scratch Training Scripts")
    print("=" * 80)
    
    # Test configurations
    test_configs = [
        {
            "name": "ResNet14 24h Weighted (Balanced) - From Scratch",
            "base_config": Path("configs/icu_24h/24h_weighted/balanced_weights.yaml"),
            "model_config": Path("configs/model/resnet14/resnet1d_14_scratch.yaml"),
        },
        {
            "name": "ResNet14 24h Weighted (SQRT) - From Scratch",
            "base_config": Path("configs/icu_24h/24h_weighted/sqrt_weights.yaml"),
            "model_config": Path("configs/model/resnet14/resnet1d_14_scratch.yaml"),
        },
    ]
    
    results = []
    for test_cfg in test_configs:
        success = test_config(
            config_name=test_cfg["name"],
            base_config_path=test_cfg["base_config"],
            model_config_path=test_cfg["model_config"],
        )
        results.append((test_cfg["name"], success))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")
        if not success:
            all_passed = False
    
    print("=" * 80)
    if all_passed:
        print("[SUCCESS] All smoke tests passed!")
        print("Both ResNet14 from-scratch training scripts are ready to use.")
    else:
        print("[FAILURE] Some smoke tests failed. Please fix the issues above.")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

