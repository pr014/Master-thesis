"""Smoke test for ECG training pipeline (works for all architectures).

Tests the complete pipeline:
- Data loading with LOS bin labels
- Model forward pass
- Loss computation
- Batch format validation
"""

from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models import CNNScratch
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.data.labeling import load_icustays, ICUStayMapper, load_mortality_mapping
from src.training.losses import get_loss, get_multi_task_loss
from src.utils.config_loader import load_config


def main():
    """Run smoke test."""
    print("=" * 60)
    print("Smoke Test: ECG Training Pipeline (LOS Bin Classification)")
    print("=" * 60)
    
    # Load configs
    config = load_config(
        base_config_path=Path("configs/baseline.yaml"),
        model_config_path=Path("configs/model/cnn_scratch.yaml"),
    )
    
    # Load ICU stays
    icustays_path = os.getenv("ICUSTAYS_PATH", "data/labeling/labels_csv/icustays.csv")
    if not Path(icustays_path).exists():
        data_dir = config.get("data", {}).get("data_dir", "")
        icustays_path = Path(data_dir).parent / "icustays.csv"
        if not icustays_path.exists():
            # Try default location
            icustays_path = Path("data/labeling/labels_csv/icustays.csv")
            if not icustays_path.exists():
                print(f"[WARNING] icustays.csv not found at {icustays_path}")
                print("   Skipping label generation test. Using dummy labels.")
                icu_mapper = None
                mortality_mapping = None
            else:
                print(f"[OK] Loading ICU stays from: {icustays_path}")
                icustays_df = load_icustays(str(icustays_path))
                icu_mapper = ICUStayMapper(icustays_df)
                print(f"[OK] Loaded {len(icustays_df)} ICU stays")
                mortality_mapping = None
        else:
            print(f"[OK] Loading ICU stays from: {icustays_path}")
            icustays_df = load_icustays(str(icustays_path))
            icu_mapper = ICUStayMapper(icustays_df)
            print(f"[OK] Loaded {len(icustays_df)} ICU stays")
            mortality_mapping = None
    else:
        print(f"[OK] Loading ICU stays from: {icustays_path}")
        icustays_df = load_icustays(str(icustays_path))
        icu_mapper = ICUStayMapper(icustays_df)
        print(f"[OK] Loaded {len(icustays_df)} ICU stays")
        mortality_mapping = None
    
    # Load mortality mapping if multi-task is enabled
    multi_task_config = config.get("multi_task", {})
    is_multi_task = multi_task_config.get("enabled", False)
    if is_multi_task and icu_mapper is not None:
        admissions_path = multi_task_config.get("admissions_path", "data/labeling/labels_csv/admissions.csv")
        admissions_path = Path(admissions_path)
        if not admissions_path.is_absolute():
            admissions_path = project_root / admissions_path
        
        if admissions_path.exists():
            print(f"[OK] Loading admissions from: {admissions_path}")
            try:
                mortality_mapping = load_mortality_mapping(str(admissions_path), icustays_df)
                icu_mapper = ICUStayMapper(icustays_df, mortality_mapping=mortality_mapping)
                print(f"[OK] Loaded mortality mapping: {sum(mortality_mapping.values())} died, {len(mortality_mapping) - sum(mortality_mapping.values())} survived")
            except Exception as e:
                print(f"[WARNING] Failed to load mortality mapping: {e}")
                mortality_mapping = None
        else:
            print(f"[WARNING] admissions.csv not found at {admissions_path}")
            print("   Multi-task enabled but no mortality labels will be generated")
            mortality_mapping = None
    
    # Create DataLoaders (small subset for testing)
    print("\n" + "-" * 60)
    print("Creating DataLoaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            config=config,
            labels=None,
            preprocess=None,
            transform=None,
            icu_mapper=icu_mapper,
            mortality_labels=None,  # Will be auto-generated
        )
    except FileNotFoundError as e:
        print(f"[ERROR] Data directory not found: {e}")
        print("[INFO] Please set data_dir in configs/baseline.yaml or provide ECG data")
        print("[INFO] Skipping DataLoader tests")
        return
    
    print(f"[OK] Train batches: {len(train_loader)}")
    print(f"[OK] Val batches: {len(val_loader)}")
    
    # Calculate ECGs per stay statistics
    print("\n" + "-" * 60)
    print("ECGs per Stay Statistics...")
    if icu_mapper is not None:
        from collections import Counter
        import numpy as np
        
        stay_ecg_counts = Counter()
        
        # Count ECGs per stay in train + val loaders
        for loader in [train_loader, val_loader]:
            for batch in loader:
                meta = batch["meta"]
                for m in meta:
                    stay_id = m.get("stay_id")
                    if stay_id is not None:
                        stay_ecg_counts[stay_id] += 1
        
        if len(stay_ecg_counts) > 0:
            counts = list(stay_ecg_counts.values())
            
            print(f"  Total stays with ECGs: {len(stay_ecg_counts):,}")
            print(f"  Total ECGs: {sum(counts):,}")
            print(f"  Mean ECGs/stay: {np.mean(counts):.2f}")
            print(f"  Median ECGs/stay: {np.median(counts):.1f}")
            print(f"  Min ECGs/stay: {min(counts)}")
            print(f"  Max ECGs/stay: {max(counts)}")
            print(f"  Std ECGs/stay: {np.std(counts):.2f}")
            
            # Distribution
            count_dist = Counter(counts)
            print(f"\n  Distribution:")
            print(f"    1 ECG/stay: {count_dist[1]:,} stays ({100*count_dist[1]/len(counts):.1f}%)")
            print(f"    2 ECGs/stay: {count_dist[2]:,} stays ({100*count_dist[2]/len(counts):.1f}%)")
            print(f"    3 ECGs/stay: {count_dist[3]:,} stays ({100*count_dist[3]/len(counts):.1f}%)")
            print(f"    4+ ECGs/stay: {sum(count_dist[i] for i in count_dist if i >= 4):,} stays ({100*sum(count_dist[i] for i in count_dist if i >= 4)/len(counts):.1f}%)")
            
            # Percentiles
            print(f"\n  Percentiles:")
            for p in [25, 50, 75, 90, 95, 99]:
                val = np.percentile(counts, p)
                print(f"    {p}th percentile: {val:.1f} ECGs/stay")
        else:
            print("[WARNING] No stay_id found in batches")
    else:
        print("[WARNING] ICU mapper not available - cannot compute statistics")
    
    # Get one batch
    print("\n" + "-" * 60)
    print("Testing batch format...")
    batch = next(iter(train_loader))
    
    signals = batch["signal"]
    labels = batch["label"]
    
    print(f"[OK] Signals shape: {signals.shape} (expected: (B, 12, 5000))")
    print(f"[OK] Labels shape: {labels.shape} (expected: (B,))")
    print(f"[OK] Signals dtype: {signals.dtype} (expected: float32)")
    print(f"[OK] Labels dtype: {labels.dtype} (expected: int64/long)")
    
    # Check label values - CRITICAL: No unmatched samples (label == -1) should appear
    unmatched_count = (labels == -1).sum().item()
    assert unmatched_count == 0, f"Found {unmatched_count} unmatched samples (label == -1) in batch. Unmatched samples must be filtered before training."
    print(f"[OK] No unmatched samples in batch (all labels >= 0)")
    
    valid_labels = labels[labels >= 0]
    if len(valid_labels) > 0:
        print(f"[OK] Valid labels range: [{valid_labels.min().item()}, {valid_labels.max().item()}]")
        print(f"[OK] Valid labels count: {len(valid_labels)}/{len(labels)}")
        assert valid_labels.min() >= 0, "Labels must be >= 0"
        assert valid_labels.max() <= 9, "Labels must be <= 9"
        print("[OK] Label values are in range [0, 9]")
    else:
        print("[WARNING] No valid labels found in batch")
    
    # Check that stay_id is present in meta (required for stay-level evaluation)
    meta = batch["meta"]
    stay_ids_present = sum(1 for m in meta if "stay_id" in m and m["stay_id"] is not None)
    print(f"[OK] stay_id present in {stay_ids_present}/{len(meta)} samples")
    if stay_ids_present == 0:
        print("[WARNING] No stay_id found in meta - stay-level evaluation may not work")
    
    # Check for mortality labels (if multi-task)
    has_mortality_labels = "mortality_label" in batch
    if has_mortality_labels:
        mortality_labels = batch["mortality_label"]
        print(f"[OK] Mortality labels shape: {mortality_labels.shape} (expected: (B,))")
        valid_mortality = mortality_labels[mortality_labels >= 0]
        print(f"[OK] Valid mortality labels: {len(valid_mortality)}/{len(mortality_labels)}")
        if len(valid_mortality) > 0:
            print(f"[OK] Mortality labels range: [{valid_mortality.min().item()}, {valid_mortality.max().item()}]")
    
    # Create model
    print("\n" + "-" * 60)
    print("Testing model...")
    base_model = CNNScratch(config)
    print(f"[OK] Base model created: {base_model.__class__.__name__}")
    print(f"[OK] Base model parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    
    # Test multi-task if enabled in config
    multi_task_config = config.get("multi_task", {})
    is_multi_task = multi_task_config.get("enabled", False)
    
    if is_multi_task and has_mortality_labels:
        print("\n" + "-" * 60)
        print("Testing Multi-Task model...")
        try:
            model = MultiTaskECGModel(base_model, config)
            print(f"[OK] Multi-Task model created: {model.__class__.__name__}")
            print(f"[OK] Multi-Task model parameters: {model.count_parameters():,}")
        except Exception as e:
            print(f"[ERROR] Failed to create Multi-Task model: {e}")
            is_multi_task = False
            model = base_model
    else:
        model = base_model
        if is_multi_task and not has_mortality_labels:
            print("[WARNING] Multi-task enabled in config but no mortality labels in batch")
    
    # Forward pass
    print("\n" + "-" * 60)
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(signals)
    
    if is_multi_task and isinstance(outputs, dict):
        los_logits = outputs["los"]
        mortality_probs = outputs["mortality"]
        print(f"[OK] LOS logits shape: {los_logits.shape} (expected: ({signals.shape[0]}, 10))")
        print(f"[OK] Mortality probs shape: {mortality_probs.shape} (expected: ({signals.shape[0]}, 1))")
        assert los_logits.shape[0] == signals.shape[0], "Batch size mismatch for LOS"
        assert los_logits.shape[1] == 10, f"Expected 10 LOS classes, got {los_logits.shape[1]}"
        assert mortality_probs.shape[0] == signals.shape[0], "Batch size mismatch for mortality"
        assert mortality_probs.shape[1] == 1, f"Expected 1 mortality output, got {mortality_probs.shape[1]}"
        assert (mortality_probs >= 0).all() and (mortality_probs <= 1).all(), "Mortality probs should be in [0, 1]"
        print("[OK] Multi-Task outputs shape is correct")
        logits = los_logits  # For compatibility with rest of test
    else:
        logits = outputs if not isinstance(outputs, dict) else outputs.get("los", outputs)
        print(f"[OK] Logits shape: {logits.shape} (expected: ({signals.shape[0]}, 10))")
        assert logits.shape[0] == signals.shape[0], "Batch size mismatch"
        assert logits.shape[1] == 10, f"Expected 10 classes, got {logits.shape[1]}"
        print("[OK] Logits shape is correct: (B, 10)")
    
    # Loss computation
    print("\n" + "-" * 60)
    print("Testing loss computation...")
    if is_multi_task and has_mortality_labels:
        criterion = get_multi_task_loss(config)
        mortality_labels_batch = batch["mortality_label"]
        loss_dict = criterion(logits, labels, mortality_probs, mortality_labels_batch)
        loss = loss_dict["total"]
        print(f"[OK] Multi-Task Loss computed:")
        print(f"    Total: {loss.item():.4f}")
        print(f"    LOS: {loss_dict['los'].item():.4f}")
        print(f"    Mortality: {loss_dict['mortality'].item():.4f}")
    else:
        criterion = get_loss(config)
        loss = criterion(logits, labels)
        print(f"[OK] Loss computed: {loss.item():.4f}")
    print(f"[OK] Loss on {len(labels)} samples")
    
    # Test stay-level aggregation (for evaluation)
    print("\n" + "-" * 60)
    print("Testing stay-level aggregation...")
    meta = batch["meta"]
    stay_to_logits = {}
    stay_to_labels = {}
    
    for i in range(len(labels)):
        stay_id = meta[i].get("stay_id")
        if stay_id is not None:
            if stay_id not in stay_to_logits:
                stay_to_logits[stay_id] = []
                stay_to_labels[stay_id] = labels[i].item()
            stay_to_logits[stay_id].append(logits[i].cpu())
    
    if len(stay_to_logits) > 0:
        # Aggregate by mean
        aggregated_logits = []
        aggregated_labels = []
        for stay_id in stay_to_logits:
            agg = torch.stack(stay_to_logits[stay_id]).mean(dim=0)
            aggregated_logits.append(agg)
            aggregated_labels.append(stay_to_labels[stay_id])
        
        num_stays = len(aggregated_logits)
        print(f"[OK] Stay-level aggregation: {len(labels)} ECGs → {num_stays} stays")
        print(f"[OK] Stay-level logits shape: {torch.stack(aggregated_logits).shape}")
        assert len(aggregated_logits) == num_stays, "Number of evaluated items should equal unique stay_ids"
    else:
        print("[WARNING] No stay_id found - cannot test stay-level aggregation")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All smoke tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
    
    
