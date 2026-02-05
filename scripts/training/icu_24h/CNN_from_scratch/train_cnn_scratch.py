"""Training script for CNN from scratch with LOS regression.

LOS Regression Task: Predicts continuous LOS in days (not binned classes).
"""

from pathlib import Path
import sys
import os
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models import CNNScratch
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.data.labeling import load_icustays, ICUStayMapper, load_mortality_mapping
from src.training import Trainer, evaluate_and_print_results
from src.training.losses import get_loss, get_multi_task_loss
from src.utils.config_loader import load_config


def main():
    """Main training function.
    
    LOS Regression Task: Predicts continuous LOS in days.
    """
    # Load configs
    # Use baseline_no_aug.yaml for true baseline training (no augmentation)
    base_config_path = Path("configs/icu_24h/baseline_no_aug.yaml")
    model_config_path = Path("configs/model/cnn_scratch.yaml")
    
    config = load_config(
        base_config_path=base_config_path,
        model_config_path=model_config_path,
    )
    
    # Log config paths for tracking
    print("="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Base config: {base_config_path}")
    print(f"Model config: {model_config_path}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")
    print(f"Task: LOS REGRESSION (continuous prediction in days)")
    print(f"Loss type: {config.get('training', {}).get('loss', {}).get('type', 'mse')}")
    print("="*60)
    
    # Load ICU stays and create mapper
    # Path can be set via environment variable or config
    icustays_env = os.getenv("ICUSTAYS_PATH")
    if icustays_env:
        env_path = Path(icustays_env)
        if env_path.exists():
            icustays_path = env_path
        else:
            print(f"Warning: ICUSTAYS_PATH is set but file does not exist: {env_path}. Falling back to default lookup.")
            icustays_env = None

    if not icustays_env:
        # Try relative to data_dir
        data_dir = config.get("data", {}).get("data_dir", "")
        if data_dir:
            icustays_path = Path(data_dir).parent.parent / "labeling" / "labels_csv" / "icustays.csv"
        else:
            # Default fallback (relative to project root)
            icustays_path = Path("data/labeling/labels_csv/icustays.csv")

    icustays_path = Path(icustays_path)
    if not icustays_path.exists():
        raise FileNotFoundError(
            f"icustays.csv not found at: {icustays_path}\n"
            f"Set ICUSTAYS_PATH environment variable or place icustays.csv in data/labeling/labels_csv."
        )
    
    print(f"Loading ICU stays from: {icustays_path}")
    icustays_df = load_icustays(str(icustays_path))
    print(f"Loaded {len(icustays_df)} ICU stays")
    
    # Check if multi-task is enabled
    multi_task_config = config.get("multi_task", {})
    is_multi_task = multi_task_config.get("enabled", False)
    
    # Load mortality mapping if multi-task is enabled
    mortality_mapping = None
    if is_multi_task:
        admissions_path = multi_task_config.get("admissions_path", "data/labeling/labels_csv/admissions.csv")
        admissions_path = Path(admissions_path)
        
        # Try to resolve relative path
        if not admissions_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent.parent
            admissions_path = project_root / admissions_path
        
        if not admissions_path.exists():
            data_dir = config.get("data", {}).get("data_dir", "")
            if data_dir:
                admissions_path = Path(data_dir).parent.parent / "labeling" / "labels_csv" / "admissions.csv"
        
        if not admissions_path.exists():
            raise FileNotFoundError(
                f"admissions.csv not found for multi-task learning at: {admissions_path}\n"
                f"Set multi_task.admissions_path in config or place admissions.csv in data/labeling directory."
            )
        
        print(f"Loading admissions from: {admissions_path}")
        mortality_mapping = load_mortality_mapping(str(admissions_path), icustays_df)
        print(f"Loaded mortality mapping: {sum(mortality_mapping.values())} died, {len(mortality_mapping) - sum(mortality_mapping.values())} survived")
    
    # Create ICU mapper with mortality mapping
    icu_mapper = ICUStayMapper(icustays_df, mortality_mapping=mortality_mapping)
    
    # Create DataLoaders (labels will be auto-generated via icu_mapper)
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        labels=None,  # Will be auto-generated
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,  # Will be auto-generated from mortality_mapping
    )
    
    # Create base model
    base_model = CNNScratch(config)
    
    # Wrap in MultiTaskECGModel if multi-task is enabled
    if is_multi_task:
        print("Creating Multi-Task model (LOS Regression + Mortality Classification)...")
        model = MultiTaskECGModel(base_model, config)
        print(f"Multi-Task model created with {model.count_parameters():,} parameters")
    else:
        model = base_model
    
    # Create loss function
    if is_multi_task:
        criterion = get_multi_task_loss(config)
        print("Using Multi-Task Loss (LOS MSE + Mortality BCE)")
    else:
        criterion = get_loss(config)
        print(f"Using Single-Task Loss (LOS MSE: {type(criterion).__name__})")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
    )
    
    # Store config paths for checkpoint saving
    trainer.config_paths = {
        "base": str(base_config_path.resolve()),
        "model": str(model_config_path.resolve()),
    }
    
    # Store job ID if available (from SLURM)
    trainer.job_id = os.getenv("SLURM_JOB_ID")
    if trainer.job_id:
        print(f"SLURM Job ID: {trainer.job_id}")
    
    # Train
    history = trainer.train()
    
    print("Training completed!")
    print(f"Best validation loss: {min(history.get('val_loss', [float('inf')])):.4f}")
    print(f"Best validation MAE: {min(history.get('val_los_mae', [float('inf')])):.4f} days")
    print(f"Best validation R²: {max(history.get('val_los_r2', [float('-inf')])):.4f}")
    
    # Test evaluation
    history = evaluate_and_print_results(trainer, test_loader, history, config)
    
    return history


if __name__ == "__main__":
    main()
