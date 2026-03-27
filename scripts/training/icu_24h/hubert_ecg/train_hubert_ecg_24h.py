"""Training script for HuBERT-ECG with two-phase training.

Phase 1: Frozen backbone (HuBERT encoder frozen, only heads trained)
Phase 2: Fine-tuning with layer-dependent learning rates

Based on Transfer Learning strategy for self-supervised pretrained models.

LOS Regression Task: Predicts continuous LOS in days (not binned classes).

Resume after timeout (checkpoints from ModelCheckpoint: HuBERT_ECG_best_<JOBID>.pt, HuBERT_ECG_epoch_*.pt):
  # Still in phase 1:
  python .../train_hubert_ecg_24h.py --resume-phase1 outputs/checkpoints/HuBERT_ECG_best_3768869.pt

  # Phase 1 done (hubert_ecg_phase1_best_*.pt exists), interrupted in phase 2:
  python .../train_hubert_ecg_24h.py --skip-phase1 --resume-phase2 outputs/checkpoints/HuBERT_ECG_best_3768869.pt

  # Skip phase 1, start phase 2 from phase-1 weights only (no phase-2 trainer checkpoint yet):
  python .../train_hubert_ecg_24h.py --skip-phase1 --phase1-weights outputs/checkpoints/hubert_ecg_phase1_best_3768869.pt

Optional env: RESUME_PHASE1, RESUME_PHASE2, PHASE1_WEIGHTS, SKIP_PHASE1=1
"""

from pathlib import Path
import argparse
import sys
import os
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.models import HuBERT_ECG
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.training import Trainer, setup_icustays_mapper, evaluate_and_print_results
from src.training.losses import get_multi_task_loss
from src.utils.config_loader import load_config


class HuBERT_ECG_Trainer(Trainer):
    """Specialized trainer for HuBERT-ECG with two-phase training."""
    
    def __init__(
        self,
        model: MultiTaskECGModel,
        train_loader,
        val_loader,
        config: dict,
        device=None,
        criterion=None,
        phase: int = 1,
    ):
        """Initialize HuBERT-ECG trainer.
        
        Args:
            phase: Training phase (1: frozen, 2: fine-tuning)
        """
        self.phase = phase
        super().__init__(model, train_loader, val_loader, config, device, criterion)
        
        # Recreate optimizer/scheduler so both use the phase-specific parameter groups.
        self.optimizer = self._create_layer_dependent_optimizer()
        # Setup phase-specific settings BEFORE creating scheduler (scheduler warmup sets LR=0 at step 0)
        if phase == 1:
            self._setup_phase1()
        elif phase == 2:
            self._setup_phase2()
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 1 (frozen) or 2 (fine-tuning)")
        self.scheduler = self._create_scheduler()
    
    def _setup_phase1(self) -> None:
        """Setup Phase 1: Frozen backbone training (H2: frozen backbone, only heads, max 50 epochs)."""
        print("="*60)
        print("Phase 1: Frozen Backbone Training")
        print("="*60)
        print("HuBERT encoder: FROZEN (no gradient updates)")
        print("Trainable: Only shared layers and regression/classification heads")
        lr = self.optimizer.param_groups[0]['lr']
        print(f"Learning rate: {lr}")
        if lr <= 0:
            raise ValueError(f"Phase 1 LR must be > 0, got {lr}. Check config optimizer.lr.")
        print("="*60)
        
        # Freeze backbone (should already be frozen from model init, but ensure it)
        base_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model
        if hasattr(base_model, 'freeze_backbone'):
            base_model.freeze_backbone()
    
    def _setup_phase2(self) -> None:
        """Setup Phase 2: Fine-tuning with layer-dependent LRs (H2: unfrozen backbone, max 20 epochs, feature extractor stays frozen)."""
        print("="*60)
        print("Phase 2: Fine-tuning with Layer-Dependent Learning Rates")
        print("="*60)
        print("HuBERT encoder: UNFROZEN (feature extractor stays frozen, transformer layers fine-tuned)")
        print("Learning rates (H2: 1e-7, 1e-5, 1e-4):")
        for i, group in enumerate(self.optimizer.param_groups):
            lr = group['lr']
            if lr <= 0:
                raise ValueError(f"Phase 2 Group {i} LR must be > 0, got {lr}. Check config.")
            print(f"  Group {i}: lr={lr:.2e}, params={len(group['params'])}")
        print("="*60)
        
        # Unfreeze backbone
        base_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model
        if hasattr(base_model, 'unfreeze_backbone'):
            base_model.unfreeze_backbone()
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        sched_config = self.config.get("training", {}).get("scheduler", {})
        sched_type = sched_config.get("type")
        
        if sched_type == "LinearScheduleWithWarmup":
            from transformers import get_linear_schedule_with_warmup
            num_epochs = self.config.get("training", {}).get("num_epochs", 50)
            steps_per_epoch = len(self.train_loader)
            total_training_steps = max(1, num_epochs * steps_per_epoch)
            warmup_pct = sched_config.get("warmup_steps_pct", 0.08)
            warmup_steps = max(1, int(total_training_steps * warmup_pct))
            
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps
            )
            
        return super()._create_scheduler()

    MIN_LR = 1e-7  # Guard: LRs must never be 0 (matches H2 / original HuBERT-ECG)

    def _create_layer_dependent_optimizer(self):
        """Create optimizer with layer-dependent learning rates (H2 / original HuBERT-ECG)."""
        opt_config = self.config.get("training", {}).get("optimizer", {})
        opt_type = opt_config.get("type", "Adam").lower()
        base_lr = opt_config.get("lr", 5e-4)
        if isinstance(base_lr, str):
            base_lr = float(base_lr)
        backbone_lr = opt_config.get("backbone_lr", 1e-5)
        if isinstance(backbone_lr, str):
            backbone_lr = float(backbone_lr)
        deep_lr = opt_config.get("deep_backbone_lr", 1e-7)
        if isinstance(deep_lr, str):
            deep_lr = float(deep_lr)
        weight_decay = opt_config.get("weight_decay", 1e-4)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)

        # Guard: LRs must never be 0 (prevents silent training failure)
        base_lr = max(float(base_lr), self.MIN_LR)
        backbone_lr = max(float(backbone_lr), self.MIN_LR)
        deep_lr = max(float(deep_lr), self.MIN_LR)
        
        # Get base model
        base_model = self.model.base_model if hasattr(self.model, 'base_model') else self.model
        
        # Phase-specific learning rates
        # Note: HuBERT_ECG architecture has classifier_dropout + heads (no shared_bn/shared_fc)
        if self.phase == 1:
            # Phase 1: Only heads are trainable (HuBERT encoder frozen)
            head_lr = base_lr  # 1e-4 typically
            param_groups = [
                {
                    'params': list(base_model.classifier_dropout.parameters()) +
                             list(base_model.los_head.parameters()) +
                             list(base_model.mortality_head.parameters()),
                    'lr': head_lr,
                }
            ]
        else:  # Phase 2
            # Phase 2: Layer-dependent learning rates
            # - Deep transformer layers + feature projection: deep_backbone_lr
            # - Last 4 transformer layers: backbone_lr
            # - Classifier dropout + Heads: lr
            head_lr = base_lr
            
            param_groups = []
            
            # hubert_model is the HuBERTECG instance inside HuBERTEncoder
            hubert_model = base_model.hubert_encoder.hubert_model
            
            if hasattr(hubert_model, 'feature_projection'):
                param_groups.append({
                    'params': hubert_model.feature_projection.parameters(),
                    'lr': deep_lr,
                })
                
            if hasattr(hubert_model, 'encoder') and hasattr(hubert_model.encoder, 'layers'):
                layers = hubert_model.encoder.layers
                num_layers = len(layers)
                if num_layers > 4:
                    param_groups.append({
                        'params': layers[:-4].parameters(),
                        'lr': deep_lr,
                    })
                    param_groups.append({
                        'params': layers[-4:].parameters(),
                        'lr': backbone_lr,
                    })
                else:
                    param_groups.append({
                        'params': layers.parameters(),
                        'lr': backbone_lr,
                    })
            else:
                # Fallback
                param_groups.append({
                    'params': base_model.hubert_encoder.parameters(),
                    'lr': backbone_lr,
                })
            
            # Classifier Heads
            param_groups.append({
                'params': list(base_model.classifier_dropout.parameters()) +
                         list(base_model.los_head.parameters()) +
                         list(base_model.mortality_head.parameters()),
                'lr': head_lr,
            })
        
        # Create optimizer with parameter groups
        betas = opt_config.get("betas", [0.9, 0.999])
        if opt_type == "adamw":
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay, betas=betas)
        elif opt_type == "adam":
            return torch.optim.Adam(param_groups, weight_decay=weight_decay, betas=betas)
        elif opt_type == "sgd":
            momentum = opt_config.get("momentum", 0.9)
            return torch.optim.SGD(param_groups, lr=base_lr, weight_decay=weight_decay, momentum=momentum)
        else:
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay, betas=betas)
    
    def train_phase1(
        self,
        max_epochs: int = 20,
        patience: int = 20,
        resume_from: Path | None = None,
    ):
        """Train Phase 1: Frozen backbone."""
        if self.phase != 1:
            raise ValueError("train_phase1() can only be called when phase=1")
        
        self.early_stopping.patience = patience
        original_max_epochs = self.config.get("training", {}).get("num_epochs", 50)
        self.config["training"]["num_epochs"] = max_epochs
        self.scheduler = self._create_scheduler()
        
        history = self.train(resume_from=resume_from)
        
        self.config["training"]["num_epochs"] = original_max_epochs
        return history
    
    def train_phase2(
        self,
        phase1_checkpoint_path=None,
        max_epochs: int = 30,
        patience: int = 15,
        resume_from: Path | None = None,
    ):
        """Train Phase 2: Fine-tuning."""
        if self.phase != 2:
            raise ValueError("train_phase2() can only be called when phase=2")
        
        if resume_from is not None and resume_from.exists():
            print(f"Resuming Phase 2 from trainer checkpoint: {resume_from}")
        elif phase1_checkpoint_path is not None and phase1_checkpoint_path.exists():
            print(f"Loading Phase 1 weights from: {phase1_checkpoint_path}")
            checkpoint = torch.load(phase1_checkpoint_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print(f"Loaded Phase 1 model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint, strict=False)
                print("Loaded Phase 1 model state dict")
        
        self.early_stopping.patience = patience
        original_max_epochs = self.config.get("training", {}).get("num_epochs", 50)
        self.config["training"]["num_epochs"] = max_epochs
        self.scheduler = self._create_scheduler()
        
        if resume_from is None:
            self.current_epoch = 0
            self.history = {
                "train_loss": [],
                "train_los_mae": [],
                "train_los_rmse": [],
                "train_los_r2": [],
                "val_loss": [],
                "val_los_mae": [],
                "val_los_rmse": [],
                "val_los_r2": [],
                "train_mortality_acc": [],
                "val_mortality_acc": [],
                "val_mortality_auc": [],
            }
        
        history = self.train(resume_from=resume_from)
        
        self.config["training"]["num_epochs"] = original_max_epochs
        return history


def main():
    """Main training function for HuBERT-ECG with two-phase training.
    
    Phase 1: Frozen backbone, only task heads trained.
    Phase 2: Fine-tuning with unfrozen backbone (layer-dependent LRs).
    
    LOS Regression Task: Predicts continuous LOS in days.
    """
    parser = argparse.ArgumentParser(description="HuBERT-ECG two-phase training (24h ICU)")
    parser.add_argument(
        "--resume-phase1",
        type=str,
        default=None,
        help="Resume Phase 1 from HuBERT_ECG_best_*.pt or HuBERT_ECG_epoch_*.pt",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 (use with --phase1-weights and/or --resume-phase2)",
    )
    parser.add_argument(
        "--phase1-weights",
        type=str,
        default=None,
        help="When skipping Phase 1: load hubert_ecg_phase1_best_*.pt (weights only) before Phase 2",
    )
    parser.add_argument(
        "--resume-phase2",
        type=str,
        default=None,
        help="Resume Phase 2 from HuBERT_ECG_best_*.pt or epoch file (full trainer state from Phase 2)",
    )
    args = parser.parse_args()

    resume_p1 = args.resume_phase1 or os.getenv("RESUME_PHASE1")
    resume_p2 = args.resume_phase2 or os.getenv("RESUME_PHASE2")
    phase1_w = args.phase1_weights or os.getenv("PHASE1_WEIGHTS")
    skip_phase1 = args.skip_phase1 or os.getenv("SKIP_PHASE1", "").lower() in ("1", "true", "yes")

    resume_phase1_path = Path(resume_p1) if resume_p1 else None
    resume_phase2_path = Path(resume_p2) if resume_p2 else None
    phase1_weights_path = Path(phase1_w) if phase1_w else None

    if skip_phase1 and resume_phase1_path is not None:
        parser.error("Cannot combine --skip-phase1 with --resume-phase1")
    if skip_phase1 and resume_phase2_path is None and phase1_weights_path is None:
        parser.error("--skip-phase1 requires --phase1-weights and/or --resume-phase2")
    if resume_phase1_path is not None and not resume_phase1_path.is_file():
        parser.error(f"--resume-phase1 file not found: {resume_phase1_path}")
    if resume_phase2_path is not None and not resume_phase2_path.is_file():
        parser.error(f"--resume-phase2 file not found: {resume_phase2_path}")
    if phase1_weights_path is not None and not phase1_weights_path.is_file():
        parser.error(f"--phase1-weights file not found: {phase1_weights_path}")

    # Load config
    model_config_path = Path("configs/model/hubert_ecg/hubert_ecg.yaml")
    config = load_config(model_config_path=model_config_path)
    
    print("="*60)
    print("Training HuBERT-ECG - Phase 1 + Phase 2 (Frozen → Unfrozen)")
    print("Task: LOS REGRESSION (continuous prediction in days)")
    print("="*60)
    print(f"Model config: {model_config_path}")
    
    model_config = config.get('model', {})
    hubert_config = model_config.get('hubert', {})
    pretrained_config = model_config.get('pretrained', {})
    
    if pretrained_config.get('enabled', False):
        cache_dir = pretrained_config.get('cache_dir', 'data/pretrained_weights/Hubert_ECG/base')
        print(f"Pretrained weights: Enabled ({cache_dir})")
    else:
        print("Pretrained weights: Disabled (random init)")
    
    demographic_config = config.get('data', {}).get('demographic_features', {})
    print(f"Demographic features: {'Enabled' if demographic_config.get('enabled', False) else 'Disabled'}")
    
    diagnosis_config = config.get('data', {}).get('diagnosis_features', {})
    print(f"Diagnosis features: {'Enabled' if diagnosis_config.get('enabled', False) else 'Disabled'}")
    print("="*60)
    
    # Load ICU stays and create mapper
    icu_mapper = setup_icustays_mapper(config)
    
    # Ensure multi-task is enabled
    if not config.get("multi_task", {}).get("enabled", False):
        config["multi_task"] = {"enabled": True}
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        labels=None,
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,
    )
    
    # Create base model with pretrained weights + frozen backbone
    print("\nCreating HuBERT-ECG model (loading pretrained weights)...")
    base_model = HuBERT_ECG(config)
    
    total_params = base_model.count_parameters()
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} (backbone frozen, heads only)")
    
    # Wrap in MultiTaskECGModel
    model = MultiTaskECGModel(base_model, config)
    
    # Loss function
    criterion = get_multi_task_loss(config)
    
    # Get SLURM job ID
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id:
        print(f"SLURM Job ID: {job_id}")

    checkpoint_dir = Path(config.get("checkpoint", {}).get("save_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    phase1_checkpoint_path = checkpoint_dir / f"hubert_ecg_phase1_best_{job_id if job_id else 'local'}.pt"

    history_phase1: dict = {}
    if skip_phase1 and phase1_weights_path is not None and resume_phase2_path is None:
        print(f"\nLoading Phase 1 weights (skip Phase 1): {phase1_weights_path}")
        p1_ck = torch.load(phase1_weights_path, map_location="cpu")
        if "model_state_dict" in p1_ck:
            model.load_state_dict(p1_ck["model_state_dict"], strict=False)
        else:
            model.load_state_dict(p1_ck, strict=False)
        h = p1_ck.get("history")
        history_phase1 = h if isinstance(h, dict) else {}

    if skip_phase1:
        print("\n" + "="*60)
        print("PHASE 1: SKIPPED (--skip-phase1)")
        print("="*60)
    else:
        # ========== PHASE 1: Frozen Backbone (Head-Only Training) ==========
        print("\n" + "="*60)
        print("PHASE 1: FROZEN BACKBONE - HEAD-ONLY TRAINING")
        print("="*60)
        if resume_phase1_path is not None:
            print(f"Resuming Phase 1 from: {resume_phase1_path}")

        trainer_phase1 = HuBERT_ECG_Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            criterion=criterion,
            phase=1,
        )
        trainer_phase1.config_paths = {"model": str(model_config_path.resolve())}
        trainer_phase1.job_id = job_id

        history_phase1 = trainer_phase1.train_phase1(
            max_epochs=config.get("training", {}).get("num_epochs", 50),
            patience=config.get("early_stopping", {}).get("patience", 20),
            resume_from=resume_phase1_path,
        )

        print("\nPhase 1 completed!")
        print(f"Best validation loss: {min(history_phase1.get('val_loss', [float('inf')])):.4f}")
        print(f"Best validation MAE:  {min(history_phase1.get('val_los_mae', [float('inf')])):.4f} days")

        save_dict = {
            "model_state_dict": model.state_dict(),
            "epoch": getattr(trainer_phase1.checkpoint, "best_epoch", trainer_phase1.current_epoch),
            "val_loss": getattr(trainer_phase1.checkpoint, "best_value", None),
            "history": history_phase1,
            "config": config,
        }
        torch.save(save_dict, phase1_checkpoint_path)
        print(f"Saved Phase 1 checkpoint to: {phase1_checkpoint_path}")
    
    # ========== PHASE 2: Unfrozen Backbone (Fine-tuning) ==========
    print("\n" + "="*60)
    print("PHASE 2: UNFROZEN BACKBONE - FINE-TUNING")
    print("="*60)
    
    trainer_phase2 = HuBERT_ECG_Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
        phase=2,
    )
    trainer_phase2.config_paths = {"model": str(model_config_path.resolve())}
    trainer_phase2.job_id = job_id
    
    # Phase 2: max 20 epochs, patience 7
    p1_for_phase2 = phase1_checkpoint_path if not skip_phase1 else None

    history_phase2 = trainer_phase2.train_phase2(
        phase1_checkpoint_path=p1_for_phase2,
        max_epochs=20,
        patience=config.get("early_stopping", {}).get("patience", 7),
        resume_from=resume_phase2_path,
    )
    
    print("\nPhase 2 completed!")
    print(f"Best validation loss: {min(history_phase2.get('val_loss', [float('inf')])):.4f}")
    print(f"Best validation MAE:  {min(history_phase2.get('val_los_mae', [float('inf')])):.4f} days")
    
    # Save Phase 2 checkpoint
    phase2_checkpoint_path = checkpoint_dir / f"hubert_ecg_phase2_best_{job_id if job_id else 'local'}.pt"
    save_dict_phase2 = {
        "model_state_dict": model.state_dict(),
        "epoch": getattr(trainer_phase2.checkpoint, "best_epoch", trainer_phase2.current_epoch),
        "val_loss": getattr(trainer_phase2.checkpoint, "best_value", None),
        "history": history_phase2,
        "config": config,
    }
    torch.save(save_dict_phase2, phase2_checkpoint_path)
    print(f"Saved Phase 2 checkpoint to: {phase2_checkpoint_path}")
    
    # ========== FINAL EVALUATION (Phase 2 model) ==========
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET (Phase 2 model)")
    print("="*60)
    
    # Load best Phase 2 model for evaluation
    if phase2_checkpoint_path.exists():
        checkpoint = torch.load(phase2_checkpoint_path, map_location=trainer_phase2.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(trainer_phase2.device)
        print(f"Loaded best Phase 2 model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    eval_trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
    )
    eval_trainer.job_id = job_id
    
    # Merge Phase 1 + Phase 2 history for logging (use Phase 2 for final metrics)
    history_combined = {
        k: history_phase1.get(k, []) + history_phase2.get(k, [])
        for k in set(history_phase1) | set(history_phase2)
    }
    history_final = evaluate_and_print_results(eval_trainer, test_loader, history_combined, config)
    
    print("\n" + "="*60)
    print("PHASE 1 + PHASE 2 TRAINING COMPLETED")
    print("="*60)
    if skip_phase1:
        print(f"Phase 1 checkpoint (skipped): {phase1_weights_path or 'n/a — resumed Phase 2 only'}")
    else:
        print(f"Phase 1 checkpoint: {phase1_checkpoint_path}")
    print(f"Phase 2 checkpoint: {phase2_checkpoint_path}")
    
    return history_phase2


if __name__ == "__main__":
    main()
