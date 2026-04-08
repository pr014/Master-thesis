"""Training script for DeepECG-SL (WCR) with two-phase training.

Phase 1: Frozen backbone (WCR encoder frozen, only heads trained)
Phase 2: Fine-tuning with layer-dependent learning rates

LOS Regression Task: Predicts continuous LOS in days.

Usage:
  python scripts/training/icu_24h/deepecg_sl/train_deepecg_sl_24h.py
  python .../train_deepecg_sl_24h.py --experiment-config configs/tuning/deepecg_sl/optuna_base_p1_no_tabular.yaml

Phase-2 LR: set ``training.optimizer.backbone_lr`` and ``training.optimizer.head_lr`` (aliases
``lr_backbone`` / ``lr_head``). If both are omitted, legacy 1e-5 / 1e-4 is used.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.models import DeepECG_SL
from src.models import MultiTaskECGModel
from src.data.ecg import create_dataloaders
from src.training import Trainer, setup_icustays_mapper, evaluate_and_print_results
from src.training.losses import get_multi_task_loss
from src.utils.config_loader import load_config


def _float_opt(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        return float(value)
    return float(value)


def _resolve_finetune_lrs(opt_config: dict, base_lr: float) -> tuple[float, float]:
    """Learning rates for phase-2 (backbone/adapter vs heads).

    If neither ``backbone_lr`` nor ``head_lr`` is set, keep legacy defaults
    (1e-5 backbone, 1e-4 heads) so existing configs without these keys behave
    unchanged. If either is set, missing side falls back to ``base_lr``
    (``training.optimizer.lr``).
    """
    backbone_key = opt_config.get("backbone_lr", opt_config.get("lr_backbone"))
    head_key = opt_config.get("head_lr", opt_config.get("lr_head"))
    if backbone_key is None and head_key is None:
        return 1e-5, 1e-4
    bb = _float_opt(backbone_key, base_lr)
    hd = _float_opt(head_key, base_lr)
    return bb, hd


class DeepECG_SL_Trainer(Trainer):
    """Specialized trainer for DeepECG-SL with two-phase training."""

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
        self.phase = phase
        super().__init__(model, train_loader, val_loader, config, device, criterion)
        self.optimizer = self._create_layer_dependent_optimizer()
        if phase == 1:
            self._setup_phase1()
        elif phase == 2:
            self._setup_phase2()
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 1 (frozen) or 2 (fine-tuning)")

    def _setup_phase1(self) -> None:
        print("=" * 60)
        print("Phase 1: Frozen Backbone Training")
        print("=" * 60)
        print("WCR encoder: FROZEN (no gradient updates)")
        print("Input adapter: FROZEN (no gradient updates)")
        print("Trainable: Only shared layers and regression/classification heads")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print("=" * 60)
        base_model = self.model.base_model if hasattr(self.model, "base_model") else self.model
        if hasattr(base_model, "freeze_backbone"):
            base_model.freeze_backbone()

    def _setup_phase2(self) -> None:
        print("=" * 60)
        print("Phase 2: Fine-tuning with Layer-Dependent Learning Rates")
        print("=" * 60)
        print("WCR encoder: UNFROZEN (with reduced learning rate)")
        print("Input adapter: UNFROZEN (with reduced learning rate)")
        print("Learning rates:")
        for i, group in enumerate(self.optimizer.param_groups):
            print(f"  Group {i}: lr={group['lr']:.2e}, params={len(group['params'])}")
        print("=" * 60)
        base_model = self.model.base_model if hasattr(self.model, "base_model") else self.model
        if hasattr(base_model, "unfreeze_backbone"):
            base_model.unfreeze_backbone()

    def _create_layer_dependent_optimizer(self):
        opt_config = self.config.get("training", {}).get("optimizer", {})
        opt_type = opt_config.get("type", "Adam").lower()
        base_lr = opt_config.get("lr", 5e-4)
        if isinstance(base_lr, str):
            base_lr = float(base_lr)
        weight_decay = opt_config.get("weight_decay", 1e-4)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        base_model = self.model.base_model if hasattr(self.model, "base_model") else self.model

        if self.phase == 1:
            head_lr = opt_config.get("head_lr", opt_config.get("lr_head"))
            if head_lr is None:
                head_lr = base_lr
            else:
                head_lr = _float_opt(head_lr, base_lr)
            param_groups = [
                {
                    "params": list(base_model.shared_bn.parameters())
                    + list(base_model.shared_fc.parameters())
                    + list(base_model.los_head.parameters())
                    + list(base_model.mortality_head.parameters()),
                    "lr": head_lr,
                }
            ]
        else:
            backbone_lr, head_lr = _resolve_finetune_lrs(opt_config, base_lr)
            param_groups = [
                {"params": base_model.wcr_encoder.parameters(), "lr": backbone_lr},
                {"params": base_model.input_adapter.parameters(), "lr": backbone_lr},
                {
                    "params": list(base_model.shared_bn.parameters())
                    + list(base_model.shared_fc.parameters()),
                    "lr": head_lr,
                },
                {"params": base_model.los_head.parameters(), "lr": head_lr},
                {"params": base_model.mortality_head.parameters(), "lr": head_lr},
            ]

        betas = opt_config.get("betas", [0.9, 0.999])
        if opt_type == "adamw":
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay, betas=betas)
        if opt_type == "adam":
            return torch.optim.Adam(param_groups, weight_decay=weight_decay, betas=betas)
        if opt_type == "sgd":
            momentum = opt_config.get("momentum", 0.9)
            return torch.optim.SGD(param_groups, lr=base_lr, weight_decay=weight_decay, momentum=momentum)
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay, betas=betas)

    def train_phase1(self, max_epochs: int = 20, patience: int = 20):
        if self.phase != 1:
            raise ValueError("train_phase1() can only be called when phase=1")
        self.early_stopping.patience = patience
        original_max_epochs = self.config.get("training", {}).get("num_epochs", 50)
        self.config["training"]["num_epochs"] = max_epochs
        history = self.train()
        self.config["training"]["num_epochs"] = original_max_epochs
        return history

    def train_phase2(self, phase1_checkpoint_path=None, max_epochs: int = 30, patience: int = 15):
        if self.phase != 2:
            raise ValueError("train_phase2() can only be called when phase=2")
        if phase1_checkpoint_path is not None and phase1_checkpoint_path.exists():
            print(f"Loading Phase 1 checkpoint from: {phase1_checkpoint_path}")
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
        history = self.train()
        self.config["training"]["num_epochs"] = original_max_epochs
        return history


def _resolve_optional_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    return Path(value)


def run_training(
    model_config_path: Path,
    experiment_config_path: Optional[Path] = None,
    trial_id: Optional[str] = None,
    sweep_id: Optional[str] = None,
    print_parameter_debug: bool = True,
) -> Dict[str, Any]:
    """Run DeepECG-SL Phase-2 training + test eval; return metrics for Optuna / logging."""
    config = load_config(
        model_config_path=model_config_path,
        experiment_config_path=experiment_config_path,
    )
    config.setdefault("runtime", {})
    config["runtime"]["model_config_path"] = str(model_config_path.resolve())
    if experiment_config_path is not None:
        config["runtime"]["experiment_config_path"] = str(experiment_config_path.resolve())
    if trial_id:
        config["runtime"]["trial_id"] = trial_id
    if sweep_id:
        config["runtime"]["sweep_id"] = sweep_id

    print("=" * 60)
    print("Training DeepECG-SL (WCR + Self-Supervised Pretraining) for 24h Dataset")
    print("Task: LOS REGRESSION (continuous prediction in days)")
    print("=" * 60)
    print(f"Model config: {model_config_path}")
    print(f"Experiment config: {experiment_config_path if experiment_config_path else 'None'}")
    print(f"Trial ID: {trial_id if trial_id else 'None'}")
    print(f"Sweep ID: {sweep_id if sweep_id else 'None'}")
    print(f"Model type: {config.get('model', {}).get('type', 'unknown')}")

    model_config = config.get("model", {})
    wcr_config = model_config.get("wcr", {})
    print(f"WCR Model: {wcr_config.get('model_name', 'wcr_77_classes')}")
    print(f"WCR d_model: {wcr_config.get('d_model', 512)}")
    pretrained_config = model_config.get("pretrained", {})
    if pretrained_config.get("enabled", False):
        cache_dir = pretrained_config.get("cache_dir", "data/pretrained_weights/deepecg_sl")
        print(f"Pretrained weights: Enabled (local cache: {cache_dir})")
    else:
        print("Pretrained weights: Disabled (training from scratch)")

    demographic_config = config.get("data", {}).get("demographic_features", {})
    print(
        f"Demographic features: {'Enabled' if demographic_config.get('enabled', False) else 'Disabled'}"
    )
    diagnosis_config = config.get("data", {}).get("diagnosis_features", {})
    print(
        f"Diagnosis features: {'Enabled' if diagnosis_config.get('enabled', False) else 'Disabled'}"
    )
    icu_unit_config = config.get("data", {}).get("icu_unit_features", {})
    print(f"ICU unit features: {icu_unit_config.get('enabled', False)}")
    sofa_cfg = config.get("data", {}).get("sofa_features", {})
    print(f"SOFA features: {'Enabled' if sofa_cfg.get('enabled', False) else 'Disabled'}")
    print("=" * 60)

    icu_mapper = setup_icustays_mapper(config)
    multi_task_config = config.get("multi_task", {})
    if not multi_task_config.get("enabled", False):
        print("Warning: Multi-task is disabled. DeepECG-SL requires multi-task learning.")
        config["multi_task"] = {"enabled": True}

    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        labels=None,
        preprocess=None,
        transform=None,
        icu_mapper=icu_mapper,
        mortality_labels=None,
    )

    print("\nCreating DeepECG-SL model...")
    base_model = DeepECG_SL(config)
    print(f"Model created with {base_model.count_parameters():,} parameters")
    print("Creating Multi-Task model (LOS Regression + Mortality Classification)...")
    model = MultiTaskECGModel(base_model, config)
    print(f"Multi-Task model created with {model.count_parameters():,} parameters")

    if print_parameter_debug:

        def count_component_params(component, name):
            total = sum(p.numel() for p in component.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in component.parameters() if not p.requires_grad)
            return total, frozen

        print("\n" + "=" * 60)
        print("DEBUG: Detailed Parameter Analysis")
        print("=" * 60)
        wcr_trainable, wcr_frozen = count_component_params(base_model.wcr_encoder, "WCR Encoder")
        adapter_trainable, adapter_frozen = count_component_params(base_model.input_adapter, "Input Adapter")
        shared_trainable, _ = count_component_params(base_model.shared_bn, "Shared BN")
        shared_trainable += sum(p.numel() for p in base_model.shared_fc.parameters() if p.requires_grad)
        shared_trainable += sum(
            p.numel() for p in base_model.shared_dropout1.parameters() if p.requires_grad
        )
        shared_trainable += sum(
            p.numel() for p in base_model.shared_dropout2.parameters() if p.requires_grad
        )
        los_trainable, _ = count_component_params(base_model.los_head, "LOS Head")
        mortality_trainable, _ = count_component_params(base_model.mortality_head, "Mortality Head")
        print(f"WCR Encoder:        {wcr_trainable:>15,} trainable, {wcr_frozen:>15,} frozen")
        print(f"Input Adapter:      {adapter_trainable:>15,} trainable, {adapter_frozen:>15,} frozen")
        print(f"Shared Layers:      {shared_trainable:>15,} trainable")
        print(f"LOS Head:           {los_trainable:>15,} trainable")
        print(f"Mortality Head:     {mortality_trainable:>15,} trainable")
        enc_t = sum(1 for p in base_model.wcr_encoder.parameters() if p.requires_grad)
        enc_n = sum(1 for p in base_model.wcr_encoder.parameters())
        print(f"\nWCR Encoder trainable tensors: {enc_t}/{enc_n}")
        print("=" * 60 + "\n")

    criterion = get_multi_task_loss(config)

    job_id = (
        os.getenv("SLURM_JOB_ID")
        or os.getenv("OPTUNA_TRIAL_JOB_ID")
        or "local"
    )
    print(f"Job ID (checkpoints / eval): {job_id}")

    print("\n" + "=" * 60)
    print("PHASE 2: FULL FINE-TUNING")
    print("=" * 60)

    checkpoint_dir = Path(config.get("checkpoint", {}).get("save_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer_phase2 = DeepECG_SL_Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
        phase=2,
    )
    trainer_phase2.config_paths = {"model": str(model_config_path.resolve())}
    if experiment_config_path is not None:
        trainer_phase2.config_paths["experiment"] = str(experiment_config_path.resolve())
    trainer_phase2.job_id = job_id

    training_cfg = config.get("training", {})
    es_cfg = config.get("early_stopping", {})
    phase2_max_epochs = int(training_cfg.get("num_epochs", 30))
    phase2_patience = int(es_cfg.get("patience", 15))
    print(f"Phase 2 budget: num_epochs={phase2_max_epochs}, early_stopping.patience={phase2_patience}")
    history_phase2 = trainer_phase2.train_phase2(
        phase1_checkpoint_path=None,
        max_epochs=phase2_max_epochs,
        patience=phase2_patience,
    )

    val_maes = history_phase2.get("val_los_mae", [])
    val_losses = history_phase2.get("val_loss", [])
    val_r2s = history_phase2.get("val_los_r2", [])
    best_val_mae = min(val_maes) if val_maes else float("inf")
    best_val_loss = min(val_losses) if val_losses else float("inf")
    best_val_r2 = max(val_r2s) if val_r2s else float("-inf")

    print("\nPhase 2 completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation MAE: {best_val_mae:.4f} days")

    phase2_checkpoint_path = checkpoint_dir / f"deepecg_sl_phase2_best_{job_id}.pt"
    if hasattr(trainer_phase2.checkpoint, "best_model_state"):
        torch.save(
            {
                "model_state_dict": trainer_phase2.checkpoint.best_model_state,
                "epoch": trainer_phase2.checkpoint.best_epoch,
                "val_loss": trainer_phase2.checkpoint.best_score,
                "history": history_phase2,
                "config": config,
            },
            phase2_checkpoint_path,
        )
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": trainer_phase2.current_epoch,
                "history": history_phase2,
                "config": config,
            },
            phase2_checkpoint_path,
        )
    print(f"Saved Phase 2 checkpoint to: {phase2_checkpoint_path}")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    if phase2_checkpoint_path.exists():
        checkpoint = torch.load(phase2_checkpoint_path, map_location=trainer_phase2.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(trainer_phase2.device)

    eval_trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
    )
    eval_trainer.job_id = job_id
    history_final = evaluate_and_print_results(eval_trainer, test_loader, history_phase2, config)

    print("\n" + "=" * 60)
    print("FULL FINE-TUNING COMPLETED")
    print("=" * 60)
    print(f"Phase 2 checkpoint: {phase2_checkpoint_path}")

    out: Dict[str, Any] = {
        "best_val_mae": best_val_mae,
        "best_val_loss": best_val_loss,
        "best_val_r2": best_val_r2,
        "phase2_checkpoint_path": str(phase2_checkpoint_path),
        "job_id": job_id,
        "history": history_final,
    }
    if "test_los_mae" in history_final:
        out["test_los_mae"] = history_final["test_los_mae"]
    if "test_los_r2" in history_final:
        out["test_los_r2"] = history_final["test_los_r2"]
    return out


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train DeepECG-SL (WCR) for 24h LOS regression")
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Base model YAML (default: configs/model/deepecg_sl/deepecg_sl.yaml).",
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        default=None,
        help="Optional override YAML merged on top of base config (tuning / Optuna trial).",
    )
    parser.add_argument(
        "--trial-id",
        type=str,
        default=None,
        help="Optional trial id for traceability.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Optional sweep / study id for traceability.",
    )
    parser.add_argument(
        "--no-parameter-debug",
        action="store_true",
        help="Skip detailed parameter-count debug block.",
    )
    args = parser.parse_args()

    model_config_path = _resolve_optional_path(args.model_config or os.getenv("MODEL_CONFIG_PATH"))
    if model_config_path is None:
        model_config_path = Path("configs/model/deepecg_sl/deepecg_sl.yaml")
    experiment_config_path = _resolve_optional_path(
        args.experiment_config or os.getenv("EXPERIMENT_CONFIG_PATH")
    )
    trial_id = args.trial_id or os.getenv("TRIAL_ID")
    sweep_id = args.sweep_id or os.getenv("SWEEP_ID")

    return run_training(
        model_config_path=model_config_path,
        experiment_config_path=experiment_config_path,
        trial_id=trial_id,
        sweep_id=sweep_id,
        print_parameter_debug=not args.no_parameter_debug,
    )


if __name__ == "__main__":
    main()
