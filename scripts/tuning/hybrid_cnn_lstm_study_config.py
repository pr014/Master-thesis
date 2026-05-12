"""
Shared Optuna study configuration for Hybrid CNN-LSTM HPO.

Used by:
  - scripts/tuning/optuna_hybrid_cnn_lstm_worker.py
  - scripts/cluster/icu_24h/hybrid_cnn_lstm/launch_optuna_hybrid_workers.sh (precreate + schema)

Single objective: minimize best validation **total loss** (multi-task combined val_loss,
consistent with fixed los/mortality loss weights in optuna_base). TPESampler + MedianPruner.

Repro / overrides via environment:
  OPTUNA_TPE_SEED
  OPTUNA_MEDIAN_N_STARTUP_TRIALS
  OPTUNA_MEDIAN_N_WARMUP_STEPS
"""
from __future__ import annotations

import os

import optuna
from optuna import pruners
from optuna.samplers import TPESampler


def tpe_seed() -> int:
    return int(os.getenv("OPTUNA_TPE_SEED", "42"))


def median_n_startup_trials() -> int:
    return int(os.getenv("OPTUNA_MEDIAN_N_STARTUP_TRIALS", "5"))


def median_n_warmup_steps() -> int:
    return int(os.getenv("OPTUNA_MEDIAN_N_WARMUP_STEPS", "0"))


def make_sampler() -> TPESampler:
    return TPESampler(seed=tpe_seed())


def make_pruner() -> pruners.MedianPruner:
    return pruners.MedianPruner(
        n_startup_trials=median_n_startup_trials(),
        n_warmup_steps=median_n_warmup_steps(),
    )


def study_config_summary() -> str:
    return (
        f"TPESampler(seed={tpe_seed()}), "
        f"direction=minimize(best_val_loss / val_loss), "
        f"MedianPruner(n_startup_trials={median_n_startup_trials()}, "
        f"n_warmup_steps={median_n_warmup_steps()})"
    )


def create_hybrid_study(*, study_name: str, storage: str) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=make_sampler(),
        pruner=make_pruner(),
    )
