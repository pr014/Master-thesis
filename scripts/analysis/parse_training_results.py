#!/usr/bin/env python3
"""
Parse training metrics from SLURM output logs.

Supports LOS REGRESSION task with metrics: MAE, RMSE, R², Median AE, Percentile Errors.

Usage:
  python scripts/analysis/parse_training_results.py --job 3010998
  python scripts/analysis/parse_training_results.py --log outputs/logs/slurm_3010998.out
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_RE_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass
class ParsedResults:
    job_id: Optional[int]
    best_val_loss: Optional[float]
    best_val_mae: Optional[float]
    best_val_r2: Optional[float]
    test_metrics: Dict[str, float]
    num_stays_evaluated: Optional[int]
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    training_history: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "best_val_loss": self.best_val_loss,
            "best_val_mae": self.best_val_mae,
            "best_val_r2": self.best_val_r2,
            "test_metrics": self.test_metrics,
            "num_stays_evaluated": self.num_stays_evaluated,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "training_history": self.training_history,
        }


def _extract_job_id(text: str) -> Optional[int]:
    m = re.search(r"\bJob ID:\s*(\d+)\b", text)
    return int(m.group(1)) if m else None


def _extract_best_val_loss(text: str) -> Optional[float]:
    m = re.search(r"\bBest validation loss:\s*(%s)\b" % _RE_FLOAT, text)
    return float(m.group(1)) if m else None


def _extract_best_val_mae(text: str) -> Optional[float]:
    m = re.search(r"\bBest Validation MAE:\s*(%s)\s*days?\b" % _RE_FLOAT, text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _extract_best_val_r2(text: str) -> Optional[float]:
    m = re.search(r"\bBest Validation R²:\s*(%s)\b" % _RE_FLOAT, text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _extract_training_metrics(text: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, List[float]]]:
    """
    Extract training and validation metrics from epoch logs.
    
    Looks for lines like:
    "Epoch 1/50 - Train Loss: 3.1234, Train LOS MAE: 2.3456, Train LOS RMSE: 3.1234, Train LOS R²: 0.5678"
    "           Val Loss: 2.9876, Val LOS MAE: 2.1234, Val LOS RMSE: 2.9876, Val LOS R²: 0.6789"
    or with mortality:
    "           Mortality - Val Acc: 0.7654, Val AUC: 0.8234"
    
    Returns:
        Tuple of (final_train_metrics, final_val_metrics, training_history)
    """
    train_metrics: Dict[str, float] = {}
    val_metrics: Dict[str, float] = {}
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_los_mae": [],
        "train_los_rmse": [],
        "train_los_r2": [],
        "val_loss": [],
        "val_los_mae": [],
        "val_los_rmse": [],
        "val_los_r2": [],
        "val_mortality_acc": [],
        "val_mortality_auc": [],
    }
    
    # Pattern to match epoch log lines (two-line format)
    # Line 1: "Epoch X/Y - Train Loss: X.XXXX, Train LOS MAE: X.XXXX, Train LOS RMSE: X.XXXX, Train LOS R²: X.XXXX"
    # Line 2: "           Val Loss: X.XXXX, Val LOS MAE: X.XXXX, Val LOS RMSE: X.XXXX, Val LOS R²: X.XXXX"
    epoch_train_pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+-\s+"
        r"Train Loss:\s+(%s),\s+"
        r"Train LOS MAE:\s+(%s),\s+"
        r"Train LOS RMSE:\s+(%s),\s+"
        r"Train LOS R²:\s+(%s)" % (_RE_FLOAT, _RE_FLOAT, _RE_FLOAT, _RE_FLOAT),
        re.IGNORECASE
    )
    
    epoch_val_pattern = re.compile(
        r"^\s+Val Loss:\s+(%s),\s+"
        r"Val LOS MAE:\s+(%s),\s+"
        r"Val LOS RMSE:\s+(%s),\s+"
        r"Val LOS R²:\s+(%s)" % (_RE_FLOAT, _RE_FLOAT, _RE_FLOAT, _RE_FLOAT),
        re.IGNORECASE
    )
    
    # Pattern for mortality metrics (optional, on next line)
    mortality_pattern = re.compile(
        r"Mortality\s*-\s*"
        r"Val Acc:\s+(%s),\s+"
        r"Val AUC:\s+(%s)" % (_RE_FLOAT, _RE_FLOAT),
        re.IGNORECASE
    )
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        # Match train metrics line
        train_match = epoch_train_pattern.search(line)
        if train_match:
            epoch_num = int(train_match.group(1))
            train_loss = float(train_match.group(3))
            train_mae = float(train_match.group(4))
            train_rmse = float(train_match.group(5))
            train_r2 = float(train_match.group(6))
            
            # Store in history
            history["train_loss"].append(train_loss)
            history["train_los_mae"].append(train_mae)
            history["train_los_rmse"].append(train_rmse)
            history["train_los_r2"].append(train_r2)
            
            # Update final metrics (last epoch wins)
            train_metrics["train_loss"] = train_loss
            train_metrics["train_los_mae"] = train_mae
            train_metrics["train_los_rmse"] = train_rmse
            train_metrics["train_los_r2"] = train_r2
            
            # Check next line for validation metrics
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                val_match = epoch_val_pattern.search(next_line)
                if val_match:
                    val_loss = float(val_match.group(1))
                    val_mae = float(val_match.group(2))
                    val_rmse = float(val_match.group(3))
                    val_r2 = float(val_match.group(4))
                    
                    history["val_loss"].append(val_loss)
                    history["val_los_mae"].append(val_mae)
                    history["val_los_rmse"].append(val_rmse)
                    history["val_los_r2"].append(val_r2)
                    
                    val_metrics["val_loss"] = val_loss
                    val_metrics["val_los_mae"] = val_mae
                    val_metrics["val_los_rmse"] = val_rmse
                    val_metrics["val_los_r2"] = val_r2
                    
                    # Check line after that for mortality metrics
                    if i + 2 < len(lines):
                        mort_line = lines[i + 2]
                        mort_match = mortality_pattern.search(mort_line)
                        if mort_match:
                            val_mort_acc = float(mort_match.group(1))
                            val_mort_auc = float(mort_match.group(2))
                            history["val_mortality_acc"].append(val_mort_acc)
                            history["val_mortality_auc"].append(val_mort_auc)
                            val_metrics["val_mortality_acc"] = val_mort_acc
                            val_metrics["val_mortality_auc"] = val_mort_auc
    
    return train_metrics, val_metrics, history


def _extract_test_block(text: str) -> Optional[str]:
    """
    Extract the block that contains final test metrics.

    We support the summary format:
       📊 TRAINING RESULTS SUMMARY (LOS REGRESSION)
       🔹 LOS Regression Performance:
         ...
       🔹 Error Percentiles:
         ...
       🔹 Mortality (Binary Classification):
         ...
       ✅ Checkpoints: ...
    """
    # Summary section for regression
    m = re.search(r"(?ms)^=+\n📊 TRAINING RESULTS SUMMARY.*?LOS REGRESSION.*?\n=+\n(.*?)(?:^=+\n✅ Checkpoints:|^✅ Checkpoints:|^End time:)", text)
    if m:
        return m.group(0)
    
    # Fallback: try without "LOS REGRESSION" in title
    m = re.search(r"(?ms)^=+\n📊 TRAINING RESULTS SUMMARY\n=+\n(.*?)(?:^=+\n✅ Checkpoints:|^✅ Checkpoints:|^End time:)", text)
    if m:
        return m.group(0)

    return None


def _parse_key_metrics(block: str) -> Tuple[Dict[str, float], Optional[int]]:
    metrics: Dict[str, float] = {}
    num_stays: Optional[int] = None

    # Key metrics for LOS REGRESSION
    patterns = {
        "test_los_loss": r"^\s*Test LOS Loss:\s*(%s)\s*$" % _RE_FLOAT,
        "test_los_mae": r"^\s*Test LOS MAE:\s*(%s)\s*days?\s*$" % _RE_FLOAT,
        "test_los_rmse": r"^\s*Test LOS RMSE:\s*(%s)\s*days?\s*$" % _RE_FLOAT,
        "test_los_r2": r"^\s*Test LOS R²:\s*(%s)\s*$" % _RE_FLOAT,
        "test_los_median_ae": r"^\s*Test LOS Median AE:\s*(%s)\s*days?\s*$" % _RE_FLOAT,
        "test_los_p25_error": r"^\s*25th percentile:\s*(%s)\s*days?\s*$" % _RE_FLOAT,
        "test_los_p50_error": r"^\s*50th percentile:\s*(%s)\s*days?\s*\(median\)\s*$" % _RE_FLOAT,
        "test_los_p75_error": r"^\s*75th percentile:\s*(%s)\s*days?\s*$" % _RE_FLOAT,
        "test_los_p90_error": r"^\s*90th percentile:\s*(%s)\s*days?\s*$" % _RE_FLOAT,
        # Mortality (overall) if multi-task results are printed in the log
        "mortality_accuracy": r"^\s*Accuracy:\s*(%s)\s*$" % _RE_FLOAT,
        "mortality_precision": r"^\s*Precision:\s*(%s)\s*$" % _RE_FLOAT,
        "mortality_recall": r"^\s*Recall:\s*(%s)\s*$" % _RE_FLOAT,
        "mortality_f1": r"^\s*F1:\s*(%s)\s*$" % _RE_FLOAT,
        "mortality_auc": r"^\s*AUC:\s*(%s)\s*$" % _RE_FLOAT,
    }

    for key, pat in patterns.items():
        m = re.search(pat, block, flags=re.MULTILINE | re.IGNORECASE)
        if m:
            metrics[key] = float(m.group(1))

    # Extract number of stays
    m = re.search(r"^\s*Test ICU Stays:\s*(\d+)\s*$", block, flags=re.MULTILINE)
    if m:
        num_stays = int(m.group(1))
    else:
        # Try alternative format
        m = re.search(r"^\s*Test ICU Stays:\s*(\d+)\s*$", block, flags=re.MULTILINE)
        if m:
            num_stays = int(m.group(1))

    return metrics, num_stays


def parse_log_text(text: str) -> ParsedResults:
    job_id = _extract_job_id(text)
    best_val_loss = _extract_best_val_loss(text)
    best_val_mae = _extract_best_val_mae(text)
    best_val_r2 = _extract_best_val_r2(text)

    test_block = _extract_test_block(text)
    test_metrics: Dict[str, float] = {}
    num_stays: Optional[int] = None

    if test_block:
        test_metrics, num_stays = _parse_key_metrics(test_block)
    
    # Extract training/validation metrics from epoch logs
    train_metrics, val_metrics, training_history = _extract_training_metrics(text)

    return ParsedResults(
        job_id=job_id,
        best_val_loss=best_val_loss,
        best_val_mae=best_val_mae,
        best_val_r2=best_val_r2,
        test_metrics=test_metrics,
        num_stays_evaluated=num_stays,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        training_history=training_history,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse training results from SLURM log (LOS REGRESSION).")
    parser.add_argument("--job", type=int, default=None, help="SLURM job id (uses outputs/logs/slurm_<job>.out)")
    parser.add_argument("--log", type=str, default=None, help="Path to slurm .out log file")
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    args = parser.parse_args()

    if not args.log and not args.job:
        raise SystemExit("Provide --job <id> or --log <path>")

    if args.log:
        log_path = Path(args.log)
    else:
        log_path = Path("outputs/logs") / f"slurm_{args.job}.out"

    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_log_text(text)

    if args.json:
        print(json.dumps(parsed.to_dict(), indent=2, sort_keys=True))
        return

    print(f"job_id: {parsed.job_id}")
    print(f"best_val_loss: {parsed.best_val_loss}")
    if parsed.best_val_mae is not None:
        print(f"best_val_mae: {parsed.best_val_mae:.4f} days")
    if parsed.best_val_r2 is not None:
        print(f"best_val_r2: {parsed.best_val_r2:.4f}")
    
    # Print test metrics first (needed for overfitting analysis)
    if parsed.test_metrics:
        print("\n🔹 Test Metrics (LOS Regression):")
        # Print LOS metrics first, then mortality metrics (preserve logical order)
        los_metric_keys = [k for k in parsed.test_metrics.keys() if k.startswith('los_') or k.startswith('test_los_')]
        mortality_metric_keys = [k for k in parsed.test_metrics.keys() if k.startswith('mortality_')]
        other_metric_keys = [k for k in parsed.test_metrics.keys() if k not in los_metric_keys and k not in mortality_metric_keys]
        
        # Print in order: LOS metrics, then mortality, then others
        for key_list in [los_metric_keys, mortality_metric_keys, other_metric_keys]:
            for k in sorted(key_list):
                value = parsed.test_metrics[k]
                # Add unit for MAE/RMSE/Median AE/Percentile errors
                if any(x in k for x in ['mae', 'rmse', 'median_ae', 'p25', 'p50', 'p75', 'p90']):
                    print(f"  {k}: {value:.4f} days")
                else:
                    print(f"  {k}: {value:.4f}")
        print(f"num_stays_evaluated: {parsed.num_stays_evaluated}")
    else:
        print("\n🔹 Test Metrics: <not found in log>")
    
    # Overfitting Analysis Section
    print("\n" + "="*80)
    print("🔍 OVERFITTING ANALYSIS")
    print("="*80)
    
    if parsed.train_metrics or parsed.val_metrics:
        # Final epoch metrics
        if parsed.train_metrics:
            print("\n📊 Final Training Metrics (last epoch):")
            for k, v in sorted(parsed.train_metrics.items()):
                if any(x in k for x in ['mae', 'rmse', 'median_ae']):
                    print(f"  {k}: {v:.4f} days")
                else:
                    print(f"  {k}: {v:.4f}")
        else:
            print("\n📊 Final Training Metrics: <not found in log>")
        
        if parsed.val_metrics:
            print("\n📊 Final Validation Metrics (last epoch):")
            for k, v in sorted(parsed.val_metrics.items()):
                if any(x in k for x in ['mae', 'rmse', 'median_ae']):
                    print(f"  {k}: {v:.4f} days")
                else:
                    print(f"  {k}: {v:.4f}")
        else:
            print("\n📊 Final Validation Metrics: <not found in log>")
        
        # Overfitting indicators
        print("\n🔬 Overfitting Indicators:")
        
        # 1. Train vs Val Loss Gap
        if parsed.train_metrics and parsed.val_metrics:
            train_loss = parsed.train_metrics.get("train_loss")
            val_loss = parsed.val_metrics.get("val_loss")
            train_mae = parsed.train_metrics.get("train_los_mae")
            val_mae = parsed.val_metrics.get("val_los_mae")
            
            if train_loss is not None and val_loss is not None:
                loss_gap = train_loss - val_loss
                print(f"  Loss Gap (Train - Val): {loss_gap:.4f}")
                if loss_gap < -0.1:
                    print("    ⚠️  WARNING: Val loss >> Train loss (strong overfitting signal)")
                elif loss_gap < -0.05:
                    print("    ⚠️  CAUTION: Val loss > Train loss (possible overfitting)")
                elif loss_gap > 0.5:
                    print("    ℹ️  Train loss > Val loss (normal, model still learning)")
                else:
                    print("    ✅ Loss gap is reasonable")
            
            # 2. Train vs Val MAE Gap (for regression)
            if train_mae is not None and val_mae is not None:
                mae_gap = train_mae - val_mae
                print(f"  MAE Gap (Train - Val): {mae_gap:.4f} days")
                if mae_gap < -0.5:
                    print("    ⚠️  WARNING: Val MAE >> Train MAE (strong overfitting signal)")
                elif mae_gap < -0.2:
                    print("    ⚠️  CAUTION: Val MAE > Train MAE (possible overfitting)")
                elif mae_gap > 1.0:
                    print("    ℹ️  Train MAE > Val MAE (normal, model still learning)")
                else:
                    print("    ✅ MAE gap is reasonable")
        
        # 3. Best Val Loss vs Final Val Loss
        if parsed.best_val_loss is not None and parsed.val_metrics:
            final_val_loss = parsed.val_metrics.get("val_loss")
            if final_val_loss is not None:
                val_loss_deterioration = final_val_loss - parsed.best_val_loss
                print(f"  Val Loss Deterioration (Final - Best): {val_loss_deterioration:.4f}")
                print(f"    Best Val Loss: {parsed.best_val_loss:.4f}")
                print(f"    Final Val Loss: {final_val_loss:.4f}")
                if val_loss_deterioration > 0.1:
                    print("    ⚠️  WARNING: Val loss increased significantly after best (overfitting)")
                elif val_loss_deterioration > 0.05:
                    print("    ⚠️  CAUTION: Val loss increased after best (possible overfitting)")
                elif val_loss_deterioration < 0:
                    print("    ✅ Final val loss improved from best (good)")
                else:
                    print("    ✅ Val loss stable")
        
        # 4. Test vs Val Comparison
        if parsed.test_metrics and parsed.val_metrics:
            test_loss = parsed.test_metrics.get("test_los_loss")
            val_loss = parsed.val_metrics.get("val_loss")
            test_mae = parsed.test_metrics.get("test_los_mae")
            val_mae = parsed.val_metrics.get("val_los_mae")
            test_r2 = parsed.test_metrics.get("test_los_r2")
            val_r2 = parsed.val_metrics.get("val_los_r2")
            
            if test_loss is not None and val_loss is not None:
                test_val_loss_gap = test_loss - val_loss
                print(f"  Test vs Val Loss Gap: {test_val_loss_gap:.4f}")
                if test_val_loss_gap > 0.2:
                    print("    ⚠️  WARNING: Test loss >> Val loss (poor generalization)")
                elif test_val_loss_gap > 0.1:
                    print("    ⚠️  CAUTION: Test loss > Val loss (possible overfitting)")
                elif test_val_loss_gap < -0.1:
                    print("    ℹ️  Test loss < Val loss (unusual, check data splits)")
                else:
                    print("    ✅ Test and Val loss are similar (good generalization)")
            
            if test_mae is not None and val_mae is not None:
                test_val_mae_gap = test_mae - val_mae
                print(f"  Test vs Val MAE Gap: {test_val_mae_gap:.4f} days")
                if test_val_mae_gap > 0.5:
                    print("    ⚠️  WARNING: Test MAE >> Val MAE (poor generalization)")
                elif test_val_mae_gap > 0.2:
                    print("    ⚠️  CAUTION: Test MAE > Val MAE (possible overfitting)")
                elif test_val_mae_gap < -0.2:
                    print("    ℹ️  Test MAE < Val MAE (unusual, check data splits)")
                else:
                    print("    ✅ Test and Val MAE are similar (good generalization)")
            
            if test_r2 is not None and val_r2 is not None:
                test_val_r2_gap = test_r2 - val_r2
                print(f"  Test vs Val R² Gap: {test_val_r2_gap:.4f}")
                if test_val_r2_gap < -0.1:
                    print("    ⚠️  WARNING: Test R² << Val R² (poor generalization)")
                elif test_val_r2_gap < -0.05:
                    print("    ⚠️  CAUTION: Test R² < Val R² (possible overfitting)")
                elif test_val_r2_gap > 0.1:
                    print("    ℹ️  Test R² > Val R² (unusual, check data splits)")
                else:
                    print("    ✅ Test and Val R² are similar (good generalization)")
        
        # 5. Training History Summary (if available)
        if parsed.training_history:
            history = parsed.training_history
            if history.get("train_loss") and history.get("val_loss"):
                train_losses = history["train_loss"]
                val_losses = history["val_loss"]
                if len(train_losses) > 1 and len(val_losses) > 1:
                    # Check if val loss is increasing in last epochs (overfitting signal)
                    last_n = min(5, len(val_losses))
                    if last_n > 1:
                        recent_val_losses = val_losses[-last_n:]
                        val_loss_trend = recent_val_losses[-1] - recent_val_losses[0]
                        print(f"\n  Training History (last {last_n} epochs):")
                        print(f"    Val Loss Trend: {val_loss_trend:+.4f}")
                        if val_loss_trend > 0.05:
                            print("    ⚠️  WARNING: Val loss increasing in recent epochs (overfitting)")
                        elif val_loss_trend > 0.02:
                            print("    ⚠️  CAUTION: Val loss slightly increasing (watch for overfitting)")
                        else:
                            print("    ✅ Val loss stable or decreasing")
                        
                        # Also check MAE trend
                        if history.get("val_los_mae") and len(history["val_los_mae"]) >= last_n:
                            recent_val_maes = history["val_los_mae"][-last_n:]
                            val_mae_trend = recent_val_maes[-1] - recent_val_maes[0]
                            print(f"    Val MAE Trend: {val_mae_trend:+.4f} days")
                            if val_mae_trend > 0.2:
                                print("    ⚠️  WARNING: Val MAE increasing in recent epochs (overfitting)")
                            elif val_mae_trend > 0.1:
                                print("    ⚠️  CAUTION: Val MAE slightly increasing (watch for overfitting)")
                            else:
                                print("    ✅ Val MAE stable or decreasing")
    else:
        print("\n⚠️  Cannot analyze overfitting: Train/Val metrics not found in log")
        print("   (Metrics will be available for jobs run after the trainer.py update)")


if __name__ == "__main__":
    main()
