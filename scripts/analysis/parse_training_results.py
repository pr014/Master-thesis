#!/usr/bin/env python3
"""
Parse training metrics from SLURM output logs.

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
    test_metrics: Dict[str, float]
    per_class: Dict[int, Dict[str, float]]
    confusion_matrix: Optional[List[List[int]]]
    num_stays_evaluated: Optional[int]
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    training_history: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "best_val_loss": self.best_val_loss,
            "test_metrics": self.test_metrics,
            "num_stays_evaluated": self.num_stays_evaluated,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
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


def _extract_training_metrics(text: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, List[float]]]:
    """
    Extract training and validation metrics from epoch logs.
    
    Looks for lines like:
    "Epoch 1/50 - Train Loss: 3.1234, Train LOS Acc: 0.2345, Val Loss: 2.9876, Val LOS Acc: 0.3456"
    or with mortality:
    "Epoch 1/50 - Train Loss: 3.1234, Train LOS Acc: 0.2345, Val Loss: 2.9876, Val LOS Acc: 0.3456"
    "           Mortality - Val Acc: 0.7654, Val AUC: 0.8234"
    
    Returns:
        Tuple of (final_train_metrics, final_val_metrics, training_history)
    """
    train_metrics: Dict[str, float] = {}
    val_metrics: Dict[str, float] = {}
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_los_acc": [],
        "val_loss": [],
        "val_los_acc": [],
        "val_mortality_acc": [],
        "val_mortality_auc": [],
    }
    
    # Pattern to match epoch log lines
    # Format: "Epoch X/Y - Train Loss: X.XXXX, Train LOS Acc: X.XXXX, Val Loss: X.XXXX, Val LOS Acc: X.XXXX"
    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+-\s+"
        r"Train Loss:\s+(%s),\s+"
        r"Train LOS Acc:\s+(%s),\s+"
        r"Val Loss:\s+(%s),\s+"
        r"Val LOS Acc:\s+(%s)" % (_RE_FLOAT, _RE_FLOAT, _RE_FLOAT, _RE_FLOAT),
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
        # Match epoch line
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            train_loss = float(epoch_match.group(3))
            train_acc = float(epoch_match.group(4))
            val_loss = float(epoch_match.group(5))
            val_acc = float(epoch_match.group(6))
            
            # Store in history
            history["train_loss"].append(train_loss)
            history["train_los_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_los_acc"].append(val_acc)
            
            # Update final metrics (last epoch wins)
            train_metrics["train_loss"] = train_loss
            train_metrics["train_los_acc"] = train_acc
            val_metrics["val_loss"] = val_loss
            val_metrics["val_los_acc"] = val_acc
            
            # Check next line for mortality metrics
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                mort_match = mortality_pattern.search(next_line)
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

    We support two formats seen in logs:
    1) Legacy:
       Test Set Results:
         Test LOS Loss: ...
         ...
         Confusion Matrix:
    2) New summary:
       📊 TRAINING RESULTS SUMMARY
       🔹 Model Performance:
         ...
       🔹 Mortality (overall):
         ...
       🔹 Confusion Matrix:
         ...
       ✅ Checkpoints: ...
    """
    # Format 1: explicit "Test Set Results:" section
    m = re.search(r"(?ms)^Test Set Results:\n(.*?)(?:^\S|^End time:)", text)
    if m:
        block = m.group(0)
        start = block.find("Test Set Results:")
        return block[start:] if start >= 0 else block

    # Format 2: summary section
    m = re.search(r"(?ms)^=+\n📊 TRAINING RESULTS SUMMARY\n=+\n(.*?)(?:^=+\n✅ Checkpoints:|^✅ Checkpoints:|^End time:)", text)
    if m:
        return m.group(0)

    return None


def _parse_key_metrics(block: str) -> Tuple[Dict[str, float], Optional[int]]:
    metrics: Dict[str, float] = {}
    num_stays: Optional[int] = None

    # Key metrics of interest (keep names stable)
    patterns = {
        "test_los_loss": r"^\s*Test LOS Loss:\s*(%s)\s*$" % _RE_FLOAT,
        "test_los_accuracy": r"^\s*Test LOS Accuracy:\s*(%s)\s*\(" % _RE_FLOAT,
        # Support multiple formats: "LOS Balanced Accuracy:" or "LOS Balanced Acc:" or "Balanced Accuracy:"
        "los_balanced_accuracy": r"^\s*(?:LOS\s+)?Balanced\s+Acc(?:uracy)?:\s*(%s)\s*\(" % _RE_FLOAT,
        "los_macro_precision": r"^\s*LOS Macro Precision:\s*(%s)\s*$" % _RE_FLOAT,
        "los_macro_recall": r"^\s*LOS Macro Recall:\s*(%s)\s*$" % _RE_FLOAT,
        "los_macro_f1": r"^\s*LOS Macro F1-Score:\s*(%s)\s*$" % _RE_FLOAT,
        # Mortality (overall) if multi-task results are printed in the log
        "mortality_accuracy": r"^\s*Accuracy:\s*(%s)\s*$" % _RE_FLOAT,
        "mortality_precision": r"^\s*Precision:\s*(%s)\s*$" % _RE_FLOAT,
        "mortality_recall": r"^\s*Recall:\s*(%s)\s*$" % _RE_FLOAT,
        "mortality_f1": r"^\s*F1:\s*(%s)\s*$" % _RE_FLOAT,
        "mortality_auc": r"^\s*AUC:\s*(%s)\s*$" % _RE_FLOAT,
    }

    for key, pat in patterns.items():
        m = re.search(pat, block, flags=re.MULTILINE)
        if m:
            metrics[key] = float(m.group(1))

    m = re.search(r"^\s*Number of ICU stays evaluated:\s*(\d+)\s*$", block, flags=re.MULTILINE)
    if m:
        num_stays = int(m.group(1))

    return metrics, num_stays


def _parse_per_class(block: str) -> Dict[int, Dict[str, float]]:
    # Lines like: "    1        0.3218       0.8164       0.4616"
    # Support both single-digit and multi-digit class indices
    # IMPORTANT: Only match lines where the third value (F1) is between 0 and 1
    # This avoids matching lines with Support values (large integers)
    per_class: Dict[int, Dict[str, float]] = {}
    for m in re.finditer(
        r"(?m)^\s*(\d+)\s+(%s)\s+(%s)\s+(%s)\s*$" % (_RE_FLOAT, _RE_FLOAT, _RE_FLOAT),
        block,
    ):
        cls = int(m.group(1))
        precision = float(m.group(2))
        recall = float(m.group(3))
        f1 = float(m.group(4))
        
        # Only accept if F1 is between 0 and 1 (valid F1-score range)
        # This filters out Support values which are large integers
        if 0.0 <= f1 <= 1.0:
            per_class[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
    return per_class


def _parse_confusion_matrix(block: str) -> Optional[List[List[int]]]:
    # Dynamically detect number of classes from header row
    if "Confusion Matrix:" not in block:
        return None

    # Find the section after "Confusion Matrix:"
    cm_section = block[block.find("Confusion Matrix:"):]
    lines = cm_section.splitlines()
    
    # Find header row - look for line with numbers starting from 0
    num_classes = None
    header_line_idx = None
    
    for i, line in enumerate(lines):
        if "Confusion Matrix" in line:
            continue
        # Extract all numbers from line
        numbers = re.findall(r'\d+', line)
        if numbers:
            try:
                parsed_nums = [int(n) for n in numbers]
                if len(parsed_nums) > 0 and parsed_nums[0] == 0:
                    # Check if consecutive starting from 0
                    if parsed_nums == list(range(len(parsed_nums))):
                        num_classes = len(parsed_nums)
                        header_line_idx = i
                        break
            except (ValueError, IndexError):
                continue
    
    if num_classes is None or header_line_idx is None:
        return None
    
    # Parse rows after header
    rows: List[List[int]] = []
    for i in range(header_line_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        
        # Extract all numbers from line
        numbers = re.findall(r'\d+', line)
        if not numbers:
            # Stop if we hit a non-number line and already have rows
            if rows:
                break
            continue
        
        try:
            parsed_nums = [int(n) for n in numbers]
            # First number is class index, rest are confusion matrix values
            if len(parsed_nums) == num_classes + 1:
                # Verify class index matches row number
                cls_idx = parsed_nums[0]
                if cls_idx == len(rows):
                    vals = parsed_nums[1:]
                    rows.append(vals)
                    # Stop when we have all rows
                    if len(rows) == num_classes:
                        break
        except (ValueError, IndexError):
            # Stop if parsing fails and we already have rows
            if rows:
                break
            continue

    # Return if we got the expected number of rows
    return rows if len(rows) == num_classes else None


def parse_log_text(text: str) -> ParsedResults:
    job_id = _extract_job_id(text)
    best_val_loss = _extract_best_val_loss(text)

    test_block = _extract_test_block(text)
    test_metrics: Dict[str, float] = {}
    per_class: Dict[int, Dict[str, float]] = {}
    cm: Optional[List[List[int]]] = None
    num_stays: Optional[int] = None

    if test_block:
        test_metrics, num_stays = _parse_key_metrics(test_block)
        per_class = _parse_per_class(test_block)
        cm = _parse_confusion_matrix(test_block)
    
    # Extract training/validation metrics from epoch logs
    train_metrics, val_metrics, training_history = _extract_training_metrics(text)

    return ParsedResults(
        job_id=job_id,
        best_val_loss=best_val_loss,
        test_metrics=test_metrics,
        per_class=per_class,
        confusion_matrix=cm,
        num_stays_evaluated=num_stays,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        training_history=training_history,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse training results from SLURM log.")
    parser.add_argument("--job", type=int, default=None, help="SLURM job id (uses outputs/logs/slurm_<job>.out)")
    parser.add_argument("--log", type=str, default=None, help="Path to slurm .out log file")
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    parser.add_argument("--cm", action="store_true", help="Print confusion matrix if present")
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
    
    # Print test metrics first (needed for overfitting analysis)
    if parsed.test_metrics:
        print("\n🔹 Test Metrics:")
        # Print LOS metrics first, then mortality metrics (preserve logical order)
        los_metric_keys = [k for k in parsed.test_metrics.keys() if k.startswith('los_') or k.startswith('test_los_')]
        mortality_metric_keys = [k for k in parsed.test_metrics.keys() if k.startswith('mortality_')]
        other_metric_keys = [k for k in parsed.test_metrics.keys() if k not in los_metric_keys and k not in mortality_metric_keys]
        
        # Print in order: LOS metrics, then mortality, then others
        for key_list in [los_metric_keys, mortality_metric_keys, other_metric_keys]:
            for k in sorted(key_list):
                print(f"  {k}: {parsed.test_metrics[k]:.4f}")
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
                print(f"  {k}: {v:.4f}")
        else:
            print("\n📊 Final Training Metrics: <not found in log>")
        
        if parsed.val_metrics:
            print("\n📊 Final Validation Metrics (last epoch):")
            for k, v in sorted(parsed.val_metrics.items()):
                print(f"  {k}: {v:.4f}")
        else:
            print("\n📊 Final Validation Metrics: <not found in log>")
        
        # Overfitting indicators
        print("\n🔬 Overfitting Indicators:")
        
        # 1. Train vs Val Loss Gap
        if parsed.train_metrics and parsed.val_metrics:
            train_loss = parsed.train_metrics.get("train_loss")
            val_loss = parsed.val_metrics.get("val_loss")
            train_acc = parsed.train_metrics.get("train_los_acc")
            val_acc = parsed.val_metrics.get("val_los_acc")
            
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
            
            # 2. Train vs Val Accuracy Gap
            if train_acc is not None and val_acc is not None:
                acc_gap = train_acc - val_acc
                print(f"  Accuracy Gap (Train - Val): {acc_gap:.4f}")
                if acc_gap > 0.15:
                    print("    ⚠️  WARNING: Large accuracy gap (Train >> Val) suggests overfitting")
                elif acc_gap > 0.10:
                    print("    ⚠️  CAUTION: Moderate accuracy gap (possible overfitting)")
                elif acc_gap < -0.05:
                    print("    ℹ️  Val accuracy > Train (unusual, check data splits)")
                else:
                    print("    ✅ Accuracy gap is reasonable")
        
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
            test_acc = parsed.test_metrics.get("test_los_accuracy")
            val_acc = parsed.val_metrics.get("val_los_acc")
            
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
            
            if test_acc is not None and val_acc is not None:
                test_val_acc_gap = test_acc - val_acc
                print(f"  Test vs Val Accuracy Gap: {test_val_acc_gap:.4f}")
                if test_val_acc_gap < -0.1:
                    print("    ⚠️  WARNING: Test accuracy << Val accuracy (poor generalization)")
                elif test_val_acc_gap < -0.05:
                    print("    ⚠️  CAUTION: Test accuracy < Val accuracy (possible overfitting)")
                elif test_val_acc_gap > 0.1:
                    print("    ℹ️  Test accuracy > Val accuracy (unusual, check data splits)")
                else:
                    print("    ✅ Test and Val accuracy are similar (good generalization)")
        
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
    else:
        print("\n⚠️  Cannot analyze overfitting: Train/Val metrics not found in log")
        print("   (Metrics will be available for jobs run after the trainer.py update)")

    if parsed.per_class:
        print("\n🔹 Per-Class Metrics (precision/recall/f1):")
        for cls in sorted(parsed.per_class.keys()):
            m = parsed.per_class[cls]
            print(f"  {cls}: p={m['precision']:.4f} r={m['recall']:.4f} f1={m['f1']:.4f}")

    # Always print confusion matrix (unless explicitly disabled)
    if parsed.confusion_matrix is not None:
        print("\n🔹 Confusion Matrix:")
        num_classes = len(parsed.confusion_matrix)
        # Print header
        print("   " + " ".join([f"{i:>6}" for i in range(num_classes)]))
        # Print rows
        for i, row in enumerate(parsed.confusion_matrix):
            row_str = f"   {i} " + " ".join([f"{val:>6}" for val in row])
            print(row_str)
    else:
        print("\n🔹 Confusion Matrix: <not found in log>")


if __name__ == "__main__":
    main()


