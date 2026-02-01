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
from dataclasses import dataclass
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

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "best_val_loss": self.best_val_loss,
            "test_metrics": self.test_metrics,
            "num_stays_evaluated": self.num_stays_evaluated,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


def _extract_job_id(text: str) -> Optional[int]:
    m = re.search(r"\bJob ID:\s*(\d+)\b", text)
    return int(m.group(1)) if m else None


def _extract_best_val_loss(text: str) -> Optional[float]:
    m = re.search(r"\bBest validation loss:\s*(%s)\b" % _RE_FLOAT, text)
    return float(m.group(1)) if m else None


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

    return ParsedResults(
        job_id=job_id,
        best_val_loss=best_val_loss,
        test_metrics=test_metrics,
        per_class=per_class,
        confusion_matrix=cm,
        num_stays_evaluated=num_stays,
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
    if not parsed.test_metrics:
        print("test_metrics: <not found in log>")
        return

    print("test_metrics:")
    for k, v in parsed.test_metrics.items():
        print(f"  {k}: {v}")
    print(f"num_stays_evaluated: {parsed.num_stays_evaluated}")

    if parsed.per_class:
        print("per_class (precision/recall/f1):")
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


