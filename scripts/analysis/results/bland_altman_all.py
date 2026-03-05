#!/usr/bin/env python3
"""Generate Bland-Altman plots for all 6 model architectures (best job per arch).

Uses best job IDs from docs/results/model_overview.csv (lowest MAE per architecture).
Output: outputs/bland_altman/bland_altman_job_<id>.png (one per architecture).

Usage
-----
python scripts/analysis/results/bland_altman_all.py
python scripts/analysis/results/bland_altman_all.py --out-dir outputs/analysis/bland_altman
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Non-interactive backend for batch plotting
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Best job ID per architecture (from model_overview.csv, lowest MAE)
# ---------------------------------------------------------------------------
BEST_JOBS = {
    "CNN Scratch": "3119696",
    "LSTM Uni": "3346712",
    "LSTM Bi": "3346713",
    "Hybrid CNN-LSTM": "3362967",
    "DeepECG-SL": "3346715",
    "HuBERT-ECG": "3221340",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bland-Altman plots for all 6 architectures (best job each)."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/bland_altman)",
    )
    parser.add_argument(
        "--archs",
        nargs="+",
        default=None,
        help="Restrict to architectures, e.g. --archs 'Hybrid CNN-LSTM' 'DeepECG-SL'",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    out_dir = Path(args.out_dir) if args.out_dir else project_root / "outputs" / "bland_altman"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs_to_run = BEST_JOBS
    if args.archs:
        jobs_to_run = {k: v for k, v in BEST_JOBS.items() if k in args.archs}
        if not jobs_to_run:
            print(f"No matching architectures for: {args.archs}")
            sys.exit(1)

    # Import after path setup
    from src.data.ecg import create_dataloaders
    from scripts.analysis.results.bland_altman import (
        _find_checkpoint,
        _get_config_from_checkpoint,
        _setup_icu_mapper,
        _build_model,
        _collect_predictions_and_meta,
        _plot_bland_altman,
    )
    import torch
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {out_dir}\n")

    for arch_name, job_id in jobs_to_run.items():
        print(f"[{arch_name}] Job {job_id}...")
        try:
            checkpoint_path = _find_checkpoint(job_id)
            config = _get_config_from_checkpoint(checkpoint_path)
            icu_mapper = _setup_icu_mapper(config)

            _, _, test_loader = create_dataloaders(
                config=config,
                labels=None,
                preprocess=None,
                transform=None,
                icu_mapper=icu_mapper,
                mortality_labels=None,
            )
            if test_loader is None:
                print(f"  [skip] No test set for job {job_id}")
                continue

            model = _build_model(config, checkpoint_path, device)
            df = _collect_predictions_and_meta(model, test_loader, device, icu_mapper)

            out_path = out_dir / f"bland_altman_job_{job_id}.png"
            _plot_bland_altman(df, job_id, out_path)
            plt.close("all")
            print(f"  -> {out_path.name}")
        except Exception as e:
            print(f"  [error] {e}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
