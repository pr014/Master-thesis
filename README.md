# Master Thesis: Machine Learning-Based ICU Stay Prediction

Deep Learning models for predicting ICU Length of Stay (LOS) and mortality from 12-lead ECG signals using MIMIC-IV-ECG dataset.

## Project Structure

```
MA-thesis-1/
├── configs/          # Configuration files
├── scripts/           # Executable scripts
├── src/               # Source code modules
├── data/              # Datasets (not in git)
├── outputs/           # Results and checkpoints (not in git)
└── docs/              # Documentation
```

## Directory Overview

### `configs/`
- **`all_icu_ecgs/`** - Configs for full ICU ECG dataset
- **`icu_24h/`** - Configs for 24-hour ICU ECG dataset
- **`model/`** - Model architecture configurations
- **`visualization/`** - Paths for visualization scripts

### `scripts/`
- **`analysis/`** - Data analysis scripts
  - `dataset/` - Dataset analysis and filtering
  - `training/` - Training results analysis
  - `mortality/` - Mortality analysis
  - `visualization/` - Plotting scripts
- **`training/`** - Model training scripts
- **`cluster/`** - SLURM job scripts for cluster execution
- **`preprocessing/`** - Data preprocessing scripts
- **`ecg_visualizing/`** - ECG visualization utilities

### `src/`
- **`data/`** - Data loading and preprocessing modules
- **`models/`** - Model architectures (CNN, LSTM, etc.)
- **`training/`** - Training loop and trainer classes
- **`evaluation/`** - Evaluation metrics and utilities
- **`utils/`** - Utility functions and helpers
- **`visualization/`** - Visualization modules

### `docs/`
- **`deployment/`** - Server deployment guides
- **`data/`** - Dataset documentation
- **`models/`** - Model architecture documentation
- **`baseline/`** - Baseline model documentation
- **`TROUBLESHOOTING.md`** - Common issues and solutions

### `data/` (not tracked in git)
- **`all_icu_ecgs/`** - Full ICU ECG dataset
- **`icu_ecgs_24h/`** - 24-hour ICU ECG dataset
- **`labeling/`** - ICU stay labels and metadata
- **`mimic_iv_ecg_all/`** - Complete MIMIC-IV-ECG dataset

### `outputs/` (not tracked in git)
- **`checkpoints/`** - Model checkpoints
- **`training/`** - Training logs and results
- **`data_analysis/`** - Analysis outputs
- **`visualizations/`** - Generated plots and figures

## Datasets

- **`icu_24h`** - ECGs from first 24 hours of ICU stay
- **`all_icu_ecgs`** - All available ICU ECGs regardless of timing

## Requirements

See `requirements.txt` for Python dependencies.