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
  - `efficientnet1d/` - EfficientNet1D-B1 configs
  - `xresnet1d_101/` - FastAI xResNet1D-101 configs
  - `resnet1d_14/` - ResNet1D-14 configs
  - `lstm/` - LSTM model configs
    - `unidirectional/` - Unidirectional LSTM configs
    - `bidirectional/` - Bidirectional LSTM configs
- **`features/`** - Feature configuration (e.g., demographic features)
- **`visualization/`** - Paths for visualization scripts
- **`weights/`** - Class weight configurations

### `scripts/`
- **`analysis/`** - Analysis and evaluation scripts
  - `parse_training_results.py` - Parse metrics from SLURM logs
  - `visualize_confusion_matrices.py` - Confusion matrix visualization
- **`training/`** - Model training scripts
  - `icu_24h/` - Training scripts for 24h dataset
    - `CNN_from_scratch/` - Baseline CNN training scripts
    - `resnet-14/` - ResNet1D-14 training scripts
    - `efficientnet1d/` - EfficientNet1D-B1 training scripts
    - `xresnet1d_101/` - FastAI xResNet1D-101 training scripts
    - `lstm/` - LSTM training scripts (unidirectional and bidirectional)
  - `all_icu_ecgs/` - Training scripts for full dataset
- **`cluster/`** - SLURM job scripts for cluster execution
  - Mirrors the structure of `scripts/training/`
- **`preprocessing/`** - Data preprocessing scripts
- **`ecg_visualizing/`** - ECG visualization utilities
- **`data/`** - Data management scripts
- **`datasplit/`** - Dataset splitting utilities
- **`scoring_models/`** - Model scoring and evaluation

### `src/`
- **`data/`** - Data loading and preprocessing modules
  - `ecg/` - ECG dataset and dataloader implementations
- **`models/`** - Model architectures
  - `core/` - Core components (BaseECGModel, MultiTaskECGModel)
  - `cnn_scratch/` - Baseline CNN from scratch
  - `efficientnet1d/` - EfficientNet1D-B1 implementation
  - `pretrained_CNN/` - Pretrained CNN models
    - `resnet1d_14/` - ResNet1D-14
    - `xresnet1d_101/` - FastAI xResNet1D-101
  - `lstm/` - LSTM models
    - `unidirectional/` - Unidirectional LSTM (LSTM1D_Unidirectional)
    - `bidirectional/` - Bidirectional LSTM (LSTM1D_Bidirectional)
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

## Model Architectures

### CNN Models
- **ResNet1D-14**: Shallow ResNet architecture with 14 layers (~1-2M parameters)
- **EfficientNet1D-B1**: 1D adaptation of EfficientNet-B1 with MBConv blocks (~7.8M parameters)
- **xResNet1D-101**: FastAI xResNet architecture with pretrained weights from PTB-XL dataset (~23M parameters)
- **CNNScratch**: Simple baseline CNN from scratch (~50-100K parameters)

### LSTM Models
- **LSTM1D_Unidirectional**: Unidirectional LSTM for ECG time series classification
  - Architecture: 1-2 LSTM layers, hidden dimension 256, pooling strategies (last/mean/max)
  - Parameters: ~280K (1 layer) or ~806K (2 layers)
  - Use case: Real-time ICU deployment scenarios (only uses past information)
  
- **LSTM1D_Bidirectional**: Bidirectional LSTM for ECG time series classification
  - Architecture: 1-2 bidirectional LSTM layers, 64 hidden units per direction (128 total), pooling strategies (last/mean/max)
  - Parameters: ~500-600K (1 layer) or ~900K (2 layers)
  - Use case: Retrospective analysis and scientific comparison (uses both past and future information)
  - Scientific justification: Based on Yildirim (2018, IEEE Access) and Lipton et al. (2016, arXiv) for ECG classification and clinical time series analysis

All models support:
- Multi-task learning (LOS classification + Mortality prediction)
- Optional demographic features (Age & Sex) via late fusion
- SQRT class weighting for imbalanced LOS classes
- Weighted BCE for mortality prediction

## Datasets

- **`icu_24h`** - ECGs from first 24 hours of ICU stay
- **`all_icu_ecgs`** - All available ICU ECGs regardless of timing

## Requirements

See `requirements.txt` for Python dependencies.