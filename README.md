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
  - `output/` - Output-specific configs (weighted_exact_days, weighted_intervals)
- **`model/`** - Model architecture configurations
  - `efficientnet1d/` - EfficientNet1D-B1 configs
  - `xresnet1d_101/` - FastAI xResNet1D-101 configs
  - `resnet1d_14/` - ResNet1D-14 configs
  - `lstm/` - LSTM model configs
    - `unidirectional/` - Unidirectional LSTM configs
    - `bidirectional/` - Bidirectional LSTM configs
  - `hybrid_cnn_lstm/` - Hybrid CNN-LSTM configs
- **`features/`** - Feature configuration
  - `demographic_features.yaml` - Demographic and diagnosis features configuration
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
    - `hybrid_cnn_lstm/` - Hybrid CNN-LSTM training scripts
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
  - `hybrid_cnn_lstm/` - Hybrid CNN-LSTM architecture
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
- **`logs/`** - SLURM job logs (slurm_*.out, slurm_*.err)
- **`training/`** - Training logs and results
- **`analysis/`** - Analysis outputs
  - `demographic_and_EHR_data/` - Demographic and diagnosis analysis visualizations
- **`visualizations/`** - Generated plots and figures

## Model Architectures

### CNN Models
- **ResNet1D-14**: Shallow ResNet architecture with 14 layers (~1-2M parameters)
- **EfficientNet1D-B1**: 1D adaptation of EfficientNet-B1 with MBConv blocks (~7.8M parameters)
- **xResNet1D-101**: FastAI xResNet architecture with pretrained weights from PTB-XL dataset (~23M parameters)
- **CNNScratch**: Simple baseline CNN from scratch (~50-100K parameters)

### LSTM Models
- **LSTM1D_Unidirectional**: Unidirectional LSTM for ECG time series classification
  - Architecture: 1-2 LSTM layers, hidden dimension 128, pooling strategies (last/mean/max)
  - Parameters: ~74K (1 layer) or ~280K (2 layers)
  - Use case: Real-time ICU deployment scenarios (only uses past information)
  
- **LSTM1D_Bidirectional**: Bidirectional LSTM for ECG time series classification
  - Architecture: 1-2 bidirectional LSTM layers, 128 hidden units per direction (256 total), pooling strategies (last/mean/max)
  - Parameters: ~150K (1 layer) or ~900K (2 layers)
  - Use case: Retrospective analysis and scientific comparison (uses both past and future information)
  - Scientific justification: Based on Yildirim (2018, IEEE Access) and Lipton et al. (2016, arXiv) for ECG classification and clinical time series analysis

### Hybrid Models
- **HybridCNNLSTM**: Combined CNN and LSTM architecture
  - Architecture: CNN feature extraction (3 conv blocks) → LSTM temporal modeling → Classification head
  - Parameters: ~700K-1M depending on configuration
  - Use case: Captures both spatial (CNN) and temporal (LSTM) patterns in ECG signals

All models support:
- Multi-task learning (LOS classification + Mortality prediction)
- Optional demographic features (Age & Sex) via late fusion
- Optional diagnosis features (ICD-10 codes) via late fusion
  - Supports top 15 most frequent diagnoses (configurable)
  - Binary encoding per diagnosis
- Weighted class weighting for imbalanced LOS classes (balanced, sqrt, or custom weights)
- Weighted BCE for mortality prediction

## Features

### Demographic Features
- **Age**: Normalized age (minmax or z-score normalization)
- **Sex**: Binary encoding (0/1) or one-hot encoding (configurable)

### Diagnosis Features (ICD-10 Codes)
- **Top 15 Diagnoses**: Most frequent ICD-10 diagnosis codes from MIMIC-IV
  - Default diagnoses: R6521, J9690, Z66, R6520, N170, A419, E872, J690, N179, J189, J449, D696, E871, N390, D62
  - Binary encoding: Each diagnosis is represented as 0 (absent) or 1 (present)
  - Missing strategy: Zero-filling for missing diagnoses (configurable)
- **Integration**: Late fusion with ECG features and demographic features
- **Performance Impact**: Diagnosis features show significant improvements:
  - +57.4% Balanced Accuracy improvement
  - +67.0% Macro F1-Score improvement
  - +32.7% Mortality prediction accuracy improvement
  - Better per-class performance, especially for rare classes

## Datasets

- **`icu_24h`** - ECGs from first 24 hours of ICU stay
- **`all_icu_ecgs`** - All available ICU ECGs regardless of timing

## Requirements

See `requirements.txt` for Python dependencies.