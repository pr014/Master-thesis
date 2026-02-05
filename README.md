# Master Thesis: Machine Learning-Based ICU Stay Prediction

Deep Learning models for predicting ICU Length of Stay (LOS) and mortality from 12-lead ECG signals using MIMIC-IV-ECG dataset.

**Task Type**: LOS Regression (continuous prediction in days) + Mortality Classification (binary)

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
  - `baseline_with_aug.yaml` - Baseline config with data augmentation (LOS regression)
  - `baseline_no_aug.yaml` - Baseline config without augmentation (LOS regression)
- **`model/`** - Model architecture configurations
  - `efficientnet1d/` - EfficientNet1D-B1 configs
  - `xresnet1d_101/` - FastAI xResNet1D-101 configs
  - `resnet14/` - ResNet1D-14 configs
  - `cnn_scratch.yaml` - CNN from scratch config
  - `lstm/` - LSTM model configs
    - `unidirectional/` - Unidirectional LSTM configs (improved: 2-layer, mean pooling, embedding)
    - `bidirectional/` - Bidirectional LSTM configs (improved: 2-layer, mean pooling, embedding)
  - `hybrid_cnn_lstm/` - Hybrid CNN-LSTM configs
  - `deepecg_sl/` - DeepECG-SL foundation model configs
  - `hubert_ecg/` - HuBERT-ECG foundation model configs
- **`features/`** - Feature configuration
  - `demographic_features.yaml` - Demographic and diagnosis features configuration

### `scripts/`
- **`analysis/`** - Analysis and evaluation scripts
  - `parse_training_results.py` - Parse regression metrics (MAE, RMSE, R²) from SLURM logs
- **`training/`** - Model training scripts
  - `icu_24h/` - Training scripts for 24h dataset
    - `CNN_from_scratch/` - Baseline CNN training scripts
    - `resnet-14/` - ResNet1D-14 training scripts
    - `efficientnet1d/` - EfficientNet1D-B1 training scripts
    - `xresnet1d_101/` - FastAI xResNet1D-101 training scripts
    - `lstm/` - LSTM training scripts (unidirectional and bidirectional)
    - `hybrid_cnn_lstm/` - Hybrid CNN-LSTM training scripts
    - `deepecg_sl/` - DeepECG-SL foundation model training (2-phase)
    - `hubert_ecg/` - HuBERT-ECG foundation model training (2-phase)
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
  - `deepecg_sl/` - DeepECG-SL foundation model (WCR Transformer Encoder)
  - `hubert_ecg/` - HuBERT-ECG foundation model (HuBERT Transformer Encoder)
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
- **`pretrained_weights/`** - Pretrained model weights (DeepECG-SL, HuBERT-ECG)

### `outputs/` (not tracked in git)
- **`checkpoints/`** - Model checkpoints
- **`logs/`** - SLURM job logs (slurm_*.out, slurm_*.err)
- **`training/`** - Training logs and results
- **`analysis/`** - Analysis outputs
  - `demographic_and_EHR_data/` - Demographic and diagnosis analysis visualizations
- **`visualizations/`** - Generated plots and figures

### `test/` (not tracked in git)
- Smoke tests for model architectures
- Regression readiness tests for foundation models

### `notebooks/` (not tracked in git)
- Jupyter notebooks for data exploration and analysis
- Training result visualizations

## Model Architectures

### CNN Models
- **ResNet1D-14**: Shallow ResNet architecture with 14 layers (~1-2M parameters)
- **EfficientNet1D-B1**: 1D adaptation of EfficientNet-B1 with MBConv blocks (~7.8M parameters)
- **xResNet1D-101**: FastAI xResNet architecture with pretrained weights from PTB-XL dataset (~23M parameters)
- **CNNScratch**: Simple baseline CNN from scratch (~50-100K parameters)

### LSTM Models
- **LSTM1D_Unidirectional**: Unidirectional LSTM for ECG time series regression
  - Architecture: 2-layer LSTM (default), hidden dimension 128, mean pooling, embedding layer (12→64)
  - Improvements: Mean pooling (uses all timesteps), 2-layer with dropout (0.2), embedding for better representation
  - Parameters: ~233K (2 layers)
  - Use case: Real-time ICU deployment scenarios (only uses past information)
  - Scientific justification: 
    - Mean pooling: Lin et al. (2013) - aggregates information across all timesteps
    - Multi-layer LSTM: Sutskever et al. (2014) - learns hierarchical temporal dependencies
    - Embedding: Mikolov et al. (2013) - increases representation capacity
  
- **LSTM1D_Bidirectional**: Bidirectional LSTM for ECG time series regression
  - Architecture: 2-layer bidirectional LSTM (default), 128 hidden units per direction (256 total), mean pooling, embedding layer (12→64)
  - Improvements: Mean pooling (uses all timesteps), 2-layer with dropout (0.2), embedding for better representation
  - Parameters: ~596K (2 layers, 2.56x unidirectional)
  - Use case: Retrospective analysis and scientific comparison (uses both past and future information)
  - Scientific justification: 
    - Bidirectional LSTM: Graves & Schmidhuber (2005) - captures forward and backward temporal dependencies
    - Mean pooling: Lin et al. (2013) - aggregates information across all timesteps
    - Multi-layer LSTM: Sutskever et al. (2014) - learns hierarchical temporal dependencies
    - Embedding: Mikolov et al. (2013) - increases representation capacity
    - Dropout: Srivastava et al. (2014) - reduces overfitting in multi-layer architectures

### Hybrid Models
- **HybridCNNLSTM**: Combined CNN and LSTM architecture
  - Architecture: CNN feature extraction (3 conv blocks) → LSTM temporal modeling → Regression head
  - Parameters: ~700K-1M depending on configuration
  - Use case: Captures both spatial (CNN) and temporal (LSTM) patterns in ECG signals

### Foundation Models (Self-Supervised Pretrained)
- **DeepECG-SL**: WCR (Wav2Vec2 with Contrastive Multi-view Coding) Transformer Encoder
  - Architecture: Input Adapter (5000→2500) → WCR Encoder (pretrained) → Global Pooling → Shared Layers → LOS/Mortality Heads
  - Pretrained: Self-supervised pretraining on large ECG dataset
  - Parameters: ~100M+ (with pretrained encoder)
  - Training: 2-phase (Phase 1: Frozen backbone, Phase 2: Fine-tuning)
  - Use case: Transfer learning from self-supervised pretraining
  
- **HuBERT-ECG**: HuBERT (Hidden Unit BERT) Transformer Encoder
  - Architecture: Feature Extractor (5 Conv layers) → HuBERT Encoder (pretrained) → Mean Pooling → LOS/Mortality Heads
  - Pretrained: Self-supervised pretraining on large ECG dataset
  - Parameters: ~93M (with pretrained encoder)
  - Training: 2-phase (Phase 1: Frozen backbone, Phase 2: Fine-tuning)
  - Use case: Transfer learning from self-supervised pretraining

All models support:
- Multi-task learning (LOS regression + Mortality prediction)
- LOS Regression: Continuous prediction in days (output dim = 1)
- Mortality Classification: Binary prediction (0/1)
- Optional demographic features (Age & Sex) via late fusion
- Optional diagnosis features (ICD-10 codes) via late fusion
  - Supports top 15 most frequent diagnoses (configurable)
  - Binary encoding per diagnosis
- Loss functions: MSE/L1/Huber for LOS regression, BCE for mortality

## Evaluation Metrics

### LOS Regression Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual LOS in days
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences (penalizes large errors more)
- **R² (Coefficient of Determination)**: Proportion of variance explained (1.0 = perfect, 0.0 = baseline mean)
- **Median Absolute Error**: Median of absolute errors (robust to outliers)
- **Percentile Errors**: P25, P50, P75, P90 error percentiles

### Mortality Classification Metrics
- **AUC-ROC**: Area Under the ROC Curve (preferred for imbalanced classification)
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

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

## Datasets

- **`icu_24h`** - ECGs from first 24 hours of ICU stay
  - Preprocessed: Bandpass filtered (0.5-50 Hz), resampled to 500 Hz, 10-second segments (5000 samples)
  - Labels: Continuous LOS in days (regression target)
- **`all_icu_ecgs`** - All available ICU ECGs regardless of timing

## Data Splitting

- **Stratified Split**: Quantile-based stratification on LOS values for train/val/test splits
- **Patient-Level Split**: Ensures no data leakage (all ECGs from same patient in same split)
- **Default Split**: 70% train, 15% validation, 15% test

## Training

### Training Scripts
- **CNN Models**: `scripts/training/icu_24h/CNN_from_scratch/`, `efficientnet1d/`, `xresnet1d_101/`
- **LSTM Models**: `scripts/training/icu_24h/lstm/` (unidirectional and bidirectional)
- **Hybrid Models**: `scripts/training/icu_24h/hybrid_cnn_lstm/`
- **Foundation Models**: 
  - `scripts/training/icu_24h/deepecg_sl/` - DeepECG-SL with 2-phase training
  - `scripts/training/icu_24h/hubert_ecg/` - HuBERT-ECG with 2-phase training

### Cluster Execution
- SLURM job scripts in `scripts/cluster/icu_24h/` mirror training script structure
- Submit with: `sbatch scripts/cluster/icu_24h/<model>/train_<model>_24h.sbatch`

## Requirements

See `requirements.txt` for Python dependencies.

### Additional Dependencies for Foundation Models
- **DeepECG-SL**: Requires `fairseq-signals` (install with: `pip install git+https://github.com/HeartWise-AI/fairseq-signals.git`)
- **HuBERT-ECG**: Requires pretrained weights in `data/pretrained_weights/Hubert_ECG/base/`