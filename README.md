# Master Thesis: Machine Learning-Based ICU Stay Prediction

Deep Learning models for predicting ICU Length of Stay (LOS) and mortality from 12-lead ECG signals using MIMIC-IV-ECG dataset.

**Task Type**: LOS Regression (continuous prediction in days) + Mortality Classification (binary)

**Supported Models**: CNN, LSTM, Hybrid CNN-LSTM, XResNet1D-PTBXL, DeepECG-SL, HuBERT-ECG, XGBoost (classical ML)

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
- **`model/`** - Model architecture configurations (standalone configs with data, training, model params)
  - `cnn_scratch.yaml` - CNN from scratch
  - `lstm/` - LSTM model configs
    - `unidirectional/` - Unidirectional LSTM (1-layer, 2-layer)
    - `bidirectional/` - Bidirectional LSTM (1-layer, 2-layer)
  - `hybrid_cnn_lstm/` - Hybrid CNN-LSTM
  - `xresnet1d_ptbxl/` - XResNet1D-101 with PTB-XL pretrained weights (transfer learning)
  - `deepecg_sl/` - DeepECG-SL foundation model
  - `hubert_ecg/` - HuBERT-ECG foundation model
- **`classical_ml/`** - XGBoost configs (handcrafted and DL features)
- **`all_icu_ecgs/`** - Configs for full ICU ECG dataset
  - `weighted/` - Class weights for classification (balanced, sqrt)
- **`features/`** - Feature configuration
  - `demographic_features.yaml` - Age & Sex features
  - `diagnosis_features.yaml` - ICD-10 diagnosis features
  - `icu_unit_features.yaml` - ICU unit type (first_careunit) as one-hot
- **`archive/`** - Legacy configs (baseline_with_aug, baseline_no_aug, etc.)

### `scripts/`
- **`analysis/`** - Analysis and evaluation scripts
  - `parse_training_results.py` - Parse regression metrics from SLURM logs: `python scripts/analysis/parse_training_results.py --job <SLURM_JOB_ID>`
  - `evaluate_subgroup.py` - Subgroup evaluation
  - `analyze_job_performance.py` - SLURM job performance analysis
  - `plot_architecture_params_accuracy.py`, `plot_architecture_params_mae.py` - Architecture comparison plots
  - `plot_los_imbalance.py` - LOS distribution visualization
- **`training/`** - Model training scripts
  - `icu_24h/` - Training scripts for 24h dataset
    - `CNN_from_scratch/` - Baseline CNN training
    - `lstm/` - LSTM training (unidirectional and bidirectional)
    - `hybrid_cnn_lstm/` - Hybrid CNN-LSTM training
    - `xresnet1d_ptbxl/` - XResNet1D-101 PTB-XL pretrained training
    - `deepecg_sl/` - DeepECG-SL (2-phase training)
    - `hubert_ecg/` - HuBERT-ECG (2-phase training)
  - `classical_ml/` - XGBoost LOS regression (handcrafted or DL features)
  - `all_icu_ecgs/` - Training scripts for full dataset
- **`cluster/`** - SLURM job scripts for cluster execution
  - `icu_24h/<model>/` - Mirrors training scripts (CNN_from_scratch, lstm, hybrid_cnn_lstm, xresnet1d_ptbxl, deepecg_sl, hubert_ecg)
  - `classical_ml/` - XGBoost training
  - `all_icu_ecgs/baseline_CNN/` - Full dataset CNN baseline
  - `analysis/` - Bland-Altman and evaluation jobs
- **`preprocessing/`** - Data preprocessing scripts
- **`ecg_visualizing/`** - ECG visualization utilities
- **`data/`** - Data management scripts
- **`datasplit/`** - Dataset splitting utilities
- **`scoring_models/`** - Model scoring and evaluation

### `src/`
- **`data/`** - Data loading and preprocessing
  - `ecg/` - ECG dataset, dataloader factory
  - `labeling/` - ICU LOS labels, icustays mapping
- **`features/`** - Feature extraction (handcrafted, DL-based for XGBoost)
- **`models/`** - Model architectures
  - `core/` - Core components (BaseECGModel, MultiTaskECGModel)
  - `cnn_scratch/` - Baseline CNN from scratch
  - `lstm/` - LSTM models
    - `unidirectional/` - Unidirectional LSTM (LSTM1D_Unidirectional)
    - `bidirectional/` - Bidirectional LSTM (LSTM1D_Bidirectional)
  - `hybrid_cnn_lstm/` - Hybrid CNN-LSTM architecture
  - `xresnet1d_ptbxl/` - XResNet1D-101 with PTB-XL pretrained weights (pure PyTorch)
  - `deepecg_sl/` - DeepECG-SL foundation model (WCR Transformer Encoder)
  - `hubert_ecg/` - HuBERT-ECG foundation model (HuBERT Transformer Encoder)
  - `classical_ml/` - XGBoost model wrapper
- **`training/`** - Training loop, trainer, losses
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
- **`pretrained_weights/`** - Pretrained model weights (PTB-XL fastai_xresnet1d101.pth, DeepECG-SL, HuBERT-ECG)

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
- **XResNet1D-PTBXL**: XResNet1D-101 with PTB-XL pretrained weights (ecg_ptbxl_benchmarking)
  - Architecture: 1:1 from ecg_ptbxl_benchmarking, pretrained on PTB-XL 71 SCP classes
  - Pure PyTorch implementation, loads FastAI checkpoint (`fastai_xresnet1d101.pth`)
  - Parameters: ~23M
  - Use case: Transfer learning from ECG classification pretraining
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

### Classical ML
- **XGBoost**: Gradient boosted trees for LOS regression
  - Features: Handcrafted (RR intervals, spectral, morphological) or DL-extracted
  - Configs: `configs/classical_ml/xgboost_handcrafted.yaml`, `xgboost_dl_features.yaml`

All DL models support:
- Multi-task learning (LOS regression + Mortality prediction)
- LOS Regression: Continuous prediction in days (output dim = 1)
- Mortality Classification: Binary prediction (0/1)
- Optional demographic features (Age & Sex) via late fusion
- Optional diagnosis features (ICD-10 codes) via late fusion
  - Supports top 15 most frequent diagnoses (configurable)
  - Binary encoding per diagnosis
- Loss functions: MSE/L1/Huber for LOS regression, BCE for mortality
- Optional ICU unit features (first_careunit) via late fusion

### Sample Weighting for Regression
- **Imbalanced LOS**: Weight MSE loss by inverse bin frequency to emphasize rare LOS values
- **Enable per model**: Add to model config under `training.loss`:
  ```yaml
  loss:
    type: "mse"
    weighted: true
    method: "balanced"   # or "sqrt"
  ```
- **Methods**: `balanced` (inverse frequency) or `sqrt` (softer weighting)
- **Binning**: Uses `data.los_binning` (strategy: `intervals` or `exact_days`, max_days: 9)
- Weights computed at runtime from training LOS distribution; each model config is independent

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

### ICU Unit Features
- **first_careunit**: ICU unit type at admission (from MIMIC-IV icustays)
- **Top 10 Units**: One-hot encoding for most common units (MICU, MICU/SICU, CVICU, SICU, CCU, TSICU, Neuro Intermediate, Neuro SICU, Neuro Stepdown, Surgery/Vascular/Intermediate)
- **Other**: Rare units encoded as zero vector; unknown/missing use `most_common` or `zero` strategy
- **Integration**: Late fusion with ECG features; enabled via `data.icu_unit_features.enabled: true`

## Datasets

- **`icu_24h`** - ECGs from first 24 hours of ICU stay
  - Preprocessed: Bandpass filtered (0.5-50 Hz), resampled to 500 Hz, 10-second segments (5000 samples)
  - Labels: Continuous LOS in days (regression target)
- **`all_icu_ecgs`** - All available ICU ECGs regardless of timing

## Data Splitting

- **Temporal Stratified Split**: Time-ordered stratification on LOS for train/val/test
- **Patient-Level Split**: All ECGs from same subject in same split (no data leakage)
- **Default Split**: ~80% train, ~10% validation, ~10% test (subject counts vary by dataset)

## Training

### Config Loading
- Each model uses a **standalone config** in `configs/model/<model>/<config>.yaml`
- No base/experiment config merge for icu_24h models; all params in model config

### Training Scripts
- **CNN Models**: `scripts/training/icu_24h/CNN_from_scratch/`, `xresnet1d_ptbxl/`
- **LSTM Models**: `scripts/training/icu_24h/lstm/train_lstm_24h.py`, `train_lstm_bi_24h.py`
- **Hybrid Models**: `scripts/training/icu_24h/hybrid_cnn_lstm/`
- **Foundation Models**: `deepecg_sl/`, `hubert_ecg/` (2-phase training)
- **Classical ML**: `scripts/training/classical_ml/train_xgboost_24h.py`

### Cluster Execution
- SLURM job scripts in `scripts/cluster/icu_24h/<model>/`
- Submit: `sbatch scripts/cluster/icu_24h/lstm/train_lstm_bi_24h.sbatch` or `sbatch scripts/cluster/icu_24h/xresnet1d_ptbxl/train_xresnet1d_ptbxl_24h.sbatch`
- Logs: `outputs/logs/slurm_<job_id>.out`, `slurm_<job_id>.err`

## Requirements

See `requirements.txt` for Python dependencies.

### Additional Dependencies for Foundation Models
- **DeepECG-SL**: `fairseq-signals` (`pip install git+https://github.com/HeartWise-AI/fairseq-signals.git`)
- **HuBERT-ECG**: Pretrained weights in `data/pretrained_weights/Hubert_ECG/base/`

## Quick Start

```bash
# Local training (example: LSTM Bidirectional or XResNet1D-PTBXL)
python scripts/training/icu_24h/lstm/train_lstm_bi_24h.py
python scripts/training/icu_24h/xresnet1d_ptbxl/train_xresnet1d_ptbxl_24h.py

# Cluster (SLURM)
sbatch scripts/cluster/icu_24h/lstm/train_lstm_bi_24h.sbatch
sbatch scripts/cluster/icu_24h/xresnet1d_ptbxl/train_xresnet1d_ptbxl_24h.sbatch

# Parse results from SLURM log
python scripts/analysis/parse_training_results.py --job <SLURM_JOB_ID>
```