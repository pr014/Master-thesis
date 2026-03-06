# Master Thesis: ICU LOS Prediction from 12-Lead ECG

Deep Learning models for predicting ICU Length of Stay (LOS) and mortality from 12-lead ECG signals (MIMIC-IV-ECG).

**Tasks**: LOS Regression (continuous) + Mortality Classification (binary)  
**Models**: CNN, LSTM, Hybrid CNN-LSTM, XResNet1D-PTBXL, DeepECG-SL, HuBERT-ECG, XGBoost

## Project Structure

```
MA-thesis-1/
├── configs/model/     # Model configs (hybrid_cnn_lstm, deepecg_sl, hubert_ecg, xresnet1d_ptbxl, lstm, cnn_scratch)
├── configs/classical_ml/  # XGBoost configs
├── scripts/training/  # Training scripts (icu_24h/, classical_ml/)
├── scripts/cluster/   # SLURM sbatch scripts
├── scripts/analysis/  # parse_training_results, evaluate_subgroup, plot_*
├── src/               # data/, models/, training/, features/
├── data/              # Datasets, pretrained weights (not in git)
└── outputs/           # Checkpoints, logs (not in git)
```

## Models (Overview)

| Model | Params | Notes |
|-------|--------|-------|
| CNNScratch | ~50-100K | Baseline from scratch |
| LSTM (Uni/Bi) | ~233K / ~596K | 2-layer, mean pooling |
| Hybrid CNN-LSTM | ~700K-1M | CNN + BiLSTM |
| XResNet1D-PTBXL | ~23M | PTB-XL pretrained |
| DeepECG-SL | ~100M | WCR Transformer, 2-phase training |
| HuBERT-ECG | ~93M | HuBERT Transformer, 2-phase training |
| XGBoost | - | Handcrafted or DL features |

All DL models support optional **late fusion** with demographics (Age & Sex), diagnosis (ICD-10), and ICU unit features.

## Quick Start

```bash
# Local
python scripts/training/icu_24h/lstm/train_lstm_bi_24h.py
python scripts/training/icu_24h/xresnet1d_ptbxl/train_xresnet1d_ptbxl_24h.py

# Cluster (SLURM)
sbatch scripts/cluster/icu_24h/xresnet1d_ptbxl/train_xresnet1d_ptbxl_24h.sbatch

# Parse results
python scripts/analysis/parse_training_results.py --job <JOB_ID>
```

## Requirements

- `requirements.txt`
- **DeepECG-SL**: `pip install git+https://github.com/HeartWise-AI/fairseq-signals.git`
- **HuBERT-ECG**: Pretrained weights in `data/pretrained_weights/Hubert_ECG/base/`

## Datasets

- **icu_24h**: ECGs from first 24h of ICU stay, bandpass 0.5–50 Hz, 500 Hz, 10 s segments (5000 samples)
- **all_icu_ecgs**: Full ICU ECG dataset
- Split: Temporal stratified, ~80/10/10 train/val/test

## Evaluation Metrics

**LOS**: MAE, RMSE, R², Median AE  
**Mortality**: AUC-ROC, F1, Precision, Recall
