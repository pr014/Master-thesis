# ICU Length-of-Stay from 12-Lead ECG

PyTorch codebase for **master thesis** work: predict **ICU length of stay (LOS, days)** from short **12-lead ECG** windows, with optional **mortality** as an auxiliary multi-task head. Data pipeline targets **MIMIC-IV-ECG**-style ICU cohorts (e.g. 24h extracts).

## Scope

- **Primary task:** LOS regression (continuous).
- **Secondary task:** Binary mortality (shared training; threshold for F1 chosen on validation).
- **Fusion:** Optional tabular inputs—demographics, ICU unit (one-hot), **SOFA** (and similar non-ECG signals where configured).

## Models

| Model | Role |
|--------|------|
| **Hybrid CNN-LSTM** | Main scratch architecture; end-to-end training. |
| **DeepECG-SL** | WCR / wav2vec-style encoder; local pretrained weights; fine-tuning (phase-2–style run in current script). |
| **HuBERT-ECG** | Foundation-style encoder; **two-phase** train (frozen backbone → partial unfreeze). |
| CNN / LSTM / CNN-from-scratch | Lighter baselines under `scripts/training/icu_24h/`. |
| **XGBoost** | Classical baseline (`scripts/training/classical_ml/`). |

**Hyperparameters:** [Optuna](https://optuna.org/) workers for Hybrid and DeepECG (`scripts/tuning/`, shared DB storage on cluster). HuBERT typically uses fixed YAML schedules.

## Layout

```
configs/model/          # hybrid_cnn_lstm, deepecg_sl, hubert_ecg, …
configs/tuning/         # Optuna base overlays + best-trial exports (as used)
scripts/training/       # icu_24h/, classical_ml/
scripts/tuning/         # Optuna workers + exports
scripts/cluster/        # SLURM sbatch
scripts/analysis/       # metrics, plots, subgroup eval
src/                    # data, models, training, features
app/                    # optional Streamlit clinic demo (DeepECG-SL inference)
data/, outputs/         # local only (not in git)
```

## Clinic demo (Streamlit)

Small **inference-only** UI under `app/`: upload a 12-lead window, optionally set tabular fields, run **DeepECG-SL** multi-task (LOS + mortality probability). Not a medical device.

**1. Dependencies** — install from repo root (includes **Streamlit** and **fairseq-signals** for DeepECG-SL):

```bash
pip install -r requirements.txt
```

If the clinic app still reports a missing `fairseq_signals` import, install the Git dependency explicitly (network required):

```bash
pip install 'git+https://github.com/HeartWise-AI/fairseq-signals.git'
```

**2. Pretrained backbone** — same as training: WCR weights under `data/pretrained_weights/deepecg_sl/` as in `configs/model/deepecg_sl/deepecg_sl.yaml`.

**3. Trained checkpoint** — a file saved by training, e.g.  
`outputs/checkpoints/DeepECG_SL_best_<SLURM_JOB_ID>.pt`  
with **`config`** and **`model_state_dict`** inside.

**4. Run the app** (from repo root):

```bash
streamlit run app/clinic_app.py
```

**5. Load the model in the UI** — sidebar:

- Either set **`MA_THESIS_CLINIC_CKPT`** to the full path before starting (used as default), **or**
- Paste the path into **“Pfad zum Checkpoint (.pt)”**. If unset, the app **prefills the newest** `outputs/checkpoints/DeepECG_SL_best_*.pt` when that file exists.
- Click **“Modell laden”** and wait until the success message appears.

More detail: `app/README.md`.

## Quick start

```bash
pip install -r requirements.txt   # includes fairseq-signals (DeepECG)

python scripts/training/icu_24h/hybrid_cnn_lstm/train_hybrid_cnn_lstm_24h.py
python scripts/training/icu_24h/deepecg_sl/train_deepecg_sl_24h.py
python scripts/training/icu_24h/hubert_ecg/train_hubert_ecg_24h.py
```

Override knobs with `--experiment-config <merged_yaml>` (see `configs/tuning/`). On the cluster, submit the matching `scripts/cluster/icu_24h/*/*.sbatch` jobs.

## Data & splits

- **Preprocessed ECG:** 500 Hz, bandpass, z-score; 10 s segments (e.g. 12×5000). Paths set in YAML (`data.data_dir`).
- **Split:** Subject-level **temporal stratified** split (~80/10/10) with optional LOS stratification inside time bands—see data chapter / `create_dataloaders`.

## Metrics

**LOS:** MAE, RMSE, R², median AE. **Mortality:** AUC-ROC; F1/precision/recall at validation-tuned threshold.

## Notes

- Place **pretrained checkpoints** under `data/pretrained_weights/` as referenced in the model YAMLs (DeepECG WCR, HuBERT-ECG).
- Set `SLURM_JOB_ID` (or tuning env) so checkpoints and logs are traceable per job.
