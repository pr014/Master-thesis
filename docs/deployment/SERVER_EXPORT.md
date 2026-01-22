# Server Export & Training Guide

## Übersicht

Dieses Dokument beschreibt, wie das CNN-Scratch Modell auf dem BW Uni Cluster trainiert wird.

## Voraussetzungen

1. **Daten auf dem Server:**
   - ECG-Daten: `data/raw/demo/ecg/mimic-iv-ecg-demo/` (oder via `DATA_DIR` environment variable)
   - ICU Stays: `icustays.csv` (Pfad via `ICUSTAYS_PATH` environment variable)

2. **Python Environment:**
   - Python 3.8+
   - PyTorch mit CUDA Support (falls GPU verwendet wird)
   - Alle Dependencies aus `requirements.txt`

## Projekt-Struktur auf Server

```
/path/to/MA-thesis-1/
├── configs/
│   ├── baseline.yaml
│   └── model/
│       └── cnn_scratch.yaml
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── scripts/
│   ├── training/
│   │   └── train_cnn_scratch.py
│   └── cluster/
│       └── train_cnn_scratch.sbatch
├── data/
│   └── icustays.csv
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── plots/
└── requirements.txt
```

## Setup auf dem Server

### 1. Projekt hochladen

```bash
# Option A: Git Clone (wenn Repository vorhanden)
git clone <repository-url>
cd MA-thesis-1

# Option B: SCP/RSYNC (vom lokalen Rechner)
scp -r MA-thesis-1 user@cluster:/path/to/destination/
```

### 2. Dependencies installieren

```bash
# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Linux
# ODER
conda create -n ma-thesis python=3.9
conda activate ma-thesis

# Dependencies installieren
pip install -r requirements.txt
```

### 3. Environment Variables setzen

```bash
# In ~/.bashrc oder im SLURM-Script
export DATA_DIR="/path/to/data"
export ICUSTAYS_PATH="${DATA_DIR}/icustays.csv"
```

## Training starten

### Option 1: SLURM Job Script (empfohlen)

```bash
# Job einreichen
sbatch scripts/cluster/train_cnn_scratch.sbatch

# Job Status prüfen
squeue -u $USER

# Logs ansehen
tail -f outputs/logs/slurm_<JOB_ID>.out
```

### Option 2: Direkt ausführen (interaktiv)

```bash
# GPU-Node anfordern
srun --gres=gpu:1 --mem=16G --time=24:00:00 --pty bash

# Environment aktivieren
source venv/bin/activate

# Environment Variables setzen
export ICUSTAYS_PATH="/path/to/icustays.csv"

# Training starten
python scripts/training/cnn_from_scratch/icu_24h/train_cnn_scratch.py
```

## SLURM Script anpassen

Das Script `scripts/cluster/train_cnn_scratch.sbatch` muss an deine Cluster-Umgebung angepasst werden:

1. **Module laden:**
   ```bash
   module load python/3.9
   module load cuda/11.8
   ```

2. **Virtual Environment:**
   ```bash
   source venv/bin/activate
   # ODER
   conda activate ma-thesis
   ```

3. **Ressourcen anpassen:**
   - `--time`: Maximale Laufzeit
   - `--mem`: RAM-Anforderung
   - `--gres=gpu:1`: GPU-Anforderung
   - `--cpus-per-task`: CPU-Kerne für DataLoader

## Konfiguration

### Pfade in `configs/baseline.yaml`

```yaml
data:
  data_dir: "data/raw/demo/ecg/mimic-iv-ecg-demo"  # Relativ zum Projekt-Root
```

**Wichtig:** Alle Pfade sind relativ zum Projekt-Root. Stelle sicher, dass die Datenstruktur auf dem Server identisch ist.

### Environment Variables

- `ICUSTAYS_PATH`: Absoluter Pfad zu `icustays.csv`
- `DATA_DIR`: Basis-Verzeichnis für Daten (optional)

## Outputs

Nach dem Training findest du:

- **Checkpoints:** `outputs/checkpoints/cnn_scratch/best_model.pt`
- **Logs:** `outputs/logs/training.log`
- **TensorBoard:** `outputs/logs/tensorboard/` (falls aktiviert)
- **SLURM Logs:** `outputs/logs/slurm_<JOB_ID>.out`

## Troubleshooting

### Problem: `FileNotFoundError: icustays.csv not found`

**Lösung:**
```bash
export ICUSTAYS_PATH="/absoluter/pfad/zu/icustays.csv"
```

### Problem: `CUDA out of memory`

**Lösung:** Batch Size in `configs/baseline.yaml` reduzieren:
```yaml
training:
  batch_size: 32  # Statt 64
```

### Problem: `ModuleNotFoundError`

**Lösung:** Dependencies installieren:
```bash
pip install -r requirements.txt
```

### Problem: Pfade funktionieren nicht

**Lösung:** Prüfe, ob alle Pfade relativ zum Projekt-Root sind:
```bash
cd /path/to/MA-thesis-1
python scripts/training/cnn_from_scratch/icu_24h/train_cnn_scratch.py
```

## Reproduzierbarkeit

- **Random Seed:** Wird in `configs/baseline.yaml` gesetzt (`seed: 42`)
- **Deterministic Operations:** Optional in `src/utils/device.py` aktivieren
- **Config Versioning:** Alle Hyperparameter in `baseline.yaml` und `cnn_scratch.yaml`

## Nächste Schritte

Nach erfolgreichem Training:

1. Bestes Modell: `outputs/checkpoints/cnn_scratch/best_model.pt`
2. Evaluation auf Test-Set (falls implementiert)
3. Metriken aus Logs extrahieren
4. Vergleich mit anderen Architekturen (LSTM, Transformer, etc.)



