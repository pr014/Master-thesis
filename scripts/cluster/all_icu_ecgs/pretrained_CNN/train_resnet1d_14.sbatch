#!/bin/bash
#SBATCH --job-name=resnet1d14
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=03:00:00          # Maximale Laufzeit (Format: HH:MM:SS oder Tage-HH:MM:SS)
#SBATCH --nodes=1                # Anzahl Knoten (meist 1)
#SBATCH --ntasks=1              # Anzahl Tasks pro Knoten (meist 1)
#SBATCH --cpus-per-task=4       # CPU-Kerne für DataLoader (4-8 empfohlen)
#SBATCH --mem=16G               # RAM-Anforderung (16GB sollte für Batch-Size 64 reichen)
#SBATCH --gres=gpu:1            # GPU-Anforderung (1 GPU)
#SBATCH --mail-type=BEGIN,END   # E-Mail bei Start und Ende
#SBATCH --mail-user=zx9981@partner.kit.edu  # Deine E-Mail-Adresse
#SBATCH --partition=gpu_a100_il,gpu_h100_il  # GPU-Partition für Training (flexibel)

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Ensure we run from project root
cd "${SLURM_SUBMIT_DIR}"

# ============================================================================
# ANPASSUNGEN FÜR BW UNI CLUSTER 3.0:
# ============================================================================

# 1. MODULE LADEN
# Prüfe verfügbare Module mit: module avail python/cuda/cudnn
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
# cuDNN nicht verfügbar oder nicht nötig

# 2. VIRTUAL ENVIRONMENT AKTIVIEREN
# Option A: venv (wenn du venv erstellt hast)
source venv/bin/activate

# Option B: conda (wenn du conda verwendest)
# conda activate ma-thesis

# 3. ENVIRONMENT VARIABLES SETZEN
# icustays.csv (robust via submit dir)
export ICUSTAYS_PATH="${SLURM_SUBMIT_DIR}/data/labeling/labels_csv/icustays.csv"
if [ ! -f "${ICUSTAYS_PATH}" ]; then
  echo "ERROR: icustays.csv not found at: ${ICUSTAYS_PATH}"
  exit 1
fi

# Optional: DATA_DIR setzen (falls benötigt)
# export DATA_DIR="/path/to/data"

# GPU-Einstellung (wird automatisch von SLURM gesetzt, aber sicherheitshalber)
export CUDA_VISIBLE_DEVICES=0

# 4. PYTHON PATH SETZEN
# Stellt sicher, dass Python das Projekt findet
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# ============================================================================
# TRAINING STARTEN
# ============================================================================

# Training starten
python scripts/training/all_icu_ecgs/pretrained_CNN/train_resnet1d_14.py

TRAINING_EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Job completed"

# ============================================================================
# ERGEBNISSE EXTRAHIEREN UND AUSGEBEN (für E-Mail)
# ============================================================================

echo ""
echo "============================= TRAINING RESULTS ============================="

# Log-Datei finden
LOG_FILE="outputs/logs/slurm_${SLURM_JOB_ID}.out"

if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "=== Best Validation Loss ==="
    grep -i "Best validation loss" "$LOG_FILE" || echo "Nicht gefunden"
    
    echo ""
    echo "=== Test Set Results ==="
    grep -A 10 "Test Set Results" "$LOG_FILE" || echo "Keine Test-Ergebnisse gefunden"
    
    echo ""
    echo "=== Final Training Metrics (letzte 5 Epochs) ==="
    grep "Epoch.*Train Loss\|Epoch.*Val Loss" "$LOG_FILE" | tail -5 || echo "Keine Epoch-Metriken gefunden"
    
    echo ""
    echo "=== Class Distribution ==="
    grep -A 1 "Class distribution" "$LOG_FILE" | tail -3 || echo "Nicht gefunden"
    
    echo ""
    echo "=== Model Checkpoints ==="
    if [ -d "outputs/checkpoints" ]; then
        ls -lh outputs/checkpoints/*.pt 2>/dev/null | tail -3 || echo "Keine Checkpoints gefunden"
    else
        echo "Checkpoint-Verzeichnis nicht gefunden"
    fi
else
    echo "Log-Datei nicht gefunden: $LOG_FILE"
fi

echo ""
echo "============================= END RESULTS ============================="
echo ""
echo "Vollständige Logs verfügbar in: $LOG_FILE"
echo "Checkpoints verfügbar in: outputs/checkpoints/"

# Exit mit Training-Exit-Code
exit $TRAINING_EXIT_CODE

