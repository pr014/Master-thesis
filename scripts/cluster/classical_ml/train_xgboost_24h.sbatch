#!/bin/bash
#SBATCH --job-name=xgboost_24h
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=04:00:00          # Maximale Laufzeit (Format: HH:MM:SS oder Tage-HH:MM:SS)
#SBATCH --nodes=1                # Anzahl Knoten (meist 1)
#SBATCH --ntasks=1              # Anzahl Tasks pro Knoten (meist 1)
#SBATCH --cpus-per-task=8       # CPU-Kerne für XGBoost (mehr Kerne = schnelleres Training)
#SBATCH --mem=32G               # RAM-Anforderung (32GB für Feature-Extraktion und XGBoost)
#SBATCH --gres=gpu:1            # GPU-Anforderung (erforderlich für Partition, wird aber nicht genutzt)
#SBATCH --mail-type=BEGIN,END   # E-Mail bei Start und Ende
#SBATCH --mail-user=zx9981@partner.kit.edu  # Deine E-Mail-Adresse
#SBATCH --partition=gpu_a100_il,gpu_h100_il  # Flexible Partition (GPU wird angefordert, aber XGBoost nutzt nur CPU)

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Ensure we run from the submit directory (robust paths)
cd "${SLURM_SUBMIT_DIR}"

# ============================================================================
# ANPASSUNGEN FÜR BW UNI CLUSTER 3.0:
# ============================================================================

# 1. MODULE LADEN
# Prüfe verfügbare Module mit: module avail python/cuda/cudnn
module load devel/python/3.12.3-gnu-14.2
# CUDA nicht nötig für XGBoost (CPU-only)

# 2. VIRTUAL ENVIRONMENT AKTIVIEREN
# Option A: venv (wenn du venv erstellt hast)
source venv/bin/activate

# Option B: conda (wenn du conda verwendest)
# conda activate ma-thesis

# 3. ENVIRONMENT VARIABLES SETZEN
export ICUSTAYS_PATH="${SLURM_SUBMIT_DIR}/data/labeling/labels_csv/icustays.csv"
if [ ! -f "${ICUSTAYS_PATH}" ]; then
  echo "ERROR: icustays.csv not found at: ${ICUSTAYS_PATH}"
  exit 1
fi

# Optional: DATA_DIR setzen (falls benötigt)
# export DATA_DIR="/path/to/data"

# 4. PYTHON PATH SETZEN
# Stellt sicher, dass Python das Projekt findet
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# ============================================================================
# TRAINING STARTEN
# ============================================================================

# Training starten
# Usage: python train_xgboost_24h.py [config_path]
# Default: configs/classical_ml/xgboost_handcrafted.yaml
# For DL features: python train_xgboost_24h.py configs/classical_ml/xgboost_dl_features.yaml

CONFIG_PATH="${1:-configs/classical_ml/xgboost_handcrafted.yaml}"
echo "Using config: ${CONFIG_PATH}"

python scripts/training/classical_ml/train_xgboost_24h.py "${CONFIG_PATH}"

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
    echo "=== Validation Results ==="
    grep -A 10 "Validation Results" "$LOG_FILE" || echo "Nicht gefunden"
    
    echo ""
    echo "=== Test Results ==="
    grep -A 10 "Test Results" "$LOG_FILE" || echo "Keine Test-Ergebnisse gefunden"
    
    echo ""
    echo "=== Feature Extraction Info ==="
    grep -E "Extracting|features shape|samples:" "$LOG_FILE" | tail -10 || echo "Keine Feature-Info gefunden"
    
    echo ""
    echo "=== Top Feature Importances ==="
    grep -A 20 "Top 20 Feature Importances" "$LOG_FILE" || echo "Nicht gefunden"
    
    echo ""
    echo "=== Model Checkpoints ==="
    if [ -d "outputs/checkpoints" ]; then
        ls -lh outputs/checkpoints/xgboost_*.pkl 2>/dev/null | tail -3 || echo "Keine Checkpoints gefunden"
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

