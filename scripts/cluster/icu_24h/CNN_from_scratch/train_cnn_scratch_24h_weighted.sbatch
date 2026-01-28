#!/bin/bash
#SBATCH --job-name=cnn_scratch_24h_weighted
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1            # GPU-Anforderung (1 GPU, jede verfügbare)
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zx9981@partner.kit.edu
#SBATCH --partition=gpu_a100_il,gpu_h100_il  # Flexible Partition: A100 oder H100

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Training with Class Weights for 24h Dataset (Balanced Method)"

# Ensure we run from the submit directory (robust paths)
cd "${SLURM_SUBMIT_DIR}"

# ============================================================================
# BW UNI CLUSTER 3.0 SETUP:
# ============================================================================

# 1. MODULE LADEN
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

# 2. VIRTUAL ENVIRONMENT AKTIVIEREN
source venv/bin/activate

# 3. ENVIRONMENT VARIABLES SETZEN
export ICUSTAYS_PATH="${SLURM_SUBMIT_DIR}/data/labeling/labels_csv/icustays.csv"
if [ ! -f "${ICUSTAYS_PATH}" ]; then
  echo "ERROR: icustays.csv not found at: ${ICUSTAYS_PATH}"
  exit 1
fi
export CUDA_VISIBLE_DEVICES=0

# 4. PYTHON PATH SETZEN
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# ============================================================================
# TRAINING STARTEN (24h Dataset with Class Weights)
# ============================================================================

# Training starten mit weighted config
python scripts/training/icu_24h/CNN_from_scratch/train_cnn_scratch_24h_weighted.py

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
    grep -A 15 "Test Set Results" "$LOG_FILE" || echo "Keine Test-Ergebnisse gefunden"
    
    echo ""
    echo "=== Balanced Accuracy & Macro Metrics ==="
    grep -E "Balanced Accuracy|Macro Precision|Macro Recall|Macro F1" "$LOG_FILE" || echo "Nicht gefunden"
    
    echo ""
    echo "=== Final Training Metrics (letzte 5 Epochs) ==="
    grep "Epoch.*Train Loss\|Epoch.*Val Loss" "$LOG_FILE" | tail -5 || echo "Keine Epoch-Metriken gefunden"
    
    echo ""
    echo "=== Class Distribution ==="
    grep -A 1 "Class distribution" "$LOG_FILE" | tail -3 || echo "Nicht gefunden"
    
    echo ""
    echo "=== Model Checkpoints ==="
    if [ -d "outputs/checkpoints" ]; then
        ls -lh outputs/checkpoints/CNNScratch_best.pt 2>/dev/null || echo "Best checkpoint nicht gefunden"
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
echo ""
echo "Config verwendet: configs/24h_weighted/balanced_weights.yaml (Balanced Method)"

# Exit mit Training-Exit-Code
exit $TRAINING_EXIT_CODE

