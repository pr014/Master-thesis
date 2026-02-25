#!/bin/bash
#SBATCH --job-name=hubert_ecg_24h_phase1
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=06:00:00          # Phase 1 only (frozen backbone, head-only): ~4-5h
#SBATCH --nodes=1                # Anzahl Knoten (meist 1)
#SBATCH --ntasks=1              # Anzahl Tasks pro Knoten (meist 1)
#SBATCH --cpus-per-task=4       # CPU-Kerne für DataLoader (4-8 empfohlen)
#SBATCH --mem=16G               # RAM-Anforderung (16GB sollte für Batch-Size 32 reichen)
#SBATCH --gres=gpu:1            # GPU-Anforderung (1 GPU, jede verfügbare)
#SBATCH --mail-type=BEGIN,END   # E-Mail bei Start und Ende
#SBATCH --mail-user=zx9981@partner.kit.edu  # Deine E-Mail-Adresse
#SBATCH --partition=gpu_a100_il,gpu_h100_il  # Flexible Partition: A100 oder H100

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Ensure we run from the project directory
PROJECT_ROOT="/home/ka/ka_aifb/ka_zx9981/workspace/ma-thesis/MA-thesis-1"

# Verify project root exists
if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "ERROR: Project root not found at: ${PROJECT_ROOT}"
    exit 1
fi

cd "${PROJECT_ROOT}"
export SLURM_SUBMIT_DIR="${PROJECT_ROOT}"
echo "Changed to project directory: ${PROJECT_ROOT}"

# ============================================================================
# ANPASSUNGEN FÜR BW UNI CLUSTER 3.0:
# ============================================================================

# 1. MODULE LADEN
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

# 2. VIRTUAL ENVIRONMENT AKTIVIEREN
if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
    echo "Activated venv from: ${PROJECT_ROOT}/venv"
else
    echo "ERROR: venv not found at ${PROJECT_ROOT}/venv/bin/activate"
    exit 1
fi

# 3. CHECK DEPENDENCIES
echo "Checking dependencies..."
python -c "import torch; import transformers; import biosppy; import scipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Required dependencies not found (torch, transformers, biosppy, scipy)"
    echo "Please install dependencies: pip install torch transformers biosppy scipy"
    exit 1
fi
echo "All dependencies available"

# 4. Fix cuDNN version incompatibility: Use PyTorch's bundled cuDNN instead of module's
TORCH_LIB_DIR=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -iv cudnn | grep -v '^$' | tr '\n' ':' | sed 's/:$//')
fi
if [ -n "$TORCH_LIB_DIR" ] && [ -d "$TORCH_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"
    echo "Using PyTorch's bundled cuDNN from: $TORCH_LIB_DIR"
fi

# 5. ENVIRONMENT VARIABLES SETZEN
export ICUSTAYS_PATH="${SLURM_SUBMIT_DIR}/data/labeling/labels_csv/icustays.csv"
if [ ! -f "${ICUSTAYS_PATH}" ]; then
  echo "ERROR: icustays.csv not found at: ${ICUSTAYS_PATH}"
  exit 1
fi

# GPU-Einstellung
export CUDA_VISIBLE_DEVICES=0

# PYTHON PATH SETZEN
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# ============================================================================
# TRAINING STARTEN
# ============================================================================

echo "============================================================"
echo "Training HuBERT-ECG - Phase 1 Only (Frozen Backbone / Head-Only)"
echo "Pretrained Weights: Enabled | Demographics: Disabled"
echo "============================================================"

# Training starten
python scripts/training/icu_24h/hubert_ecg/train_hubert_ecg_24h.py

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
    echo "=== Phase 1 Results ==="
    grep -i "Phase 1.*Best validation loss" "$LOG_FILE" || grep -A 2 "Phase 1 completed" "$LOG_FILE" || echo "Phase 1 results not found"
    
    echo ""
    echo "=== Phase 2 Results ==="
    grep -i "Phase 2.*Best validation loss" "$LOG_FILE" || grep -A 2 "Phase 2 completed" "$LOG_FILE" || echo "Phase 2 results not found"
    
    echo ""
    echo "=== Test Set Results ==="
    grep -A 10 "TRAINING RESULTS SUMMARY" "$LOG_FILE" || echo "Keine Test-Ergebnisse gefunden"
    
    echo ""
    echo "=== Model Checkpoints ==="
    if [ -d "outputs/checkpoints" ]; then
        ls -lh outputs/checkpoints/hubert_ecg_*${SLURM_JOB_ID}*.pt 2>/dev/null || echo "Keine HuBERT-ECG Checkpoints gefunden"
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

