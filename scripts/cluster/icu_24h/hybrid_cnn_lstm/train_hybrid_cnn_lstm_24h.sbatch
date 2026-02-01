#!/bin/bash
#SBATCH --job-name=hybrid_cnn_lstm_24h_weighted
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=03:00:00          # Maximale Laufzeit (Format: HH:MM:SS oder Tage-HH:MM:SS)
#SBATCH --nodes=1                # Anzahl Knoten (meist 1)
#SBATCH --ntasks=1              # Anzahl Tasks pro Knoten (meist 1)
#SBATCH --cpus-per-task=4       # CPU-Kerne für DataLoader (4-8 empfohlen)
#SBATCH --mem=16G               # RAM-Anforderung (16GB sollte für Batch-Size 64 reichen)
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
# Use absolute path to project root (hardcoded for reliability)
PROJECT_ROOT="/home/ka/ka_aifb/ka_zx9981/workspace/ma-thesis/MA-thesis-1"

# Verify project root exists
if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "ERROR: Project root not found at: ${PROJECT_ROOT}"
    echo "Current directory: $(pwd)"
    echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-not set}"
    exit 1
fi

cd "${PROJECT_ROOT}"
export SLURM_SUBMIT_DIR="${PROJECT_ROOT}"
echo "Changed to project directory: ${PROJECT_ROOT}"
echo "Verification: venv exists: $([ -f "${PROJECT_ROOT}/venv/bin/activate" ] && echo 'YES' || echo 'NO')"

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
if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
    echo "Activated venv from: ${PROJECT_ROOT}/venv"
else
    echo "ERROR: venv not found at ${PROJECT_ROOT}/venv/bin/activate"
    exit 1
fi

# Fix cuDNN version incompatibility: Use PyTorch's bundled cuDNN instead of module's
# PyTorch was compiled against cuDNN 9.8.0, but module system provides 9.7.1
# Strategy: Get PyTorch's lib directory and prepend it, then remove all cuDNN paths
TORCH_LIB_DIR=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)

# Remove ALL cuDNN paths from LD_LIBRARY_PATH (more aggressive)
if [ -n "$LD_LIBRARY_PATH" ]; then
    # Split by colon, filter out any path containing cudnn (case-insensitive), rejoin
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -iv cudnn | grep -v '^$' | tr '\n' ':' | sed 's/:$//')
fi

# Also check and clean CUDA_HOME if it exists
if [ -n "$CUDA_HOME" ]; then
    # Don't remove CUDA_HOME, but ensure it doesn't point to incompatible cuDNN
    unset CUDA_HOME
fi

# Prepend PyTorch's lib directory to ensure bundled cuDNN is found first
if [ -n "$TORCH_LIB_DIR" ] && [ -d "$TORCH_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"
    echo "Using PyTorch's bundled cuDNN from: $TORCH_LIB_DIR"
    echo "LD_LIBRARY_PATH (first 3 entries): $(echo $LD_LIBRARY_PATH | cut -d: -f1-3)"
    
    # Force PyTorch to use its bundled cuDNN via LD_PRELOAD
    CUDNN_LIB="${TORCH_LIB_DIR}/libcudnn.so"
    if [ -f "$CUDNN_LIB" ]; then
        export LD_PRELOAD="${CUDNN_LIB}:${LD_PRELOAD}"
        echo "Forcing cuDNN via LD_PRELOAD: $CUDNN_LIB"
    else
        echo "WARNING: libcudnn.so not found in $TORCH_LIB_DIR"
        # Try to find it
        CUDNN_LIB=$(find "${TORCH_LIB_DIR}" -name "libcudnn*.so*" | head -1)
        if [ -n "$CUDNN_LIB" ]; then
            export LD_PRELOAD="${CUDNN_LIB}:${LD_PRELOAD}"
            echo "Found and using: $CUDNN_LIB"
        fi
    fi
else
    echo "WARNING: Could not find PyTorch lib directory"
fi

# Verify cuDNN fix before training
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'cuDNN available: {torch.backends.cudnn.is_available()}'); print(f'cuDNN enabled: {torch.backends.cudnn.enabled}')" 2>&1 || echo "WARNING: Could not verify cuDNN"

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

# GPU-Einstellung (wird automatisch von SLURM gesetzt, aber sicherheitshalber)
export CUDA_VISIBLE_DEVICES=0

# 4. PYTHON PATH SETZEN
# Stellt sicher, dass Python das Projekt findet
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# ============================================================================
# TRAINING STARTEN
# ============================================================================

echo "============================================================"
echo "Training Hybrid CNN-LSTM with SQRT Class Weights for 24h Dataset"
echo "============================================================"

# Training starten
python scripts/training/icu_24h/hybrid_cnn_lstm/train_hybrid_cnn_lstm_24h.py

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
    grep -A 10 "TRAINING RESULTS SUMMARY" "$LOG_FILE" || echo "Keine Test-Ergebnisse gefunden"
    
    echo ""
    echo "=== Final Training Metrics (letzte 5 Epochs) ==="
    grep "Epoch.*Train Loss\|Epoch.*Val Loss" "$LOG_FILE" | tail -5 || echo "Keine Epoch-Metriken gefunden"
    
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

