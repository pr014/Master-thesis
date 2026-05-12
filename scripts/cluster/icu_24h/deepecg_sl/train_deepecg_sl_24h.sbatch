#!/bin/bash
#SBATCH --job-name=deepecg_sl_24h_weighted
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=02:00:00          # Budget 2h — Skript nur Phase 2; bei Timeout ggf. erhöhen
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

# 3. INSTALL FAIRSEQ-SIGNALS IF NEEDED
echo "Checking for fairseq-signals..."
python -c "import fairseq_signals" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "fairseq-signals not found. Installing..."
    # Fix: pip 25.3 rejects omegaconf 2.0.x due to metadata warnings, but fairseq-signals needs 2.0.x for 'II' symbol
    # Solution: Install omegaconf 2.0.6 explicitly first, then install fairseq-signals
    echo "Installing compatible omegaconf version (2.0.6) first..."
    pip install "omegaconf==2.0.6" || {
        echo "WARNING: Failed to install omegaconf 2.0.6, trying alternative approach..."
        # Alternative: Downgrade pip temporarily to avoid metadata validation
        pip install "pip<24.1" --quiet 2>/dev/null || true
        pip install "omegaconf==2.0.6" || pip install "omegaconf>=2.0.0,<2.1"
    }
    # Now install fairseq-signals (it should use the already-installed omegaconf 2.0.6)
    echo "Installing fairseq-signals..."
    pip install git+https://github.com/HeartWise-AI/fairseq-signals.git
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install fairseq-signals"
        exit 1
    fi
    # Verify that omegaconf still has the II symbol
    python -c "from omegaconf import II; print('✓ omegaconf II import successful')" || {
        echo "ERROR: omegaconf II import failed after installation"
        echo "Reinstalling omegaconf 2.0.6..."
        pip install --force-reinstall --no-deps "omegaconf==2.0.6"
    }
    echo "fairseq-signals installed successfully"
else
    echo "fairseq-signals already installed"
    # Verify omegaconf version is compatible
    python -c "from omegaconf import II" 2>/dev/null || {
        echo "WARNING: omegaconf II import failed. Reinstalling omegaconf 2.0.6..."
        pip install --force-reinstall --no-deps "omegaconf==2.0.6"
    }
fi

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
# HuggingFace API Key für gated Models (heartwise/wcr_*) – falls Modell nicht lokal:
# export HUGGINGFACE_API_KEY="hf_your_token"
# Oder Modell vorab nach data/pretrained_weights/deepecg_sl/WCR_77_classes/ legen
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
echo "Training DeepECG-SL (WCR + Self-Supervised Pretraining) for 24h Dataset"
echo "Two-Phase Training: Phase 1 (Frozen) + Phase 2 (Fine-tuning)"
echo "============================================================"

if [ -n "${MODEL_CONFIG_PATH:-}" ]; then
    echo "MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH}"
fi
if [ -n "${EXPERIMENT_CONFIG_PATH:-}" ]; then
    echo "EXPERIMENT_CONFIG_PATH=${EXPERIMENT_CONFIG_PATH}"
fi
if [ -n "${TRIAL_ID:-}" ]; then
    echo "TRIAL_ID=${TRIAL_ID}"
fi
if [ -n "${SWEEP_ID:-}" ]; then
    echo "SWEEP_ID=${SWEEP_ID}"
fi

CLI_ARGS=("$@")
if [ -n "${TRAIN_EXTRA_ARGS:-}" ]; then
    read -r -a EXTRA_ARGS <<< "${TRAIN_EXTRA_ARGS}"
    CLI_ARGS+=("${EXTRA_ARGS[@]}")
fi
echo "Forwarded CLI args: ${CLI_ARGS[*]:-(none)}"

python scripts/training/icu_24h/deepecg_sl/train_deepecg_sl_24h.py "${CLI_ARGS[@]}"

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
        ls -lh outputs/checkpoints/deepecg_sl_*${SLURM_JOB_ID}*.pt 2>/dev/null || echo "Keine DeepECG-SL Checkpoints gefunden"
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

