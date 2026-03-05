#!/bin/bash
#SBATCH --job-name=xresnet_ptbxl_24h
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zx9981@partner.kit.edu
#SBATCH --partition=gpu_a100_il,gpu_h100_il

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

PROJECT_ROOT="/home/ka/ka_aifb/ka_zx9981/workspace/ma-thesis/MA-thesis-1"

if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "ERROR: Project root not found at: ${PROJECT_ROOT}"
    exit 1
fi

cd "${PROJECT_ROOT}"
export SLURM_SUBMIT_DIR="${PROJECT_ROOT}"

module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
    echo "Activated venv from: ${PROJECT_ROOT}/venv"
else
    echo "ERROR: venv not found at ${PROJECT_ROOT}/venv/bin/activate"
    exit 1
fi

# cuDNN fix for PyTorch
TORCH_LIB_DIR=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -iv cudnn | grep -v '^$' | tr '\n' ':' | sed 's/:$//')
fi
if [ -n "$TORCH_LIB_DIR" ] && [ -d "$TORCH_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"
    CUDNN_LIB="${TORCH_LIB_DIR}/libcudnn.so"
    if [ -f "$CUDNN_LIB" ]; then
        export LD_PRELOAD="${CUDNN_LIB}:${LD_PRELOAD}"
    fi
fi

export ICUSTAYS_PATH="${SLURM_SUBMIT_DIR}/data/labeling/labels_csv/icustays.csv"
if [ ! -f "${ICUSTAYS_PATH}" ]; then
    echo "ERROR: icustays.csv not found at: ${ICUSTAYS_PATH}"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

echo "============================================================"
echo "Training XResNet1D-101 (PTB-XL Pretrained) for 24h Dataset"
echo "Task: LOS Regression + Mortality (Multi-Task)"
echo "============================================================"

python scripts/training/icu_24h/xresnet1d_ptbxl/train_xresnet1d_ptbxl_24h.py

TRAINING_EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Job completed"

echo ""
echo "============================= TRAINING RESULTS ============================="
LOG_FILE="outputs/logs/slurm_${SLURM_JOB_ID}.out"
if [ -f "$LOG_FILE" ]; then
    grep -i "Best validation loss" "$LOG_FILE" || echo "Nicht gefunden"
    grep -A 10 "TRAINING RESULTS SUMMARY" "$LOG_FILE" || echo "Keine Test-Ergebnisse gefunden"
    ls -lh outputs/checkpoints/*.pt 2>/dev/null | tail -3 || echo "Keine Checkpoints gefunden"
fi
echo "============================= END RESULTS ============================="

exit $TRAINING_EXIT_CODE
