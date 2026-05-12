#!/bin/bash
#SBATCH --job-name=hubert_ecg_p2_resume
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zx9981@partner.kit.edu
#SBATCH --partition=gpu_a100_il,gpu_h100_il

# Resume Phase 2 only from a prior job's trainer checkpoint (e.g. after TIMEOUT).
# Set RESUME_CKPT to the HuBERT_ECG_best_<OLD_JOBID>.pt from the interrupted run.
#
# Continuation of HuBERT-ECG run 4644580 (A7.1 / H1, base weights, unfrozen):
#   - Phase 1 finished (hubert_ecg_phase1_best_4644580.pt was saved)
#   - Phase 2 ran >=2 epochs before SLURM TIMEOUT; latest trainer state is in
#     HuBERT_ECG_best_4644580.pt (contains model + optimizer + scheduler + RNG + history + epoch).
#   - Default below resumes that Phase 2 exactly where it stopped.
# Override at submit time with: sbatch --export=ALL,RESUME_CKPT=outputs/checkpoints/HuBERT_ECG_best_<JOBID>.pt ...

RESUME_CKPT="${RESUME_CKPT:-outputs/checkpoints/HuBERT_ECG_best_4644580.pt}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Resume checkpoint: ${RESUME_CKPT}"

PROJECT_ROOT="/home/ka/ka_aifb/ka_zx9981/workspace/ma-thesis/MA-thesis-1"

if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "ERROR: Project root not found at: ${PROJECT_ROOT}"
    exit 1
fi

cd "${PROJECT_ROOT}"
export SLURM_SUBMIT_DIR="${PROJECT_ROOT}"
echo "Changed to project directory: ${PROJECT_ROOT}"

module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
else
    echo "ERROR: venv not found"
    exit 1
fi

TORCH_LIB_DIR=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -iv cudnn | grep -v '^$' | tr '\n' ':' | sed 's/:$//')
fi
if [ -n "$TORCH_LIB_DIR" ] && [ -d "$TORCH_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"
fi

export ICUSTAYS_PATH="${SLURM_SUBMIT_DIR}/data/labeling/labels_csv/icustays.csv"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

if [ ! -f "${RESUME_CKPT}" ]; then
    echo "ERROR: Resume checkpoint not found: ${RESUME_CKPT}"
    exit 1
fi

echo "============================================================"
echo "HuBERT-ECG — resume Phase 2 only (--skip-phase1 --resume-phase2)"
echo "============================================================"

python scripts/training/icu_24h/hubert_ecg/train_hubert_ecg_24h.py \
    --skip-phase1 \
    --resume-phase2 "${RESUME_CKPT}"

TRAINING_EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: ${TRAINING_EXIT_CODE}"

exit $TRAINING_EXIT_CODE
