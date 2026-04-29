#!/bin/bash
#SBATCH --job-name=optuna_hybrid_cnn_lstm
#SBATCH --output=outputs/logs/slurm_optuna_hybrid_%j.out
#SBATCH --error=outputs/logs/slurm_optuna_hybrid_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zx9981@partner.kit.edu
#SBATCH --partition=gpu_a100_il,gpu_h100_il

# Optuna worker for Hybrid CNN-LSTM. Set before sbatch, e.g.:
#   export OPTUNA_STORAGE="postgresql://user:pass@host:5432/optuna"
#   export OPTUNA_STUDY_NAME="hybrid_cnn_lstm_v1"
#   export N_TRIALS_PER_JOB=100
# Or submit many workers in one go (same pattern as manual DeepECG runs):
#   ./scripts/cluster/icu_24h/hybrid_cnn_lstm/launch_optuna_hybrid_workers.sh
# Optional: fixed date folder for artifacts (default: today)
#   export OPTUNA_OUTPUT_DATE="2026-04-03"
# Trial YAMLs + exports: outputs/tuning/hybrid_cnn_lstm/${OPTUNA_OUTPUT_DATE:-<today>}/${OPTUNA_STUDY_NAME}/
# Prefer PostgreSQL for many parallel jobs; avoid SQLite on shared NFS with heavy concurrency.

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

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
else
    echo "ERROR: venv not found"
    exit 1
fi

python -c "import optuna" 2>/dev/null || pip install "optuna>=4.8.0,<5"

TORCH_LIB_DIR=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -iv cudnn | grep -v '^$' | tr '\n' ':' | sed 's/:$//')
fi
if [ -n "$TORCH_LIB_DIR" ] && [ -d "$TORCH_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"
fi

export ICUSTAYS_PATH="${SLURM_SUBMIT_DIR}/data/labeling/labels_csv/icustays.csv"
if [ ! -f "${ICUSTAYS_PATH}" ]; then
  echo "ERROR: icustays.csv not found at: ${ICUSTAYS_PATH}"
  exit 1
fi
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

if [ -z "${OPTUNA_STORAGE:-}" ]; then
    echo "ERROR: Set OPTUNA_STORAGE (e.g. postgresql://... or sqlite:////abs/path/optuna.db)"
    exit 1
fi
if [ -z "${OPTUNA_STUDY_NAME:-}" ]; then
    echo "ERROR: Set OPTUNA_STUDY_NAME"
    exit 1
fi

N_TRIALS="${N_TRIALS_PER_JOB:-1}"
echo "OPTUNA_STORAGE=${OPTUNA_STORAGE}"
echo "OPTUNA_STUDY_NAME=${OPTUNA_STUDY_NAME}"
echo "N_TRIALS_PER_JOB=${N_TRIALS}"
if [ -n "${OPTUNA_OUTPUT_DATE:-}" ]; then
  echo "OPTUNA_OUTPUT_DATE=${OPTUNA_OUTPUT_DATE}"
fi

# CSV export is on by default in the worker; set OPTUNA_EXPORT_CSV=0 to skip.
EXTRA=()
if [ "${OPTUNA_EXPORT_CSV:-1}" = "0" ]; then
    EXTRA+=(--no-export-csv-after)
fi

OUTPUT_DATE_ARGS=()
if [ -n "${OPTUNA_OUTPUT_DATE:-}" ]; then
  OUTPUT_DATE_ARGS+=(--output-date "${OPTUNA_OUTPUT_DATE}")
fi

python scripts/tuning/optuna_hybrid_cnn_lstm_worker.py \
    --storage "${OPTUNA_STORAGE}" \
    --study-name "${OPTUNA_STUDY_NAME}" \
    --n-trials "${N_TRIALS}" \
    "${OUTPUT_DATE_ARGS[@]}" \
    "${EXTRA[@]}" \
    "$@"

exit $?
