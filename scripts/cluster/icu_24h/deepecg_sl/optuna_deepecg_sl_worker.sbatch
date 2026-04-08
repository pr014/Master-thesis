#!/bin/bash
#SBATCH --job-name=optuna_deepecg_sl
#SBATCH --output=outputs/logs/slurm_optuna_deepecg_%j.out
#SBATCH --error=outputs/logs/slurm_optuna_deepecg_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zx9981@partner.kit.edu
#SBATCH --partition=gpu_a100_il,gpu_h100_il
# Avoid nodes that were stuck as Unavailable (ReqNodeNotAvail); remove if obsolete.
#SBATCH --exclude=uc2n901

# Optuna worker for DeepECG-SL. Many workers: use
#   ./scripts/cluster/icu_24h/deepecg_sl/launch_optuna_deepecg_sl_workers.sh
# Single job, e.g.:
#   export OPTUNA_FINAL_ENV="${PROJECT_ROOT}/outputs/tuning/deepecg_sl/2026-04-05/env.sh"
#   sbatch scripts/cluster/icu_24h/deepecg_sl/optuna_deepecg_sl_worker.sbatch
# (env.sh sets OPTUNA_STORAGE SQLite, OPTUNA_STUDY_NAME, OPTUNA_OUTPUT_DATE — see README in that folder.)
#
# Or set manually:
#   export OPTUNA_STORAGE="postgresql://user:pass@host:5432/optuna"
#   export OPTUNA_STUDY_NAME="..."
# Optional: OPTUNA_OUTPUT_DATE, OPTUNA_DEEPECG_SEARCH_SPACE, OPTUNA_BASE_TUNING_YAML, N_TRIALS_PER_JOB
# Trial YAMLs + exports: outputs/tuning/deepecg_sl/${OPTUNA_OUTPUT_DATE}/${OPTUNA_STUDY_NAME}/
# Many parallel workers: prefer PostgreSQL; SQLite on shared NFS can lock.

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

if [ -n "${OPTUNA_FINAL_ENV:-}" ] && [ -f "${OPTUNA_FINAL_ENV}" ]; then
    echo "Sourcing OPTUNA_FINAL_ENV=${OPTUNA_FINAL_ENV}"
    # shellcheck disable=SC1090
    source "${OPTUNA_FINAL_ENV}"
fi

module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
else
    echo "ERROR: venv not found"
    exit 1
fi

python -c "import fairseq_signals" 2>/dev/null || {
    echo "ERROR: fairseq_signals missing in venv; run a normal DeepECG train job once or pip install fairseq-signals"
    exit 1
}

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
SEARCH_SPACE="${OPTUNA_DEEPECG_SEARCH_SPACE:-p2}"
BASE_YAML="${OPTUNA_BASE_TUNING_YAML:-configs/tuning/deepecg_sl/optuna_base_p1_tabular.yaml}"
echo "OPTUNA_STORAGE=${OPTUNA_STORAGE}"
echo "OPTUNA_STUDY_NAME=${OPTUNA_STUDY_NAME}"
echo "N_TRIALS_PER_JOB=${N_TRIALS}"
echo "OPTUNA_DEEPECG_SEARCH_SPACE=${SEARCH_SPACE}"
echo "OPTUNA_BASE_TUNING_YAML=${BASE_YAML}"
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

python scripts/tuning/optuna_deepecg_sl_worker.py \
    --storage "${OPTUNA_STORAGE}" \
    --study-name "${OPTUNA_STUDY_NAME}" \
    --n-trials "${N_TRIALS}" \
    --search-space "${SEARCH_SPACE}" \
    --base-tuning-yaml "${BASE_YAML}" \
    "${OUTPUT_DATE_ARGS[@]}" \
    "${EXTRA[@]}" \
    "$@"

exit $?
