#!/usr/bin/env bash
# Submit many parallel Optuna workers for DeepECG-SL (one shared study / SQLite DB).
#
# Usage (repo root):
#   ./scripts/cluster/icu_24h/deepecg_sl/launch_optuna_deepecg_sl_workers.sh
#
# Optional (export before run):
#   DEEPECG_OPTUNA_RUN_DIR=.../outputs/tuning/deepecg_sl/2026-04-05
#   NUM_JOBS=20              # default: 20 parallel workers (each runs N_TRIALS_PER_JOB trials)
#   N_TRIALS_PER_JOB=1       # default: 1
#   OPTUNA_EXPORT_CSV=0      # default: 0 (avoids many writers overwriting CSV on NFS+SQLite)
#   DRY_RUN=1
#
# Job IDs are written to ${DEEPECG_OPTUNA_RUN_DIR}/submitted_sbatch_job_ids.txt

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

unset OPTUNA_DEEPECG_SMOKE_OBJECTIVE || true

DEEPECG_OPTUNA_RUN_DIR="${DEEPECG_OPTUNA_RUN_DIR:-${PROJECT_ROOT}/outputs/tuning/deepecg_sl/2026-04-05}"
export OPTUNA_FINAL_ENV="${DEEPECG_OPTUNA_RUN_DIR}/env.sh"

NUM_JOBS="${NUM_JOBS:-20}"
N_TRIALS_PER_JOB="${N_TRIALS_PER_JOB:-1}"
OPTUNA_EXPORT_CSV="${OPTUNA_EXPORT_CSV:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${OPTUNA_FINAL_ENV}" ]]; then
  echo "ERROR: Missing ${OPTUNA_FINAL_ENV}"
  exit 1
fi

# shellcheck disable=SC1090
source "${OPTUNA_FINAL_ENV}"

export N_TRIALS_PER_JOB
export OPTUNA_EXPORT_CSV

STUDY_DIR="${PROJECT_ROOT}/outputs/tuning/deepecg_sl/${OPTUNA_OUTPUT_DATE}/${OPTUNA_STUDY_NAME}"
mkdir -p "${STUDY_DIR}/trial_configs" "${STUDY_DIR}/exports" "${DEEPECG_OPTUNA_RUN_DIR}/storage"
JOB_LIST="${DEEPECG_OPTUNA_RUN_DIR}/submitted_sbatch_job_ids.txt"
: > "${JOB_LIST}"

echo "OPTUNA_FINAL_ENV=${OPTUNA_FINAL_ENV}"
echo "OPTUNA_STORAGE=${OPTUNA_STORAGE}"
echo "OPTUNA_STUDY_NAME=${OPTUNA_STUDY_NAME}"
echo "OPTUNA_OUTPUT_DATE=${OPTUNA_OUTPUT_DATE}"
echo "NUM_JOBS=${NUM_JOBS}  N_TRIALS_PER_JOB=${N_TRIALS_PER_JOB}  OPTUNA_EXPORT_CSV=${OPTUNA_EXPORT_CSV}"
echo "Job IDs -> ${JOB_LIST}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 -> no sbatch."
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found."
  exit 1
fi

PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

case "${OPTUNA_STORAGE}" in
  sqlite:///*)
    echo "Initializing Optuna SQLite study/schema once (avoids parallel CREATE TABLE race)..."
    "${PYTHON_BIN}" -c "import optuna; optuna.create_study(study_name='${OPTUNA_STUDY_NAME}', storage='${OPTUNA_STORAGE}', direction='minimize', load_if_exists=True)"
    ;;
esac

SBATCH_SCRIPT="${PROJECT_ROOT}/scripts/cluster/icu_24h/deepecg_sl/optuna_deepecg_sl_worker.sbatch"
for ((i = 1; i <= NUM_JOBS; i++)); do
  jid="$(sbatch --export=ALL "${SBATCH_SCRIPT}" | awk '{print $4}')"
  echo "${jid}" >> "${JOB_LIST}"
  echo "Submitted ${i}/${NUM_JOBS}  job_id=${jid}"
done

echo "Done. Submitted ${NUM_JOBS} jobs."
echo "Monitor: squeue -u \"\$USER\" -n optuna_deepecg_sl"
echo "After completion: export CSV via scripts/tuning/export_optuna_deepecg_sl_study.py if OPTUNA_EXPORT_CSV=0."
