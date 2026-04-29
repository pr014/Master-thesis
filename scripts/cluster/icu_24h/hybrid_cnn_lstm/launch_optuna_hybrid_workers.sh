#!/usr/bin/env bash
# Launch the Hybrid CNN-LSTM Optuna tuning job.
#
# Usage (from repo root, after: chmod +x .../launch_optuna_hybrid_workers.sh):
#   ./scripts/cluster/icu_24h/hybrid_cnn_lstm/launch_optuna_hybrid_workers.sh
#
# Optional overrides (export before running):
#   OPTUNA_OUTPUT_DATE=2026-04-03          # default: today (YYYY-MM-DD)
#   OPTUNA_STUDY_NAME=my_study             # default: hybrid_cnn_lstm_hpo_tab_<YYYYMMDD>
#   NUM_JOBS=1                             # default: 1 worker/job id
#   N_TRIALS_PER_JOB=100                   # default: 100 trials in that one job
#   OPTUNA_EXPORT_CSV=0                    # default: 1 (same as DeepECG); use 0 to skip per-job CSV races on NFS+SQLite
#   DRY_RUN=1                              # print what would be submitted, do not sbatch
#
# Artifacts:
#   outputs/tuning/hybrid_cnn_lstm/${OPTUNA_OUTPUT_DATE}/${OPTUNA_STUDY_NAME}/
#     storage/optuna.db
#     trial_configs/
#     exports/optuna_trials.csv            # all trials, written if OPTUNA_EXPORT_CSV=1
#     exports/optuna_best_trial.csv        # best trial params + stored metrics, written if OPTUNA_EXPORT_CSV=1
#
# SQLite on shared NFS + many writers: prefer PostgreSQL for OPTUNA_STORAGE if you see DB lock issues.
# Parallel workers must not all call create_study on an empty DB at once (SQLAlchemy CREATE TABLE race).
# This script runs a one-process optuna.create_study(..., load_if_exists=True) before sbatch.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

unset OPTUNA_HYBRID_SMOKE_OBJECTIVE || true

OPTUNA_OUTPUT_DATE="${OPTUNA_OUTPUT_DATE:-$(date +%F)}"
DATE_COMPACT="${OPTUNA_OUTPUT_DATE//-/}"
if [[ -z "${OPTUNA_STUDY_NAME:-}" ]]; then
  OPTUNA_STUDY_NAME="hybrid_cnn_lstm_hpo_tab_${DATE_COMPACT}"
fi
NUM_JOBS="${NUM_JOBS:-1}"
N_TRIALS_PER_JOB="${N_TRIALS_PER_JOB:-100}"
OPTUNA_EXPORT_CSV="${OPTUNA_EXPORT_CSV:-1}"
DRY_RUN="${DRY_RUN:-0}"

STUDY_DIR="${PROJECT_ROOT}/outputs/tuning/hybrid_cnn_lstm/${OPTUNA_OUTPUT_DATE}/${OPTUNA_STUDY_NAME}"
STORAGE_DIR="${STUDY_DIR}/storage"
mkdir -p "${STORAGE_DIR}" "${STUDY_DIR}/trial_configs" "${STUDY_DIR}/exports"

DB_FILE="${STORAGE_DIR}/optuna.db"
# Optuna/SQLAlchemy: absolute path -> sqlite:///<absolute>  (three slashes after colon + leading / in path)
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///${DB_FILE}}"

export OPTUNA_OUTPUT_DATE
export OPTUNA_STUDY_NAME
export OPTUNA_STORAGE
export N_TRIALS_PER_JOB
export OPTUNA_EXPORT_CSV

JOB_LIST="${STUDY_DIR}/submitted_sbatch_job_ids.txt"
: > "${JOB_LIST}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "OPTUNA_OUTPUT_DATE=${OPTUNA_OUTPUT_DATE}"
echo "OPTUNA_STUDY_NAME=${OPTUNA_STUDY_NAME}"
echo "OPTUNA_STORAGE=${OPTUNA_STORAGE}"
echo "NUM_JOBS=${NUM_JOBS}  N_TRIALS_PER_JOB=${N_TRIALS_PER_JOB}  OPTUNA_EXPORT_CSV=${OPTUNA_EXPORT_CSV}"
echo "STUDY_DIR=${STUDY_DIR}"
echo "Job IDs will be appended to: ${JOB_LIST}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 -> no sbatch will be submitted."
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this script on the cluster login node."
  exit 1
fi

PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

case "${OPTUNA_STORAGE}" in
  sqlite:///*)
    echo "Initializing Optuna SQLite study/schema once (avoids parallel CREATE TABLE race)..."
    echo "  (same TPESampler + MedianPruner as optuna_hybrid_cnn_lstm_worker — see scripts/tuning/hybrid_cnn_lstm_study_config.py)"
    "${PYTHON_BIN}" -c "
import sys
from pathlib import Path
root = Path('${PROJECT_ROOT}')
sys.path[:0] = [str(root / 'scripts' / 'tuning')]
import hybrid_cnn_lstm_study_config as c
c.create_hybrid_study(study_name='${OPTUNA_STUDY_NAME}', storage='${OPTUNA_STORAGE}')
print(' ', c.study_config_summary())
"
    ;;
esac

SBATCH_SCRIPT="${PROJECT_ROOT}/scripts/cluster/icu_24h/hybrid_cnn_lstm/optuna_hybrid_cnn_lstm_worker.sbatch"
for ((i = 1; i <= NUM_JOBS; i++)); do
  # --export=ALL: forward current exported env into the job (matches typical DeepECG manual workflow)
  jid="$(sbatch --export=ALL "${SBATCH_SCRIPT}" | awk '{print $4}')"
  echo "${jid}" >> "${JOB_LIST}"
  echo "Submitted ${i}/${NUM_JOBS}  job_id=${jid}"
done

echo "Done. Submitted ${NUM_JOBS} jobs."
echo "Monitor: squeue -u \"\$USER\""
echo "After completion, check study dir and exports/optuna_trials.csv plus exports/optuna_best_trial.csv."
