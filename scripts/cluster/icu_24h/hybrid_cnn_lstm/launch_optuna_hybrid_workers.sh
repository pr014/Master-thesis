#!/usr/bin/env bash
# Launch many Optuna worker jobs for Hybrid CNN-LSTM (same env pattern as DeepECG optuna_deepecg_sl_worker.sbatch).
#
# Usage (from repo root, after: chmod +x .../launch_optuna_hybrid_workers.sh):
#   ./scripts/cluster/icu_24h/hybrid_cnn_lstm/launch_optuna_hybrid_workers.sh
#
# Optional overrides (export before running):
#   OPTUNA_OUTPUT_DATE=2026-04-03          # default: today (YYYY-MM-DD)
#   OPTUNA_STUDY_NAME=my_study             # default: p1 -> hybrid_cnn_lstm_hpo_p1_tab_<YYYYMMDD>; p2 -> ..._p2_tab_<YYYYMMDD>
#   OPTUNA_HYBRID_SEARCH_SPACE=p2          # default: unset -> p1 narrow; p2 = wide (worker default base YAML: tabular features ON)
#   NUM_JOBS=25                            # default: 25 parallel workers (each 1 trial)
#   N_TRIALS_PER_JOB=1                     # default: 1 (recommended for parallel)
#   OPTUNA_EXPORT_CSV=0                    # default: 1 (same as DeepECG); use 0 to skip per-job CSV races on NFS+SQLite
#   DRY_RUN=1                              # print what would be submitted, do not sbatch
#
# Artifacts:
#   outputs/tuning/hybrid_cnn_lstm/${OPTUNA_OUTPUT_DATE}/${OPTUNA_STUDY_NAME}/
#     storage/optuna.db
#     trial_configs/
#     exports/optuna_trials.csv            # written by each worker if OPTUNA_EXPORT_CSV=1 (last writer wins)
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
  if [[ "${OPTUNA_HYBRID_SEARCH_SPACE:-}" == "p2" ]]; then
    OPTUNA_STUDY_NAME="hybrid_cnn_lstm_hpo_p2_tab_${DATE_COMPACT}"
  else
    OPTUNA_STUDY_NAME="hybrid_cnn_lstm_hpo_p1_tab_${DATE_COMPACT}"
  fi
fi
NUM_JOBS="${NUM_JOBS:-25}"
N_TRIALS_PER_JOB="${N_TRIALS_PER_JOB:-1}"
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
export OPTUNA_HYBRID_SEARCH_SPACE="${OPTUNA_HYBRID_SEARCH_SPACE:-}"

JOB_LIST="${STUDY_DIR}/submitted_sbatch_job_ids.txt"
: > "${JOB_LIST}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "OPTUNA_OUTPUT_DATE=${OPTUNA_OUTPUT_DATE}"
echo "OPTUNA_STUDY_NAME=${OPTUNA_STUDY_NAME}"
echo "OPTUNA_STORAGE=${OPTUNA_STORAGE}"
echo "OPTUNA_HYBRID_SEARCH_SPACE=${OPTUNA_HYBRID_SEARCH_SPACE:-<unset -> p1>}"
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
    "${PYTHON_BIN}" -c "import optuna; optuna.create_study(study_name='${OPTUNA_STUDY_NAME}', storage='${OPTUNA_STORAGE}', direction='minimize', load_if_exists=True)"
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
echo "After completion, check study dir and exports/optuna_trials.csv (or re-export from DB if you used OPTUNA_EXPORT_CSV=0)."
