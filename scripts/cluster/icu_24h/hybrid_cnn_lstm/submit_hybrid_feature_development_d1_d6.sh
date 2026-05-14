#!/usr/bin/env bash
# Submit 6 separate SLURM jobs: Hybrid CNN-LSTM with P1+AG4+H4 base and D1…D6 feature overlays.
#
# Usage (from repo root on login node):
#   bash scripts/cluster/icu_24h/hybrid_cnn_lstm/submit_hybrid_feature_development_d1_d6.sh
#
# Each submission → one Job ID (six jobs total). Logs: outputs/logs/slurm_<jobid>.out
#
# Override sbatch script if needed (full path recommended):
#   SBATCH_SCRIPT=/path/to/MA-thesis-1/scripts/cluster/icu_24h/hybrid_cnn_lstm/train_hybrid_cnn_lstm_24h.sbatch bash ...

set -euo pipefail

# Script lives at scripts/cluster/icu_24h/hybrid_cnn_lstm/ — four .. to repo root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

SBATCH_SCRIPT="${SBATCH_SCRIPT:-${ROOT}/scripts/cluster/icu_24h/hybrid_cnn_lstm/train_hybrid_cnn_lstm_24h.sbatch}"
BASE_MODEL="${ROOT}/configs/model/hybrid_cnn_lstm/hybrid_cnn_lstm_p1_ag4_h4.yaml"
EXP_DIR="${ROOT}/configs/experiments/hybrid_cnn_lstm/feature_development"
SWEEP_ID="${SWEEP_ID:-hybrid_feature_dev_p1_ag4_h4}"

for d in 1 2 3 4 5 6; do
  EXP="${EXP_DIR}/d${d}.yaml"
  TRIAL_ID="D${d}"
  echo "Submitting ${TRIAL_ID} → sbatch --job-name=hybrid_${TRIAL_ID} ${SBATCH_SCRIPT} …"
  sbatch \
    --job-name="hybrid_${TRIAL_ID}" \
    "${SBATCH_SCRIPT}" \
    --model-config "${BASE_MODEL}" \
    --experiment-config "${EXP}" \
    --trial-id "${TRIAL_ID}" \
    --sweep-id "${SWEEP_ID}"
done

echo "Done: 6 jobs submitted (check squeue / e-mail)."
