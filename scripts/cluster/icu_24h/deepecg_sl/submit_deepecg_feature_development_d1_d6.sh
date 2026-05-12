#!/usr/bin/env bash
# Submit 6 separate SLURM jobs: DeepECG-SL with P1+AG4+H3 base and D1…D6 feature overlays.
#
# Usage (from repo root on login node):
#   bash scripts/cluster/icu_24h/deepecg_sl/submit_deepecg_feature_development_d1_d6.sh
#
# Each submission → one Job ID (six jobs total). Logs: outputs/logs/slurm_<jobid>.out
#
# Override sbatch script if needed (full path recommended):
#   SBATCH_SCRIPT=/path/to/MA-thesis-1/scripts/cluster/icu_24h/deepecg_sl/train_deepecg_sl_24h.sbatch bash ...

set -euo pipefail

# Script lives at scripts/cluster/icu_24h/deepecg_sl/ — four .. to repo root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

SBATCH_SCRIPT="${SBATCH_SCRIPT:-${ROOT}/scripts/cluster/icu_24h/deepecg_sl/train_deepecg_sl_24h.sbatch}"
BASE_MODEL="${ROOT}/configs/model/deepecg_sl/deepecg_sl_p1_ag4_h3.yaml"
EXP_DIR="${ROOT}/configs/experiments/deepecg_sl/feature_development"
SWEEP_ID="${SWEEP_ID:-deepecg_feature_dev_p1_ag4_h3}"

for d in 1 2 3 4 5 6; do
  EXP="${EXP_DIR}/d${d}.yaml"
  TRIAL_ID="D${d}"
  echo "Submitting ${TRIAL_ID} → sbatch --job-name=deepecg_${TRIAL_ID} ${SBATCH_SCRIPT} …"
  sbatch \
    --job-name="deepecg_${TRIAL_ID}" \
    "${SBATCH_SCRIPT}" \
    --model-config "${BASE_MODEL}" \
    --experiment-config "${EXP}" \
    --trial-id "${TRIAL_ID}" \
    --sweep-id "${SWEEP_ID}"
done

echo "Done: 6 jobs submitted (check squeue / e-mail)."
