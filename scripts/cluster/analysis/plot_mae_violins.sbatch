#!/bin/bash
#SBATCH --job-name=mae_violins
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il,gpu_h100_il
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zx9981@partner.kit.edu

# Usage:
#   sbatch scripts/cluster/analysis/plot_mae_violins.sbatch
#
# Optional overrides:
#   VIOLIN_CONFIG=configs/analysis/mae_violin_baseline.yaml sbatch scripts/cluster/analysis/plot_mae_violins.sbatch
#   VIOLIN_EXTRA_ARGS="--error-level ecg --out outputs/results/models/violin/custom.pdf" sbatch ...
#
# One plot group from a multi-group YAML (see mae_violin_10days_four_plots.yaml header):
#   VIOLIN_CONFIG=configs/analysis/mae_violin_10days_four_plots.yaml \
#   VIOLIN_EXTRA_ARGS='--plot-group-index 0' sbatch scripts/cluster/analysis/plot_mae_violins.sbatch
#
# Four parallel tasks (array must be on ``sbatch`` or as #SBATCH in this file — not a shell comment):
#   export VIOLIN_CONFIG=configs/analysis/mae_violin_10days_four_plots.yaml
#   sbatch --array=0-3 scripts/cluster/analysis/plot_mae_violins.sbatch
# Each task gets ``--plot-group-index $SLURM_ARRAY_TASK_ID`` automatically (unless you already set it in VIOLIN_EXTRA_ARGS).

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
  echo "Array job: ${SLURM_ARRAY_JOB_ID}  task_id: ${SLURM_ARRAY_TASK_ID:-}"
fi
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start time: $(date)"

PROJECT_ROOT="/home/ka/ka_aifb/ka_zx9981/workspace/ma-thesis/MA-thesis-1"
cd "${PROJECT_ROOT}"
mkdir -p outputs/logs outputs/results/models/violin

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export ICUSTAYS_PATH="${PROJECT_ROOT}/data/labeling/labels_csv/icustays.csv"

module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/venv/bin/activate"
else
  echo "ERROR: venv not found at ${PROJECT_ROOT}/venv/bin/activate"
  exit 1
fi

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -iv cudnn | grep -v '^$' | tr '\n' ':' | sed 's/:$//')
fi
TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || true)
if [ -n "${TORCH_LIB_DIR}" ] && [ -d "${TORCH_LIB_DIR}" ]; then
  export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

CONFIG_PATH="${VIOLIN_CONFIG:-configs/analysis/mae_violin_baseline.yaml}"
# In a job array, map each task to ``--plot-group-index`` (YAML plot_groups order: 0..3).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [[ "${VIOLIN_EXTRA_ARGS:-}" != *--plot-group-index* ]]; then
  VIOLIN_EXTRA_ARGS="${VIOLIN_EXTRA_ARGS:+${VIOLIN_EXTRA_ARGS} }--plot-group-index ${SLURM_ARRAY_TASK_ID}"
fi
EXTRA=()
if [ -n "${VIOLIN_EXTRA_ARGS:-}" ]; then
  read -r -a EXTRA <<< "${VIOLIN_EXTRA_ARGS}"
fi

python "${PROJECT_ROOT}/scripts/analysis/results/plot_mae_violins.py" \
  --config "${CONFIG_PATH}" \
  "${EXTRA[@]}"

echo "End time: $(date)"
