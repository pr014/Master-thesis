#!/bin/bash
#SBATCH --job-name=viol10_deepecg
#SBATCH --output=outputs/logs/viol10_deepecg_%j.out
#SBATCH --error=outputs/logs/viol10_deepecg_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il,gpu_h100_il
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zx9981@partner.kit.edu

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
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

EXTRA=()
if [ -n "${VIOLIN_EXTRA_ARGS:-}" ]; then
  read -r -a EXTRA <<< "${VIOLIN_EXTRA_ARGS}"
fi

python "${PROJECT_ROOT}/scripts/analysis/results/plot_mae_violins.py" \
  --config "${PROJECT_ROOT}/configs/analysis/mae_violin_10days_feature_development_deepecg.yaml" \
  "${EXTRA[@]}"

echo "End time: $(date)"
