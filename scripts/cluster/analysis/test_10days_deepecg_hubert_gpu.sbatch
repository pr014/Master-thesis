#!/bin/bash
#SBATCH --job-name=test10d_deepecg_hubert
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=01:00:00   # Inference-only; raise if Slurm kills before DeepECG+HuBERT finish
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il,gpu_h100_il

# Finishes the baseline test_10days_split batch on GPU: DeepECG-SL (3119699) + HuBERT (3436209).
# Uses the SAME default job order as test_10days.py + --resume so completed jobs are skipped
# and baseline_10days_split.csv keeps ALL rows (do not pass only two --job IDs — that would
# overwrite the CSV with just those columns).
#
# Submit:
#   sbatch scripts/cluster/analysis/test_10days_deepecg_hubert_gpu.sbatch
#
# Checkpoints needed for the heavy models:
#   outputs/checkpoints/DeepECG_SL_best_3119699.pt
#   outputs/checkpoints/HuBERT_ECG_best_3436209.pt

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start time: $(date)"

PROJECT_ROOT="/home/ka/ka_aifb/ka_zx9981/workspace/ma-thesis/MA-thesis-1"
cd "${PROJECT_ROOT}"
mkdir -p outputs/logs

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/venv/bin/activate"
fi

# Prefer PyTorch's bundled libs over a conflicting system cuDNN (same as other DL sbatch jobs)
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

OUT_DIR="${PROJECT_ROOT}/outputs/results/test_10days_split"
CSV="${OUT_DIR}/baseline_10days_split.csv"

# Explicit seven-job list (matches legacy baseline evaluation order). Pass --resume to skip finished rows.
python "${PROJECT_ROOT}/scripts/test_10days/test_10days.py" \
  --resume \
  --device cuda \
  --out-dir "${OUT_DIR}" \
  --csv "${CSV}" \
  --job 3400484 \
  --job 3119696 \
  --job 3119697 \
  --job 3119698 \
  --job 3217131 \
  --job 3119699 \
  --job 3436209

echo "End time: $(date)"
