#!/bin/bash
#SBATCH --job-name=bland_altman
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_il,gpu_h100_il

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

PROJECT_ROOT="/home/ka/ka_aifb/ka_zx9981/workspace/ma-thesis/MA-thesis-1"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export ICUSTAYS_PATH="${PROJECT_ROOT}/data/labeling/labels_csv/icustays.csv"

module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

# cuDNN fix (like other DL jobs)
if [ -n "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -iv cudnn | grep -v '^$' | tr '\n' ':' | sed 's/:$//')
fi
TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$TORCH_LIB_DIR" ] && [ -d "$TORCH_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"
fi

echo "Generating Bland-Altman plots for Hybrid CNN-LSTM and DeepECG-SL (best jobs)..."
python scripts/analysis/results/bland_altman_all.py \
    --archs "Hybrid CNN-LSTM" "DeepECG-SL" \
    --out-dir outputs/bland_altman

echo ""
echo "End time: $(date)"
echo "Output: outputs/bland_altman/bland_altman_job_3362967.png (Hybrid)"
echo "Output: outputs/bland_altman/bland_altman_job_3346715.png (DeepECG)"
