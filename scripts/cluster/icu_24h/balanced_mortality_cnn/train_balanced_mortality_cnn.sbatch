#!/bin/bash
#SBATCH --job-name=bal_mort_cnn
#SBATCH --output=outputs/logs/slurm_balanced_mortality_cnn_%j.out
#SBATCH --error=outputs/logs/slurm_balanced_mortality_cnn_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zx9981@partner.kit.edu
#SBATCH --partition=gpu_a100_il,gpu_h100_il

# Isolated pipeline: CNNScratch binary mortality (balanced_mortality + manifest).
# Augmentations: Gaussian noise, amplitude scaling, baseline wander, lead dropout (see experiment YAML).

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
cd "${SLURM_SUBMIT_DIR}"

module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

MANIFEST="${SLURM_SUBMIT_DIR}/data/icu_ecgs_24h/balanced_mortality/manifest.csv"
if [ ! -f "${MANIFEST}" ]; then
  echo "ERROR: manifest not found: ${MANIFEST}"
  exit 1
fi

python scripts/training/icu_24h/balanced_mortality_cnn/train_balanced_mortality_cnn.py \
  --config configs/experiments/balanced_mortality_cnn.yaml

EXIT_CODE=$?
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

LOG_FILE="outputs/logs/slurm_balanced_mortality_cnn_${SLURM_JOB_ID}.out"
if [ -f "$LOG_FILE" ]; then
  echo "=== [balanced_mortality_cnn] tail metrics ==="
  grep "\[balanced_mortality_cnn\]" "$LOG_FILE" | tail -20 || true
fi
echo "Checkpoints: outputs/checkpoints/balanced_mortality_cnn/"
exit $EXIT_CODE
