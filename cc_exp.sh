#!/bin/bash
#SBATCH --job-name=value-plasticity
#SBATCH --output=slurm/logs/main_%A_%a.out
#SBATCH --error=slurm/logs/main_%A_%a.err
#SBATCH --array=0-29
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --account=def-lelis

set -u

echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Array task : ${SLURM_ARRAY_TASK_ID:-0}"
echo "Host       : $(hostname)"
echo "Start      : $(date)"
echo ""

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm/logs

# Environment
module --force purge
module load StdEnv/2023 gcc/12.3

source ~/ENV/rl/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg

SEED="${SLURM_ARRAY_TASK_ID:-0}"
BASE_OUTPUT_DIR="${PLASTICITY_OUTPUT_DIR:-results/value_prediction_plasticity}"
RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/seed_${SEED}"

mkdir -p "$RUN_OUTPUT_DIR"

echo "Seed       : $SEED"
echo "Output dir : $RUN_OUTPUT_DIR"
echo ""

# Optional sanity check
python -c "import torch, matplotlib; print('torch', torch.__version__); print('matplotlib', matplotlib.__version__)"

# Run one seed per array task. The final arguments enforce separate per-seed outputs.
python -u main.py "$@" --num-seeds 1 --seed-offset "$SEED" --output-dir "$RUN_OUTPUT_DIR"
EXIT_CODE=$?

echo ""
echo "End : $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
