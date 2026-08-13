#!/bin/bash
#SBATCH --account=aip-lelis
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=llmexp-setup
#SBATCH --output=/scratch/aghakasi/TestResearch/slurm/logs/setup_%j.out
#SBATCH --error=/scratch/aghakasi/TestResearch/slurm/logs/setup_%j.err
#
# One-time (or refresh-on-demand) venv build for llm_exploration.
# Run with: sbatch slurm/setup_env.sh
#
# All packages come from the local Compute Canada wheelhouse (avail_wheels),
# so this installs with --no-index and never touches the internet.

set -euo pipefail

REPO_ROOT="/scratch/aghakasi/TestResearch"
VENV_DIR="$SCRATCH/venvs/llm_exploration"

module purge
module load StdEnv/2023 python/3.11.5

mkdir -p "$(dirname "$VENV_DIR")"
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --no-index --upgrade pip
pip install --no-index -r "$REPO_ROOT/llm_exploration/requirements.txt"

echo "Venv ready at $VENV_DIR"
