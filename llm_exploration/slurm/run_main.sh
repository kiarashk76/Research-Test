#!/bin/bash
#SBATCH --account=aip-lelis
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=llmexp-main
#SBATCH --output=/scratch/aghakasi/TestResearch/slurm/logs/main_%j.out
#SBATCH --error=/scratch/aghakasi/TestResearch/slurm/logs/main_%j.err
#
# Runs llm_exploration/main.py (run_multi_seed_experiment over SimpleGridEnv +
# ProgrammaticLLMAgent). No GPU requested: every agent path currently in
# main.py's __main__ block runs with device="cpu".
#
# Prereqs (one-time):
#   1. sbatch slurm/setup_env.sh                     # builds the venv on $SCRATCH
#   2. cp slurm/secrets.env.example slurm/secrets.env # fill in real API key, chmod 600
#
# Submit with: sbatch slurm/run_main.sh

set -euo pipefail

REPO_ROOT="/scratch/aghakasi/TestResearch"
SECRETS_FILE="$REPO_ROOT/slurm/secrets.env"

source "~/ENV/llm/bin/activate"

# Job-scoped caches must live on $SCRATCH, not $HOME or the wiped /tmp.
export HF_HOME="$SCRATCH/.cache/huggingface"
export XDG_CACHE_HOME="$SCRATCH/.cache"

if [[ ! -f "$SECRETS_FILE" ]]; then
    echo "Missing $SECRETS_FILE — copy slurm/secrets.env.example and fill in API credentials." >&2
    exit 1
fi
source "$SECRETS_FILE"

cd "$REPO_ROOT/llm_exploration"
python main.py
