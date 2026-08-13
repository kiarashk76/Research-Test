#!/bin/bash
#SBATCH --account=aip-lelis
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=exp-main
#SBATCH --output=/home/aghakasi/scratch/TestResearch/llm_exploration/logs/main_%j.out
#SBATCH --error=/home/aghakasi/scratch/TestResearch/llm_exploration/logs/main_%j.err

set -euo pipefail

# ============================================================================
# Experiment Configuration
# ============================================================================

ENV_NAME="SimpleGridEnv"
AGENT_NAME="SimpleLLMAgent"
LLM_NAME="GEMINI"

ENV_OVERRIDES='{
    "size": 5,
    "max_steps": 50
}'

AGENT_OVERRIDES='{
    "n_actions": 25
}'

SEEDS="23 45 68"

MAX_STEPS=200
NUM_EPISODES=""

TAG="test"


# ============================================================================
# Setup
# ============================================================================

source ~/ENV/llm/bin/activate
source /scratch/aghakasi/TestResearch/llm_exploration/llm/llm_info.sh

export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd /scratch/aghakasi/TestResearch/llm_exploration


# ============================================================================
# Run
# ============================================================================

ARGS=(
    --env "$ENV_NAME"
    --agent "$AGENT_NAME"
    --llm "$LLM_NAME"
    --env-overrides "$ENV_OVERRIDES"
    --agent-overrides "$AGENT_OVERRIDES"
    --seeds $SEEDS
    --tag "$TAG"
)

if [[ -n "$MAX_STEPS" ]]; then
    ARGS+=(--max-steps "$MAX_STEPS")
fi

if [[ -n "$NUM_EPISODES" ]]; then
    ARGS+=(--num-episodes "$NUM_EPISODES")
fi

python main.py "${ARGS[@]}"