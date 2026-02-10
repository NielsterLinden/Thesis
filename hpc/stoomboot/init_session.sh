#!/bin/bash
# HPC session init: activate conda, cd to project, git pull, set WandB env.
# Usage: source hpc/stoomboot/init_session.sh   (from project root)
#    or: source /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/init_session.sh
#    or: add alias thesis='source .../init_session.sh' to ~/.bashrc and type 'thesis'

set -e

PROJECT_DIR="/project/atlas/users/nterlind/Thesis-Code"
CONDA_ENV="/data/atlas/users/nterlind/venvs/thesis-ml"
WANDB_DIR="/data/atlas/users/nterlind/outputs/wandb"

# 1. Activate conda
set +u
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
  echo "Conda: $CONDA_DEFAULT_ENV"
else
  echo "Warning: conda not found in PATH"
fi
set -u

# 2. Go to project
cd "$PROJECT_DIR"
echo "Project: $PROJECT_DIR"

# 3. Git pull
git pull
echo "Branch: $(git branch --show-current)"

# 4. WandB env
export WANDB_DIR="$WANDB_DIR"
export WANDB_MODE=online

# Load API key from one of these (keeps key out of repo):
if [ -f "$PROJECT_DIR/hpc/stoomboot/.wandb_env" ]; then
  source "$PROJECT_DIR/hpc/stoomboot/.wandb_env"
elif [ -f "$HOME/.wandb_api_key" ]; then
  export WANDB_API_KEY=$(cat "$HOME/.wandb_api_key")
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "Hint: set WANDB_API_KEY via ~/.wandb_api_key or hpc/stoomboot/.wandb_env"
else
  echo "WandB: ready (mode=$WANDB_MODE)"
fi

# Optional: show job count (if condor available)
if command -v condor_q &>/dev/null; then
  COUNT=$(condor_q -nobatch 2>/dev/null | tail -1)
  echo "Jobs: $COUNT"
fi

echo "Session ready."
