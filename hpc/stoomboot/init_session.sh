#!/bin/bash
# HPC session init: activate conda, cd to project, set WandB env.
# Usage: source hpc/stoomboot/init_session.sh   (from project root)
#    or: source /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/init_session.sh
#    or: add alias thesis='source .../init_session.sh' to ~/.bashrc and type 'thesis'

# If this script is sourced, shell options (e.g. -e, -u) would persist and can
# inadvertently terminate your SSH session on a harmless error/typo. Save and
# restore options to keep interactive shells friendly.
__THESIS_OLD_SET_OPTS="$(set +o)"
set -e

PROJECT_DIR="/project/atlas/users/nterlind/Thesis-Code"
CONDA_ENV="/data/atlas/users/nterlind/venvs/thesis-ml"
WANDB_DIR="/data/atlas/users/nterlind/outputs/wandb"
WANDB_CACHE_DIR="/data/atlas/users/nterlind/outputs/wandb_cache"

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

# 3. WandB env (cache on /data avoids ~/.cache/wandb artifact quota on home)
export WANDB_DIR="$WANDB_DIR"
export WANDB_CACHE_DIR="$WANDB_CACHE_DIR"
mkdir -p "$WANDB_CACHE_DIR" 2>/dev/null || true
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

# Optional: condor summary (match "Total for <user>:" from condor_q output, not the last line)
if command -v condor_q &>/dev/null; then
  _condor_out=$(condor_q 2>/dev/null || true)
  COUNT=$(printf '%s\n' "$_condor_out" | grep -F "Total for ${USER}:" | head -1)
  [ -z "$COUNT" ] && COUNT=$(printf '%s\n' "$_condor_out" | grep -F "Total for query:" | head -1)
  unset _condor_out
  echo "Jobs: ${COUNT:-}"
fi

echo "Session ready."

# Restore prior shell options (important when sourced).
eval "$__THESIS_OLD_SET_OPTS"
unset __THESIS_OLD_SET_OPTS
