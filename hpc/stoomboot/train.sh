#!/bin/bash
set -exo pipefail

echo "=== Stoomboot Thesis ML Training Job ==="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${CONDOR_JOB_ID:-N/A}"

# Navigate to code directory
cd /project/atlas/users/nterlind/Thesis-Code

# WandB: write local files to large filesystem (not project dir, avoids quota)
export WANDB_DIR=/data/atlas/users/nterlind/outputs/wandb
export WANDB_MODE=online

# Activate conda environment
set +u
eval "$(conda shell.bash hook)"
conda activate /data/atlas/users/nterlind/venvs/thesis-ml
set -u
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

# Run training (all settings come from configs)
echo "Starting training..."
python -m thesis_ml.cli.train "$@"

EXIT_CODE=$?
echo "Job completed at: $(date) with exit code: $EXIT_CODE"
exit $EXIT_CODE
