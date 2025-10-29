#!/bin/bash
set -exo pipefail  # Added 'x' for verbose output

echo "=== Stoomboot Thesis ML Job ==="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${CONDOR_JOB_ID:-N/A}"
echo "User: $(whoami)"
echo "PATH: $PATH"
echo "Current directory contents:"
ls -la

# Navigate to code directory
echo "Changing to code directory..."
cd /project/atlas/users/nterlind/Thesis-Code
echo "Now in: $(pwd)"

# Activate conda environment
echo "Activating conda environment..."
set +u
eval "$(conda shell.bash hook)"
conda activate /data/atlas/users/nterlind/venvs/thesis-ml
set -u
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# Run training (all settings come from configs)
echo "Starting training..."
python -m thesis_ml.train "$@"

EXIT_CODE=$?
echo "Job completed at: $(date) with exit code: $EXIT_CODE"
exit $EXIT_CODE
