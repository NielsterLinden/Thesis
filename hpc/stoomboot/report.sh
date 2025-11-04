#!/bin/bash
set -exo pipefail

echo "=== Stoomboot Thesis ML Report Job ==="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${CONDOR_JOB_ID:-N/A}"

# Navigate to code directory
cd /project/atlas/users/nterlind/Thesis-Code

# Activate conda environment
set +u
eval "$(conda shell.bash hook)"
conda activate /data/atlas/users/nterlind/venvs/thesis-ml
set -u
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

# Run report generation
echo "Starting report generation..."
python -m thesis_ml.reports "$@"

EXIT_CODE=$?
echo "Job completed at: $(date) with exit code: $EXIT_CODE"
exit $EXIT_CODE
