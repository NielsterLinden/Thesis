#!/bin/bash
set -euo pipefail

echo "=== Stoomboot Thesis ML Job ==="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${CONDOR_JOB_ID:-N/A}"

# Navigate to code directory
cd /project/atlas/users/nterlind/Thesis-Code

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /data/atlas/users/nterlind/venvs/thesis-ml

# Run training (all settings come from configs)
python -m thesis_ml.train "$@"

EXIT_CODE=$?
echo "Job completed at: $(date) with exit code: $EXIT_CODE"
exit $EXIT_CODE
