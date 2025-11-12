#!/bin/bash
set -exo pipefail

echo "=== Stoomboot Thesis ML Training Job ==="
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

# Run training (all settings come from configs)
echo "Starting training..."
python -m thesis_ml.cli.train "$@"

EXIT_CODE=$?
echo "Job completed at: $(date) with exit code: $EXIT_CODE"

# Wait for HTCondor to close files, then organize into grouped directory
echo "Organizing log files into grouped directory..."
sleep 2
OUTPUT_DIR="/data/atlas/users/nterlind/logs/train_${CONDOR_CLUSTER_ID}"
mkdir -p "$OUTPUT_DIR"

# Try to move files, with error handling
if mv "/data/atlas/users/nterlind/logs/train_${CONDOR_CLUSTER_ID}.out" "${OUTPUT_DIR}/train_${CONDOR_CLUSTER_ID}.out" 2>/dev/null; then
    echo "Moved .out file successfully"
else
    echo "Warning: Could not move .out file (may still be in use)"
fi

if mv "/data/atlas/users/nterlind/logs/train_${CONDOR_CLUSTER_ID}.err" "${OUTPUT_DIR}/train_${CONDOR_CLUSTER_ID}.err" 2>/dev/null; then
    echo "Moved .err file successfully"
else
    echo "Warning: Could not move .err file (may still be in use)"
fi

if mv "/data/atlas/users/nterlind/logs/train_${CONDOR_CLUSTER_ID}.log" "${OUTPUT_DIR}/train_${CONDOR_CLUSTER_ID}.log" 2>/dev/null; then
    echo "Moved .log file successfully"
else
    echo "Warning: Could not move .log file (may still be in use)"
fi

exit $EXIT_CODE
