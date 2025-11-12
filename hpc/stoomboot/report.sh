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
python -m thesis_ml.cli.reports "$@"

EXIT_CODE=$?
echo "Job completed at: $(date) with exit code: $EXIT_CODE"

# Close stdout/stderr to allow HTCondor to close the files
exec 1>&-
exec 2>&-

# Small delay to ensure files are closed, then organize into grouped directory
sleep 0.5
OUTPUT_DIR="/data/atlas/users/nterlind/logs/report_${CONDOR_CLUSTER_ID}"
mkdir -p "$OUTPUT_DIR"
mv "/data/atlas/users/nterlind/logs/report_${CONDOR_CLUSTER_ID}.out" "${OUTPUT_DIR}/report_${CONDOR_CLUSTER_ID}.out" 2>/dev/null || true
mv "/data/atlas/users/nterlind/logs/report_${CONDOR_CLUSTER_ID}.err" "${OUTPUT_DIR}/report_${CONDOR_CLUSTER_ID}.err" 2>/dev/null || true
mv "/data/atlas/users/nterlind/logs/report_${CONDOR_CLUSTER_ID}.log" "${OUTPUT_DIR}/report_${CONDOR_CLUSTER_ID}.log" 2>/dev/null || true

exit $EXIT_CODE
