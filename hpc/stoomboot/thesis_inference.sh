#!/bin/bash
# GPU Condor/local runner for thesis-ml inference jobs (e.g. eval Stage B).
# Same conda + ulimit pattern as train.sh; arguments are passed to stage_b_inference.py.
set -exo pipefail

echo "=== Stoomboot Thesis ML inference ==="
ulimit -n 8192 2>/dev/null || ulimit -n 4096 2>/dev/null || true
echo "ulimit -n: $(ulimit -n)"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${CONDOR_JOB_ID:-N/A}"

cd /project/atlas/users/nterlind/Thesis-Code

set +u
eval "$(conda shell.bash hook)"
conda activate /data/atlas/users/nterlind/venvs/thesis-ml
set -u
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

export PYTHONPATH=/project/atlas/users/nterlind/Thesis-Code/src
echo "Starting stage_b_inference.py ..."
python wandb_cleanup/eval_pipeline/stage_b_inference.py "$@"

EXIT_CODE=$?
echo "Job completed at: $(date) with exit code: $EXIT_CODE"
exit $EXIT_CODE
