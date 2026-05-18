#!/bin/bash
set -exo pipefail

echo "=== Stoomboot Re-eval Job (CPU) ==="
ulimit -n 8192 2>/dev/null || true
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

cd /project/atlas/users/nterlind/Thesis-Code

export WANDB_DIR=/data/atlas/users/nterlind/outputs/wandb
export WANDB_CACHE_DIR=/data/atlas/users/nterlind/outputs/wandb_cache
export WANDB_DATA_DIR=/data/atlas/users/nterlind/outputs/wandb_data
export TMPDIR=/data/atlas/users/nterlind/outputs/tmp
export WANDB_MODE=online
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$TMPDIR"

set +u
eval "$(conda shell.bash hook)"
conda activate /data/atlas/users/nterlind/venvs/thesis-ml
set -u

echo "Python: $(python --version)"
python scripts/one_off/ch8_optuna_reeval.py --device auto

EXIT_CODE=$?
echo "Job completed at: $(date) with exit code: $EXIT_CODE"
exit $EXIT_CODE
