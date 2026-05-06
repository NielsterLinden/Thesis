#!/bin/bash
set -exo pipefail

echo "=== Stoomboot Thesis ML Training Job ==="
# DataLoader workers + W&B open many FDs; default login/batch limits are often too low.
ulimit -n 8192 2>/dev/null || ulimit -n 4096 2>/dev/null || true
echo "ulimit -n: $(ulimit -n)"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${CONDOR_JOB_ID:-N/A}"

# Navigate to code directory
cd /project/atlas/users/nterlind/Thesis-Code

# WandB: write local files to large filesystem (not project dir, avoids quota).
# WANDB_CACHE_DIR: artifact staging for log_artifact() defaults to ~/.cache/wandb (small home quota).
WANDB_CACHE_ROOT=/data/atlas/users/nterlind/outputs/wandb_cache
mkdir -p "$WANDB_CACHE_ROOT"
export WANDB_DIR=/data/atlas/users/nterlind/outputs/wandb
export WANDB_CACHE_DIR="$WANDB_CACHE_ROOT"
export WANDB_DATA_DIR=/data/atlas/users/nterlind/outputs/wandb_data
export TMPDIR=/data/atlas/users/nterlind/outputs/tmp
export WANDB_MODE=online
mkdir -p "$WANDB_DATA_DIR" "$TMPDIR"

# Activate conda environment
set +u
eval "$(conda shell.bash hook)"
conda activate /data/atlas/users/nterlind/venvs/thesis-ml
set -u
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

# Optional NVIDIA CUDA MPS client env (multi-process GPU sharing). The MPS *daemon*
# (nvidia-cuda-mps-control / nvidia-cuda-mps-server) must already be running on the node.
if [[ "${THESIS_CUDA_MPS:-0}" == "1" ]]; then
  _scratch="${TMPDIR:-${_CONDOR_SCRATCH_DIR:-/tmp}}"
  MPS_ROOT="${_scratch}/thesis_cuda_mps_${USER:-user}_$$"
  mkdir -p "$MPS_ROOT/pipe" "$MPS_ROOT/log"
  export CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe"
  export CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log"
  echo "THESIS_CUDA_MPS=1 CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
  echo "THESIS_CUDA_MPS=1 CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"
fi

# Run training (all settings come from configs)
echo "Starting training..."
python -m thesis_ml.cli.train "$@"

EXIT_CODE=$?
echo "Job completed at: $(date) with exit code: $EXIT_CODE"
exit $EXIT_CODE
