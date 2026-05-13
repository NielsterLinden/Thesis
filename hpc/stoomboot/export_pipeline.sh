#!/usr/bin/env bash
set -eo pipefail

source ~/.bashrc || true

set -u

REPO=/project/atlas/users/nterlind/Thesis-Code
PYTHON=/data/atlas/users/nterlind/venvs/thesis-ml/bin/python3

cd "$REPO"

echo "[export_pipeline] $(date) starting Stage 01 raw export"
"$PYTHON" scripts/wandb/wandb_export_to_analysis_ready/01_raw_export.py \
    --out thesis_results/01_raw_export.csv

echo "[export_pipeline] $(date) Stage 01 done"
