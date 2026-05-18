#!/bin/bash
# submit_ch8_optuna.sh — HTCondor submission for Ch8 Optuna/TPE hyperparameter search.
#
# 150 Optuna trials, sequential (n_jobs=1), objective: maximise test_auroc.
# Task: 4t-vs-all-background (G3). Expected runtime: ~62h (budget: 72h).
#
# Usage:
#   bash hpc/submit_ch8_optuna.sh           # submit
#   bash hpc/submit_ch8_optuna.sh --dry-run # print without submitting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

ARGS="hydra/sweeper=optuna env=stoomboot loop=transformer_classifier \
classifier/experiment=thesis_experiments/ch8_candidates/cand_optuna --multirun"

echo "=== Ch8 Optuna submission ($([ $DRY_RUN -eq 1 ] && echo DRY-RUN || echo LIVE)) ==="
echo "150 Optuna/TPE trials, sequential, ~62h expected"
echo ""

cd "$REPO_ROOT"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "  DRY: condor_submit hpc/stoomboot/train.sub +JobCategory=long arguments=$ARGS"
else
  condor_submit hpc/stoomboot/train.sub \
    -append "arguments = $ARGS" \
    -append '+JobCategory = "long"'
  echo "  Submitted."
fi

echo ""
echo "=== Done. ==="
