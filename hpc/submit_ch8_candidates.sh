#!/bin/bash
# submit_ch8_candidates.sh — HTCondor submission for Ch8 candidate validation runs.
#
# Submits 3 experiments simultaneously (9 training runs total):
#   Set M: cand_m1, cand_m2, cand_m3 — marginal-greedy candidates, 50 epochs, 3 seeds
#
# (cand01/02/03 surrogate-strategy candidates already completed in earlier run.)
#
# Each YAML sweeps classifier.trainer.seed: 42,123,456 via hydra.sweeper.params.
# JobCategory: short  |  request_gpus: 1  (inherited from train.sub)
#
# Usage:
#   bash hpc/submit_ch8_candidates.sh           # submit M-set
#   bash hpc/submit_ch8_candidates.sh --dry-run # print without submitting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

JOBS=(
  thesis_experiments/ch8_candidates/cand_m1
  thesis_experiments/ch8_candidates/cand_m2
  thesis_experiments/ch8_candidates/cand_m3
)

echo "=== Ch8 candidate submission ($([ $DRY_RUN -eq 1 ] && echo DRY-RUN || echo LIVE)) ==="
echo "Submitting ${#JOBS[@]} experiments × 3 seeds = $((${#JOBS[@]}*3)) training runs"
echo ""

cd "$REPO_ROOT"
for exp_path in "${JOBS[@]}"; do
  args="env=stoomboot loop=transformer_classifier classifier/experiment=${exp_path} --multirun"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  DRY: condor_submit hpc/stoomboot/train.sub +JobCategory=short arguments=$args"
  else
    condor_submit hpc/stoomboot/train.sub \
      -append "arguments = $args" \
      -append '+JobCategory = "short"'
    echo "  Submitted: $exp_path"
  fi
done

echo ""
echo "=== Done. ==="
