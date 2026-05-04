#!/bin/bash
# db_completeness_submit.sh — Throttled HTCondor job submission.
# Usage:
#   bash db_completeness_submit.sh           # submit pending jobs (max 4 concurrent)
#   bash db_completeness_submit.sh --dry-run # print without submitting
#   bash db_completeness_submit.sh --status  # show queue + remaining
#
# Already-submitted entries are marked "DONE: ..." in the queue file so
# re-running the script after interruption is safe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QUEUE="$SCRIPT_DIR/db_completeness_queue.txt"
MAX_JOBS=3
DRY_RUN=0
STATUS_ONLY=0

for arg in "$@"; do
  if [[ "$arg" == "--dry-run"  ]]; then DRY_RUN=1; fi
  if [[ "$arg" == "--status"   ]]; then STATUS_ONLY=1; fi
done

condor_running() {
  local n
  n=$(condor_q 2>/dev/null | grep -c "^[0-9]") || n=0
  echo "$n"
}

pending_count() {
  local n
  n=$(grep -c "^thesis_experiments" "$QUEUE" 2>/dev/null) || n=0
  echo "$n"
}

if [[ $STATUS_ONLY -eq 1 ]]; then
  echo "HTCondor jobs active : $(condor_running)"
  echo "Queue entries pending : $(pending_count)"
  exit 0
fi

echo "=== db_completeness throttled submission ==="
echo "Queue : $QUEUE"
echo "Limit : MAX_JOBS=$MAX_JOBS"
echo "Mode  : $([ $DRY_RUN -eq 1 ] && echo DRY-RUN || echo LIVE)"
echo ""

while IFS= read -r line; do
  # Skip comments, blank lines, already-submitted entries
  if [[ -z "$line" || "$line" == "#"* || "$line" == "DONE:"* ]]; then continue; fi

  exp_path="$line"

  # Wait until a slot is free
  while true; do
    running=$(condor_running)
    if (( running < MAX_JOBS )); then break; fi
    echo "  [$(date +%H:%M:%S)] $running/$MAX_JOBS jobs active — waiting 60s..."
    sleep 60
  done

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  DRY: $exp_path"
  else
    echo "  Submitting ($( condor_running ) active): $exp_path"
    cd "$REPO_ROOT"
    condor_submit hpc/stoomboot/train.sub \
      -append "arguments = env=stoomboot classifier/experiment=${exp_path} --multirun"
    # Mark as done in queue file so re-runs skip this entry
    sed -i "s|^${exp_path}$|DONE: ${exp_path}|" "$QUEUE"
    sleep 5  # give condor a moment to register the new job
  fi

done < "$QUEUE"

echo ""
echo "=== Done. Active jobs: $(condor_running). Pending: $(pending_count). ==="
