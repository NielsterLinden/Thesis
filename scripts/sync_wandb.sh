#!/bin/bash
# Sync offline W&B runs from HPC to cloud
# Run this from a login node with internet access after jobs complete
#
# Usage:
#   ./scripts/sync_wandb.sh                          # Sync all runs
#   ./scripts/sync_wandb.sh /path/to/specific/run    # Sync specific run

set -e

# Default output root on Stoomboot HPC
DEFAULT_OUTPUT_ROOT="/data/atlas/users/nterlind/outputs"

if [ -n "$1" ]; then
    # Specific path provided
    WANDB_DIR="$1"
    if [ ! -d "$WANDB_DIR" ]; then
        echo "Error: Directory not found: $WANDB_DIR"
        exit 1
    fi
    echo "Syncing W&B data from: $WANDB_DIR"
    wandb sync "$WANDB_DIR"
else
    # Sync all offline runs
    RUNS_DIR="${DEFAULT_OUTPUT_ROOT}/runs"

    if [ ! -d "$RUNS_DIR" ]; then
        echo "Error: Runs directory not found: $RUNS_DIR"
        exit 1
    fi

    echo "Scanning for offline W&B runs in: $RUNS_DIR"

    # Find all wandb directories
    WANDB_DIRS=$(find "$RUNS_DIR" -type d -name "wandb" 2>/dev/null)

    if [ -z "$WANDB_DIRS" ]; then
        echo "No W&B directories found"
        exit 0
    fi

    COUNT=0
    for dir in $WANDB_DIRS; do
        # Check if there are offline runs to sync
        if ls "$dir"/offline-run-* 1> /dev/null 2>&1; then
            echo "Syncing: $dir"
            wandb sync "$dir" || echo "Warning: Failed to sync $dir"
            COUNT=$((COUNT + 1))
        fi
    done

    echo ""
    echo "Synced $COUNT W&B directories"
fi

echo "Done!"
