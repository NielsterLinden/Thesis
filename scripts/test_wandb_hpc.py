#!/usr/bin/env python
"""Minimal test for direct WandB upload from HPC.

Run on HPC with:
    export WANDB_DIR=/data/atlas/users/nterlind/outputs/wandb_test
    export WANDB_MODE=online
    python scripts/test_wandb_hpc.py

Verify:
- No wandb/ dir is created under the project directory.
- A run named "wandb_hpc_test" appears in the thesis-ml project on WandB.
- Local files only under $WANDB_DIR.
"""

from __future__ import annotations

import wandb


def main() -> None:
    run = wandb.init(
        project="thesis-ml",
        entity="nterlind-nikhef",
        name="wandb_hpc_test",
    )
    for step in range(5):
        wandb.log({"dummy_loss": 1.0 / (step + 1), "step": step}, step=step)
    run.finish()
    print("Done. Check WandB UI for run 'wandb_hpc_test'.")


if __name__ == "__main__":
    main()
