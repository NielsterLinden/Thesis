# Scripts Reference

One-paragraph summary of each script. Run from project root. See [ENV_AND_DATA.md](ENV_AND_DATA.md) for data and output paths.

## Dry Runs

No scripts. Use CLI overrides: add `env=local data.limit_samples=50 phase1.trainer.epochs=1 classifier.trainer.epochs=1` to any training command. See **`.cursor/rules/dry-runs.mdc`** for full instructions (new experiments, debugging, cleanup).

## WandB (`scripts/wandb/`)

| Script | Purpose |
|--------|---------|
| `backfill_labels.py` | Stamp old WandB runs with new config defaults |
| `cleanup_wandb.py` | Delete runs/artifacts from W&B |
| `migrate_runs_to_wandb.py` | Migrate Facts runs to W&B retroactively |
| `sync_wandb.sh` | Sync offline W&B runs from HPC to cloud |
| `test_wandb_hpc.py` | Minimal WandB upload test on HPC |

## Check (`scripts/check/`)

| Script | Purpose |
|--------|---------|
| `check_ids.py` | Verify H5 dataset token IDs match Hydra config |
| `check_logging_import.py` | Lint: ensure files using logger have `import logging` |
| `check_run_completeness.py` | Validate run dirs (facts, checkpoints) |

## One-Off (`scripts/one_off/`)

| Script | Purpose |
|--------|---------|
| `backfill_meta.py` | Backfill facts/meta.json from .hydra/config.yaml |
| `compare_binned_datasets.py` | Compare binned H5 datasets |
| `create_binned_dataset.py` | Create Ambre-style binned H5 from raw data |
| `evaluate_vq_vae.py` | VQ-VAE quality analysis |
| `migrate_legacy_runs.py` | Migrate legacy experiment dirs to multirun structure |
| `refactor_imports.py` | Rewrite imports for codebase refactoring |
| `rename_jobs_padding.py` | Rename job folders (job0 → job000) |
