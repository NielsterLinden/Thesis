# Scripts Reference

One-paragraph summary of each script in `scripts/`. Run from project root.


| Script                           | Purpose                                                                                                                  | Example                                                                                |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `backfill_meta.py`               | Backfill `facts/meta.json` for existing runs from `.hydra/config.yaml`. Infers metadata; never guesses `process_groups`. | `python scripts/backfill_meta.py --runs-dir outputs/runs --dry-run`                    |
| `check_ids.py`                   | Verify H5 dataset token IDs match Hydra config. Loads config and inspects train/val/test splits.                         | `python scripts/check_ids.py`                                                          |
| `check_logging_import.py`        | Lint: ensure files using `logging.getLogger(__name__)` have `import logging`.                                            | `python scripts/check_logging_import.py`                                               |
| `check_run_completeness.py`      | Validate run directories (facts, checkpoints). Reports missing or incomplete runs.                                       | `python scripts/check_run_completeness.py outputs/runs`                                |
| `compare_binned_datasets.py`     | Compare our binned H5 dataset with Ambre's pre-binned dataset. Inspects structure and statistics.                        | `python scripts/compare_binned_datasets.py --ours X.h5 --ambres Y.h5`                  |
| `create_binned_dataset.py`       | Create Ambre-style binned H5 from raw 4tops data. Run interactively on HPC login node.                                   | `python scripts/create_binned_dataset.py --input raw.h5 --output binned.h5 --n-bins 5` |
| `evaluate_vq_vae.py`             | Evaluate VQ-VAE quality for downstream transformer use. Analyzes codebook usage, reconstruction, training metrics.       | `python scripts/evaluate_vq_vae.py --run-dir outputs/runs/run_*_vq`                    |
| `migrate_legacy_runs.py`         | Migrate legacy experiment dirs with numbered subdirs (0, 1, 2...) to new multirun structure.                             | `python scripts/migrate_legacy_runs.py --input old_exp/ --output outputs/`             |
| `refactor_imports.py`            | Rewrite imports for codebase refactoring. Applies predefined old→new mappings.                                           | `python scripts/refactor_imports.py`                                                   |
| `rename_jobs_padding.py`         | Rename job folders to 3-digit zero-padding (job0 → job000) for correct sorting.                                          | `python scripts/rename_jobs_padding.py --runs-dir outputs/runs --execute`              |
| `wandb/backfill_labels.py`       | Stamp old WandB runs with new config defaults. Use when adding config keys that old runs lack.                           | `python scripts/wandb/backfill_labels.py --labels '{"training/loss_type":"bce"}'`      |
| `wandb/cleanup_wandb.py`         | Delete runs and/or artifacts from W&B. Optional `--group` to target a specific experiment.                               | `python scripts/wandb/cleanup_wandb.py --project thesis-ml --execute`                  |
| `wandb/migrate_runs_to_wandb.py` | Migrate existing Facts runs to W&B retroactively. Reads facts, uploads to W&B with metadata.                             | `python scripts/wandb/migrate_runs_to_wandb.py --sweep-dir outputs/multiruns/exp_*`    |
| `wandb/sync_wandb.sh`            | Sync offline W&B runs from HPC to cloud. Run on login node after jobs complete.                                          | `./scripts/wandb/sync_wandb.sh`                                                        |
| `wandb/test_wandb_hpc.py`        | Minimal test for direct WandB upload from HPC. Verifies API key and connectivity.                                        | `python scripts/wandb/test_wandb_hpc.py`                                               |
