#!/usr/bin/env python
"""Migrate existing Facts runs to W&B retroactively.

Facts system remains the source of truth - W&B is a mirror for visualization.

Usage:
    # Dry run first (recommended)
    python scripts/migrate_runs_to_wandb.py --runs-dir /data/atlas/users/nterlind/outputs/runs --dry-run

    # Migrate all runs
    python scripts/migrate_runs_to_wandb.py --runs-dir /data/atlas/users/nterlind/outputs/runs

    # Migrate specific sweep
    python scripts/migrate_runs_to_wandb.py --sweep-dir /data/atlas/users/nterlind/outputs/multiruns/exp_*

    # Custom project/entity
    python scripts/migrate_runs_to_wandb.py --runs-dir ... --project my-project --entity my-team

    # Limit number of runs to migrate
    python scripts/migrate_runs_to_wandb.py --runs-dir ... --limit 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thesis_ml.facts.readers import _read_cfg, _read_events, _read_scalars, discover_runs

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _extract_group_from_run(run_dir: Path, cfg: dict) -> str | None:
    """Extract experiment/sweep group name from run directory or config.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory
    cfg : dict
        Hydra config dict

    Returns
    -------
    str | None
        Group name (e.g., "exp_20260128-111834_my_experiment") or None
    """
    run_name = run_dir.name

    # Try to extract experiment name from run directory name
    # Format: run_YYYYMMDD-HHMMSS_experimentname_jobN
    if run_name.startswith("run_"):
        parts = run_name.split("_")
        if len(parts) >= 3:
            # Extract experiment name (everything between timestamp and job number)
            # run_20260128-111834_my_experiment_job0 -> my_experiment
            timestamp = parts[1]  # YYYYMMDD-HHMMSS
            # Find job suffix
            job_idx = None
            for i, part in enumerate(parts):
                if part.startswith("job"):
                    job_idx = i
                    break

            if job_idx and job_idx > 2:
                experiment_name = "_".join(parts[2:job_idx])
                return f"exp_{timestamp}_{experiment_name}"

    # Try to get from Hydra overrides in config
    hydra_cfg = cfg.get("hydra", {})
    sweep_dir = hydra_cfg.get("sweep", {}).get("dir", "")
    if sweep_dir:
        sweep_path = Path(sweep_dir)
        if sweep_path.name.startswith("exp_"):
            return sweep_path.name

    return None


def _extract_tags_from_cfg(cfg: dict) -> list[str]:
    """Extract meaningful tags from config.

    Parameters
    ----------
    cfg : dict
        Hydra config dict

    Returns
    -------
    list[str]
        List of tags (e.g., ["autoencoder", "vq", "mlp"])
    """
    tags = []

    # Detect loop type
    loop = cfg.get("loop", "")
    if loop:
        tags.append(loop)

    # Phase 1 configs
    phase1 = cfg.get("phase1", {})
    if phase1:
        encoder = phase1.get("encoder", {})
        if isinstance(encoder, dict) and encoder.get("name"):
            tags.append(f"enc:{encoder['name']}")

        latent = phase1.get("latent_space", {})
        if isinstance(latent, dict) and latent.get("name"):
            tags.append(f"lat:{latent['name']}")

    # Classifier configs
    classifier = cfg.get("classifier", {})
    if classifier:
        model = classifier.get("model", {})
        if isinstance(model, dict) and model.get("name"):
            tags.append(f"model:{model['name']}")

    return tags


def migrate_run(
    run_dir: Path,
    project: str,
    entity: str | None,
    dry_run: bool,
    upload_artifacts: bool = True,
    group_override: str | None = None,
) -> bool:
    """Migrate a single run to W&B.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory
    project : str
        W&B project name
    entity : str | None
        W&B entity (username or team)
    dry_run : bool
        If True, only log what would be done
    upload_artifacts : bool
        If True, upload model artifacts
    group_override : str | None
        Override group name (useful for sweep migrations)

    Returns
    -------
    bool
        True if migration succeeded (or dry run), False otherwise
    """
    run_name = run_dir.name

    try:
        # Read facts
        cfg, _ = _read_cfg(run_dir)
        scalars = _read_scalars(run_dir)
        _events = _read_events(run_dir)  # Read to verify file exists, not used in migration
    except FileNotFoundError as e:
        logger.warning("Skipping %s: %s", run_name, e)
        return False
    except Exception as e:
        logger.warning("Skipping %s due to error: %s", run_name, e)
        return False

    # Get validation metrics
    val_df = scalars[scalars["split"] == "val"] if "split" in scalars.columns else scalars
    n_epochs = len(val_df)

    if n_epochs == 0:
        logger.warning("Skipping %s: no validation data", run_name)
        return False

    # Extract group and tags
    group = group_override or _extract_group_from_run(run_dir, cfg)
    tags = _extract_tags_from_cfg(cfg)

    if dry_run:
        group_str = f" [group: {group}]" if group else ""
        tags_str = f" [tags: {', '.join(tags)}]" if tags else ""
        logger.info("[DRY RUN] Would migrate: %s (%d epochs)%s%s", run_name, n_epochs, group_str, tags_str)
        return True

    # Import wandb here to avoid import errors if not installed
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        return False

    try:
        # Always use resume="allow" to prevent duplicates on re-run
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=tags if tags else None,
            config=cfg,
            resume="allow",
        )

        # Define metrics for proper axes
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

        # Log all epochs from scalars.csv
        for _, row in val_df.iterrows():
            epoch = int(row["epoch"]) if "epoch" in row and pd.notna(row["epoch"]) else 0

            metrics: dict[str, float | None] = {"epoch": epoch}

            # Add available metrics
            if "train_loss" in row and pd.notna(row["train_loss"]):
                metrics["train/loss"] = float(row["train_loss"])
            if "val_loss" in row and pd.notna(row["val_loss"]):
                metrics["val/loss"] = float(row["val_loss"])

            # Add metric_* columns
            for col in row.index:
                if col.startswith("metric_") and pd.notna(row[col]):
                    metric_name = col.replace("metric_", "")
                    metrics[f"val/{metric_name}"] = float(row[col])

            # Add performance metrics
            if "epoch_time_s" in row and pd.notna(row["epoch_time_s"]):
                metrics["perf/epoch_time_s"] = float(row["epoch_time_s"])
            if "throughput" in row and pd.notna(row["throughput"]):
                metrics["perf/throughput"] = float(row["throughput"])

            # Remove None values
            metrics = {k: v for k, v in metrics.items() if v is not None}

            wandb.log(metrics, step=epoch)

        # Upload model artifact if exists
        if upload_artifacts:
            model_path = run_dir / "best_val.pt"
            if model_path.exists():
                art = wandb.Artifact(f"model-{run_name}", type="model")
                art.add_file(str(model_path))
                wandb.log_artifact(art)
                logger.info("Uploaded artifact: model-%s", run_name)

            # Also try model.json for BDT
            json_model_path = run_dir / "model.json"
            if json_model_path.exists() and not model_path.exists():
                art = wandb.Artifact(f"model-{run_name}", type="model")
                art.add_file(str(json_model_path))
                wandb.log_artifact(art)
                logger.info("Uploaded artifact: model-%s (json)", run_name)

        run.finish()
        logger.info("Migrated: %s (%d epochs)", run_name, n_epochs)
        return True

    except Exception as e:
        logger.error("Failed to migrate %s: %s", run_name, e)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing Facts runs to W&B retroactively.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--runs-dir",
        type=Path,
        help="Path to runs directory containing run_* folders",
    )
    input_group.add_argument(
        "--sweep-dir",
        type=str,
        help="Path to multirun directory (can contain wildcards)",
    )
    input_group.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        help="Explicit list of run directories to migrate",
    )

    # W&B options
    parser.add_argument(
        "--project",
        type=str,
        default="thesis-ml",
        help="W&B project name (default: thesis-ml)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (username or team)",
    )

    # Migration options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually uploading",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip uploading model artifacts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of runs to migrate",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Override group name for all runs (useful for manual grouping)",
    )

    args = parser.parse_args()

    # Discover runs
    if args.runs_dir:
        # Scan directory for run_* folders
        if not args.runs_dir.exists():
            logger.error("Directory not found: %s", args.runs_dir)
            sys.exit(1)

        run_dirs = sorted([d for d in args.runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    elif args.sweep_dir:
        # Use discover_runs for sweep directory
        try:
            run_dirs = discover_runs(sweep_dir=args.sweep_dir, run_dirs=None)
        except Exception as e:
            logger.error("Failed to discover runs: %s", e)
            sys.exit(1)
    else:
        run_dirs = args.run_dirs

    if not run_dirs:
        logger.error("No runs found to migrate")
        sys.exit(1)

    # Apply limit
    if args.limit:
        run_dirs = run_dirs[: args.limit]

    logger.info("Found %d runs to migrate", len(run_dirs))

    if args.dry_run:
        logger.info("DRY RUN MODE - no actual uploads will be made")

    # Determine group override
    # If migrating from a sweep dir, use the sweep name as group
    group_override = args.group
    if not group_override and args.sweep_dir:
        # Extract group from sweep directory name
        sweep_path = Path(args.sweep_dir.rstrip("*").rstrip("/"))
        if sweep_path.name.startswith("exp_"):
            group_override = sweep_path.name
            logger.info("Using sweep directory as group: %s", group_override)

    # Migrate runs
    success_count = 0
    fail_count = 0

    for run_dir in run_dirs:
        run_path = Path(run_dir) if not isinstance(run_dir, Path) else run_dir
        if migrate_run(
            run_path,
            project=args.project,
            entity=args.entity,
            dry_run=args.dry_run,
            upload_artifacts=not args.no_artifacts,
            group_override=group_override,
        ):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    logger.info("")
    logger.info("Migration complete:")
    logger.info("  Succeeded: %d", success_count)
    logger.info("  Failed: %d", fail_count)
    logger.info("  Total: %d", success_count + fail_count)

    if args.dry_run:
        logger.info("")
        logger.info("This was a dry run. Run without --dry-run to actually migrate.")


if __name__ == "__main__":
    main()
