#!/usr/bin/env python
"""Migrate existing Facts runs to W&B retroactively.

Facts system remains the source of truth - W&B is a mirror for visualization.
Extracts comprehensive config metadata for maximum divisibility in WandB dashboards.

Usage:
    # Dry run first (recommended)
    python scripts/migrate_runs_to_wandb.py --runs-dir /data/atlas/users/nterlind/outputs/runs --dry-run

    # Migrate all runs
    python scripts/migrate_runs_to_wandb.py --runs-dir /data/atlas/users/nterlind/outputs/runs

    # Migrate from both HPC and local
    python scripts/migrate_runs_to_wandb.py \
        --hpc-runs-dir /data/atlas/users/nterlind/outputs/runs \
        --local-runs-dir ~/Projects/Thesis-Code/outputs/runs

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
from typing import Any

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thesis_ml.facts.readers import _read_cfg, _read_events, _read_scalars, discover_runs
from thesis_ml.utils.wandb_utils import _safe_get, extract_wandb_config

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


def _extract_tags_from_cfg(cfg: dict[str, Any]) -> list[str]:
    """Extract meaningful tags from config for quick filtering.

    Tags are for quick categorical filtering in WandB.
    Detailed config values go into the config dict (via extract_wandb_config).

    Parameters
    ----------
    cfg : dict
        Hydra config dict

    Returns
    -------
    list[str]
        List of tags (e.g., ["transformer", "rotary", "pre-norm"])
    """
    tags = []

    # Model type tag
    loop = _safe_get(cfg, "loop", "")
    if loop:
        if "transformer" in loop:
            tags.append("transformer")
        elif "mlp" in loop:
            tags.append("mlp")
        elif "bdt" in loop:
            tags.append("bdt")
        elif "ae" in loop:
            tags.append("autoencoder")
        else:
            tags.append(loop)
    elif _safe_get(cfg, "phase1"):
        tags.append("autoencoder")

    # Positional encoding tag
    pos = _safe_get(cfg, "classifier.model.positional")
    if pos:
        tags.append(f"pe:{pos}")

    # Normalization policy tag
    norm = _safe_get(cfg, "classifier.model.norm.policy")
    if norm:
        tags.append(f"norm:{norm}")

    # Tokenizer tag
    tok = _safe_get(cfg, "classifier.tokenizer") or _safe_get(cfg, "tokenizer")
    if isinstance(tok, dict) and tok.get("name"):
        tags.append(f"tok:{tok['name']}")
    elif tok:
        tags.append(f"tok:{tok}")

    # Pooling tag
    pool = _safe_get(cfg, "classifier.model.pooling")
    if pool:
        tags.append(f"pool:{pool}")

    # Autoencoder-specific tags
    phase1 = _safe_get(cfg, "phase1")
    if phase1:
        enc = _safe_get(cfg, "phase1.encoder")
        if isinstance(enc, dict) and enc.get("name"):
            tags.append(f"enc:{enc['name']}")
        lat = _safe_get(cfg, "phase1.latent_space")
        if isinstance(lat, dict) and lat.get("name"):
            tags.append(f"lat:{lat['name']}")

    return tags


def migrate_run(
    run_dir: Path,
    project: str,
    entity: str | None,
    dry_run: bool,
    upload_artifacts: bool = True,
    group_override: str | None = None,
    source_location: str = "unknown",
) -> bool:
    """Migrate a single run to W&B with comprehensive metadata.

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
    source_location : str
        Source of the run ("hpc", "local", or other identifier)

    Returns
    -------
    bool
        True if migration succeeded (or dry run), False otherwise
    """
    run_name = run_dir.name

    # === Validation: Check for valid config ===
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    legacy_cfg_path = run_dir / "cfg.yaml"

    if not hydra_cfg_path.exists() and not legacy_cfg_path.exists():
        logger.warning("Skipping %s: no config file found", run_name)
        return False

    # === Read config ===
    try:
        cfg, _ = _read_cfg(run_dir)
    except FileNotFoundError as e:
        logger.warning("Skipping %s: config not found: %s", run_name, e)
        return False
    except Exception as e:
        logger.warning("Skipping %s: config read error: %s", run_name, e)
        return False

    # === Read scalars ===
    try:
        scalars = _read_scalars(run_dir)
    except FileNotFoundError:
        logger.warning("Skipping %s: no scalars.csv found", run_name)
        return False
    except Exception as e:
        logger.warning("Skipping %s: scalars read error: %s", run_name, e)
        return False

    # === Validate scalars ===
    if scalars.empty:
        logger.warning("Skipping %s: empty scalars.csv", run_name)
        return False

    # Get validation metrics
    val_df = scalars[scalars["split"] == "val"] if "split" in scalars.columns else scalars
    n_epochs = len(val_df)

    if n_epochs == 0:
        logger.warning("Skipping %s: no validation data", run_name)
        return False

    # === Read events (optional, for completeness check) ===
    try:
        _events = _read_events(run_dir)
    except Exception:
        _events = None  # Events are optional

    # === Extract metadata ===
    group = group_override or _extract_group_from_run(run_dir, cfg)
    tags = _extract_tags_from_cfg(cfg)
    wandb_config = extract_wandb_config(cfg, source_location=source_location)

    # Add run directory to config for traceability
    wandb_config["source/run_dir"] = str(run_dir)

    # === Dry run logging ===
    if dry_run:
        group_str = f" [group: {group}]" if group else ""
        tags_str = f" [tags: {', '.join(tags)}]" if tags else ""
        model_type = wandb_config.get("model/type", "unknown")
        size_label = wandb_config.get("model/size_label", "")
        size_str = f" [{size_label}]" if size_label else ""
        logger.info(
            "[DRY RUN] Would migrate: %s (%d epochs, %s%s)%s%s",
            run_name,
            n_epochs,
            model_type,
            size_str,
            group_str,
            tags_str,
        )
        return True

    # === Import wandb ===
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        return False

    # === Migrate to WandB ===
    try:
        # Always use resume="allow" to prevent duplicates on re-run
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=tags if tags else None,
            config=wandb_config,  # Use extracted config for max divisibility
            resume="allow",
        )

        # Define metrics for proper axes
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("perf/*", step_metric="epoch")

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
            if "max_memory_mib" in row and pd.notna(row["max_memory_mib"]):
                metrics["perf/max_memory_mib"] = float(row["max_memory_mib"])

            # Remove None values
            metrics = {k: v for k, v in metrics.items() if v is not None}

            wandb.log(metrics, step=epoch)

        # Upload model artifact if exists
        if upload_artifacts:
            # Extract key metadata for artifact
            artifact_metadata = {
                "model_type": wandb_config.get("model/type"),
                "size_label": wandb_config.get("model/size_label"),
                "params_est": wandb_config.get("model/params_est"),
                "source": source_location,
            }
            artifact_metadata = {k: v for k, v in artifact_metadata.items() if v is not None}

            model_path = run_dir / "best_val.pt"
            if model_path.exists():
                art = wandb.Artifact(f"model-{run_name}", type="model", metadata=artifact_metadata)
                art.add_file(str(model_path))
                wandb.log_artifact(art)
                logger.info("Uploaded artifact: model-%s", run_name)

            # Also try model.json for BDT
            json_model_path = run_dir / "model.json"
            if json_model_path.exists() and not model_path.exists():
                art = wandb.Artifact(f"model-{run_name}", type="model", metadata=artifact_metadata)
                art.add_file(str(json_model_path))
                wandb.log_artifact(art)
                logger.info("Uploaded artifact: model-%s (json)", run_name)

        run.finish()
        model_type = wandb_config.get("model/type", "unknown")
        logger.info("Migrated: %s (%d epochs, %s)", run_name, n_epochs, model_type)
        return True

    except Exception as e:
        logger.error("Failed to migrate %s: %s", run_name, e)
        return False


def _discover_runs_in_dir(runs_dir: Path) -> list[Path]:
    """Discover run directories in a given directory.

    Parameters
    ----------
    runs_dir : Path
        Directory containing run_* folders

    Returns
    -------
    list[Path]
        Sorted list of run directories
    """
    if not runs_dir.exists():
        return []

    run_dirs = []
    for d in runs_dir.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            # Validate that it has a config file
            has_hydra = (d / ".hydra" / "config.yaml").exists()
            has_legacy = (d / "cfg.yaml").exists()
            if has_hydra or has_legacy:
                run_dirs.append(d)

    return sorted(run_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing Facts runs to W&B retroactively with comprehensive metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options - now supports multiple sources
    input_group = parser.add_argument_group("Input sources (at least one required)")
    input_group.add_argument(
        "--runs-dir",
        type=Path,
        help="Path to runs directory containing run_* folders",
    )
    input_group.add_argument(
        "--hpc-runs-dir",
        type=Path,
        help="HPC runs directory (e.g., /data/atlas/users/nterlind/outputs/runs)",
    )
    input_group.add_argument(
        "--local-runs-dir",
        type=Path,
        help="Local runs directory",
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

    # Collect all runs from all sources with their source location
    all_runs: list[tuple[Path, str]] = []

    # HPC runs
    if args.hpc_runs_dir:
        if args.hpc_runs_dir.exists():
            hpc_runs = _discover_runs_in_dir(args.hpc_runs_dir)
            logger.info("Found %d runs in HPC directory: %s", len(hpc_runs), args.hpc_runs_dir)
            for r in hpc_runs:
                all_runs.append((r, "hpc"))
        else:
            logger.warning("HPC directory not found: %s", args.hpc_runs_dir)

    # Local runs
    if args.local_runs_dir:
        if args.local_runs_dir.exists():
            local_runs = _discover_runs_in_dir(args.local_runs_dir)
            logger.info("Found %d runs in local directory: %s", len(local_runs), args.local_runs_dir)
            for r in local_runs:
                all_runs.append((r, "local"))
        else:
            logger.warning("Local directory not found: %s", args.local_runs_dir)

    # Generic runs-dir (backward compatibility)
    if args.runs_dir:
        if args.runs_dir.exists():
            generic_runs = _discover_runs_in_dir(args.runs_dir)
            logger.info("Found %d runs in directory: %s", len(generic_runs), args.runs_dir)
            for r in generic_runs:
                all_runs.append((r, "unknown"))
        else:
            logger.error("Directory not found: %s", args.runs_dir)
            sys.exit(1)

    # Sweep directory
    if args.sweep_dir:
        try:
            sweep_runs = discover_runs(sweep_dir=args.sweep_dir, run_dirs=None)
            logger.info("Found %d runs in sweep: %s", len(sweep_runs), args.sweep_dir)
            for r in sweep_runs:
                all_runs.append((r, "sweep"))
        except Exception as e:
            logger.error("Failed to discover runs in sweep: %s", e)
            sys.exit(1)

    # Explicit run directories
    if args.run_dirs:
        for r in args.run_dirs:
            if r.exists():
                all_runs.append((r, "explicit"))
            else:
                logger.warning("Run directory not found: %s", r)

    # Validate we have runs to migrate
    if not all_runs:
        logger.error("No runs found to migrate. Provide at least one of: --runs-dir, --hpc-runs-dir, --local-runs-dir, --sweep-dir, --run-dirs")
        sys.exit(1)

    # Apply limit
    if args.limit:
        all_runs = all_runs[: args.limit]

    logger.info("")
    logger.info("=" * 60)
    logger.info("Total runs to migrate: %d", len(all_runs))
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - no actual uploads will be made")
        logger.info("")

    # Determine group override
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

    for run_path, source_location in all_runs:
        result = migrate_run(
            run_path,
            project=args.project,
            entity=args.entity,
            dry_run=args.dry_run,
            upload_artifacts=not args.no_artifacts,
            group_override=group_override,
            source_location=source_location,
        )
        if result:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Migration complete:")
    logger.info("  Succeeded: %d", success_count)
    logger.info("  Skipped/Failed: %d", fail_count)
    logger.info("  Total processed: %d", success_count + fail_count)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("")
        logger.info("This was a dry run. Run without --dry-run to actually migrate.")


if __name__ == "__main__":
    main()
