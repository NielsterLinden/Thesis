#!/usr/bin/env python
"""Delete old toy models and runs from W&B.

This script cleans up the W&B project to prepare for a fresh migration
of all runs with proper metadata.

Usage:
    # Dry run (default) - see what would be deleted
    python scripts/cleanup_wandb.py --project thesis-ml

    # Actually delete everything
    python scripts/cleanup_wandb.py --project thesis-ml --execute

    # Delete only runs from a specific group (e.g. failed experiment)
    python scripts/cleanup_wandb.py --project thesis-ml --group "exp_20260210-135259_exp_binning_vs_direct" --execute

    # Delete only runs (keep artifacts)
    python scripts/cleanup_wandb.py --project thesis-ml --runs-only --execute

    # Delete only artifacts (keep runs)
    python scripts/cleanup_wandb.py --project thesis-ml --artifacts-only --execute
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cleanup_wandb(
    project: str,
    entity: str | None = None,
    dry_run: bool = True,
    delete_runs: bool = True,
    delete_artifacts: bool = True,
    group: str | None = None,
) -> tuple[int, int]:
    """Delete runs and/or artifacts from a W&B project.

    Parameters
    ----------
    project : str
        W&B project name
    entity : str | None
        W&B entity (username or team). If None, uses default.
    dry_run : bool
        If True, only log what would be deleted
    delete_runs : bool
        If True, delete all runs
    delete_artifacts : bool
        If True, delete all model artifacts
    group : str | None
        If set, only delete runs in this group (e.g. exp_20260210-135259_exp_binning_vs_direct)

    Returns
    -------
    tuple[int, int]
        (runs_deleted, artifacts_deleted)
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        sys.exit(1)

    api = wandb.Api()

    # Build project path
    project_path = f"{entity}/{project}" if entity else project

    runs_deleted = 0
    artifacts_deleted = 0

    # Delete runs
    if delete_runs:
        try:
            all_runs = list(api.runs(project_path))
            if group:
                runs = [r for r in all_runs if getattr(r, "group", None) == group]
                logger.info("Found %d runs in group '%s' (of %d total)", len(runs), group, len(all_runs))
            else:
                runs = all_runs
                logger.info("Found %d runs in project '%s'", len(runs), project_path)

            for run in runs:
                if dry_run:
                    logger.info("[DRY RUN] Would delete run: %s (%s)", run.name, run.id)
                else:
                    try:
                        run.delete()
                        logger.info("Deleted run: %s (%s)", run.name, run.id)
                        runs_deleted += 1
                    except Exception as e:
                        logger.warning("Failed to delete run %s: %s", run.name, e)
        except Exception as e:
            logger.error("Failed to list runs: %s", e)

    # Delete model artifacts
    if delete_artifacts:
        try:
            # Get all artifact types first
            artifact_types = ["model", "dataset", "checkpoint"]

            for artifact_type in artifact_types:
                try:
                    artifacts = list(api.artifacts(type_name=artifact_type, project=project_path))
                    if artifacts:
                        logger.info(
                            "Found %d '%s' artifacts in project '%s'",
                            len(artifacts),
                            artifact_type,
                            project_path,
                        )

                        for art in artifacts:
                            if dry_run:
                                logger.info(
                                    "[DRY RUN] Would delete %s artifact: %s",
                                    artifact_type,
                                    art.name,
                                )
                            else:
                                try:
                                    art.delete()
                                    logger.info("Deleted %s artifact: %s", artifact_type, art.name)
                                    artifacts_deleted += 1
                                except Exception as e:
                                    logger.warning("Failed to delete artifact %s: %s", art.name, e)
                except wandb.CommError:
                    # No artifacts of this type
                    pass
        except Exception as e:
            logger.error("Failed to list artifacts: %s", e)

    return runs_deleted, artifacts_deleted


def main():
    parser = argparse.ArgumentParser(
        description="Delete all old toy models and runs from W&B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

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
        help="W&B entity (username or team). If not specified, uses default.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete (default is dry-run mode)",
    )
    parser.add_argument(
        "--runs-only",
        action="store_true",
        help="Only delete runs, keep artifacts",
    )
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Only delete artifacts, keep runs",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Only delete runs in this group (e.g. exp_20260210-135259_exp_binning_vs_direct)",
    )

    args = parser.parse_args()

    # Determine what to delete
    delete_runs = not args.artifacts_only
    delete_artifacts = not args.runs_only

    dry_run = not args.execute

    if dry_run:
        logger.info("DRY RUN MODE - no actual deletions will be made")
        logger.info("Run with --execute to actually delete")
    else:
        # Confirmation prompt
        project_path = f"{args.entity}/{args.project}" if args.entity else args.project
        if args.group:
            project_path = f"{project_path} (group={args.group})"
        what = []
        if delete_runs:
            what.append("runs")
        if delete_artifacts:
            what.append("artifacts")
        what_str = " and ".join(what)

        logger.warning("=" * 60)
        logger.warning("WARNING: This will DELETE all %s from '%s'", what_str, project_path)
        logger.warning("=" * 60)

        response = input("Type 'DELETE' to confirm: ")
        if response != "DELETE":
            logger.info("Aborted.")
            sys.exit(0)

    runs_deleted, artifacts_deleted = cleanup_wandb(
        project=args.project,
        entity=args.entity,
        dry_run=dry_run,
        delete_runs=delete_runs,
        delete_artifacts=delete_artifacts,
        group=args.group,
    )

    # Summary
    logger.info("")
    logger.info("=" * 40)
    if dry_run:
        logger.info("DRY RUN COMPLETE")
        logger.info("Would delete runs: (see above)")
        logger.info("Would delete artifacts: (see above)")
        logger.info("")
        logger.info("Run with --execute to actually delete")
    else:
        logger.info("CLEANUP COMPLETE")
        logger.info("Runs deleted: %d", runs_deleted)
        logger.info("Artifacts deleted: %d", artifacts_deleted)
    logger.info("=" * 40)


if __name__ == "__main__":
    main()
