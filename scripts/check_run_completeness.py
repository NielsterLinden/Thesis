#!/usr/bin/env python3
"""Check completeness of training runs.

This script checks whether training runs have all required outputs to be
considered complete and usable for report generation.

Usage:
    # Check specific experiment runs
    python scripts/check_run_completeness.py \
        --base-dir /data/atlas/users/nterlind/outputs/runs \
        --pattern "run_20260211-093121_exp_binning_vs_direct_test_job*"

    # Check all runs in a directory
    python scripts/check_run_completeness.py \
        --base-dir /data/atlas/users/nterlind/outputs/runs

    # Verbose output with details per run
    python scripts/check_run_completeness.py \
        --base-dir /data/atlas/users/nterlind/outputs/runs \
        --pattern "run_*_exp_binning_vs_direct*" \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RunStatus:
    """Status of a training run."""

    run_dir: Path
    is_complete: bool
    has_hydra_config: bool
    has_scalars: bool
    has_events: bool
    has_on_train_end: bool
    has_checkpoint: bool
    num_epochs: int | None
    missing_items: list[str]
    error_message: str | None = None


def check_run_completeness(run_dir: Path, verbose: bool = False) -> RunStatus:
    """Check if a single run is complete.

    A complete run should have:
    - .hydra/config.yaml (or cfg.yaml as fallback)
    - facts/scalars.csv with multiple rows
    - facts/events.jsonl with on_train_end event
    - At least one checkpoint file (best_val.pt, last.pt, or model.pt)

    Parameters
    ----------
    run_dir : Path
        Path to the run directory
    verbose : bool
        If True, log details about each check

    Returns
    -------
    RunStatus
        Status object with completeness information
    """
    missing_items = []
    error_message = None

    if verbose:
        logger.info(f"Checking: {run_dir.name}")

    # Check 1: Hydra config
    hydra_config = run_dir / ".hydra" / "config.yaml"
    cfg_fallback = run_dir / "cfg.yaml"
    has_hydra_config = hydra_config.exists() or cfg_fallback.exists()

    if not has_hydra_config:
        missing_items.append(".hydra/config.yaml (or cfg.yaml)")
        if verbose:
            logger.warning("  ❌ Missing config file")

    # Check 2: Scalars CSV
    scalars_csv = run_dir / "facts" / "scalars.csv"
    has_scalars = False
    num_epochs = None

    if not scalars_csv.exists():
        missing_items.append("facts/scalars.csv")
        if verbose:
            logger.warning("  ❌ Missing scalars.csv")
    else:
        try:
            df = pd.read_csv(scalars_csv)
            # Should have multiple rows (at least 2: header + 1 epoch)
            if len(df) > 0:
                has_scalars = True
                # Count unique epochs
                if "epoch" in df.columns:
                    num_epochs = df["epoch"].max() + 1  # epochs are 0-indexed
                if verbose:
                    logger.info(f"  ✓ scalars.csv ({len(df)} rows, {num_epochs} epochs)")
            else:
                missing_items.append("scalars.csv (empty)")
                if verbose:
                    logger.warning("  ❌ scalars.csv is empty")
        except Exception as e:
            error_message = f"Error reading scalars.csv: {e}"
            missing_items.append("scalars.csv (corrupt)")
            if verbose:
                logger.error(f"  ❌ Error reading scalars.csv: {e}")

    # Check 3: Events JSONL
    events_jsonl = run_dir / "facts" / "events.jsonl"
    has_events = events_jsonl.exists()
    has_on_train_end = False

    if not has_events:
        missing_items.append("facts/events.jsonl")
        if verbose:
            logger.warning("  ❌ Missing events.jsonl")
    else:
        # Check for on_train_end event
        try:
            with open(events_jsonl) as f:
                for line in f:
                    event = json.loads(line)
                    if event.get("moment") == "on_train_end":
                        has_on_train_end = True
                        break

            if has_on_train_end:
                if verbose:
                    logger.info("  ✓ events.jsonl with on_train_end")
            else:
                missing_items.append("on_train_end event")
                if verbose:
                    logger.warning("  ⚠️  events.jsonl exists but no on_train_end event")
        except Exception as e:
            error_message = f"Error reading events.jsonl: {e}"
            if verbose:
                logger.error(f"  ❌ Error reading events.jsonl: {e}")

    # Check 4: Checkpoint files
    checkpoint_files = [
        run_dir / "best_val.pt",
        run_dir / "last.pt",
        run_dir / "model.pt",
    ]
    has_checkpoint = any(ckpt.exists() for ckpt in checkpoint_files)

    if has_checkpoint:
        found_checkpoints = [ckpt.name for ckpt in checkpoint_files if ckpt.exists()]
        if verbose:
            logger.info(f"  ✓ Checkpoints: {', '.join(found_checkpoints)}")
    else:
        missing_items.append("checkpoint files (.pt)")
        if verbose:
            logger.warning("  ❌ No checkpoint files found")

    # Overall completeness
    is_complete = has_hydra_config and has_scalars and has_events and has_on_train_end and has_checkpoint

    if verbose:
        if is_complete:
            logger.info("  ✅ Run is COMPLETE")
        else:
            logger.warning(f"  ⚠️  Run is INCOMPLETE (missing: {', '.join(missing_items)})")

    return RunStatus(
        run_dir=run_dir,
        is_complete=is_complete,
        has_hydra_config=has_hydra_config,
        has_scalars=has_scalars,
        has_events=has_events,
        has_on_train_end=has_on_train_end,
        has_checkpoint=has_checkpoint,
        num_epochs=num_epochs,
        missing_items=missing_items,
        error_message=error_message,
    )


def check_multiple_runs(
    base_dir: Path,
    pattern: str = "run_*",
    verbose: bool = False,
) -> list[RunStatus]:
    """Check completeness of multiple runs.

    Parameters
    ----------
    base_dir : Path
        Base directory containing run directories
    pattern : str
        Glob pattern to match run directories (default: "run_*")
    verbose : bool
        If True, log details about each run

    Returns
    -------
    list[RunStatus]
        List of status objects for each run
    """
    if not base_dir.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        return []

    # Find all matching run directories
    run_dirs = sorted(base_dir.glob(pattern))

    if not run_dirs:
        logger.warning(f"No runs found matching pattern '{pattern}' in {base_dir}")
        return []

    logger.info(f"Found {len(run_dirs)} runs matching '{pattern}'")
    logger.info("")

    statuses = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue

        status = check_run_completeness(run_dir, verbose=verbose)
        statuses.append(status)

        if verbose:
            logger.info("")  # Blank line between runs

    return statuses


def print_summary(statuses: list[RunStatus]) -> None:
    """Print summary of run completeness checks.

    Parameters
    ----------
    statuses : list[RunStatus]
        List of status objects
    """
    if not statuses:
        logger.info("No runs to summarize.")
        return

    complete = [s for s in statuses if s.is_complete]
    incomplete = [s for s in statuses if not s.is_complete]

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total runs checked: {len(statuses)}")
    logger.info(f"Complete runs: {len(complete)} ({100 * len(complete) / len(statuses):.1f}%)")
    logger.info(f"Incomplete runs: {len(incomplete)} ({100 * len(incomplete) / len(statuses):.1f}%)")
    logger.info("")

    if incomplete:
        logger.info("INCOMPLETE RUNS:")
        for status in incomplete:
            logger.info(f"  - {status.run_dir.name}")
            logger.info(f"    Missing: {', '.join(status.missing_items)}")
            if status.error_message:
                logger.info(f"    Error: {status.error_message}")

    logger.info("")

    # Statistics
    has_config = sum(1 for s in statuses if s.has_hydra_config)
    has_scalars = sum(1 for s in statuses if s.has_scalars)
    has_events = sum(1 for s in statuses if s.has_events)
    has_train_end = sum(1 for s in statuses if s.has_on_train_end)
    has_ckpt = sum(1 for s in statuses if s.has_checkpoint)

    logger.info("COMPONENT STATISTICS:")
    logger.info(f"  Hydra config:     {has_config}/{len(statuses)} ({100 * has_config / len(statuses):.1f}%)")
    logger.info(f"  Scalars CSV:      {has_scalars}/{len(statuses)} ({100 * has_scalars / len(statuses):.1f}%)")
    logger.info(f"  Events JSONL:     {has_events}/{len(statuses)} ({100 * has_events / len(statuses):.1f}%)")
    logger.info(f"  on_train_end:     {has_train_end}/{len(statuses)} ({100 * has_train_end / len(statuses):.1f}%)")
    logger.info(f"  Checkpoint:       {has_ckpt}/{len(statuses)} ({100 * has_ckpt / len(statuses):.1f}%)")

    # Epoch statistics
    epochs = [s.num_epochs for s in statuses if s.num_epochs is not None]
    if epochs:
        logger.info("")
        logger.info("EPOCH STATISTICS:")
        logger.info(f"  Min epochs: {min(epochs)}")
        logger.info(f"  Max epochs: {max(epochs)}")
        logger.info(f"  Mean epochs: {sum(epochs) / len(epochs):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Check completeness of training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Base directory containing run directories",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="run_*",
        help="Glob pattern to match run directories (default: 'run_*')",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed information for each run",
    )

    args = parser.parse_args()

    statuses = check_multiple_runs(
        base_dir=args.base_dir,
        pattern=args.pattern,
        verbose=args.verbose,
    )

    print_summary(statuses)

    # Exit with error code if any runs are incomplete
    incomplete_count = sum(1 for s in statuses if not s.is_complete)
    if incomplete_count > 0:
        logger.warning(f"⚠️  {incomplete_count} incomplete runs found")
        return 1
    else:
        logger.info("✅ All runs are complete!")
        return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
