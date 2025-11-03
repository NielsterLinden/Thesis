#!/usr/bin/env python3
"""Migrate legacy experiment directory with numbered subdirectories to new structure.

Usage:
    python scripts/migrate_legacy_runs.py /path/to/old/experiment/dir/
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def extract_experiment_info(exp_dir: Path) -> tuple[str, str]:
    """Extract timestamp and experiment name from directory name.

    Expected format: exp_YYYYMMDD-HHMMSS_name

    Parameters
    ----------
    exp_dir : Path
        Path to experiment directory

    Returns
    -------
    tuple[str, str]
        (timestamp, experiment_name)
    """
    name = exp_dir.name
    match = re.match(r"exp_(\d{8}-\d{6})_(.+)", name)
    if not match:
        raise ValueError(f"Directory name '{name}' does not match expected pattern 'exp_YYYYMMDD-HHMMSS_name'")
    return match.group(1), match.group(2)


def validate_run_directory(run_dir: Path) -> bool:
    """Validate that a directory contains run artifacts.

    Parameters
    ----------
    run_dir : Path
        Path to potential run directory

    Returns
    -------
    bool
        True if directory contains required artifacts
    """
    hydra_cfg = run_dir / ".hydra" / "config.yaml"
    legacy_cfg = run_dir / "cfg.yaml"
    return hydra_cfg.exists() or legacy_cfg.exists()


def migrate_experiment(old_exp_dir: Path, output_root: Path) -> dict:
    """Migrate legacy experiment directory to new structure.

    Parameters
    ----------
    old_exp_dir : Path
        Path to old experiment directory (e.g., exp_20251031-152750_compare_globals_heads)
    output_root : Path
        Root output directory (e.g., outputs/)

    Returns
    -------
    dict
        Summary report with migration details
    """
    old_exp_dir = Path(old_exp_dir).resolve()
    output_root = Path(output_root).resolve()

    if not old_exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {old_exp_dir}")
    if not old_exp_dir.is_dir():
        raise ValueError(f"Path is not a directory: {old_exp_dir}")

    # Extract timestamp and experiment name
    timestamp, exp_name = extract_experiment_info(old_exp_dir)

    # Create new directories
    runs_dir = output_root / "runs"
    multiruns_dir = output_root / "multiruns"
    runs_dir.mkdir(parents=True, exist_ok=True)
    multiruns_dir.mkdir(parents=True, exist_ok=True)

    new_multirun_dir = multiruns_dir / f"exp_{timestamp}_{exp_name}"
    new_multirun_dir.mkdir(parents=True, exist_ok=True)

    # Find all numbered subdirectories
    numbered_dirs = []
    for item in old_exp_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            numbered_dirs.append((int(item.name), item))

    numbered_dirs.sort(key=lambda x: x[0])

    migrated_runs = []
    skipped = []

    # Migrate each numbered subdirectory
    for job_num, old_run_dir in numbered_dirs:
        if not validate_run_directory(old_run_dir):
            skipped.append((job_num, old_run_dir, "Missing config files"))
            continue

        # Create new run directory name
        new_run_id = f"run_{timestamp}_{exp_name}_job{job_num}"
        new_run_dir = runs_dir / new_run_id

        if new_run_dir.exists():
            skipped.append((job_num, old_run_dir, f"Target already exists: {new_run_dir}"))
            continue

        # Move directory
        try:
            shutil.move(str(old_run_dir), str(new_run_dir))
            migrated_runs.append((job_num, new_run_dir))
        except Exception as e:
            skipped.append((job_num, old_run_dir, f"Error during move: {e}"))

    # Copy multirun.yaml if it exists
    old_multirun_yaml = old_exp_dir / "multirun.yaml"
    if old_multirun_yaml.exists():
        shutil.copy2(str(old_multirun_yaml), str(new_multirun_dir / "multirun.yaml"))

    # Validate migration
    validation_errors = []
    for job_num, expected_run_dir in migrated_runs:
        if not expected_run_dir.exists():
            validation_errors.append(f"Job {job_num}: Expected run directory not found: {expected_run_dir}")
        elif not validate_run_directory(expected_run_dir):
            validation_errors.append(f"Job {job_num}: Run directory missing config files: {expected_run_dir}")

    return {
        "old_experiment_dir": str(old_exp_dir),
        "new_multirun_dir": str(new_multirun_dir),
        "timestamp": timestamp,
        "experiment_name": exp_name,
        "total_jobs": len(numbered_dirs),
        "migrated": len(migrated_runs),
        "skipped": len(skipped),
        "migrated_runs": [(job_num, str(run_dir)) for job_num, run_dir in migrated_runs],
        "skipped_details": [(job_num, str(run_dir), reason) for job_num, run_dir, reason in skipped],
        "validation_errors": validation_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy experiment directory with numbered subdirectories to new structure")
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to old experiment directory (e.g., /data/atlas/users/nterlind/outputs/experiments/exp_20251031-152750_compare_globals_heads/)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root output directory (default: outputs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually moving files",
    )

    args = parser.parse_args()

    old_exp_dir = Path(args.experiment_dir).resolve()
    output_root = Path(args.output_root).resolve()

    if args.dry_run:
        print(f"[DRY RUN] Would migrate from: {old_exp_dir}")
        print(f"[DRY RUN] To output root: {output_root}")

        # Count numbered directories
        numbered_dirs = [item for item in old_exp_dir.iterdir() if item.is_dir() and item.name.isdigit()]
        print(f"[DRY RUN] Found {len(numbered_dirs)} numbered subdirectories")
        return

    try:
        report = migrate_experiment(old_exp_dir, output_root)

        print("=" * 80)
        print("Migration Summary")
        print("=" * 80)
        print(f"Old experiment directory: {report['old_experiment_dir']}")
        print(f"New multirun directory: {report['new_multirun_dir']}")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Experiment name: {report['experiment_name']}")
        print(f"Total jobs found: {report['total_jobs']}")
        print(f"Successfully migrated: {report['migrated']}")
        print(f"Skipped: {report['skipped']}")

        if report["migrated_runs"]:
            print("\nMigrated runs:")
            for job_num, run_dir in report["migrated_runs"]:
                print(f"  Job {job_num}: {run_dir}")

        if report["skipped_details"]:
            print("\nSkipped runs:")
            for job_num, run_dir, reason in report["skipped_details"]:
                print(f"  Job {job_num}: {run_dir}")
                print(f"    Reason: {reason}")

        if report["validation_errors"]:
            print("\nValidation errors:")
            for error in report["validation_errors"]:
                print(f"  ERROR: {error}")

        if not report["validation_errors"] and report["migrated"] > 0:
            print("\n✓ Migration completed successfully!")

    except Exception as e:
        print(f"ERROR: Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
