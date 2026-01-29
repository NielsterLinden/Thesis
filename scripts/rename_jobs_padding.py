#!/usr/bin/env python
"""Rename job folders to use 3-digit zero-padding (job0 -> job000)."""

import argparse
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Pattern to match job suffix: _job followed by 1-3 digits at the end
JOB_PATTERN = re.compile(r"^(.+_job)(\d{1,3})$")


def rename_jobs(runs_dir: Path, dry_run: bool = True) -> tuple[int, int]:
    """
    Rename all run folders to use 3-digit zero-padded job numbers.

    Args:
        runs_dir: Path to the runs directory
        dry_run: If True, only print what would be renamed

    Returns:
        Tuple of (renamed_count, skipped_count)
    """
    if not runs_dir.exists():
        logger.error(f"Directory does not exist: {runs_dir}")
        return 0, 0

    renamed = 0
    skipped = 0
    already_correct = 0

    # Get all run directories
    run_dirs = sorted(runs_dir.iterdir())

    for run_path in run_dirs:
        if not run_path.is_dir():
            continue

        name = run_path.name
        match = JOB_PATTERN.match(name)

        if not match:
            # Not a job folder, skip
            continue

        prefix = match.group(1)  # e.g., "run_20251126-185501_binary_1v1_5tops_job"
        job_num = match.group(2)  # e.g., "0", "11", "123"

        # Check if already 3 digits
        if len(job_num) == 3:
            already_correct += 1
            continue

        # Create new name with 3-digit padding
        new_name = f"{prefix}{int(job_num):03d}"
        new_path = run_path.parent / new_name

        if new_path.exists():
            logger.warning(f"Target already exists, skipping: {name} -> {new_name}")
            skipped += 1
            continue

        if dry_run:
            logger.info(f"[DRY RUN] Would rename: {name} -> {new_name}")
        else:
            run_path.rename(new_path)
            logger.info(f"Renamed: {name} -> {new_name}")

        renamed += 1

    return renamed, skipped, already_correct


def main():
    parser = argparse.ArgumentParser(description="Rename job folders to use 3-digit zero-padding")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Path to runs directory containing run_* folders",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the renames (default is dry-run)",
    )
    args = parser.parse_args()

    dry_run = not args.execute

    if dry_run:
        logger.info("DRY RUN MODE - no actual renames will be made")
        logger.info("Run with --execute to actually rename")
    else:
        logger.info("EXECUTE MODE - will rename folders")
        response = input("Type 'RENAME' to confirm: ")
        if response.strip() != "RENAME":
            logger.info("Aborted.")
            return

    logger.info(f"Scanning: {args.runs_dir}")
    renamed, skipped, already_correct = rename_jobs(args.runs_dir, dry_run=dry_run)

    logger.info("")
    logger.info("=" * 50)
    if dry_run:
        logger.info("DRY RUN COMPLETE")
        logger.info(f"Would rename: {renamed}")
    else:
        logger.info("RENAME COMPLETE")
        logger.info(f"Renamed: {renamed}")
    logger.info(f"Skipped (conflict): {skipped}")
    logger.info(f"Already correct (3 digits): {already_correct}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
