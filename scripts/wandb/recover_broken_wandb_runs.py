#!/usr/bin/env python
"""Re-upload four thesis-ml runs whose cloud records return HTTP 500.

Reads metrics/config from HPC facts directories under ``outputs/runs/`` and
creates **new** W&B runs in ``thesis-ml`` (new run ids). The old broken ids
are stored in config as ``recovery/broken_wandb_id`` for traceability.

After this script succeeds, run V2 backfill on the **new** run ids (use
``--backfill`` to run it automatically).

Default paths match Stoomboot; override with env ``THESIS_RUNS_ROOT`` if needed.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from migrate_runs_to_wandb import migrate_run  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# (broken_wandb_id, facts folder name under outputs/runs/)
RECOVERY_TABLE: list[tuple[str, str]] = [
    (
        "1iuxis2l",
        "run_20260320-141518_phd_exp1_4t_vs_bg_sizes_and_pe_job000",
    ),
    (
        "7ipzizh6",
        "run_20260320-123722_builtjes_baseline_test_job000",
    ),
    (
        "wqo0y6mw",
        "run_20251126-150519_compare_positional_encodings_job009",
    ),
    (
        "17ovy6fy",
        "run_20260320-163524_OrthogonalSweep_A_g04_attention_type_job003",
    ),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--project", default="thesis-ml")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "nterlind-nikhef"))
    p.add_argument(
        "--runs-root",
        default=os.environ.get(
            "THESIS_RUNS_ROOT",
            "/data/atlas/users/nterlind/outputs/runs",
        ),
        type=Path,
    )
    p.add_argument(
        "--name-suffix",
        default="_recovered",
        help="Appended to W&B display name (default: _recovered)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--backfill",
        action="store_true",
        help="After migrate, run scripts/wandb/backfill_labels.py --mode v2 on new ids",
    )
    p.add_argument(
        "--report-path",
        default="wandb_cleanup/backfill_recovered_four.csv",
        help="CSV path for optional V2 backfill step",
    )
    p.add_argument(
        "--check-existing-names",
        action="store_true",
        help="Fetch all run names first to skip duplicates (slow on large projects).",
    )
    args = p.parse_args()

    runs_root: Path = args.runs_root
    if not runs_root.is_dir():
        logger.error("runs root missing: %s", runs_root)
        return 1

    existing: set[str] | None = None
    if args.check_existing_names:
        from migrate_runs_to_wandb import get_existing_wandb_runs  # noqa: PLC0415

        logger.info("Checking existing run names in %s/%s ...", args.entity, args.project)
        existing = get_existing_wandb_runs(args.project, args.entity)
    else:
        logger.info("Skipping full-project name fetch (use --check-existing-names to enable).")

    mapping_rows: list[tuple[str, str, str]] = []  # broken_id, folder, new_id or ""

    for broken_id, folder in RECOVERY_TABLE:
        run_dir = runs_root / folder
        if not run_dir.is_dir():
            logger.error("Missing facts dir for %s: %s", broken_id, run_dir)
            return 1
        extra = {"recovery/broken_wandb_id": broken_id}
        result = migrate_run(
            run_dir,
            project=args.project,
            entity=args.entity,
            dry_run=args.dry_run,
            upload_artifacts=False,
            group_override=None,
            source_location="hpc_recover_broken_wandb",
            existing_runs=existing,
            name_suffix=args.name_suffix,
            extra_config=extra,
            extra_tags=["recovered-broken-wandb", f"replaces-{broken_id}"],
        )
        display = f"{folder}{args.name_suffix}"
        if isinstance(result, str) and result != "exists":
            mapping_rows.append((broken_id, folder, result))
            if existing is not None:
                existing.add(display)
        elif result == "exists":
            logger.warning(
                "Skipped %s: W&B name %s already exists — use a different --name-suffix or remove the run",
                broken_id,
                display,
            )
            mapping_rows.append((broken_id, folder, ""))
        elif result is True:
            mapping_rows.append((broken_id, folder, "(dry-run)"))
            if existing is not None:
                existing.add(display)
        else:
            logger.error("Failed to migrate %s (%s)", broken_id, run_dir)
            return 1

    map_path = Path("wandb_cleanup/recovered_wandb_id_map.csv")
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with map_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["broken_wandb_id", "facts_folder", "new_wandb_id"])
        w.writerows(mapping_rows)
    logger.info("Wrote mapping %s", map_path)

    if args.dry_run:
        logger.info("Dry run complete. Re-run without --dry-run to upload.")
        return 0

    new_ids = [row[2] for row in mapping_rows if row[2] and row[2] != "(dry-run)"]
    if len(new_ids) != len(RECOVERY_TABLE):
        logger.error("Not all runs were uploaded (see %s). Fix collisions then retry.", map_path)
        return 1

    ids_path = Path("wandb_cleanup/recovered_run_ids.txt")
    ids_path.write_text("\n".join(new_ids) + "\n", encoding="utf-8")
    logger.info("Wrote %s", ids_path)

    if not args.backfill:
        logger.info(
            "Next: python scripts/wandb/backfill_labels.py --mode v2 --execute " "--run-ids-file %s --write-empty-v2 --report-path %s",
            ids_path,
            args.report_path,
        )
        return 0

    repo_root = _THIS_DIR.parent.parent
    cmd = [
        sys.executable,
        str(repo_root / "scripts/wandb/backfill_labels.py"),
        "--mode",
        "v2",
        "--execute",
        "--entity",
        args.entity,
        "--project",
        args.project,
        "--run-ids-file",
        str(ids_path),
        "--write-empty-v2",
        "--report-path",
        str(args.report_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(repo_root))


if __name__ == "__main__":
    sys.exit(main())
