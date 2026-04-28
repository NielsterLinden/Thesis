#!/usr/bin/env python
"""Delete W&B runs/artifacts, or undo a V2 axis backfill.

Modes
-----
``--mode delete`` (default, legacy)
    Delete runs and/or model artifacts from a W&B project. Supports
    ``--group`` and ``--runs-only`` / ``--artifacts-only`` as before.

``--mode undo_backfill_v2`` (new)
    Non-destructive reset of V2-written config keys. If a pre-backfill
    snapshot artifact is available, restore keys from it. Otherwise, either
    skip the run (default) or strip the V2 keys via ``--strip-if-no-snapshot``.

Examples
--------
.. code-block:: bash

    # Legacy dry-run delete (default)
    python scripts/wandb/cleanup_wandb.py --project thesis-ml

    # Undo V2 for all runs, restoring from the most recent snapshot artifact
    python scripts/wandb/cleanup_wandb.py --mode undo_backfill_v2 \
        --project thesis-ml --execute

    # Undo V2 for a subset of runs listed in a file
    python scripts/wandb/cleanup_wandb.py --mode undo_backfill_v2 \
        --run-ids-file reports/phaseB_failed.txt --execute
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from v2_axes import V2_AXES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy delete mode
# ---------------------------------------------------------------------------


def cleanup_wandb(
    project: str,
    entity: str | None = None,
    dry_run: bool = True,
    delete_runs: bool = True,
    delete_artifacts: bool = True,
    group: str | None = None,
) -> tuple[int, int]:
    """Delete runs and/or artifacts from a W&B project (legacy)."""
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        sys.exit(1)

    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project

    runs_deleted = 0
    artifacts_deleted = 0

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

    if delete_artifacts:
        try:
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
                    pass
        except Exception as e:
            logger.error("Failed to list artifacts: %s", e)

    return runs_deleted, artifacts_deleted


# ---------------------------------------------------------------------------
# Undo V2 backfill mode
# ---------------------------------------------------------------------------


def _load_run_ids_file(path: str | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.error("run-ids-file not found: %s", path)
        sys.exit(1)
    return {line.strip() for line in p.read_text().splitlines() if line.strip()}


def _load_latest_snapshot(api: Any, project_path: str, artifact_name: str) -> dict[str, dict[str, Any]]:
    """Load the most recent ``v2_backfill_snapshot`` artifact as a dict."""
    try:
        candidates = list(api.artifacts(type_name="v2_backfill_snapshot", project=project_path))
    except Exception as e:
        logger.warning("No v2_backfill_snapshot artifacts found (%s)", e)
        return {}
    if not candidates:
        logger.warning("No v2_backfill_snapshot artifacts found in %s", project_path)
        return {}
    # Prefer the artifact whose name contains the configured name; else use newest.
    snap = None
    for a in candidates:
        if a.name.startswith(artifact_name) or artifact_name in a.name:
            snap = a
            break
    if snap is None:
        snap = candidates[-1]
    logger.info("Using snapshot artifact: %s", snap.name)
    try:
        snap_dir = Path(snap.download())
    except Exception as e:
        logger.error("Failed to download snapshot artifact: %s", e)
        return {}
    # Find the JSON payload.
    json_files = list(snap_dir.glob("*.json"))
    if not json_files:
        logger.error("Snapshot artifact has no JSON file.")
        return {}
    with json_files[0].open("r", encoding="utf-8") as f:
        return json.load(f)


def undo_backfill_v2(
    project: str,
    entity: str | None,
    dry_run: bool,
    run_ids_file: str | None,
    artifact_name: str,
    strip_if_no_snapshot: bool,
) -> tuple[int, int, int]:
    """Restore V2 config keys from snapshot (or strip them)."""
    try:
        import wandb  # type: ignore  # noqa: F401
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        sys.exit(1)
    import wandb

    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    v2_keys = [a.target_key for a in V2_AXES]

    allowlist = _load_run_ids_file(run_ids_file)
    snapshot = _load_latest_snapshot(api, project_path, artifact_name)
    if snapshot:
        logger.info("Snapshot covers %d runs", len(snapshot))
    else:
        logger.warning("No snapshot loaded — %s mode.", "STRIP" if strip_if_no_snapshot else "SKIP-unknown-runs")

    try:
        all_runs = list(api.runs(project_path, per_page=500))
    except Exception as e:
        logger.error("Failed to fetch runs: %s", e)
        sys.exit(1)

    def _unwrap(v: Any) -> Any:
        if isinstance(v, dict):
            keys = set(v.keys())
            if keys and keys.issubset({"value", "desc"}):
                return v.get("value")
        return v

    def _safe_cfg(r: Any) -> dict:
        cfg = r.config
        if isinstance(cfg, str):
            try:
                parsed = json.loads(cfg)
                cfg = parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError):
                cfg = {}
        try:
            raw = dict(cfg)
        except (TypeError, ValueError):
            try:
                raw = dict(cfg.items())
            except Exception:
                try:
                    raw = {k: cfg[k] for k in list(cfg)}
                except Exception:
                    return {}
        return {k: _unwrap(v) for k, v in raw.items()}

    def _write_cfg(r: Any, updates: dict[str, Any]) -> None:
        """Merge ``updates`` into ``r._attrs['config']`` and call ``r.update()``.

        ``Run.update()`` in wandb 0.22.x takes no arguments and serialises
        ``_attrs['config']`` via the ``json_config`` property; that requires
        the attr to be a plain dict. Some legacy runs store it as a JSON
        string, so we normalise it first.
        """
        attrs = getattr(r, "_attrs", None)
        if attrs is None:
            raise RuntimeError(f"run {getattr(r, 'id', '?')} has no _attrs")
        normalised = _safe_cfg(r)
        normalised.update(updates)
        attrs["config"] = normalised
        r.update()

    restored_runs, stripped_runs, skipped_runs = 0, 0, 0
    for run in all_runs:
        if allowlist is not None and run.id not in allowlist:
            continue
        cfg = _safe_cfg(run)
        run_snap = snapshot.get(run.id)
        if run_snap is not None:
            to_write: dict[str, Any] = {}
            for k in v2_keys:
                prior = run_snap.get(k, None)
                if prior is None:
                    # Key didn't exist pre-backfill. Best approximation: set to "".
                    to_write[k] = ""
                else:
                    to_write[k] = prior
            if dry_run:
                logger.info("[DRY RUN] Would RESTORE %d V2 keys on %s (%s)", len(to_write), run.name, run.id)
            else:
                cfg.update(to_write)
                _write_cfg(run, to_write)
                logger.info("Restored %d V2 keys on %s (%s)", len(to_write), run.name, run.id)
            restored_runs += 1
        else:
            if not strip_if_no_snapshot:
                skipped_runs += 1
                continue
            to_write = {k: "" for k in v2_keys if k in cfg}
            if not to_write:
                skipped_runs += 1
                continue
            if dry_run:
                logger.info("[DRY RUN] Would STRIP %d V2 keys on %s (%s)", len(to_write), run.name, run.id)
            else:
                cfg.update(to_write)
                _write_cfg(run, to_write)
                logger.info("Stripped %d V2 keys on %s (%s)", len(to_write), run.name, run.id)
            stripped_runs += 1

    return restored_runs, stripped_runs, skipped_runs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Delete old W&B runs/artifacts, or undo a V2 backfill.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=("delete", "undo_backfill_v2"), default="delete")
    parser.add_argument("--project", type=str, default="thesis-ml")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually execute (default is dry-run mode).")
    parser.add_argument("--runs-only", action="store_true", help="delete mode: only delete runs, keep artifacts.")
    parser.add_argument("--artifacts-only", action="store_true", help="delete mode: only delete artifacts, keep runs.")
    parser.add_argument("--group", type=str, default=None, help="delete mode: only delete runs in this group.")
    parser.add_argument("--run-ids-file", type=str, default=None, help="undo mode: newline-delimited file of wandb run ids.")
    parser.add_argument("--snapshot-artifact-name", type=str, default="v2_backfill_snapshot", help="undo mode: name prefix of snapshot artifact.")
    parser.add_argument("--strip-if-no-snapshot", action="store_true", help="undo mode: strip V2 keys for runs not covered by snapshot.")
    args = parser.parse_args()

    dry_run = not args.execute
    if dry_run:
        logger.info("DRY RUN MODE - no actual changes will be made")
        logger.info("Run with --execute to actually execute")

    if args.mode == "delete":
        delete_runs = not args.artifacts_only
        delete_artifacts = not args.runs_only
        if not dry_run:
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
        logger.info("")
        logger.info("=" * 40)
        if dry_run:
            logger.info("DRY RUN COMPLETE — run with --execute to actually delete")
        else:
            logger.info("CLEANUP COMPLETE")
            logger.info("Runs deleted: %d", runs_deleted)
            logger.info("Artifacts deleted: %d", artifacts_deleted)
        logger.info("=" * 40)
        return

    # undo_backfill_v2
    if not dry_run:
        project_path = f"{args.entity}/{args.project}" if args.entity else args.project
        logger.warning("=" * 60)
        logger.warning("WARNING: This will RESET V2 config keys on runs in '%s'", project_path)
        logger.warning("  Mode: %s", "STRIP (no-snapshot runs get V2 keys cleared)" if args.strip_if_no_snapshot else "RESTORE-only (skip uncovered)")
        logger.warning("=" * 60)
        response = input("Type 'UNDO' to confirm: ")
        if response != "UNDO":
            logger.info("Aborted.")
            sys.exit(0)

    restored, stripped, skipped = undo_backfill_v2(
        project=args.project,
        entity=args.entity,
        dry_run=dry_run,
        run_ids_file=args.run_ids_file,
        artifact_name=args.snapshot_artifact_name,
        strip_if_no_snapshot=args.strip_if_no_snapshot,
    )
    logger.info("")
    logger.info("=" * 40)
    logger.info("UNDO V2 BACKFILL %s", "DRY RUN COMPLETE" if dry_run else "COMPLETE")
    logger.info("Runs restored from snapshot : %d", restored)
    logger.info("Runs stripped (no snapshot) : %d", stripped)
    logger.info("Runs skipped                : %d", skipped)
    logger.info("=" * 40)


if __name__ == "__main__":
    main()
