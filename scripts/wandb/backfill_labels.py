#!/usr/bin/env python
"""Stamp or V2-derive WandB run config keys.

This script now has two modes:

- ``--mode stamp`` (default, legacy) — write a fixed ``{key: value}`` dict
  to matching runs. Useful when you add a new config field and want old runs
  to show a default value.
- ``--mode v2`` — compute V2 axis/sub-axis values per run via
  :func:`scripts.wandb.v2_axes.derive_v2_axes`, overwrite V2 target keys,
  and emit a CSV report. See ``docs/backfill_plan/`` for the full spec.

Examples
--------
Legacy stamp mode (unchanged CLI):

.. code-block:: bash

    # Dry run
    python scripts/wandb/backfill_labels.py --dry-run --labels '{"training/loss_type": "bce"}'
    # Apply
    python scripts/wandb/backfill_labels.py --labels '{"training/loss_type": "bce"}' --execute

V2 mode (new):

.. code-block:: bash

    # Phase A: full dry run to CSV
    python scripts/wandb/backfill_labels.py --mode v2 \
        --report-path reports/backfill_dryrun.csv

    # Phase B: pilot on 5 runs
    python scripts/wandb/backfill_labels.py --mode v2 --execute --limit 5 \
        --report-path reports/backfill_pilot.csv --snapshot-before-write

    # Phase C: full backfill with snapshot + resume support
    python scripts/wandb/backfill_labels.py --mode v2 --execute \
        --report-path reports/backfill_phaseC.csv --snapshot-before-write
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import re
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from v2_axes import V2_AXES, derive_v2_axes  # noqa: E402

_TARGET_TO_AXIS_ID: dict[str, str] = {a.target_key: a.id for a in V2_AXES}

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("BACKFILL_DEBUG") else logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy stamp mode
# ---------------------------------------------------------------------------


def backfill_labels(
    labels: dict[str, Any],
    project: str = "thesis-ml",
    entity: str | None = "nterlind-nikhef",
    dry_run: bool = True,
    force: bool = False,
    group_filter: str | None = None,
    tag_filter: str | None = None,
) -> tuple[int, int, int]:
    """Stamp fixed labels onto runs (legacy behaviour). Returns (updated, skipped, errors)."""
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        sys.exit(1)

    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project

    try:
        all_runs = list(api.runs(project_path))
    except Exception as e:
        logger.error("Failed to fetch runs: %s", e)
        sys.exit(1)

    group_re = re.compile(group_filter) if group_filter else None
    updated, skipped, errors = 0, 0, 0

    for run in all_runs:
        run_group = getattr(run, "group", None) or ""
        run_tags = getattr(run, "tags", []) or []

        if group_re and not group_re.search(run_group):
            continue
        if tag_filter and tag_filter not in run_tags:
            continue

        try:
            config = _read_run_config(run)
            to_update: dict[str, Any] = {}
            for key, val in labels.items():
                if key in config and not force:
                    continue
                to_update[key] = val

            if not to_update:
                skipped += 1
                continue

            if dry_run:
                logger.info("[DRY RUN] Would update %s (%s): %s", run.name, run.id, to_update)
                updated += 1
            else:
                config.update(to_update)
                _write_run_config(run, to_update)
                logger.info("Updated %s (%s): %s", run.name, run.id, to_update)
                updated += 1
        except Exception as e:
            errors += 1
            logger.warning("Failed %s (%s): %s", run.name, run.id, e)

    return updated, skipped, errors


# ---------------------------------------------------------------------------
# V2 mode helpers
# ---------------------------------------------------------------------------


REPORT_COLUMNS = [
    "run_id",
    "run_name",
    "experiment_name",
    "keys_written",
    "keys_left_empty_by_prereq",
    "keys_empty_missing_config",
    "keys_skipped_already_set",
    "unresolved_flags",
    "error",
]


def _unwrap_wandb_value(v: Any) -> Any:
    """Strip W&B's on-disk ``{"value": X, "desc": ...}`` envelope if present.

    The Public API normally hands back unwrapped values via ``run.config[k]``,
    but iterating ``dict(run.config)`` or reading a JSONL dump can surface the
    raw storage shape. We detect that shape specifically (a dict whose keys
    are a subset of ``{"value", "desc"}``) and return the inner value.
    """
    if isinstance(v, dict):
        keys = set(v.keys())
        if keys and keys.issubset({"value", "desc"}):
            return v.get("value")
    return v


def _coerce_config_value(cfg: Any) -> dict:
    """Return a plain dict from whatever shape ``run.config`` presents.

    Handles:
    - dict-like that iterates as ``(k, v)`` tuples   (normal case)
    - dict-like that iterates as bare keys           (some wandb versions)
    - plain string containing JSON                   (legacy double-encoded runs)
    - each entry wrapped as ``{"value": X, "desc": ...}``  (raw W&B shape)
    """
    if cfg is None:
        return {}
    if isinstance(cfg, str):
        try:
            parsed = json.loads(cfg)
            if isinstance(parsed, dict):
                cfg = parsed
            else:
                return {}
        except (json.JSONDecodeError, ValueError):
            return {}
    raw: dict
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
    return {k: _unwrap_wandb_value(v) for k, v in raw.items()}


def _read_run_config(run: Any) -> dict:
    """Read ``run.config`` into a plain dict — robust against legacy shapes."""
    return _coerce_config_value(run.config)


def _write_run_config(run: Any, updates: dict[str, Any]) -> None:
    """Merge ``updates`` into the run config and persist via ``run.update()``."""
    cfg = getattr(run, "config", None)
    if cfg is None:
        raise RuntimeError(f"run {getattr(run, 'id', '?')} has no config")
    if isinstance(cfg, str):
        # W&B sometimes stores config as a JSON string; normalise then write via
        # ``_attrs`` (same approach as ``cleanup_wandb.undo_backfill_v2``).
        normalised = dict(_coerce_config_value(cfg))
        normalised.update(updates)
        attrs = getattr(run, "_attrs", None)
        if attrs is None:
            raise RuntimeError(f"run {getattr(run, 'id', '?')} has no _attrs for legacy config write")
        attrs["config"] = normalised
        run.update()
        return
    try:
        cfg.update(updates)
    except Exception:
        for k, v in updates.items():
            cfg[k] = v
    run.update()


def _experiment_name(run: Any) -> str:
    """Best-effort experiment name: group > meta.experiment_name > ''."""
    grp = getattr(run, "group", None)
    if grp:
        return str(grp)
    try:
        cfg = _read_run_config(run)
        for k in ("meta.experiment_name", "meta/experiment_name", "experiment_name"):
            if k in cfg and cfg[k]:
                return str(cfg[k])
    except Exception:
        pass
    return ""


def _load_run_ids_file(path: str | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.error("run-ids-file not found: %s", path)
        sys.exit(1)
    ids = {line.strip() for line in p.read_text().splitlines() if line.strip()}
    logger.info("Loaded %d run_ids from %s", len(ids), path)
    return ids


def _load_resume_run_ids(path: str | None) -> set[str]:
    """Return set of run_ids already present in a prior report CSV."""
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        logger.warning("resume-from-report not found: %s (continuing without skip-set)", path)
        return set()
    ids: set[str] = set()
    with p.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = row.get("run_id") or ""
            if rid:
                ids.add(rid)
    logger.info("Resume-set loaded %d ids from %s", len(ids), path)
    return ids


def _write_report(path: str, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in REPORT_COLUMNS})
    logger.info("Report written: %s (%d rows)", path, len(rows))


def _log_report_artifact(
    report_path: str,
    project: str,
    entity: str | None,
    summary: dict[str, Any],
    tag: str = "backfill_v2",
) -> None:
    """Log the report as a wandb Artifact under a throwaway run."""
    try:
        import wandb  # type: ignore
    except ImportError:
        logger.warning("wandb not installed; skipping artifact logging")
        return
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            job_type="backfill_v2",
            name=f"backfill_v2_report_{int(time.time())}",
            tags=[tag],
            reinit=True,
        )
        art = wandb.Artifact(name="v2_backfill_report", type="v2_backfill_report")
        art.add_file(report_path)
        run.log_artifact(art)
        for k, v in summary.items():
            with contextlib.suppress(Exception):
                run.summary[k] = v
        run.finish()
        logger.info("Report artifact logged to %s/%s", entity or "default", project)
    except Exception as e:
        logger.warning("Artifact logging failed: %s", e)


def _log_snapshot_artifact(
    snapshot: dict[str, dict[str, Any]],
    project: str,
    entity: str | None,
    tag: str = "backfill_v2",
    artifact_name: str = "v2_backfill_snapshot",
) -> None:
    """Log the pre-write snapshot of V2 keys as a wandb Artifact."""
    try:
        import wandb  # type: ignore
    except ImportError:
        logger.warning("wandb not installed; skipping snapshot logging")
        return
    try:
        tmp_dir = Path("wandb_cleanup/_snapshots")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        snap_path = tmp_dir / f"v2_snapshot_{int(time.time())}.json"
        snap_path.write_text(json.dumps(snapshot, indent=2, default=str))
        run = wandb.init(
            project=project,
            entity=entity,
            job_type="backfill_v2_snapshot",
            name=f"backfill_v2_snapshot_{int(time.time())}",
            tags=[tag],
            reinit=True,
        )
        art = wandb.Artifact(name=artifact_name, type="v2_backfill_snapshot")
        art.add_file(str(snap_path))
        run.log_artifact(art)
        run.summary["n_runs_snapshotted"] = len(snapshot)
        run.finish()
        logger.info("Snapshot artifact logged (%d runs, file=%s)", len(snapshot), snap_path)
    except Exception as e:
        logger.warning("Snapshot logging failed: %s", e)


def _iter_batches(seq: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        yield seq
        return
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ---------------------------------------------------------------------------
# V2 mode main
# ---------------------------------------------------------------------------


class _DumpRun:
    """Minimal duck-type of ``wandb.apis.public.Run`` for offline derivation.

    Exposes only what the v2 derivation loop reads: ``id``, ``name``,
    ``group``, ``tags``, and ``config`` (already a dict). Cannot be updated.
    """

    __slots__ = ("id", "name", "group", "tags", "state", "config")

    def __init__(self, record: dict[str, Any]) -> None:
        self.id = record.get("id", "")
        self.name = record.get("name", "")
        self.group = record.get("group")
        self.tags = list(record.get("tags") or [])
        self.state = record.get("state", "")
        cfg = record.get("config") or {}
        if not isinstance(cfg, dict):
            cfg = _coerce_config_value(cfg)
        self.config = cfg


def _load_dump(path: str) -> list[_DumpRun]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--from-dump path does not exist: {path}")
    runs: list[_DumpRun] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Malformed JSONL line %d in %s: %s", i, path, e)
                continue
            runs.append(_DumpRun(record))
    logger.info("Loaded %d runs from dump %s", len(runs), path)
    return runs


def run_v2_mode(args: argparse.Namespace) -> int:
    from_dump = bool(getattr(args, "from_dump", None))

    if not from_dump:
        try:
            import wandb  # type: ignore  # noqa: F401
        except ImportError:
            logger.error("wandb not installed. Install with: pip install wandb")
            return 1
        import wandb  # type: ignore

    dry_run = not args.execute or from_dump
    overwrite = True
    if args.no_overwrite:
        overwrite = False
    write_empty_v2 = bool(getattr(args, "write_empty_v2", False))

    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    source_label = f"dump={args.from_dump}" if from_dump else f"api={project_path}"
    logger.info(
        "V2 mode | %s | dry_run=%s | overwrite=%s | write_empty_v2=%s",
        source_label,
        dry_run,
        overwrite,
        write_empty_v2,
    )
    logger.info("V2 axes registered: %d target keys", len(V2_AXES))

    group_re = re.compile(args.group) if args.group else None
    exp_re = re.compile(args.experiment_filter) if args.experiment_filter else None
    id_allowlist = _load_run_ids_file(args.run_ids_file)
    skip_ids = _load_resume_run_ids(args.resume_from_report)

    t0 = time.time()
    if from_dump:
        try:
            all_dump_runs = _load_dump(args.from_dump)
        except Exception as e:
            logger.error("Failed to load dump: %s", e)
            return 1
        filtered: list[Any] = []
        for run in all_dump_runs:
            if id_allowlist is not None and run.id not in id_allowlist:
                continue
            if run.id in skip_ids:
                continue
            g = run.group or ""
            if group_re and not group_re.search(g):
                continue
            if exp_re and not exp_re.search(g):
                continue
            if args.tag and args.tag not in run.tags:
                continue
            filtered.append(run)
            if args.limit and len(filtered) >= args.limit:
                break
        n_seen = len(all_dump_runs)
    else:
        api = wandb.Api()  # type: ignore[name-defined]
        server_filters: dict[str, Any] = {}
        if id_allowlist is not None:
            server_filters["name"] = {"$in": sorted(id_allowlist)}
        try:
            runs_iter = api.runs(project_path, filters=server_filters or None, per_page=500)
        except Exception as e:
            logger.error("Failed to start runs iterator: %s", e)
            return 1
        filtered = []
        n_seen = 0
        for run in runs_iter:
            n_seen += 1
            if run.id in skip_ids:
                continue
            g = getattr(run, "group", None) or ""
            tags = getattr(run, "tags", []) or []
            if group_re and not group_re.search(g):
                continue
            if exp_re and not exp_re.search(g):
                continue
            if args.tag and args.tag not in tags:
                continue
            filtered.append(run)
            if args.limit and len(filtered) >= args.limit:
                break

    logger.info(
        "Enumerated %d runs in %.1fs; %d pass filters",
        n_seen,
        time.time() - t0,
        len(filtered),
    )

    # --- pre-write snapshot
    snapshot: dict[str, dict[str, Any]] = {}

    # --- derivation loop
    report_rows: list[dict[str, Any]] = []
    total_keys_written = 0
    n_runs_with_errors = 0
    n_runs_with_unresolved = 0
    v2_keys = [a.target_key for a in V2_AXES]

    for batch in _iter_batches(filtered, args.rate_limit_batch_size):
        for run in batch:
            row: dict[str, Any] = {
                "run_id": run.id,
                "run_name": getattr(run, "name", ""),
                "experiment_name": _experiment_name(run),
                "keys_written": 0,
                "keys_left_empty_by_prereq": 0,
                "keys_empty_missing_config": 0,
                "keys_skipped_already_set": 0,
                "unresolved_flags": "",
                "error": "",
            }
            try:
                cfg = _read_run_config(run)
                flag_bucket: dict[str, list[str]] = {
                    "keys_left_empty_by_prereq": [],
                    "keys_empty_missing_config": [],
                    "unresolved_flags": [],
                }
                v2_values = derive_v2_axes(cfg, flag_bucket=flag_bucket)

                # Capture pre-state of V2 keys for snapshot.
                if args.snapshot_before_write and not dry_run:
                    snapshot[run.id] = {k: cfg.get(k, None) for k in v2_keys}

                to_write: dict[str, Any] = {}
                skipped_keys: list[str] = []
                prereq_keys = set(flag_bucket["keys_left_empty_by_prereq"])
                missing_keys = set(flag_bucket["keys_empty_missing_config"])
                for key, new_val in v2_values.items():
                    axis_id = _TARGET_TO_AXIS_ID.get(key)
                    # Default behaviour keeps historical semantics: don't write
                    # keys that resolve to empty via prereq or missing-config.
                    # Optional --write-empty-v2 writes these as explicit "" to
                    # force a uniform key set on every run.
                    if (not write_empty_v2) and (axis_id in prereq_keys or axis_id in missing_keys):
                        continue
                    current = cfg.get(key, None)
                    current_str = "" if current is None else str(current)
                    # When --write-empty-v2 is enabled, treat an absent key as
                    # different from an explicit empty string so we can materialize
                    # the full V2 key set on the run.
                    if current_str == new_val and not (write_empty_v2 and current is None and new_val == ""):
                        # Already correct — either a true idempotent no-op or a
                        # pre-existing match.
                        skipped_keys.append(key)
                        continue
                    if not overwrite and current is not None and current_str != "":
                        skipped_keys.append(key)
                        continue
                    to_write[key] = new_val

                row["keys_written"] = len(to_write)
                row["keys_left_empty_by_prereq"] = len(flag_bucket["keys_left_empty_by_prereq"])
                row["keys_empty_missing_config"] = len(flag_bucket["keys_empty_missing_config"])
                row["keys_skipped_already_set"] = len(skipped_keys)
                row["unresolved_flags"] = ";".join(flag_bucket["unresolved_flags"])

                if dry_run:
                    logger.info(
                        "[DRY RUN] %s (%s) keys_to_write=%d empty_by_prereq=%d missing=%d",
                        run.name,
                        run.id,
                        len(to_write),
                        row["keys_left_empty_by_prereq"],
                        row["keys_empty_missing_config"],
                    )
                else:
                    if to_write:
                        cfg.update(to_write)
                        _write_run_config(run, to_write)
                        logger.info(
                            "Wrote %d V2 keys to %s (%s)",
                            len(to_write),
                            run.name,
                            run.id,
                        )
                    else:
                        logger.info("No-op (idempotent) for %s (%s)", run.name, run.id)

                total_keys_written += len(to_write)
                if flag_bucket["unresolved_flags"]:
                    n_runs_with_unresolved += 1
            except Exception as e:
                import traceback

                row["error"] = repr(e)
                n_runs_with_errors += 1
                logger.warning("Failed %s (%s): %s", getattr(run, "name", ""), run.id, e)
                logger.debug("Traceback:\n%s", traceback.format_exc())

            if row["error"] == "":
                total = row["keys_written"] + row["keys_left_empty_by_prereq"] + row["keys_empty_missing_config"] + row["keys_skipped_already_set"]
                if total != len(V2_AXES):
                    diff = total - len(V2_AXES)
                    row["unresolved_flags"] = ((row["unresolved_flags"] + ";") if row["unresolved_flags"] else "") + f"row_total_mismatch:{total}_vs_{len(V2_AXES)}_diff:{diff}"

            report_rows.append(row)

        if args.rate_limit_sleep_seconds and not dry_run:
            time.sleep(args.rate_limit_sleep_seconds)

    # --- emit outputs
    if args.report_path:
        _write_report(args.report_path, report_rows)

    # --- summary
    logger.info("=" * 60)
    logger.info("V2 backfill summary")
    logger.info("  runs_processed        : %d", len(report_rows))
    logger.info("  total_keys_written    : %d", total_keys_written)
    logger.info("  runs_with_unresolved  : %d", n_runs_with_unresolved)
    logger.info("  runs_with_errors      : %d", n_runs_with_errors)
    logger.info("  dry_run               : %s", dry_run)
    logger.info("=" * 60)

    # --- log artifacts
    if not dry_run and args.snapshot_before_write and snapshot:
        _log_snapshot_artifact(
            snapshot,
            project=args.project,
            entity=args.entity,
            artifact_name=args.snapshot_artifact_name,
        )
    if not dry_run and args.report_path and args.log_report_artifact:
        _log_report_artifact(
            args.report_path,
            project=args.project,
            entity=args.entity,
            summary={
                "n_runs_written": len(report_rows),
                "total_keys_written": total_keys_written,
                "runs_with_unresolved_flags": n_runs_with_unresolved,
                "runs_with_errors": n_runs_with_errors,
            },
        )

    return 0 if n_runs_with_errors == 0 else 2


# ---------------------------------------------------------------------------
# Stamp mode wrapper
# ---------------------------------------------------------------------------


def run_stamp_mode(args: argparse.Namespace) -> int:
    if not args.labels:
        logger.error("--labels is required in stamp mode.")
        return 1
    try:
        labels = json.loads(args.labels)
        if not isinstance(labels, dict):
            raise ValueError("labels must be a JSON object")
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON for --labels: %s", e)
        return 1
    except ValueError as e:
        logger.error("%s", e)
        return 1

    dry_run = not args.execute
    if dry_run:
        logger.info("DRY RUN - no changes will be made. Use --execute to apply.")

    updated, skipped, errors = backfill_labels(
        labels=labels,
        project=args.project,
        entity=args.entity or None,
        dry_run=dry_run,
        force=args.force or args.overwrite,
        group_filter=args.group,
        tag_filter=args.tag,
    )
    logger.info("")
    logger.info("Summary: updated=%d, skipped=%d, errors=%d", updated, skipped, errors)
    return 0 if errors == 0 else 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stamp (legacy) or V2-derive WandB run config keys.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=("stamp", "v2"), default="stamp", help="Which backfill path to use (default: stamp).")

    # Stamp-mode flags (back-compat)
    p.add_argument("--labels", type=str, default=None, help='stamp mode: JSON dict e.g. \'{"training/loss_type": "bce"}\'')
    p.add_argument("--force", action="store_true", help="stamp mode: overwrite existing key.")

    # Shared W&B target
    p.add_argument("--project", type=str, default="thesis-ml")
    p.add_argument("--entity", type=str, default="nterlind-nikhef")

    # Dry-run / execute
    p.add_argument("--dry-run", action="store_true", default=True, help="Preview only, no writes (default: True)")
    p.add_argument("--execute", action="store_true", help="Actually update runs (disables dry-run)")

    # V2 overwrite toggles
    p.add_argument("--overwrite", action="store_true", help="v2 mode: default TRUE; explicit flag forces overwrite also in stamp mode.")
    p.add_argument("--no-overwrite", action="store_true", help="v2 mode: do NOT overwrite V2 keys that already have a value.")

    # Run filters
    p.add_argument("--group", type=str, default=None, help="Regex to filter runs by group name.")
    p.add_argument("--experiment-filter", type=str, default=None, help="Alias for --group.")
    p.add_argument("--tag", type=str, default=None, help="Filter runs that have this tag.")
    p.add_argument("--limit", type=int, default=0, help="Process at most this many filtered runs (0 = all).")
    p.add_argument("--run-ids-file", type=str, default=None, help="Newline-delimited file of wandb run ids to include.")

    # V2 reporting / rate-limit / snapshot
    p.add_argument("--report-path", type=str, default=None, help="v2 mode: CSV report path (recommended).")
    p.add_argument("--rate-limit-batch-size", type=int, default=50)
    p.add_argument("--rate-limit-sleep-seconds", type=float, default=2.0)
    p.add_argument("--resume-from-report", type=str, default=None, help="v2 mode: skip run_ids already present in a prior report CSV.")
    p.add_argument("--snapshot-before-write", action="store_true", help="v2 mode: capture V2-key pre-state and log as wandb artifact.")
    p.add_argument("--snapshot-artifact-name", type=str, default="v2_backfill_snapshot")
    p.add_argument("--log-report-artifact", action="store_true", help="v2 mode: log report CSV + summary scalars as wandb artifact.")
    p.add_argument("--write-empty-v2", action="store_true", help="v2 mode: also write derived empty-string values so all V2 keys exist on each run.")

    # Offline derivation source
    p.add_argument("--from-dump", type=str, default=None, help="v2 mode: path to a JSONL produced by scripts/wandb/dump_runs.py. " "Derivation reads from the dump instead of calling the W&B API. " "Implies --dry-run (no writes).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "v2":
        # --from-dump always implies dry-run; forbid --execute pairing.
        if args.from_dump and args.execute:
            logger.error(
                "--from-dump is a local derivation pass and cannot be combined with --execute. " "Use it to produce a report, then rerun without --from-dump to write to W&B.",
            )
            sys.exit(1)
        # Enforce --report-path for v2 executions (dry-run may skip).
        if args.execute and not args.report_path:
            logger.error("--report-path is required when --mode v2 and --execute are set.")
            sys.exit(1)
        # Default to a timestamped report if none given for dry-run.
        if not args.report_path:
            default_dir = Path("wandb_cleanup")
            default_dir.mkdir(parents=True, exist_ok=True)
            args.report_path = str(default_dir / f"backfill_dryrun_{int(time.time())}.csv")
            logger.info("No --report-path given; defaulting to %s", args.report_path)
        rc = run_v2_mode(args)
    else:
        rc = run_stamp_mode(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
