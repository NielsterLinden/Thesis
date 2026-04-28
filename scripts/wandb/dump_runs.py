#!/usr/bin/env python
"""One-shot dump of every run in a W&B project to a local JSONL file.

Motivation
----------
The backfill derivation pass is a pure function of ``run.config``. The W&B
Public API lazy-loads each run's config, which costs ~0.5–1 s per run over
the network. Doing 1002 lazy fetches every time you want to iterate on the
derivation logic is slow and flaky.

Instead we fetch once, cache to disk, and re-derive locally on an indexable
file. The backfill script has a matching ``--from-dump <path>`` mode that
reads this file instead of calling the API.

Output formats
--------------
Primary: ``wandb_cleanup/wandb_dump.jsonl`` — one JSON object per run,
preserving nested dicts. Fields per line:

.. code-block:: json

    {
      "id": "z8n245or",
      "name": "run_20251029-150416_run",
      "group": null,
      "tags": [],
      "state": "finished",
      "created_at": "2025-10-29T15:04:16Z",
      "config": {... full run.config ...},
      "summary": {... run.summary._json_dict ...}
    }

Optional (when ``--csv`` is passed): a flat ``wandb_dump.csv`` with one row
per run, the keys stringified; useful for Excel spot-checks but lossy on
nested values. Prefer JSONL for programmatic use.

Resilience
----------
Some legacy runs have ``run.config`` double-encoded as a JSON string at the
backend. The dumper detects that and decodes transparently so the resulting
dump always has dict-typed config.

Usage
-----
.. code-block:: bash

    # Full project dump (default output: wandb_cleanup/wandb_dump.jsonl)
    python scripts/wandb/dump_runs.py

    # Custom output path + CSV companion
    python scripts/wandb/dump_runs.py \
        --out wandb_cleanup/wandb_dump.jsonl --csv

    # Limit to the first 20 runs (quick smoke)
    python scripts/wandb/dump_runs.py --limit 20 \
        --out wandb_cleanup/wandb_dump_smoke.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _unwrap_wandb_value(v: Any) -> Any:
    """Strip the W&B on-disk ``{"value": X, "desc": ...}`` envelope if present."""
    if isinstance(v, dict):
        keys = set(v.keys())
        if keys and keys.issubset({"value", "desc"}):
            return v.get("value")
    return v


def _coerce_config(cfg: Any) -> dict:
    """Return a plain, unwrapped dict for a run's config.

    Handles:
    - dict-like that iterates as (k, v) tuples  (normal case)
    - dict-like that iterates as bare keys      (some wandb versions)
    - plain string containing JSON              (double-encoded legacy runs)
    - each entry wrapped as ``{"value": X, "desc": ...}`` (raw W&B shape)
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
        except json.JSONDecodeError:
            return {}
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


def _coerce_summary(run: Any) -> dict:
    try:
        return dict(run.summary._json_dict)
    except Exception:
        try:
            return dict(run.summary)
        except Exception:
            return {}


def _strip_private(cfg: dict) -> dict:
    """Drop wandb-internal keys that start with '_'."""
    return {k: v for k, v in cfg.items() if not (isinstance(k, str) and k.startswith("_"))}


def dump_project(
    entity: str,
    project: str,
    out_path: Path,
    csv_path: Path | None,
    limit: int | None,
    include_private: bool,
    per_page: int,
) -> int:
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        return 1

    api = wandb.Api(timeout=120)
    project_path = f"{entity}/{project}"
    logger.info("Pulling runs from %s (per_page=%d)", project_path, per_page)

    t0 = time.time()
    runs_iter = api.runs(project_path, per_page=per_page)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_coerced_str = 0
    n_empty_cfg = 0
    n_err = 0
    keys_seen: set[str] = set()

    with out_path.open("w", encoding="utf-8") as fout:
        for i, run in enumerate(runs_iter):
            if limit and i >= limit:
                break
            try:
                raw_cfg = run.config
                cfg = _coerce_config(raw_cfg)
                if isinstance(raw_cfg, str):
                    n_coerced_str += 1
                if not cfg:
                    n_empty_cfg += 1
                if not include_private:
                    cfg = _strip_private(cfg)

                record = {
                    "id": run.id,
                    "name": getattr(run, "name", ""),
                    "group": getattr(run, "group", None),
                    "tags": list(getattr(run, "tags", []) or []),
                    "state": getattr(run, "state", ""),
                    "created_at": getattr(run, "created_at", ""),
                    "config": cfg,
                    "summary": _coerce_summary(run),
                }
                fout.write(json.dumps(record, default=str))
                fout.write("\n")
                keys_seen.update(k for k in cfg if isinstance(k, str))
                n_ok += 1
                if n_ok % 50 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "  %d runs dumped (%.1fs, %.2f runs/s)",
                        n_ok,
                        elapsed,
                        n_ok / max(elapsed, 1e-6),
                    )
            except Exception as e:
                n_err += 1
                logger.warning("Skip %s: %s", getattr(run, "id", "?"), e)

    elapsed = time.time() - t0
    logger.info(
        "JSONL dump complete: %d runs (%d string-coerced, %d empty-cfg, %d errors) in %.1fs -> %s",
        n_ok,
        n_coerced_str,
        n_empty_cfg,
        n_err,
        elapsed,
        out_path,
    )
    logger.info("Union of config keys across dump: %d", len(keys_seen))

    if csv_path is not None:
        _write_flat_csv(out_path, csv_path, sorted(keys_seen))
    return 0


def _write_flat_csv(jsonl_path: Path, csv_path: Path, all_keys: list[str]) -> None:
    """Write a flat CSV: columns = [id, name, group, state, <all config keys>].

    Dict/list values are JSON-serialised; None -> empty cell. This is lossy
    but handy for Excel browsing. The canonical source remains the JSONL.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("r", encoding="utf-8") as fin, csv_path.open("w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        header = ["id", "name", "group", "state", "created_at"] + all_keys
        w.writerow(header)
        n = 0
        for line in fin:
            rec = json.loads(line)
            cfg = rec.get("config") or {}
            row = [
                rec.get("id", ""),
                rec.get("name", ""),
                rec.get("group") or "",
                rec.get("state", ""),
                rec.get("created_at", ""),
            ]
            for k in all_keys:
                v = cfg.get(k)
                if v is None:
                    row.append("")
                elif isinstance(v, dict | list | tuple):
                    row.append(json.dumps(v, default=str))
                else:
                    row.append(str(v))
            w.writerow(row)
            n += 1
    logger.info("Flat CSV written: %s (%d rows, %d columns)", csv_path, n, len(header))


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump a W&B project to JSONL.")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "nterlind-nikhef"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "thesis-ml"))
    ap.add_argument(
        "--out",
        default="wandb_cleanup/wandb_dump.jsonl",
        help="Output JSONL path (default: wandb_cleanup/wandb_dump.jsonl)",
    )
    ap.add_argument("--csv", action="store_true", help="Also emit a flat CSV at <out>.csv (lossy on nested values)")
    ap.add_argument("--limit", type=int, default=None, help="Dump only the first N runs")
    ap.add_argument("--include-private", action="store_true", help="Keep keys that start with '_' (wandb-internal). Default: strip.")
    ap.add_argument("--per-page", type=int, default=500)
    args = ap.parse_args()

    out_path = Path(args.out)
    csv_path = out_path.with_suffix(".csv") if args.csv else None
    return dump_project(
        entity=args.entity,
        project=args.project,
        out_path=out_path,
        csv_path=csv_path,
        limit=args.limit,
        include_private=args.include_private,
        per_page=args.per_page,
    )


if __name__ == "__main__":
    sys.exit(main())
