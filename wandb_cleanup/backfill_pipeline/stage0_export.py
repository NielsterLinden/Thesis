#!/usr/bin/env python
"""Stage 0: export W&B runs to CSV using a fixed column schema (see PLAN.md).

Columns come from ``wandb_cleanup/wandb_mcp_columns_thesis-ml.csv`` (same
union as ``_export_wandb_columns.py``): top-level summary/config keys per run.
Each data column is ``{kind}/{column_name}`` (e.g. ``summary/train/loss``) so
``_wandb`` can appear once per kind.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

META_RUN_COLS = [
    "meta_run/id",
    "meta_run/name",
    "meta_run/created_at",
    "meta_run/state",
    "meta_run/tags",
    "meta_run/group",
    "meta_run/project",
]

DEFAULT_SCHEMA = Path(__file__).resolve().parent.parent / "wandb_mcp_columns_thesis-ml.csv"


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(16):
        if (p / "pyproject.toml").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Could not find repo root (pyproject.toml)")


def _load_wandb_env_if_needed() -> None:
    """Set WANDB_API_KEY from hpc/stoomboot/.wandb_env if unset (aligned with thesis_ml.utils.wandb_utils)."""
    if os.environ.get("WANDB_API_KEY"):
        return
    root = _repo_root()
    env_file = root / "hpc" / "stoomboot" / ".wandb_env"
    if not env_file.is_file():
        return
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if "WANDB_API_KEY" in line and "=" in line:
                if line.startswith("export "):
                    line = line[7:]
                key, _, val = line.partition("=")
                if key.strip() == "WANDB_API_KEY" and val:
                    val = val.strip().strip("'\"").strip()
                    if val:
                        os.environ["WANDB_API_KEY"] = val
                        return
    except OSError:
        return


def _import_dump_runs() -> Any:
    root = _repo_root()
    mod_path = root / "scripts" / "wandb" / "dump_runs.py"
    spec = importlib.util.spec_from_file_location("dump_runs_stage0", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_schema(path: Path) -> list[tuple[str, str]]:
    """Load (kind, column_name) pairs in file order. kind is ``summary`` or ``config``."""
    if not path.is_file():
        raise FileNotFoundError(f"Schema CSV not found: {path}")
    out: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or "column_name" not in r.fieldnames or "kind" not in r.fieldnames:
            raise ValueError(f"Schema CSV must have column_name,kind header: {path}")
        for row in r:
            name = (row.get("column_name") or "").strip()
            kind = (row.get("kind") or "").strip().lower()
            if not name:
                continue
            if kind not in ("summary", "config"):
                raise ValueError(f"Invalid kind {kind!r} for column {name!r} in {path}")
            out.append((kind, name))
    if not out:
        raise ValueError(f"No schema rows in {path}")
    return out


def _schema_headers(schema: list[tuple[str, str]]) -> list[str]:
    return [f"{kind}/{name}" for kind, name in schema]


def _cell_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, dict | list | tuple):
        return json.dumps(v, sort_keys=True, separators=(",", ":"), default=str)
    if isinstance(v, bool):
        return json.dumps(v)
    if isinstance(v, int | float):
        return json.dumps(v)
    if isinstance(v, str):
        return v
    return str(v)


def _created_at_sort_key(run: Any) -> tuple[Any, str]:
    raw = getattr(run, "created_at", None)
    rid = getattr(run, "id", "")
    if raw is None:
        return (datetime.min, rid)
    if isinstance(raw, datetime):
        return (raw, rid)
    s = str(raw)
    try:
        s2 = s[:-1] + "+00:00" if s.endswith("Z") else s
        return (datetime.fromisoformat(s2.replace("Z", "+00:00")), rid)
    except (ValueError, TypeError):
        return (s, rid)


def _meta_row(run: Any, project: str) -> dict[str, str]:
    tags = list(getattr(run, "tags", []) or [])
    created = getattr(run, "created_at", "")
    created_s = created.isoformat() if isinstance(created, datetime) else (str(created) if created is not None else "")
    return {
        "meta_run/id": getattr(run, "id", ""),
        "meta_run/name": str(getattr(run, "name", "") or ""),
        "meta_run/created_at": created_s,
        "meta_run/state": str(getattr(run, "state", "") or ""),
        "meta_run/tags": _cell_str(tags),
        "meta_run/group": str(getattr(run, "group", "") or ""),
        "meta_run/project": project,
    }


def export_stage0(
    entity: str,
    project: str,
    out_csv: Path,
    schema_csv: Path,
    per_page: int,
    limit: int | None,
) -> int:
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        return 1

    try:
        schema = _load_schema(schema_csv)
    except (OSError, ValueError) as e:
        logger.error("%s", e)
        return 1

    _load_wandb_env_if_needed()
    dr = _import_dump_runs()
    coerce_config = dr._coerce_config
    coerce_summary = dr._coerce_summary

    data_headers = _schema_headers(schema)
    header = list(META_RUN_COLS) + data_headers

    api = wandb.Api(timeout=180)
    path = f"{entity}/{project}"
    logger.info(
        "Fetching runs from %s (per_page=%d, schema=%d cols from %s)",
        path,
        per_page,
        len(data_headers),
        schema_csv.name,
    )

    runs_sorted: list[Any] = []
    try:
        it = api.runs(path, per_page=per_page)
        for i, run in enumerate(it, start=1):
            if limit is not None and i > limit:
                break
            runs_sorted.append(run)
            if i == 1 or i % 50 == 0:
                print(f"\r  fetch {i} runs...", end="", file=sys.stderr, flush=True)
    except Exception as e:
        print(file=sys.stderr)
        logger.error("Failed to list runs (auth or network): %s", e)
        return 1

    print(file=sys.stderr)
    logger.info("Fetched %d run(s), sorting by created_at...", len(runs_sorted))

    runs_sorted.sort(key=_created_at_sort_key)

    rows: list[dict[str, str]] = []
    n_skip = 0
    total = len(runs_sorted)

    for idx, run in enumerate(runs_sorted, start=1):
        rid = getattr(run, "id", "?")
        rname = str(getattr(run, "name", "") or "")[:40]
        print(f"\r  export {idx}/{total}  {rid}  {rname}", end="", file=sys.stderr, flush=True)
        try:
            # Do not strip ``_*`` keys: schema may include ``_wandb`` etc.; only schema columns are written.
            cfg = coerce_config(run.config)
            sm = coerce_summary(run)

            row: dict[str, str] = _meta_row(run, project)
            for kind, name in schema:
                h = f"{kind}/{name}"
                if kind == "config":
                    row[h] = _cell_str(cfg.get(name))
                else:
                    row[h] = _cell_str(sm.get(name))
            rows.append(row)
        except SystemExit:
            raise
        except Exception as e:
            n_skip += 1
            logger.warning("Skip run %s: %s", rid, e)

    print(file=sys.stderr)

    if not rows:
        logger.error("No runs exported (%d skipped).", n_skip)
        return 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([row.get(c, "") for c in header])

    logger.info(
        "Wrote %s (%d rows, %d columns, %d skipped) -> %s",
        out_csv.name,
        len(rows),
        len(header),
        n_skip,
        out_csv,
    )
    return 0


def main() -> int:
    root = _repo_root()
    default_snap = root / "wandb_cleanup" / "backfill_pipeline" / "snapshots" / f"{time.strftime('%Y-%m-%d')}_raw" / "00_raw_export.csv"

    ap = argparse.ArgumentParser(description="Stage 0: W&B project → wide CSV (fixed MCP schema)")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "nterlind-nikhef"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "thesis-ml"))
    ap.add_argument(
        "--schema-csv",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="Column schema: column_name,kind rows (default: wandb_cleanup/wandb_mcp_columns_thesis-ml.csv)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=default_snap,
        help="Output CSV path (default: snapshots/<today>_raw/00_raw_export.csv under backfill_pipeline)",
    )
    ap.add_argument("--per-page", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="Max runs (smoke test)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    code = export_stage0(
        entity=args.entity,
        project=args.project,
        out_csv=args.out_csv.resolve(),
        schema_csv=args.schema_csv.resolve(),
        per_page=args.per_page,
        limit=args.limit,
    )
    logger.info("Done in %.1fs", time.perf_counter() - t0)
    return code


if __name__ == "__main__":
    sys.exit(main())
