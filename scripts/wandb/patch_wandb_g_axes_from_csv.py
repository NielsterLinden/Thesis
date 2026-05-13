#!/usr/bin/env python3
"""Patch W&B run.config for G axes (and optional meta) from the cleaned CSV.

Reads ``meta_run/id`` from rows matching ``cohort=db_completeness_2026_05`` and
applies the same G1/G2/G3 values as ``build_04_cleaned_analysis_ready.py``,
using W&B flat keys ``axes/G1_Task Type``, ``axes/G2_Model Family``,
``axes/G3_Classification Task``.

Default: dry-run (print only). Use ``--execute`` to call ``run.update()``.

Requires ``WANDB_API_KEY`` (see repo ``CLAUDE.md`` / ``hpc/stoomboot/.wandb_env``).

Usage::

    python3 scripts/wandb/patch_wandb_g_axes_from_csv.py \\
        --csv thesis_results/04_cleaned_backfilled_analysis_ready.csv --limit 3
    python3 scripts/wandb/patch_wandb_g_axes_from_csv.py --csv ... --execute
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKFILL = REPO_ROOT / "scripts" / "wandb" / "backfill_labels.py"

ENTITY = "nterlind-nikhef"
PROJECT = "thesis-ml"

TAG_COHORT = "cohort=db_completeness_2026_05"
COL_TAGS = "meta_run/tags"
COL_ID = "meta_run/id"

G1 = "transformer_classifier"
G2 = "transformer"
G3 = "ttH+ttW+ttWW+ttZ | 4t"

WB_KEYS = {
    "axes/G1_Task Type": G1,
    "axes/G2_Model Family": G2,
    "axes/G3_Classification Task": G3,
    # Align with migrate_runs_to_wandb / facts naming for filters
    "meta.class_def_str": G3,
    "meta.process_groups_key": "ttH+ttW+ttWW+ttZ|4t",
}


def _load_helpers():
    spec = importlib.util.spec_from_file_location("backfill_labels_mod", _BACKFILL)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._write_run_config, mod._read_run_config, mod._ensure_attrs_summary_is_object


def _wandb_key_from_env() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    for candidate in [
        REPO_ROOT / "hpc" / "stoomboot" / ".wandb_env",
    ]:
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            k, _, v = line.partition("=")
            if k.strip() == "WANDB_API_KEY" and v.strip():
                os.environ["WANDB_API_KEY"] = v.strip().strip("'\"")
                return


def _cohort_run_ids(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[str] = []
    for r in rows:
        if TAG_COHORT not in (r.get(COL_TAGS) or ""):
            continue
        rid = (r.get(COL_ID) or "").strip()
        if rid:
            out.append(rid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "thesis_results" / "04_cleaned_backfilled_analysis_ready.csv",
    )
    ap.add_argument("--execute", action="store_true", help="Apply patches (default: dry-run).")
    ap.add_argument("--limit", type=int, default=None, help="Max runs to process.")
    ap.add_argument("--sleep", type=float, default=0.15, help="Seconds between API calls.")
    args = ap.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    run_ids = _cohort_run_ids(csv_path)
    if args.limit is not None:
        run_ids = run_ids[: args.limit]

    print(f"[info] cohort run_ids from CSV: {len(run_ids)}")
    if not run_ids:
        return 0

    _wandb_key_from_env()
    if not os.environ.get("WANDB_API_KEY"):
        print("error: WANDB_API_KEY not set", file=sys.stderr)
        return 1

    write_cfg, read_cfg, ensure_summ = _load_helpers()

    import wandb

    api = wandb.Api(timeout=180)
    path = f"{ENTITY}/{PROJECT}"

    would, ok, skip, err = 0, 0, 0, 0
    for i, rid in enumerate(run_ids, 1):
        print(f"[{i}/{len(run_ids)}] {rid} ...", flush=True)
        try:
            run = api.run(f"{path}/{rid}")
        except Exception as e:
            print(f"  error load run: {e}", file=sys.stderr)
            err += 1
            continue

        cfg = read_cfg(run)
        to_apply = {k: v for k, v in WB_KEYS.items() if cfg.get(k) != v}
        if not to_apply:
            print("  skip (already set)")
            skip += 1
            time.sleep(args.sleep)
            continue

        print(f"  would set: {list(to_apply.keys())}")
        if not args.execute:
            would += 1
            time.sleep(args.sleep)
            continue

        try:
            ensure_summ(run)
            write_cfg(run, to_apply)
            print("  updated")
            ok += 1
        except Exception as e:
            print(f"  error update: {e}", file=sys.stderr)
            err += 1
        time.sleep(args.sleep)

    if args.execute:
        print(f"[done] updated={ok} skipped={skip} errors={err}")
    else:
        print(f"[done] dry-run would_patch={would} skipped={skip} errors={err} (pass --execute to apply)")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
