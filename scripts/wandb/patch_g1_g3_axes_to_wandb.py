#!/usr/bin/env python3
"""Patch W&B run.config for G1/G3 axis repair cohorts (thesis_results audit).

Updates flat keys:
  - ``axes/G1_Task Type`` (39 runs: ``transformer`` -> ``transformer_classifier``)
  - ``axes/G3_Classification Task``, ``meta.process_groups_key``, ``meta.class_def_str``
    (49 runs: align with true task from ``meta.row_key`` / manifest)

Default: **dry-run** (print only). Pass ``--execute`` to persist via Public API
(``run.config[k] = v`` then ``run.update()``).

Requires ``wandb`` (``pip install -e '.[wandb]'``) and ``WANDB_API_KEY``.

Usage::

    python3 scripts/wandb/patch_g1_g3_axes_to_wandb.py --limit 3
    python3 scripts/wandb/patch_g1_g3_axes_to_wandb.py --execute
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = REPO_ROOT / "scripts" / "thesis_results" / "g1_g3_patch_manifest.py"

ENTITY = "nterlind-nikhef"
PROJECT = "thesis-ml"


def _load_manifest():
    spec = importlib.util.spec_from_file_location("g1_g3_patch_manifest", _MANIFEST)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _wandb_key_from_env() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    for candidate in [REPO_ROOT / "hpc" / "stoomboot" / ".wandb_env"]:
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


def _read_cfg(run) -> dict[str, object]:
    """Best-effort flat snapshot of run.config for diffing."""
    try:
        raw = dict(run.config)
    except Exception:
        raw = {}
    out: dict[str, object] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = v
        except Exception:
            continue
    return out


def _write_cfg(run, updates: dict[str, object]) -> None:
    for k, v in updates.items():
        run.config[str(k)] = v
    run.update()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--print-manifest",
        action="store_true",
        help="Print patch counts + sample keys (no wandb import; for CI / quota smoke).",
    )
    ap.add_argument("--execute", action="store_true", help="Write to W&B (default: dry-run).")
    ap.add_argument("--limit", type=int, default=None, help="Max runs to process.")
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds between API calls.")
    args = ap.parse_args()

    mod = _load_manifest()
    all_ids = sorted(mod.all_patched_run_ids())

    if args.print_manifest:
        n_g1 = len(mod.G1_FIX_IDS)
        n_g3 = len(mod.G3_COHORT_A1_IDS | mod.G3_COHORT_A2_IDS)
        print(f"[manifest] unique_run_ids={len(all_ids)}  g1_only_cohort={n_g1}  g3_cohort={n_g3}")
        sample = "45avja8u"
        print(f"[manifest] sample {sample}: {mod.wandb_patch_for_run(sample)}")
        return 0
    if args.limit is not None:
        all_ids = all_ids[: args.limit]

    print(f"[info] unique run_ids in manifest: {len(mod.all_patched_run_ids())}")
    print(f"[info] processing: {len(all_ids)}")

    dry_run = not args.execute
    _wandb_key_from_env()
    if not os.environ.get("WANDB_API_KEY"):
        print("error: WANDB_API_KEY not set (export or hpc/stoomboot/.wandb_env)", file=sys.stderr)
        return 1

    try:
        import wandb
    except ImportError:
        print("error: wandb not installed (pip install -e '.[wandb]')", file=sys.stderr)
        return 1

    api = wandb.Api(timeout=180)
    wb_path = f"{ENTITY}/{PROJECT}"

    ok, skip, err = 0, 0, 0
    for i, rid in enumerate(all_ids, 1):
        print(f"[{i}/{len(all_ids)}] {rid} ...", flush=True)
        want = mod.wandb_patch_for_run(rid)
        if not want:
            print("  skip: manifest empty")
            skip += 1
            time.sleep(args.sleep)
            continue

        try:
            run = api.run(f"{wb_path}/{rid}")
        except Exception as e:
            print(f"  error: load run: {e}", file=sys.stderr)
            err += 1
            time.sleep(args.sleep)
            continue

        cur = _read_cfg(run)
        to_apply = {k: v for k, v in want.items() if str(cur.get(k, "")) != str(v)}
        if not to_apply:
            print("  skip: already correct")
            skip += 1
            time.sleep(args.sleep)
            continue

        print(f"  {'[dry-run] would set' if dry_run else 'setting'}: {to_apply}")

        if dry_run:
            ok += 1
            time.sleep(args.sleep)
            continue

        try:
            _write_cfg(run, to_apply)
            print("  updated")
            ok += 1
        except Exception as e:
            print(f"  error: update: {e}", file=sys.stderr)
            err += 1

        time.sleep(args.sleep)

    label = "dry-run would_patch" if dry_run else "updated"
    print(f"\n[done] {label}={ok}  skipped={skip}  errors={err}")
    if dry_run and ok > 0:
        print("[info] Pass --execute to apply.")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
