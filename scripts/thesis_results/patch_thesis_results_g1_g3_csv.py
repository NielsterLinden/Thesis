#!/usr/bin/env python3
"""Patch G1/G3 columns in all four thesis_results CSVs (manifest-driven).

- Backs up each file to ``*.bak.<timestamp>`` before overwrite.
- Only updates columns that exist in that file's header.
- ``01_raw_export.csv`` carries ``config/meta.*``; ``02``–``04`` only G1/G3 when present.

Usage::

    python3 scripts/thesis_results/patch_thesis_results_g1_g3_csv.py --dry-run
    python3 scripts/thesis_results/patch_thesis_results_g1_g3_csv.py --execute
    python3 scripts/thesis_results/patch_thesis_results_g1_g3_csv.py --verify
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = REPO_ROOT / "scripts" / "thesis_results" / "g1_g3_patch_manifest.py"

CSV_FILES = [
    REPO_ROOT / "thesis_results" / "01_raw_export.csv",
    REPO_ROOT / "thesis_results" / "02_eval_combined.csv",
    REPO_ROOT / "thesis_results" / "03_analysis_ready.csv",
    REPO_ROOT / "thesis_results" / "04_cleaned_backfilled_analysis_ready.csv",
]


def _load_manifest():
    spec = importlib.util.spec_from_file_location("g1_g3_patch_manifest", _MANIFEST)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _norm_g3(s: str) -> str:
    t = str(s).strip()
    while " | " in t:
        t = t.replace(" | ", "|")
    return t.replace(" ", "")


def _row_key_suffix(rk: str) -> str:
    if "::" not in rk:
        return ""
    return rk.split("::", 1)[1].strip()


def verify_01(mod, path: Path) -> int:
    """Return number of assertion failures."""
    bad = 0
    col_g3 = mod.COL_G3
    col_pg = mod.COL_PG
    col_g1 = mod.COL_G1
    col_row = mod.COL_ROW
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = (row.get("meta_run/id") or "").strip()
            if rid in mod.G3_COHORT_A1_IDS | mod.G3_COHORT_A2_IDS:
                if _norm_g3(row.get(col_g3, "") or "") != _row_key_suffix(row.get(col_row, "") or ""):
                    print(f"[verify fail] G3 vs row_key {rid}", file=sys.stderr)
                    bad += 1
                if (row.get(col_pg, "") or "").strip() != _row_key_suffix(row.get(col_row, "") or ""):
                    print(f"[verify fail] process_groups_key vs row_key {rid}", file=sys.stderr)
                    bad += 1
            if rid in mod.G1_FIX_IDS:
                if (row.get(col_g1, "") or "").strip() != mod.G1_TARGET:
                    print(f"[verify fail] G1 not fixed {rid}", file=sys.stderr)
                    bad += 1
    return bad


def verify_all(mod) -> int:
    fails = 0
    p01 = REPO_ROOT / "thesis_results" / "01_raw_export.csv"
    if p01.is_file():
        fails += verify_01(mod, p01)
    col_g1, col_g3 = mod.COL_G1, mod.COL_G3
    for name, path in [("02", CSV_FILES[1]), ("03", CSV_FILES[2]), ("04", CSV_FILES[3])]:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = (row.get("meta_run/id") or "").strip()
                if rid in mod.G3_COHORT_A1_IDS | mod.G3_COHORT_A2_IDS:
                    exp = (
                        mod.G3_DISPLAY_5WAY
                        if rid in mod.G3_COHORT_A1_IDS
                        else mod.G3_DISPLAY_TTH
                    )
                    if (row.get(col_g3, "") or "").strip() != exp:
                        print(f"[verify fail] {name} G3 {rid}", file=sys.stderr)
                        fails += 1
                if rid in mod.G1_FIX_IDS:
                    if (row.get(col_g1, "") or "").strip() != mod.G1_TARGET:
                        print(f"[verify fail] {name} G1 {rid}", file=sys.stderr)
                        fails += 1
    return fails


def patch_file(mod, path: Path, dry_run: bool, ts: str) -> tuple[int, int]:
    """Returns (n_rows_changed, n_cells_written)."""
    if not path.is_file():
        print(f"[warn] missing file: {path}")
        return 0, 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    changed_rows = 0
    cells = 0
    for row in rows:
        rid = (row.get("meta_run/id") or "").strip()
        patch = mod.csv_patch_for_run(rid)
        if not patch:
            continue
        touched = False
        for col, val in patch.items():
            if col not in fieldnames:
                continue
            old = row.get(col, "")
            if str(old) == str(val):
                continue
            row[col] = val
            cells += 1
            touched = True
        if touched:
            changed_rows += 1

    if dry_run or changed_rows == 0:
        print(f"[{path.name}] rows_touched={changed_rows} cells={cells} dry_run={dry_run}")
        return changed_rows, cells

    bak = path.with_name(path.name + f".bak.{ts}")
    shutil.copy2(path, bak)
    print(f"[{path.name}] backup -> {bak.name}")

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"[{path.name}] wrote rows_touched={changed_rows} cells={cells}")
    return changed_rows, cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Print counts only.")
    g.add_argument("--execute", action="store_true", help="Write backups + patched CSVs.")
    g.add_argument("--verify", action="store_true", help="Assert postconditions on current CSVs.")
    args = ap.parse_args()

    mod = _load_manifest()

    if args.verify:
        n = verify_all(mod)
        if n:
            print(f"[verify] FAILED with {n} checks", file=sys.stderr)
            return 1
        print("[verify] OK")
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dry = args.dry_run
    total_r, total_c = 0, 0
    for p in CSV_FILES:
        r, c = patch_file(mod, p, dry_run=dry, ts=ts)
        total_r += r
        total_c += c

    print(f"\n[summary] files={len(CSV_FILES)} rows_touched_total={total_r} cells_total={total_c}")
    if dry:
        print("[info] Pass --execute to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
