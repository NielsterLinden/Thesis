#!/usr/bin/env python3
"""Build ``thesis_results/04_cleaned_backfilled_analysis_ready.csv`` from ``03_analysis_ready.csv``.

1. Drop rows with no ``eval_v2/test_auroc`` (Tier-0 incomplete).
2. For rows tagged ``cohort=db_completeness_2026_05`` with empty ``G3``, set G1/G2/G3
   to the values implied by ``db_completeness`` configs (signal vs background 1 vs 2–5).

Writes:
  - ``thesis_results/04_cleaned_backfilled_analysis_ready.csv`` (canonical)
  - ``thesis_results/archive/<date>_04_cleaned_backfilled_analysis_ready.csv`` (copy)

Usage::

    python3 scripts/thesis_results/build_04_cleaned_analysis_ready.py          # counts only
    python3 scripts/thesis_results/build_04_cleaned_analysis_ready.py --write
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

COL_G1 = "config/axes/G1_Task Type"
COL_G2 = "config/axes/G2_Model Family"
COL_G3 = "config/axes/G3_Classification Task"
COL_AUROC = "eval_v2/test_auroc"
TAG_COHORT = "cohort=db_completeness_2026_05"

# Matches ``build_class_def_str`` for canonical binary signal-vs-background (process 1 vs 2–5).
G1_FILL = "transformer_classifier"
G2_FILL = "transformer"
G3_FILL = "ttH+ttW+ttWW+ttZ | 4t"


def _empty_auroc(row: dict[str, str]) -> bool:
    v = (row.get(COL_AUROC) or "").strip()
    return v == "" or v.lower() == "nan"


def _needs_g_backfill(row: dict[str, str]) -> bool:
    tags = row.get("meta_run/tags") or ""
    if TAG_COHORT not in tags:
        return False
    g3 = (row.get(COL_G3) or "").strip()
    return g3 == "" or g3.lower() == "nan"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "thesis_results" / "03_analysis_ready.csv",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "thesis_results" / "04_cleaned_backfilled_analysis_ready.csv",
    )
    ap.add_argument(
        "--archive-dir",
        type=Path,
        default=REPO_ROOT / "thesis_results" / "archive",
    )
    ap.add_argument("--date", type=str, default=date.today().isoformat(), help="Archive filename date prefix")
    ap.add_argument("--write", action="store_true", help="Write output CSV + archive copy (default: print counts only).")
    args = ap.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        print(f"error: input not found: {inp}", file=sys.stderr)
        return 1

    with inp.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        if not header:
            print("error: empty CSV", file=sys.stderr)
            return 1
        rows_in = list(reader)

    for col in (COL_G1, COL_G2, COL_G3, COL_AUROC, "meta_run/tags"):
        if col not in header:
            print(f"error: missing column {col!r}", file=sys.stderr)
            return 1

    n_in = len(rows_in)
    dropped = [r for r in rows_in if _empty_auroc(r)]
    kept = [r for r in rows_in if not _empty_auroc(r)]
    backfill = [r for r in kept if _needs_g_backfill(r)]

    print(f"[counts] input_rows={n_in}")
    print(f"[counts] dropped_no_auroc={len(dropped)}")
    print(f"[counts] kept={len(kept)}")
    print(f"[counts] backfill_g_axes_cohort={len(backfill)}")

    if not args.write:
        print("[mode] dry-run (pass --write to emit files)")
        return 0

    out_rows: list[dict[str, str]] = []
    n_bf = 0
    for r in kept:
        row = dict(r)
        if _needs_g_backfill(row):
            row[COL_G1] = G1_FILL
            row[COL_G2] = G2_FILL
            row[COL_G3] = G3_FILL
            n_bf += 1
        out_rows.append(row)

    if n_bf != len(backfill):
        print("error: backfill count mismatch", file=sys.stderr)
        return 1

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        w.writeheader()
        w.writerows(out_rows)

    arch = args.archive_dir.resolve()
    arch.mkdir(parents=True, exist_ok=True)
    arch_copy = arch / f"{args.date}_04_cleaned_backfilled_analysis_ready.csv"
    shutil.copy2(out, arch_copy)

    print(f"[io] wrote {out} ({len(out_rows)} rows)")
    print(f"[io] archive copy {arch_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
