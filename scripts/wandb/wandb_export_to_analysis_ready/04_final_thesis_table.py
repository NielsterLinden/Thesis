#!/usr/bin/env python3
"""Stage 04 — Final thesis table → thesis_results/04_cleaned_backfilled_analysis_ready.csv

Reads 03_analysis_ready.csv and:
  1. Drops rows with empty/nan eval_v2/test_auroc.
  2. Drops rows where axes_complete == "False".
  3. Drops the axes_complete helper column itself.
  4. Archives existing 04 to thesis_results/archive/<YYYY-MM-DD>_pre_freeze/ before writing.
  5. Atomically writes 04_cleaned_backfilled_analysis_ready.csv + dated archive copy.

Usage::

    python3 scripts/wandb/wandb_export_to_analysis_ready/04_final_thesis_table.py
    python3 scripts/wandb/wandb_export_to_analysis_ready/04_final_thesis_table.py \\
        --in thesis_results/03_analysis_ready.csv \\
        --out thesis_results/04_cleaned_backfilled_analysis_ready.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

COL_AUROC = "eval_v2/test_auroc"
COL_COMPLETE = "axes_complete"
CANONICAL_OUT_NAME = "04_cleaned_backfilled_analysis_ready.csv"


def _empty_auroc(row: dict[str, str]) -> bool:
    v = (row.get(COL_AUROC) or "").strip()
    return v == "" or v.lower() == "nan"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=REPO_ROOT / "thesis_results" / "03_analysis_ready.csv")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "thesis_results" / CANONICAL_OUT_NAME)
    ap.add_argument("--archive-dir", type=Path,
                    default=REPO_ROOT / "thesis_results" / "archive")
    ap.add_argument("--date", type=str, default=date.today().isoformat())
    args = ap.parse_args()

    inp = args.inp.resolve()
    if not inp.is_file():
        print(f"error: input not found: {inp}", file=sys.stderr)
        return 1

    with inp.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("error: empty CSV", file=sys.stderr)
            return 1
        in_header = list(reader.fieldnames)
        rows_in = list(reader)

    if COL_AUROC not in in_header:
        print(f"error: missing column {COL_AUROC!r} — run Stage 03 first", file=sys.stderr)
        return 1
    if COL_COMPLETE not in in_header:
        print(f"error: missing column {COL_COMPLETE!r} — run Stage 03 first", file=sys.stderr)
        return 1

    n_in = len(rows_in)
    n_no_auroc = sum(1 for r in rows_in if _empty_auroc(r))
    n_incomplete = sum(1 for r in rows_in if not _empty_auroc(r) and r.get(COL_COMPLETE) == "False")

    kept = [r for r in rows_in if not _empty_auroc(r) and r.get(COL_COMPLETE) != "False"]
    print(f"[info] input={n_in}  dropped_no_auroc={n_no_auroc}  dropped_incomplete={n_incomplete}  kept={len(kept)}")

    out_header = [c for c in in_header if c != COL_COMPLETE]

    out = args.out.resolve()
    arch_dir = args.archive_dir.resolve()
    freeze_dir = arch_dir / f"{args.date}_pre_freeze"

    # Archive any existing canonical 04 before overwriting
    if out.is_file():
        freeze_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, freeze_dir / out.name)
        print(f"[info] archived existing {out.name} to {freeze_dir}/")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_header, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
    tmp.replace(out)

    # Dated archive copy
    arch_dir.mkdir(parents=True, exist_ok=True)
    arch_copy = arch_dir / f"{args.date}_{CANONICAL_OUT_NAME}"
    shutil.copy2(out, arch_copy)

    print(f"[info] wrote {out} ({len(kept)} rows, {len(out_header)} cols)")
    print(f"[info] dated archive: {arch_copy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
