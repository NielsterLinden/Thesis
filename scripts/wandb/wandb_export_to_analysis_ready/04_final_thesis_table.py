#!/usr/bin/env python3
"""Stage 04 — Build ``thesis_results/04_cleaned_backfilled_analysis_ready.csv`` from 03.

Two explicit exclusion passes (no global AUROC cut — we do **not** drop rows
because ``eval_v2/test_auroc < 0.65``; only the cohort rules below so removals
are attributable to known-bad experiment batches).

**Section 1 — April 2026 ch5.** Drop when both hold:

1. ``meta_run/created_at`` starts with ``2026-04``.
2. ``meta_run/group`` contains ``_ch5_``.

**Section 2 — Named failed W&B cohorts.** Drop when ``meta_run/group`` contains
any of the substrings in ``FAILED_COHORT_RULES`` (first match wins for
per-cohort accounting if a row ever matched more than one; rules are disjoint in
practice).

All other rows and columns are copied unchanged from the input (including
``axes_complete`` if present).

Before overwriting an existing ``--out`` file, a copy is written under
``thesis_results/archive/<date>_pre_04_rewrite_<basename>``.

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
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

COL_CREATED = "meta_run/created_at"
COL_GROUP = "meta_run/group"
CANONICAL_OUT_NAME = "04_cleaned_backfilled_analysis_ready.csv"

# (substring on meta_run/group, log slug for dropped_failed_cohort_<slug>)
FAILED_COHORT_RULES: tuple[tuple[str, str], ...] = (
    ("builtjes_baseline", "builtjes_baseline"),
    ("OrthogonalSweep_B_g16", "orthogonal_sweep_b_g16"),
    ("order_pe_attention_4t_vs_bg", "order_pe_attention_4t_vs_bg"),
)


def _is_april_2026_ch5(row: dict[str, str]) -> bool:
    """Section 1: True if this row should be excluded."""
    created = (row.get(COL_CREATED) or "").strip()
    group = (row.get(COL_GROUP) or "").strip()
    return created.startswith("2026-04") and "_ch5_" in group


def _failed_cohort_slug(row: dict[str, str]) -> str | None:
    """Section 2: log slug for the first matching failed cohort, or None."""
    group = (row.get(COL_GROUP) or "").strip()
    for needle, slug in FAILED_COHORT_RULES:
        if needle in group:
            return slug
    return None


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
        fieldnames = list(reader.fieldnames)
        rows_in = list(reader)

    for col in (COL_CREATED, COL_GROUP):
        if col not in fieldnames:
            print(f"error: missing column {col!r}", file=sys.stderr)
            return 1

    n_in = len(rows_in)
    n_drop_ch5 = sum(1 for r in rows_in if _is_april_2026_ch5(r))
    after_ch5 = [r for r in rows_in if not _is_april_2026_ch5(r)]
    print(f"[info] input={n_in}  dropped_april_ch5={n_drop_ch5}  after_section1={len(after_ch5)}")

    fail_counts: Counter[str] = Counter()
    for r in after_ch5:
        slug = _failed_cohort_slug(r)
        if slug is not None:
            fail_counts[slug] += 1
    n_drop_fail = sum(fail_counts.values())
    kept = [r for r in after_ch5 if _failed_cohort_slug(r) is None]
    for _needle, slug in FAILED_COHORT_RULES:
        n = fail_counts.get(slug, 0)
        print(f"[info]   dropped_failed_cohort_{slug}={n}")
    print(f"[info] dropped_failed_cohorts_total={n_drop_fail}  final_kept={len(kept)}")

    out = args.out.resolve()
    arch_dir = args.archive_dir.resolve()

    if out.is_file():
        arch_dir.mkdir(parents=True, exist_ok=True)
        backup = arch_dir / f"{args.date}_pre_04_rewrite_{out.name}"
        shutil.copy2(out, backup)
        print(f"[info] archived existing {out.name} -> {backup}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
    tmp.replace(out)

    print(f"[info] wrote {out} ({len(kept)} data rows, {len(fieldnames)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
