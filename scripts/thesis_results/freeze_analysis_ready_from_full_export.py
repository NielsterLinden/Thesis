#!/usr/bin/env python3
"""Build a frozen ``03_analysis_ready.csv`` from the 160-col W&B full export.

Row rules (aligned with ``scripts/wandb/eval_pipeline/reduce_to_analysis_csv.py``):
  1. Keep only ``config/axes/G2_Model Family == transformer`` (drops early MLP/BDT runs).
  2. Remove April-2026 chapter-5 training rows: ``ch5`` in ``meta_run/group`` (case-insensitive)
     and ``meta_run/created_at`` starts with ``2026-04``.
  3. Do **not** drop rows solely for missing ``eval_v2/test_auroc`` (flag in QC sidecar).

Writes:
  - ``thesis_results/archive/<date>_03_analysis_ready.csv`` — filtered rows, same columns as input
  - ``thesis_results/archive/<date>_03_analysis_ready_removed_rows.csv`` — audit trail
  - ``thesis_results/archive/<date>_03_analysis_ready_qc_flags.csv`` — per kept run flags
  - ``thesis_results/archive/<date>_03_analysis_ready_README.md`` — provenance

Use ``--update-working`` to copy the frozen CSV to ``thesis_results/03_analysis_ready.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

G2_COL = "config/axes/G2_Model Family"
G1_PREFIX = "config/axes/G1_"
G2_PREFIX = "config/axes/G2_"
G3_PREFIX = "config/axes/G3_"


def nonempty(v: str) -> bool:
    return v is not None and str(v).strip() != ""


def is_ch5_april(row: dict[str, str]) -> bool:
    g = (row.get("meta_run/group") or "").lower()
    c = row.get("meta_run/created_at") or ""
    return "ch5" in g and c.startswith("2026-04")


def is_transformer(row: dict[str, str]) -> bool:
    v = row.get(G2_COL)
    return nonempty(v) and str(v).strip() == "transformer"


def g_columns(header: list[str], prefix: str) -> list[str]:
    return [h for h in header if h.startswith(prefix)]


def row_qc_flags(header: list[str], row: dict[str, str]) -> dict[str, str]:
    flags: dict[str, str] = {"meta_run/id": row.get("meta_run/id", "")}

    g1_cols = g_columns(header, G1_PREFIX)
    g2_cols = g_columns(header, G2_PREFIX)
    g3_cols = g_columns(header, G3_PREFIX)

    def any_nonempty(cols: list[str]) -> bool:
        return any(nonempty(row.get(c, "")) for c in cols)

    flags["qc/missing_g1"] = "true" if g1_cols and not any_nonempty(g1_cols) else "false"
    flags["qc/missing_g2"] = "true" if g2_cols and not any_nonempty(g2_cols) else "false"
    flags["qc/missing_g3"] = "true" if g3_cols and not any_nonempty(g3_cols) else "false"

    axis_cols = [h for h in header if h.startswith("config/axes/")]
    flags["qc/no_formal_axes"] = (
        "true" if axis_cols and not any(nonempty(row.get(c, "")) for c in axis_cols) else "false"
    )

    flags["qc/missing_eval_test_auroc"] = (
        "true" if not nonempty(row.get("eval_v2/test_auroc", "")) else "false"
    )
    flags["qc/missing_eval_spec_version"] = (
        "true" if not nonempty(row.get("eval_v2/spec_version", "")) else "false"
    )
    flags["qc/missing_meta_needs_review"] = (
        "true" if not nonempty(row.get("config/meta.needs_review", "")) else "false"
    )

    has_auroc = nonempty(row.get("eval_v2/test_auroc", ""))
    roc_empty = not nonempty(row.get("eval_v2/roc_fpr", ""))
    flags["qc/artifact_roc_missing_with_auroc"] = (
        "true" if has_auroc and roc_empty else "false"
    )

    return flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "thesis_results/archive/02_analysis_reduces_columns_fullrows.csv",
        help="Full 160-column export (default: archive snapshot)",
    )
    ap.add_argument(
        "--date",
        type=str,
        default=date.today().isoformat(),
        help="YYYY-MM-DD prefix for output filenames (default: today)",
    )
    ap.add_argument(
        "--archive-dir",
        type=Path,
        default=REPO_ROOT / "thesis_results/archive",
        help="Directory for dated frozen outputs",
    )
    ap.add_argument(
        "--update-working",
        action="store_true",
        help="Copy frozen CSV to thesis_results/03_analysis_ready.csv",
    )
    args = ap.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        print(f"error: input not found: {inp}", file=sys.stderr)
        return 1

    arch = args.archive_dir.resolve()
    arch.mkdir(parents=True, exist_ok=True)
    prefix = args.date + "_03_analysis_ready"
    out_frozen = arch / f"{prefix}.csv"
    out_removed = arch / f"{prefix}_removed_rows.csv"
    out_qc = arch / f"{prefix}_qc_flags.csv"
    out_readme = arch / f"{prefix}_README.md"

    with inp.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    if G2_COL not in header:
        print(f"error: missing column {G2_COL!r}", file=sys.stderr)
        return 1

    kept: list[dict[str, str]] = []
    removed: list[dict[str, str]] = []

    for row in rows:
        reasons: list[str] = []
        if is_ch5_april(row):
            reasons.append("ch5_april_2026-04")
        if not is_transformer(row):
            reasons.append("non_transformer")
        if reasons:
            removed.append(
                {
                    "meta_run/id": row.get("meta_run/id", ""),
                    "meta_run/group": row.get("meta_run/group", ""),
                    "meta_run/created_at": row.get("meta_run/created_at", ""),
                    G2_COL: row.get(G2_COL, ""),
                    "removal_reasons": ";".join(reasons),
                }
            )
            continue
        kept.append(row)

    with out_frozen.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)

    rem_fields = ["meta_run/id", "meta_run/group", "meta_run/created_at", G2_COL, "removal_reasons"]
    with out_removed.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rem_fields)
        w.writeheader()
        w.writerows(removed)

    qc_fields = [
        "meta_run/id",
        "qc/missing_g1",
        "qc/missing_g2",
        "qc/missing_g3",
        "qc/no_formal_axes",
        "qc/missing_eval_test_auroc",
        "qc/missing_eval_spec_version",
        "qc/missing_meta_needs_review",
        "qc/artifact_roc_missing_with_auroc",
    ]
    with out_qc.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=qc_fields)
        w.writeheader()
        for row in kept:
            flags = row_qc_flags(header, row)
            w.writerow({k: flags.get(k, "") for k in qc_fields})

    readme = {
        "source_input": str(inp),
        "output_frozen": str(out_frozen),
        "date": args.date,
        "row_counts": {
            "input_rows": len(rows),
            "kept_rows": len(kept),
            "removed_rows": len(removed),
        },
        "rules": [
            f"Keep only rows where {G2_COL!r} == 'transformer'",
            "Remove rows where meta_run/group contains 'ch5' (case-insensitive) AND meta_run/created_at starts with '2026-04'",
            "Do not drop rows solely for missing eval_v2/test_auroc (see QC file)",
        ],
    }
    out_readme.write_text(
        "# Frozen analysis CSV\n\n"
        f"Generated by `python scripts/thesis_results/freeze_analysis_ready_from_full_export.py`.\n\n"
        "```json\n"
        + json.dumps(readme, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )

    if args.update_working:
        working = REPO_ROOT / "thesis_results" / "03_analysis_ready.csv"
        working.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_frozen, working)

    print(f"[io] input:  {inp} ({len(rows)} rows)")
    print(f"[io] frozen: {out_frozen} ({len(kept)} rows)")
    print(f"[io] removed audit: {out_removed} ({len(removed)} rows)")
    print(f"[io] qc flags: {out_qc}")
    print(f"[io] readme: {out_readme}")
    if args.update_working:
        print(f"[io] working copy: {REPO_ROOT / 'thesis_results' / '03_analysis_ready.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
