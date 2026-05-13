#!/usr/bin/env python3
"""Stage 03 — Transformer filter + axes_complete flag → thesis_results/03_analysis_ready.csv

Reads 02_eval_combined.csv and:
  1. Keeps only rows where config/axes/G2_Model Family == "transformer".
  2. Adds boolean column axes_complete (True iff all primary axis families are non-empty).
  3. Writes a sidecar missing report at thesis_results/pipeline_v2/03_missing_report.csv.

Usage::

    python3 scripts/wandb/wandb_export_to_analysis_ready/03_transformer_filter.py
    python3 scripts/wandb/wandb_export_to_analysis_ready/03_transformer_filter.py \\
        --in thesis_results/02_eval_combined.csv --out thesis_results/03_analysis_ready.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

COL_G2 = "config/axes/G2_Model Family"
COL_AUROC = "eval_v2/test_auroc"

# Primary axis families required for a row to be considered complete.
# B1 sub-axes (L*, T*, S*, G*) are conditional — excluded from completeness.
PRIMARY_AXIS_FAMILIES = [
    "G1_", "G2_", "G3_",
    "H1_", "H2_", "H3_", "H4_",
    "F1_", "F1-moe_", "F1-eff_", "C1_",
    "D1_", "D2_", "D3_",
    "E1_",
    "T1_",
    "A1_", "A2_", "A3_", "A4_", "A5_",
    "C2_",
    "P1_", "P2_",
    "R1_", "R2_", "R3_", "R4_", "R5_",
    "B1_",
]


def _axes_col_for_family(header: list[str], family: str) -> str | None:
    """Return the first config/axes/* column whose suffix starts with `family`."""
    for col in header:
        if col.startswith("config/axes/") and col[len("config/axes/"):].startswith(family):
            return col
    return None


def _is_empty(v: str) -> bool:
    return not v or v.strip().lower() in ("", "nan", "none", "null")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=REPO_ROOT / "thesis_results" / "02_eval_combined.csv")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "thesis_results" / "03_analysis_ready.csv")
    ap.add_argument("--missing-report", type=Path,
                    default=REPO_ROOT / "thesis_results" / "pipeline_v2" / "03_missing_report.csv")
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

    if COL_G2 not in in_header:
        print(f"error: missing column {COL_G2!r} — run Stage 02 first", file=sys.stderr)
        return 1

    # Resolve family → column mapping once
    family_cols: dict[str, str | None] = {
        fam: _axes_col_for_family(in_header, fam) for fam in PRIMARY_AXIS_FAMILIES
    }
    missing_families = [f for f, c in family_cols.items() if c is None]
    if missing_families:
        print(f"[warn] no column found for families: {missing_families} — will mark those rows incomplete")

    # Filter + flag
    n_in = len(rows_in)
    kept: list[dict] = []
    n_dropped_g2 = 0
    n_dropped_auroc = 0
    examples_g2: list[str] = []
    examples_auroc: list[str] = []
    missing_report: list[dict] = []

    for row in rows_in:
        run_id = row.get("meta_run/id", "")
        g2 = (row.get(COL_G2) or "").strip().lower()
        if g2 != "transformer":
            n_dropped_g2 += 1
            if len(examples_g2) < 5:
                examples_g2.append(f"  {run_id!r:40s}  G2={row.get(COL_G2, '')!r}")
            continue

        auroc = row.get(COL_AUROC, "")
        if _is_empty(auroc):
            n_dropped_auroc += 1
            if len(examples_auroc) < 5:
                examples_auroc.append(f"  {run_id!r:40s}  {COL_AUROC}=<empty>")
            continue

        missing_for_row: list[str] = []

        for fam, col in family_cols.items():
            if col is None or _is_empty(row.get(col, "")):
                missing_for_row.append(fam.rstrip("_"))
                if col is None:
                    missing_report.append({"run_id": run_id, "missing_axis": f"NO_COL:{fam}"})
                else:
                    missing_report.append({"run_id": run_id, "missing_axis": col[len("config/axes/"):]})

        row["axes_complete"] = "False" if missing_for_row else "True"
        kept.append(row)

    n_complete = sum(1 for r in kept if r["axes_complete"] == "True")
    n_incomplete = len(kept) - n_complete
    print(f"[info] input={n_in}  dropped_non_transformer={n_dropped_g2}  dropped_no_auroc={n_dropped_auroc}  kept={len(kept)}")
    print(f"[info] axes_complete=True:{n_complete}  axes_complete=False:{n_incomplete}")
    if examples_g2:
        print(f"[info] examples dropped (non-transformer, first {len(examples_g2)}):")
        for ex in examples_g2:
            print(ex)
    if examples_auroc:
        print(f"[info] examples dropped (no {COL_AUROC}, first {len(examples_auroc)}):")
        for ex in examples_auroc:
            print(ex)

    out_header = in_header + ["axes_complete"]
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_header, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
    tmp.replace(out)
    print(f"[info] wrote {out} ({len(kept)} rows, {len(out_header)} cols)")

    # Sidecar missing report
    rep = args.missing_report.resolve()
    rep.parent.mkdir(parents=True, exist_ok=True)
    with rep.open("w", newline="", encoding="utf-8") as f:
        w2 = csv.DictWriter(f, fieldnames=["run_id", "missing_axis"])
        w2.writeheader()
        w2.writerows(missing_report)
    print(f"[info] missing report: {rep} ({len(missing_report)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
