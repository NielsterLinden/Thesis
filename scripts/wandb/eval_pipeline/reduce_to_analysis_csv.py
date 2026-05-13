"""
Reduce the post-Phase-2 W&B export to an analysis-ready CSV.

Input:  02_eval_combined.csv  (1002 rows x 579 cols)
Output: 03_analysis_ready.csv (~931 rows x ~163 cols)

The output feeds:
  1. XGBoost surrogate + SHAP + fANOVA on axes -> eval_v2 metrics (Ch 7)
  2. W&B push of eval_v2/* back to runs (Phase 2 of backfill)
  3. Pareto plots and final model selection (Ch 8, Ch 9)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Formal V2 axis ID prefixes (axis name appears after "config/axes/")
FORMAL_AXIS_PREFIXES = (
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "B1",
    "C1",
    "C2",
    "D1",
    "D2",
    "D3",
    "E1",
    "F1",
    "G1",
    "G2",
    "G3",
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H10",
    "K1",
    "K2",
    "K3",
    "K4",
    "K5",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "L7",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "P1",
    "P2",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "R7",
    "R8",
    "R9",
    "R10",
    "R11",
    "R12",
    "R13",
    "R14",
    "R15",
    "R16",
    "R17",
    "S1",
    "S2",
    "T1",
)

META_RUN_KEEP = [
    "meta_run/id",
    "meta_run/name",
    "meta_run/created_at",
    "meta_run/state",
    "meta_run/tags",
    "meta_run/group",
    "meta_run/project",
]

EXTRA_META_KEEP = [
    "config/meta.needs_review",
    "config/meta.row_key",
    "config/meta.process_groups_key",
    "config/meta.class_def_str",
]


def is_formal_axis(col: str) -> bool:
    """A formal V2 axis column has form config/axes/<ID>_<name> where <ID>
    starts with one of the FORMAL_AXIS_PREFIXES followed by '_' or '-'."""
    if not col.startswith("config/axes/"):
        return False
    tail = col[len("config/axes/") :]
    # Must match the prefix and then a separator, not be a substring.
    return any(tail.startswith(p + "_") or tail.startswith(p + "-") for p in FORMAL_AXIS_PREFIXES)


def select_columns(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    keep: list[str] = []

    # 1. Formal V2 axes
    keep.extend(c for c in cols if is_formal_axis(c))

    # 2. All eval_v2/* metrics
    keep.extend(c for c in cols if c.startswith("eval_v2/"))

    # 3. meta_run/* (run identification and grouping)
    keep.extend(c for c in META_RUN_KEEP if c in cols)

    # 4. Selected config/meta.* (only needs_review)
    keep.extend(c for c in EXTRA_META_KEEP if c in cols)

    # Preserve original CSV column order for readability.
    keep_set = set(keep)
    return [c for c in cols if c in keep_set]


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    # Transformers only.
    df = df[df["config/axes/G2_Model Family"] == "transformer"].copy()
    n1 = len(df)
    print(f"[rows] non-transformers dropped: {n0 - n1} ({n0} -> {n1})")

    # Must have eval_v2/test_auroc (covers the rare 1v1 tasks and the
    # 7 transformer rows where the inference pipeline produced no metrics).
    df = df[df["eval_v2/test_auroc"].notna()].copy()
    n2 = len(df)
    print(f"[rows] missing eval_v2/test_auroc dropped: {n1 - n2} ({n1} -> {n2})")

    # We deliberately KEEP crashed runs that have valid eval_v2 metrics and
    # checkpoint_status=success. Their crash is post-evaluation; eval_v2 is a
    # re-evaluation on a fixed test split. meta_run/state is retained so any
    # downstream notebook can still subset on state if needed.
    return df


def report_zero_variance_axes(df: pd.DataFrame) -> None:
    """Single-valued axis columns are kept in the CSV but should be excluded
    at surrogate fit time. Print them so Cursor (and the human reading this)
    knows which to drop downstream."""
    axes = [c for c in df.columns if is_formal_axis(c)]
    zero_var = [c for c in axes if df[c].nunique(dropna=False) <= 1]
    print(f"[diag] zero-variance formal axes (drop at fit time, n={len(zero_var)}):")
    for c in zero_var:
        sample_val = df[c].iloc[0] if len(df) else None
        print(f"         {c}  =  {sample_val!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("02_eval_combined.csv"))
    ap.add_argument("--output", type=Path, default=Path("03_analysis_ready.csv"))
    args = ap.parse_args()

    print(f"[io] reading {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"[io] input shape: {df.shape}")

    df = filter_rows(df)
    keep_cols = select_columns(df)
    df_out = df[keep_cols].copy()

    print(f"[io] output shape: {df_out.shape}")
    print("[io] output columns by family:")
    families = {
        "config/axes/": sum(1 for c in keep_cols if c.startswith("config/axes/")),
        "eval_v2/": sum(1 for c in keep_cols if c.startswith("eval_v2/")),
        "meta_run/": sum(1 for c in keep_cols if c.startswith("meta_run/")),
        "config/meta.": sum(1 for c in keep_cols if c.startswith("config/meta.")),
    }
    for k, v in families.items():
        print(f"         {k:20s} {v}")

    report_zero_variance_axes(df_out)

    print(f"[io] writing {args.output}")
    df_out.to_csv(args.output, index=False)
    print("[done]")


if __name__ == "__main__":
    main()
