"""ch5_typepair_plots.py — Exp 5C AUROC bar (entry point C).

Seed-averaged AUROC over the type-pair B1-T sub-axes from W&B groups
`exp_20260511-193832_ch5_typepair_p1` (init=none, no freeze axis: 3 runs) and
`exp_20260511-200334_ch5_typepair_p2` (2 inits × 2 freeze × 3 seeds = 12 runs).
15 runs total.

Source of truth: thesis_results/04_cleaned_backfilled_analysis_ready.csv
Metric column:   eval_v2/test_auroc
Axis columns:    config/axes/B1-T1_Type-Pair Initialization
                 config/axes/B1-T2_Type-Pair Freeze Table
Seed column:     config/axes/R5_Seed

Note: the actual `B1-T1` values present in the CSV are `none`, `binary`,
`fixed_coupling` (not `physics` — that was the colloquial name; the implemented
init mode is `fixed_coupling`).

Layout choice (stated in the evidence note):
  Single grouped bar over the 5 distinct cells:
    none, binary-free, binary-frozen, fixed_coupling-free, fixed_coupling-frozen.
  `none` is grouped on its own and coloured as the baseline (no freeze axis),
  binary/fixed_coupling use the physics-bias teal with hatch indicating
  frozen-vs-free. Single panel reads cleaner than two side-by-side panels at 5
  cells.

Output: /data/atlas/users/nterlind/outputs/reports/report_ch5_typepair/
Format: PDF at 300 DPI via thesis style system.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_ml.reports.plots.style import (
    apply_thesis_style,
    axis_color,
    figure_size,
)

apply_thesis_style()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path("/project/atlas/users/nterlind/Thesis-Code")
CSV_PATH = REPO_ROOT / "thesis_results" / "04_cleaned_backfilled_analysis_ready.csv"
OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch5_typepair")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Column names ─────────────────────────────────────────────────────────────

GROUP_COL = "meta_run/group"
T1_COL = "config/axes/B1-T1_Type-Pair Initialization"
T2_COL = "config/axes/B1-T2_Type-Pair Freeze Table"
SEED_COL = "config/axes/R5_Seed"
AUROC_COL = "eval_v2/test_auroc"

GROUPS = (
    "exp_20260511-193832_ch5_typepair_p1",  # init=none, no freeze sweep
    "exp_20260511-200334_ch5_typepair_p2",  # init∈{binary,fixed_coupling} × freeze∈{F,T}
)

# Display order: 5 cells. Each entry: (init, freeze, label).
# Freeze values in the CSV are strings ("True"/"False").
_CELL_ORDER: list[tuple[str, str | None, str]] = [
    ("none", None, "none\n(baseline)"),
    ("binary", "False", "binary\nfree"),
    ("binary", "True", "binary\nfrozen"),
    ("fixed_coupling", "False", "fixed-coupling\nfree"),
    ("fixed_coupling", "True", "fixed-coupling\nfrozen"),
]


def _cell_rows(df: pd.DataFrame, init: str, freeze: str | None) -> pd.DataFrame:
    if freeze is None:
        return df[df[T1_COL] == init]
    return df[(df[T1_COL] == init) & (df[T2_COL].astype(str) == freeze)]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for init, freeze, label in _CELL_ORDER:
        cell = _cell_rows(df, init, freeze)
        vals = cell[AUROC_COL].dropna()
        rows.append({
            "init": init,
            "freeze": freeze if freeze is not None else "(n/a)",
            "label": label.replace("\n", " "),
            "mean": float(vals.mean()) if len(vals) else np.nan,
            "std": float(vals.std()) if len(vals) > 1 else np.nan,
            "count": int(len(vals)),
        })
    return pd.DataFrame(rows)


def plot_5c_bar(df: pd.DataFrame, agg: pd.DataFrame, fname: str) -> Path:
    fig, ax = plt.subplots(figsize=figure_size("full"))

    labels = [lbl for _, _, lbl in _CELL_ORDER]
    x = np.arange(len(labels))
    means = agg["mean"].to_numpy()
    stds = agg["std"].to_numpy()

    # Colours and hatches:
    #   - none: baseline grey, no hatch
    #   - binary / fixed_coupling: physics-bias teal
    #   - frozen variants: hatched
    colors: list[str] = []
    hatches: list[str] = []
    for init, freeze, _ in _CELL_ORDER:
        if init == "none":
            colors.append(axis_color("baseline"))
            hatches.append("")
        else:
            colors.append(axis_color("B"))
            hatches.append("//" if freeze == "True" else "")

    bars = ax.bar(
        x,
        means,
        width=0.6,
        color=colors,
        yerr=stds,
        capsize=3,
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Per-seed scatter behind bars.
    for i, (init, freeze, _) in enumerate(_CELL_ORDER):
        cell = _cell_rows(df, init, freeze)
        ax.scatter(
            np.full(len(cell), x[i]),
            cell[AUROC_COL].to_numpy(),
            color="black",
            alpha=0.5,
            s=20,
            zorder=3,
        )

    # Reference baseline line at the `none` mean.
    none_mean = float(means[0])
    ax.axhline(
        none_mean,
        color=axis_color("baseline"),
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("B1-T1 Type-Pair Init × B1-T2 Freeze Table")
    ax.set_ylabel("Test AUROC (seed mean, n=3)")

    y_lo = float(np.nanmin(means - np.nan_to_num(stds)))
    y_hi = float(np.nanmax(means + np.nan_to_num(stds)))
    pad = max(0.0005, 0.25 * (y_hi - y_lo))
    ax.set_ylim(y_lo - pad, y_hi + pad)

    out_path = OUT_DIR / f"{fname}.pdf"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    logger.info("Saved %s", out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    sub = df[df[GROUP_COL].isin(GROUPS)].copy()
    # Normalise freeze column to string (CSV has it as object with "True"/"False").
    sub[T2_COL] = sub[T2_COL].astype(str)
    logger.info("Loaded %d rows for groups %s", len(sub), GROUPS)
    if len(sub) != 15:
        logger.warning("Expected 15 rows for Exp 5C; got %d", len(sub))

    agg = _aggregate(sub)
    logger.info("Aggregated AUROC stats (Exp 5C):\n%s", agg.to_string(index=False))

    table_path = OUT_DIR / "exp5c_auroc_summary.csv"
    agg.to_csv(table_path, index=False)
    logger.info("Saved %s", table_path)

    plot_5c_bar(sub, agg, "figure-auroc_bar_by_typepair")


if __name__ == "__main__":
    main()
