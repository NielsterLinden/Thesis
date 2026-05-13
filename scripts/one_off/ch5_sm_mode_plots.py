"""ch5_sm_mode_plots.py — Exp 5D AUROC bar (entry point C).

Seed-averaged AUROC over the SM-interaction-mode axis B1-S1 from W&B group
`exp_20260511-214641_ch5_sm_mode`. 9 runs total (3 modes × 3 seeds).

Source of truth: thesis_results/04_cleaned_backfilled_analysis_ready.csv
Metric column:   eval_v2/test_auroc
Axis column:     config/axes/B1-S1_SM Interaction Mode
Seed column:     config/axes/R5_Seed

Layout: single-row 3-bar chart. No baseline reference cell in this group
(the `none` SM-interaction baseline is recorded in Exp 5A — see
`ch5_B1_bias_families.md` Section 6 for the cross-reference).

Output: /data/atlas/users/nterlind/outputs/reports/report_ch5_sm_mode/
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
OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch5_sm_mode")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Column names ─────────────────────────────────────────────────────────────

GROUP_COL = "meta_run/group"
S1_COL = "config/axes/B1-S1_SM Interaction Mode"
SEED_COL = "config/axes/R5_Seed"
AUROC_COL = "eval_v2/test_auroc"

EXP5D_GROUP = "exp_20260511-214641_ch5_sm_mode"

_DISPLAY_ORDER: list[tuple[str, str]] = [
    ("binary", "binary"),
    ("fixed_coupling", "fixed\ncoupling"),
    ("running_coupling", "running\ncoupling"),
]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(S1_COL)[AUROC_COL]
    out = grp.agg(["mean", "std", "count"]).reset_index()
    out = out.set_index(S1_COL).reindex([v for v, _ in _DISPLAY_ORDER]).reset_index()
    return out


def plot_5d_bar(df: pd.DataFrame, agg: pd.DataFrame, fname: str) -> Path:
    fig, ax = plt.subplots(figsize=figure_size("full"))

    labels = [lbl for _, lbl in _DISPLAY_ORDER]
    x = np.arange(len(labels))
    means = agg["mean"].to_numpy()
    stds = agg["std"].to_numpy()

    ax.bar(
        x,
        means,
        width=0.6,
        color=axis_color("B"),
        yerr=stds,
        capsize=3,
        edgecolor="black",
        linewidth=0.5,
    )

    for i, (s1, _) in enumerate(_DISPLAY_ORDER):
        seeds = df[df[S1_COL] == s1]
        ax.scatter(
            np.full(len(seeds), x[i]),
            seeds[AUROC_COL].to_numpy(),
            color="black",
            alpha=0.5,
            s=20,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("B1-S1 — SM Interaction Mode")
    ax.set_ylabel("Test AUROC (seed mean, n=3)")

    y_lo = float(np.nanmin(means - stds))
    y_hi = float(np.nanmax(means + stds))
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
    sub = df[df[GROUP_COL] == EXP5D_GROUP].copy()
    logger.info("Loaded %d rows for group %s", len(sub), EXP5D_GROUP)
    if len(sub) != 9:
        logger.warning("Expected 9 rows for Exp 5D; got %d", len(sub))

    agg = _aggregate(sub)
    logger.info("Aggregated AUROC stats (Exp 5D):\n%s", agg.to_string(index=False))

    table_path = OUT_DIR / "exp5d_auroc_summary.csv"
    agg.to_csv(table_path, index=False)
    logger.info("Saved %s", table_path)

    plot_5d_bar(sub, agg, "figure-auroc_bar_by_sm_mode")


if __name__ == "__main__":
    main()
