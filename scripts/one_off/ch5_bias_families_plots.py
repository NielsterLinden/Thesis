"""ch5_bias_families_plots.py — Exp 5A AUROC bar (entry point C).

Seed-averaged AUROC bar over `config/axes/B1_Bias Activation Set` for the six
bias-set levels (3 seeds each) trained in W&B group
`exp_20260511-144128_ch5_bias_families`. 18 runs total.

Source of truth: thesis_results/04_cleaned_backfilled_analysis_ready.csv
Metric column:   eval_v2/test_auroc
Axis column:     config/axes/B1_Bias Activation Set
Seed column:     config/axes/R5_Seed

Error bars: per-bias-set seed standard deviation (n=3). Individual seed values
are overplotted as scatter behind the bars (alpha=0.5, zorder=3).

Output: /data/atlas/users/nterlind/outputs/reports/report_ch5_bias_families/
Format: PDF at 300 DPI via thesis style system.

Style compliance:
  - apply_thesis_style() called at module level
  - figure_size("full") for the bar chart
  - axis_color("B") for bias-family bars (physics biases group)
  - axis_color("baseline") for the `none` reference bar
  - no titles; captions live in LaTeX
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
OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch5_bias_families")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Column names (confirmed from CSV) ────────────────────────────────────────

GROUP_COL = "meta_run/group"
B1_COL = "config/axes/B1_Bias Activation Set"
SEED_COL = "config/axes/R5_Seed"
AUROC_COL = "eval_v2/test_auroc"

EXP5A_GROUP = "exp_20260511-144128_ch5_bias_families"

# Display order + short labels for the 6 bias-activation-set levels.
# `none` is the baseline reference; placed first.
_DISPLAY_ORDER: list[tuple[str, str]] = [
    ("none", "none\n(baseline)"),
    ("lorentz_scalar", "lorentz\nscalar"),
    ("typepair_kinematic", "type-pair\nkinematic"),
    ("sm_interaction", "SM\ninteraction"),
    ("global_conditioned", "global-\nconditioned"),
    (
        "lorentz_scalar+typepair_kinematic+sm_interaction+global_conditioned",
        "all four\ncombined",
    ),
]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-aggregate AUROC per B1 value. Returns a frame ordered by _DISPLAY_ORDER."""
    grp = df.groupby(B1_COL)[AUROC_COL]
    out = grp.agg(["mean", "std", "count"]).reset_index()
    out = out.set_index(B1_COL).reindex([v for v, _ in _DISPLAY_ORDER]).reset_index()
    return out


def plot_5a_bar(df: pd.DataFrame, agg: pd.DataFrame, fname: str) -> Path:
    """Bar chart: mean AUROC per bias family with seed-std error bars + seed scatter."""
    fig, ax = plt.subplots(figsize=figure_size("full"))

    labels = [lbl for _, lbl in _DISPLAY_ORDER]
    x = np.arange(len(labels))

    means = agg["mean"].to_numpy()
    stds = agg["std"].to_numpy()

    # Colors: baseline (`none`) in near-black, other 5 in the physics-bias teal.
    colors = [axis_color("baseline")] + [axis_color("B")] * (len(labels) - 1)

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

    # Per-seed scatter behind bars.
    for i, (b1_value, _) in enumerate(_DISPLAY_ORDER):
        seed_rows = df[df[B1_COL] == b1_value]
        ax.scatter(
            np.full(len(seed_rows), x[i]),
            seed_rows[AUROC_COL].to_numpy(),
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
    ax.set_xlabel("B1 — Bias Activation Set")
    ax.set_ylabel("Test AUROC (seed mean, n=3)")

    # Tight y-range around the data so seed spread is visible.
    y_lo = float(np.nanmin(means - stds))
    y_hi = float(np.nanmax(means + stds))
    pad = max(0.001, 0.25 * (y_hi - y_lo))
    ax.set_ylim(y_lo - pad, y_hi + pad)

    out_path = OUT_DIR / f"{fname}.pdf"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    logger.info("Saved %s", out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    sub = df[df[GROUP_COL] == EXP5A_GROUP].copy()
    logger.info("Loaded %d rows for group %s", len(sub), EXP5A_GROUP)

    expected_levels = {v for v, _ in _DISPLAY_ORDER}
    present_levels = set(sub[B1_COL].dropna().unique())
    missing = expected_levels - present_levels
    if missing:
        logger.warning("Missing B1 levels in CSV: %s", missing)

    agg = _aggregate(sub)
    logger.info("Aggregated AUROC stats (Exp 5A):\n%s", agg.to_string(index=False))

    # Save numeric companion table next to the figure.
    table_path = OUT_DIR / "exp5a_auroc_summary.csv"
    agg.to_csv(table_path, index=False)
    logger.info("Saved %s", table_path)

    plot_5a_bar(sub, agg, "figure-auroc_bar_by_bias_family")


if __name__ == "__main__":
    main()
