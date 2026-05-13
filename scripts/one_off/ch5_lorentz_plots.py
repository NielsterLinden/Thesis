"""ch5_lorentz_plots.py — Exp 5B AUROC bar (entry point C).

Seed-averaged AUROC over the Lorentz-scalar B1-L sub-axes from W&B groups
`exp_20260511-144127_ch5_lorentz_p1` (gate=off) and
`exp_20260511-150429_ch5_lorentz_p2` (gate=on). 60 runs total
(5 feature sets × 2 MLP types × 2 sparse-gating modes × 3 seeds).

Source of truth: thesis_results/04_cleaned_backfilled_analysis_ready.csv
Metric column:   eval_v2/test_auroc
Axis columns:    config/axes/B1-L1_Lorentz Feature Set
                 config/axes/B1-L2_Lorentz MLP Type
                 config/axes/B1-L5_Lorentz Sparse Gating
Seed column:     config/axes/R5_Seed

Layout choice (stated in the evidence note):
  Two-panel layout — gate=off on the left, gate=on on the right, each panel
  shows feature-set on the x-axis with MLP type (kan / standard) as hue.
  Rationale: 5 feature sets × 2 MLP types × 2 gating modes = 20 cells. A
  single grouped bar with 4 hues per feature-set would be too crowded; two
  panels separate the gating dimension cleanly while keeping the
  feature-set × MLP-type comparison intact within each panel.

Output: /data/atlas/users/nterlind/outputs/reports/report_ch5_lorentz/
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
OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch5_lorentz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Column names ─────────────────────────────────────────────────────────────

GROUP_COL = "meta_run/group"
L1_COL = "config/axes/B1-L1_Lorentz Feature Set"
L2_COL = "config/axes/B1-L2_Lorentz MLP Type"
L5_COL = "config/axes/B1-L5_Lorentz Sparse Gating"
SEED_COL = "config/axes/R5_Seed"
AUROC_COL = "eval_v2/test_auroc"

GROUPS = (
    "exp_20260511-144127_ch5_lorentz_p1",  # gate=off
    "exp_20260511-150429_ch5_lorentz_p2",  # gate=on
)

# Display order for the 5 feature sets, smallest -> largest.
_FEATURE_ORDER: list[tuple[str, str]] = [
    ("['deltaR']", "ΔR"),
    ("['m2']", "m²"),
    ("['m2', 'deltaR']", "m², ΔR"),
    (
        "['log_kt', 'z', 'deltaR', 'log_m2']",
        "log kT, z,\nΔR, log m²",
    ),
    (
        "['m2', 'deltaR', 'log_m2', 'log_kt', 'z', 'deltaR_ptw']",
        "m², ΔR, log m²,\nlog kT, z, ΔR/pT",
    ),
]

_MLP_ORDER: list[str] = ["standard", "kan"]
_GATE_ORDER: list[bool] = [False, True]
_GATE_LABEL: dict[bool, str] = {False: "Sparse gating: off", True: "Sparse gating: on"}


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate AUROC across seeds for each (L1, L2, L5) cell."""
    grp = df.groupby([L1_COL, L2_COL, L5_COL])[AUROC_COL]
    out = grp.agg(["mean", "std", "count"]).reset_index()
    return out


def plot_5b_two_panel(df: pd.DataFrame, agg: pd.DataFrame, fname: str) -> Path:
    """Two-panel grouped bar — left=gate off, right=gate on; hue=MLP type."""
    width_in, height_in = figure_size("full")
    fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in), sharey=True)

    n_features = len(_FEATURE_ORDER)
    n_mlp = len(_MLP_ORDER)
    bar_w = 0.36
    x = np.arange(n_features)

    # Colours: MLP type via the categorical palette; we keep `standard` near
    # baseline-grey and `kan` in the physics-bias teal to anchor reading.
    mlp_colors = {
        "standard": axis_color("baseline"),
        "kan": axis_color("B"),
    }

    # Track y-range for a tight common axis.
    y_means_stds: list[float] = []

    for j, gate in enumerate(_GATE_ORDER):
        ax = axes[j]
        for i, mlp in enumerate(_MLP_ORDER):
            offset = (i - (n_mlp - 1) / 2.0) * bar_w
            means: list[float] = []
            stds: list[float] = []
            for fset, _ in _FEATURE_ORDER:
                cell = agg[(agg[L1_COL] == fset) & (agg[L2_COL] == mlp) & (agg[L5_COL] == gate)]
                if len(cell) != 1:
                    logger.warning("Unexpected cell count for (%s,%s,%s): %d",
                                   fset, mlp, gate, len(cell))
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(float(cell["mean"].iloc[0]))
                    stds.append(float(cell["std"].iloc[0]))
            ax.bar(
                x + offset,
                means,
                width=bar_w,
                color=mlp_colors[mlp],
                yerr=stds,
                capsize=3,
                edgecolor="black",
                linewidth=0.5,
                label=mlp,
            )
            # Per-seed scatter behind bars.
            for k, (fset, _) in enumerate(_FEATURE_ORDER):
                seeds = df[
                    (df[L1_COL] == fset)
                    & (df[L2_COL] == mlp)
                    & (df[L5_COL] == gate)
                ]
                ax.scatter(
                    np.full(len(seeds), x[k] + offset),
                    seeds[AUROC_COL].to_numpy(),
                    color="black",
                    alpha=0.5,
                    s=20,
                    zorder=3,
                )
            for m, s in zip(means, stds):
                if not np.isnan(m):
                    y_means_stds.append(m - (s if not np.isnan(s) else 0.0))
                    y_means_stds.append(m + (s if not np.isnan(s) else 0.0))

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in _FEATURE_ORDER])
        ax.set_xlabel(f"B1-L1 — Lorentz Feature Set\n({_GATE_LABEL[gate]})")
        if j == 0:
            ax.set_ylabel("Test AUROC (seed mean, n=3)")
        ax.legend(loc="lower right")

    # Tight common y-range.
    y_lo = float(np.nanmin(y_means_stds))
    y_hi = float(np.nanmax(y_means_stds))
    pad = max(0.0005, 0.25 * (y_hi - y_lo))
    for ax in axes:
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
    logger.info("Loaded %d rows for groups %s", len(sub), GROUPS)
    if len(sub) != 60:
        logger.warning("Expected 60 rows for Exp 5B; got %d", len(sub))

    agg = _aggregate(sub)
    logger.info("Aggregated AUROC stats (Exp 5B):\n%s", agg.to_string(index=False))

    table_path = OUT_DIR / "exp5b_auroc_summary.csv"
    agg.to_csv(table_path, index=False)
    logger.info("Saved %s", table_path)

    plot_5b_two_panel(sub, agg, "figure-auroc_bar_by_lorentz")


if __name__ == "__main__":
    main()
