from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from thesis_ml.monitoring.io_utils import save_figure
from thesis_ml.reports.plots.style import apply_thesis_style, figure_size

apply_thesis_style()


def plot_scatter_colored(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_by: str,
    title: str = "",
    figs_dir: Path = None,
    fname: str = None,
    fig_cfg: dict = None,
    annotate_col: str | None = None,
) -> None:
    """Generic scatter with color grouping and optional annotations"""
    cur = df.dropna(subset=[x_col, y_col, color_by]).copy()

    fig, ax = plt.subplots(figsize=figure_size("full"))
    for group_val in sorted(cur[color_by].unique()):
        sub = cur[cur[color_by] == group_val]
        ax.scatter(sub[x_col], sub[y_col], label=str(group_val), alpha=0.7, s=100)

        if annotate_col:
            for _, row in sub.iterrows():
                ax.annotate(str(row[annotate_col]), (row[x_col], row[y_col]), alpha=0.7)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
