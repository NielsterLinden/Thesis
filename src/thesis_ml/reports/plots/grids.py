from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_ml.monitoring.io_utils import save_figure
from thesis_ml.reports.plots.style import apply_thesis_style, figure_size

apply_thesis_style()


def plot_grid_heatmap(df: pd.DataFrame, row_col: str, col_col: str, value_col: str, title: str = "", figs_dir: Path = None, fname: str = None, fig_cfg: dict = None) -> None:
    """Generic grid heatmap for 2D parameter sweeps"""
    cur = df.dropna(subset=[row_col, col_col, value_col]).copy()

    # Convert to numeric if needed
    if cur[row_col].dtype == object:
        with contextlib.suppress(Exception):
            cur[row_col] = cur[row_col].astype(float)

    pt = cur.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean")

    fig, ax = plt.subplots(figsize=figure_size("full", aspect=1.0))
    im = ax.imshow(pt.values, aspect="auto", cmap="viridis")

    # Annotate cells
    for i in range(len(pt.index)):
        for j in range(len(pt.columns)):
            val = pt.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val < pt.values.mean() else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=text_color)

    ax.set_xticks(range(len(pt.columns)))
    ax.set_xticklabels(pt.columns)
    ax.set_yticks(range(len(pt.index)))
    ax.set_yticklabels(pt.index)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    fig.colorbar(im, ax=ax, label=value_col)

    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
