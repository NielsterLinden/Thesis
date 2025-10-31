from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from thesis_ml.plots.io_utils import save_figure


def plot_loss_vs_time(
    df: pd.DataFrame,
    per_epoch: dict[str, pd.DataFrame],
    order: list[str],
    figs_dir: Path,
    fig_cfg: dict,
    metric: str = "val_loss",
    fname: str = "figure-loss_vs_time",
) -> None:
    """Generic: plot metric vs cumulative time"""
    fig, ax = plt.subplots()
    for run_dir in order:
        if run_dir not in per_epoch:
            continue
        cur = per_epoch[run_dir].copy()
        cur = cur[cur["split"] == "val"] if "split" in cur.columns else cur
        cur["cum_time_s"] = cur["epoch_time_s"].astype(float).cumsum()
        ax.plot(cur["cum_time_s"], cur[metric].astype(float), label=Path(run_dir).name)
    ax.set_xlabel("wall-clock seconds (cumulative)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs time")
    ax.legend()
    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
