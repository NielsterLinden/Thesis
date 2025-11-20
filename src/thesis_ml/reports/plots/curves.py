from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from thesis_ml.monitoring.io_utils import save_figure


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
    ax.set_xlabel("wall-clock seconds (cumulative)", fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_title(f"{metric} vs time", fontsize=16)
    ax.legend(fontsize=12)
    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)


def _color_for_latent(latent_space: str | None) -> str:
    mapping = {"none": "tab:blue", "linear": "tab:orange", "vq": "tab:green"}
    return mapping.get(str(latent_space).lower() if latent_space else None, "tab:gray")


def _linestyle_for_beta(beta: float | None) -> str:
    if beta is None:
        return "-"
    try:
        b = float(beta)
    except Exception:
        return "-"
    if abs(b) < 1e-12 or b == 0.0:
        return ":"  # dotted for 0
    if abs(b - 1.0) < 1e-9:
        return "-"  # solid for 1
    return "--"  # dashed for others (e.g., 10)


def plot_all_val_curves(
    runs_df: pd.DataFrame,
    per_epoch: dict[str, pd.DataFrame],
    figs_dir: Path,
    fig_cfg: dict,
    fname: str = "figure-all_val_curves",
) -> None:
    """Plot all runs' validation loss vs epoch with individual run legends."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Generate distinct colors for each run
    num_runs = len(runs_df)
    colors = plt.cm.tab20(np.linspace(0, 1, max(num_runs, 20)))

    # Track runs and their best values for legend
    legend_handles = []
    run_data = []

    for idx, (_, row) in enumerate(runs_df.iterrows()):
        rd = str(row.get("run_dir"))
        if rd not in per_epoch:
            continue
        hist = per_epoch[rd]
        cur = hist.copy()
        cur = cur[cur["split"] == "val"] if "split" in cur.columns else cur
        if "val_loss" not in cur.columns:
            continue
        epochs = cur["epoch"].astype(int).values
        vals = cur["val_loss"].astype(float).values

        # Find best (lowest) value and its epoch
        best_idx = np.argmin(vals)
        best_val = vals[best_idx]
        best_epoch = epochs[best_idx]

        # Get run name (extract from path)
        run_name = Path(rd).name

        # Assign color to this run
        color = colors[idx % len(colors)]

        # Plot the curve
        ax.plot(
            epochs,
            vals,
            color=color,
            linewidth=1.5,
            alpha=0.9,
        )

        # Store data for legend
        run_data.append((run_name, color, best_val, best_epoch))

    # Create legend with run name and best values
    for run_name, color, best_val, best_epoch in run_data:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2,
                label=f"{run_name}\n(best: {best_val:.4f}, epoch: {best_epoch})",
            )
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="best",
            fontsize=9,
            framealpha=0.9,
        )

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Validation Loss", fontsize=14)
    ax.set_title("Validation loss per run", fontsize=16)
    ax.grid(True, alpha=0.2)
    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)


def plot_all_train_curves(
    runs_df: pd.DataFrame,
    per_epoch: dict[str, pd.DataFrame],
    figs_dir: Path,
    fig_cfg: dict,
    fname: str = "figure-all_train_curves",
) -> None:
    """Plot all runs' training loss vs epoch with individual run legends."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Generate distinct colors for each run
    num_runs = len(runs_df)
    colors = plt.cm.tab20(np.linspace(0, 1, max(num_runs, 20)))

    # Track runs and their best values for legend
    legend_handles = []
    run_data = []

    for idx, (_, row) in enumerate(runs_df.iterrows()):
        rd = str(row.get("run_dir"))
        if rd not in per_epoch:
            continue
        hist = per_epoch[rd]
        cur = hist.copy()
        cur = cur[cur["split"] == "train"] if "split" in cur.columns else cur
        if "train_loss" not in cur.columns:
            continue
        epochs = cur["epoch"].astype(int).values
        vals = cur["train_loss"].astype(float).values

        # Find best (lowest) value and its epoch
        best_idx = np.argmin(vals)
        best_val = vals[best_idx]
        best_epoch = epochs[best_idx]

        # Get run name (extract from path)
        run_name = Path(rd).name

        # Assign color to this run
        color = colors[idx % len(colors)]

        # Plot the curve
        ax.plot(
            epochs,
            vals,
            color=color,
            linewidth=1.5,
            alpha=0.9,
        )

        # Store data for legend
        run_data.append((run_name, color, best_val, best_epoch))

    # Create legend with run name and best values
    for run_name, color, best_val, best_epoch in run_data:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2,
                label=f"{run_name}\n(best: {best_val:.4f}, epoch: {best_epoch})",
            )
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="best",
            fontsize=9,
            framealpha=0.9,
        )

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Training Loss", fontsize=14)
    ax.set_title("Training loss per run", fontsize=16)
    ax.grid(True, alpha=0.2)
    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
