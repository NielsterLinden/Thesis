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


def plot_curves_grouped_by(
    runs_df: pd.DataFrame,
    per_epoch: dict[str, pd.DataFrame],
    group_col: str,
    metric: str,
    figs_dir: Path,
    fig_cfg: dict,
    fname: str | None = None,
    title: str | None = None,
    show_individual: bool = True,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """Plot training/validation curves grouped by a hyperparameter.

    Shows mean curves with optional individual run curves behind.
    Similar to notebook-style grouped plots.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information, must contain group_col and run_dir
    per_epoch : dict[str, pd.DataFrame]
        Per-epoch metrics for each run (key is run_dir string)
    group_col : str
        Column name to group by (e.g., 'positional', 'dropout', 'norm_policy')
    metric : str
        Metric column name (e.g., 'val_loss', 'val_auroc', 'train_loss')
    figs_dir : Path
        Directory to save figures
    fig_cfg : dict
        Figure configuration (fig_format, dpi)
    fname : str | None
        Base filename for saved figure (auto-generated if None)
    title : str | None
        Plot title (auto-generated if None)
    show_individual : bool
        If True, show individual runs behind mean curves (default: True)
    figsize : tuple[float, float]
        Figure size (default: (12, 8))
    """
    if group_col not in runs_df.columns:
        return  # Skip if column doesn't exist

    fig, ax = plt.subplots(figsize=figsize)

    # Get unique group values and assign colors
    groups = sorted(runs_df[group_col].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    color_map = {g: colors[i] for i, g in enumerate(groups)}

    for group_val in groups:
        group_runs = runs_df[runs_df[group_col] == group_val]
        color = color_map[group_val]

        all_epochs = []
        all_values = []

        for _, row in group_runs.iterrows():
            rd = str(row.get("run_dir"))
            if rd not in per_epoch:
                continue

            hist = per_epoch[rd].copy()
            # Handle split column if present
            if "split" in hist.columns:
                if "val" in metric:
                    hist = hist[hist["split"] == "val"]
                elif "train" in metric:
                    hist = hist[hist["split"] == "train"]

            if metric not in hist.columns:
                continue

            epochs = hist["epoch"].astype(int).values
            values = hist[metric].astype(float).values

            # Plot individual curve if requested
            if show_individual:
                ax.plot(epochs, values, color=color, alpha=0.2, linewidth=1, zorder=1)

            all_epochs.append(epochs)
            all_values.append(values)

        # Compute and plot mean curve
        if all_values:
            # Find common epoch range
            max_epochs = max(len(e) for e in all_epochs)
            mean_values = np.full(max_epochs, np.nan)
            std_values = np.full(max_epochs, np.nan)

            for ep in range(max_epochs):
                vals_at_ep = [v[ep] for v, e in zip(all_values, all_epochs, strict=False) if len(v) > ep]
                if vals_at_ep:
                    mean_values[ep] = np.mean(vals_at_ep)
                    std_values[ep] = np.std(vals_at_ep) if len(vals_at_ep) > 1 else 0.0

            epochs_range = np.arange(max_epochs)
            valid_mask = ~np.isnan(mean_values)

            ax.plot(
                epochs_range[valid_mask],
                mean_values[valid_mask],
                color=color,
                linewidth=2.5,
                marker="o",
                markersize=4,
                label=f"{group_val} (n={len(all_values)})",
                zorder=2,
            )

            # Add shaded error region
            if show_individual and np.any(std_values[valid_mask] > 0):
                ax.fill_between(
                    epochs_range[valid_mask],
                    mean_values[valid_mask] - std_values[valid_mask],
                    mean_values[valid_mask] + std_values[valid_mask],
                    color=color,
                    alpha=0.15,
                    zorder=1,
                )

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=14)
    ax.set_title(title or f"{metric.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}", fontsize=16)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    save_figure(fig, figs_dir, fname or f"figure-{metric}_by_{group_col}", fig_cfg)
    plt.close(fig)


def plot_val_auroc_curves(
    runs_df: pd.DataFrame,
    per_epoch: dict[str, pd.DataFrame],
    figs_dir: Path,
    fig_cfg: dict,
    fname: str = "figure-all_val_auroc_curves",
) -> None:
    """Plot all runs' validation AUROC vs epoch with individual run legends.

    Similar to plot_all_val_curves but for AUROC metric.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information
    per_epoch : dict[str, pd.DataFrame]
        Per-epoch metrics for each run (key is run_dir string)
    figs_dir : Path
        Directory to save figures
    fig_cfg : dict
        Figure configuration (fig_format, dpi)
    fname : str
        Base filename for saved figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Generate distinct colors for each run
    num_runs = len(runs_df)
    colors = plt.cm.tab20(np.linspace(0, 1, max(num_runs, 20)))

    legend_handles = []
    run_data = []

    for idx, (_, row) in enumerate(runs_df.iterrows()):
        rd = str(row.get("run_dir"))
        if rd not in per_epoch:
            continue
        hist = per_epoch[rd]
        cur = hist.copy()
        cur = cur[cur["split"] == "val"] if "split" in cur.columns else cur

        # Try different AUROC column names
        auroc_col = None
        for col in ["val_auroc", "auroc", "metric_auroc"]:
            if col in cur.columns:
                auroc_col = col
                break

        if auroc_col is None:
            continue

        epochs = cur["epoch"].astype(int).values
        vals = cur[auroc_col].astype(float).values

        # Find best (highest) value and its epoch
        best_idx = np.argmax(vals)
        best_val = vals[best_idx]
        best_epoch = epochs[best_idx]

        run_name = Path(rd).name
        color = colors[idx % len(colors)]

        ax.plot(epochs, vals, color=color, linewidth=1.5, alpha=0.9)
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
        ax.legend(handles=legend_handles, loc="best", fontsize=9, framealpha=0.9)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Validation AUROC", fontsize=14)
    ax.set_title("Validation AUROC per run", fontsize=16)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.2)
    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)


def plot_val_auroc_grouped_by(
    runs_df: pd.DataFrame,
    per_epoch: dict[str, pd.DataFrame],
    group_col: str,
    figs_dir: Path,
    fig_cfg: dict,
    fname: str | None = None,
    title: str | None = None,
    show_individual: bool = True,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """Plot validation AUROC curves grouped by a hyperparameter.

    Convenience wrapper around plot_curves_grouped_by for AUROC metric.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information, must contain group_col and run_dir
    per_epoch : dict[str, pd.DataFrame]
        Per-epoch metrics for each run (key is run_dir string)
    group_col : str
        Column name to group by (e.g., 'positional', 'dropout')
    figs_dir : Path
        Directory to save figures
    fig_cfg : dict
        Figure configuration (fig_format, dpi)
    fname : str | None
        Base filename for saved figure (auto-generated if None)
    title : str | None
        Plot title (auto-generated if None)
    show_individual : bool
        If True, show individual runs behind mean curves (default: True)
    figsize : tuple[float, float]
        Figure size (default: (12, 8))
    """
    # Try different AUROC column names
    for auroc_col in ["val_auroc", "auroc", "metric_auroc"]:
        # Check if any run has this column
        has_col = False
        for rd in per_epoch:
            if auroc_col in per_epoch[rd].columns:
                has_col = True
                break
        if has_col:
            plot_curves_grouped_by(
                runs_df=runs_df,
                per_epoch=per_epoch,
                group_col=group_col,
                metric=auroc_col,
                figs_dir=figs_dir,
                fig_cfg=fig_cfg,
                fname=fname or f"figure-val_auroc_by_{group_col}",
                title=title or f"Validation AUROC by {group_col.replace('_', ' ').title()}",
                show_individual=show_individual,
                figsize=figsize,
            )
            return
