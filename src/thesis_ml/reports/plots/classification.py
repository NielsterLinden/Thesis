"""Classification plotting functions (matching anomaly detection style)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_ml.monitoring.io_utils import save_figure

logger = logging.getLogger(__name__)


def _extract_job_number(run_id: str) -> int:
    """Extract job number from run_id for numerical sorting.

    Parameters
    ----------
    run_id : str
        Run ID string (e.g., "run_20251114-153438_compare_norm_pos_pool_job1")

    Returns
    -------
    int
        Job number if found, otherwise 0
    """
    match = re.search(r"_job(\d+)$", run_id)
    if match:
        return int(match.group(1))
    return 0  # Default for runs without job number


def _sort_run_ids_numerically(run_ids: list[str]) -> list[str]:
    """Sort run IDs numerically by job number.

    Parameters
    ----------
    run_ids : list[str]
        List of run ID strings

    Returns
    -------
    list[str]
        Sorted list of run IDs (numerically by job number, then lexicographically)
    """
    return sorted(run_ids, key=lambda rid: (_extract_job_number(rid), rid))


def _get_class_names(n_classes: int, label_groups: list[dict] | None = None, signal_class_idx: int | None = None, background_class_idx: int | None = None) -> list[str]:
    """Get class names for multi-class classification.

    Supports both multi-class and binary classification. If label_groups is provided,
    uses names from the groups. Otherwise, falls back to ProcessID-based names.

    Parameters
    ----------
    n_classes : int
        Number of classes
    label_groups : list[dict] | None
        Optional list of label groups with 'name' and 'labels' keys.
        If provided, class names are taken from group names.
    signal_class_idx : int | None
        For backward compatibility: class index for signal (0-indexed).
        Only used if n_classes=2 and label_groups is None.
    background_class_idx : int | None
        For backward compatibility: class index for background (0-indexed).
        Only used if n_classes=2 and label_groups is None.

    Returns
    -------
    list[str]
        List of class names (one per class, in order)
    """
    # If label_groups provided, use names from groups
    if label_groups is not None:
        if len(label_groups) != n_classes:
            logger.warning(f"label_groups length ({len(label_groups)}) != n_classes ({n_classes}), using fallback names")
        else:
            return [group.get("name", f"Class-{i}") for i, group in enumerate(label_groups)]

    # Fallback: ProcessID-based names
    # Default mapping: ProcessID 1 → "4t", 2 → "ttH", 3 → "ttW", 4 → "ttWW", 5 → "ttZ"
    process_id_names = {1: "4t", 2: "ttH", 3: "ttW", 4: "ttWW", 5: "ttZ"}

    # For binary classification with backward compatibility
    if n_classes == 2:
        if signal_class_idx is not None:
            # Legacy binary mode: use signal/background names
            class_name_map = {0: "4t", 1: "ttH"}
            signal_name = class_name_map.get(signal_class_idx, f"Signal-{signal_class_idx}")
            if background_class_idx is None:
                background_class_idx = 1 - signal_class_idx if signal_class_idx < 2 else 0
            background_name = class_name_map.get(background_class_idx, f"Background-{background_class_idx}")
            # Return in class index order
            if signal_class_idx == 0:
                return [signal_name, background_name]
            else:
                return [background_name, signal_name]
        else:
            # Default binary names
            return ["4t", "ttH"]

    # Multi-class: use ProcessID-based names for first n_classes ProcessIDs
    class_names = []
    for i in range(n_classes):
        process_id = i + 1  # ProcessIDs are 1-indexed
        name = process_id_names.get(process_id, f"Class-{i}")
        class_names.append(name)

    return class_names


def plot_roc_curves(
    inference_results: dict[str, dict[str, Any]],
    inference_figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-roc_curves",
) -> None:
    """Plot ROC curves for each model (matching anomaly detection style).

    Parameters
    ----------
    inference_results : dict[str, dict[str, Any]]
        Nested dict: {run_id: {metrics...}}
    inference_figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    fname : str
        Base filename for saved figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort run IDs numerically by job number
    sorted_run_ids = _sort_run_ids_numerically(list(inference_results.keys()))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_run_ids)))

    for i, run_id in enumerate(sorted_run_ids):
        metrics = inference_results[run_id]
        roc_curves = metrics.get("roc_curves", {})

        if not roc_curves:
            continue

        # For binary classification, plot single curve
        if len(roc_curves) == 1:
            class_idx = list(roc_curves.keys())[0]
            curve = roc_curves[class_idx]
            auroc = metrics.get("auroc", 0.0)
            ax.plot(
                curve["fpr"],
                curve["tpr"],
                label=f"{run_id} (AUROC={auroc:.3f})",
                color=colors[i],
                linewidth=2,
            )
        # For multi-class, plot first class as example
        else:
            # Plot first available class
            class_idx = list(roc_curves.keys())[0]
            curve = roc_curves[class_idx]
            auroc = metrics.get("auroc", 0.0)
            ax.plot(
                curve["fpr"],
                curve["tpr"],
                label=f"{run_id} (class {class_idx}, AUROC={auroc:.3f})",
                color=colors[i],
                linewidth=2,
                alpha=0.7,
            )

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (0.5)")

    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("ROC Curves", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    save_figure(fig, inference_figs_dir, fname, fig_cfg)
    plt.close(fig)


def plot_confusion_matrix(
    inference_results: dict[str, dict[str, Any]],
    inference_figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-confusion_matrices",
) -> None:
    """Plot confusion matrices (one per model).

    Parameters
    ----------
    inference_results : dict[str, dict[str, Any]]
        Nested dict: {run_id: {metrics...}}
    inference_figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    fname : str
        Base filename for saved figure
    """
    for run_id, metrics in inference_results.items():
        cm_normalized = metrics.get("confusion_matrix_normalized", [])
        if not cm_normalized or len(cm_normalized) == 0:
            continue

        cm = np.array(cm_normalized)
        n_classes = cm.shape[0]

        # Get class names from label_groups if available, otherwise use defaults
        label_groups = metrics.get("label_groups", None)
        class_names = _get_class_names(n_classes=n_classes, label_groups=label_groups)

        fig, ax = plt.subplots(figsize=(8, 6))
        # Set colorbar limits to [0, 1] for normalized confusion matrix
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label("Normalized Count", fontsize=12)

        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion Matrix: {run_id}",
        )
        ax.set_xlabel("Predicted", fontsize=14)
        ax.set_ylabel("True", fontsize=14)
        ax.set_title(f"Confusion Matrix: {run_id}", fontsize=16)

        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12,
                )

        plt.tight_layout()
        save_figure(fig, inference_figs_dir, f"{fname}_{run_id}", fig_cfg)
        plt.close(fig)


def plot_metrics_comparison(
    inference_results: dict[str, dict[str, Any]],
    inference_figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-metrics_comparison",
) -> None:
    """Bar chart comparing metrics across models (similar to plot_auroc_comparison).

    Parameters
    ----------
    inference_results : dict[str, dict[str, Any]]
        Nested dict: {run_id: {metrics...}}
    inference_figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    fname : str
        Base filename for saved figure
    """
    # Sort run IDs numerically by job number
    run_ids = _sort_run_ids_numerically(list(inference_results.keys()))
    x = np.arange(len(run_ids))
    width = 0.25

    metrics_to_plot = ["accuracy", "auroc", "f1_macro"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics_to_plot):
        values = []
        for rid in run_ids:
            val = inference_results[rid].get(metric, 0.0)
            # Handle None values (e.g., AUROC might be None for single class)
            if val is None:
                val = 0.0
            values.append(val)

        ax.bar(
            x + i * width,
            values,
            width,
            label=metric.replace("_", " ").title(),
            color=colors[i],
            alpha=0.7,
        )

    ax.set_xlabel("Model (Run ID)", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Classification Performance Comparison", fontsize=16)
    ax.set_xticks(x + width)
    ax.set_xticklabels(run_ids, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    save_figure(fig, inference_figs_dir, fname, fig_cfg)
    plt.close(fig)


def plot_score_distributions(
    inference_results: dict[str, dict[str, Any]],
    inference_figs_dir: Path,
    fig_cfg: dict[str, Any],
    signal_class_idx: int = 1,
    fname: str = "figure-score_distributions",
) -> None:
    """Plot normalized step-histograms of classifier scores for signal vs background.

    Matches the style of anomaly detection reconstruction-score histograms.
    Shows signal and background score distributions, optimal threshold (Youden's J),
    and overlap area.

    Parameters
    ----------
    inference_results : dict[str, dict[str, Any]]
        Nested dict: {run_id: {metrics...}} with per_event_scores and per_event_labels
    inference_figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    signal_class_idx : int
        Which class index represents signal (default: 1 for binary classification)
    fname : str
        Base filename for saved figure
    """
    from sklearn.metrics import roc_curve

    for run_id, metrics in inference_results.items():
        # Extract per-event scores and labels (only available for binary classification)
        per_event_scores = metrics.get("per_event_scores")
        per_event_labels = metrics.get("per_event_labels")

        if per_event_scores is None or per_event_labels is None:
            continue  # Skip if not binary classification or data not available

        scores = np.array(per_event_scores)  # [N] - p(signal | event)
        labels = np.array(per_event_labels)  # [N] - true labels

        # Split into signal and background
        signal_scores = scores[labels == signal_class_idx]
        background_scores = scores[labels != signal_class_idx]

        if len(signal_scores) == 0 or len(background_scores) == 0:
            continue  # Need both classes

        # Get class names
        # Determine background class index (the other class)
        unique_labels = np.unique(labels)
        background_class_idx = unique_labels[unique_labels != signal_class_idx][0] if len(unique_labels) > 1 else None

        # Get label_groups from metrics if available
        label_groups = metrics.get("label_groups", None)
        class_names = _get_class_names(n_classes=2, label_groups=label_groups, signal_class_idx=signal_class_idx, background_class_idx=background_class_idx)
        signal_name = class_names[signal_class_idx]
        background_name = class_names[background_class_idx] if background_class_idx is not None else class_names[1 - signal_class_idx]

        # Create figure matching anomaly detection style
        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute optimal threshold using Youden's J
        # Convert labels to binary: signal=1, background=0
        y_binary = (labels == signal_class_idx).astype(int)
        fpr, tpr, thresholds = roc_curve(y_binary, scores)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]

        # Compute overlap area (common area of normalized histograms)
        # Use same bins for both distributions
        score_min = min(scores.min(), background_scores.min(), signal_scores.min())
        score_max = max(scores.max(), background_scores.max(), signal_scores.max())
        bins = np.linspace(score_min, score_max, 50)

        # Compute normalized histograms
        bg_counts, bg_edges = np.histogram(background_scores, bins=bins)
        sig_counts, sig_edges = np.histogram(signal_scores, bins=bins)

        # Normalize by total count
        bg_normalized = bg_counts / len(background_scores)
        sig_normalized = sig_counts / len(signal_scores)

        # Compute overlap (minimum of the two normalized distributions at each bin)
        overlap = np.minimum(bg_normalized, sig_normalized)
        overlap_area = np.sum(overlap) * (bins[1] - bins[0])  # Integrate

        # Plot normalized step histograms (matching anomaly detection style)
        # Use step plot (no fill) with consistent line width
        ax.step(
            bg_edges[:-1],
            bg_normalized,
            where="post",
            label=f"Background - {background_name}",
            color="#1f77b4",  # Blue
            linewidth=2,
            alpha=0.8,
        )
        ax.step(
            sig_edges[:-1],
            sig_normalized,
            where="post",
            label=f"Signal - {signal_name}",
            color="#ff7f0e",  # Orange
            linewidth=2,
            alpha=0.8,
        )

        # Plot optimal threshold line
        ax.axvline(
            optimal_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Optimal threshold ({optimal_threshold:.3f})",
        )

        # Annotate overlap area
        overlap_pct = overlap_area * 100
        ax.text(
            0.98,
            0.98,
            f"Overlap: {overlap_pct:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=11,
        )

        # Set labels and title
        ax.set_xlabel("Classifier Score (p(signal | event))", fontsize=14)
        ax.set_ylabel("bin count / N", fontsize=14)
        ax.set_title(f"Score Distributions: {run_id}", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)

        # Set x-axis limits to tightly cover the distributions
        ax.set_xlim([score_min, score_max])

        plt.tight_layout()
        save_figure(fig, inference_figs_dir, f"{fname}_{run_id}", fig_cfg)
        plt.close(fig)


def plot_metrics_by_axis(
    runs_df: pd.DataFrame,
    axis_col: str,
    metric_col: str,
    inference_results: dict[str, dict[str, Any]] | None,
    inference_figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-metrics_by_axis",
    title: str | None = None,
) -> None:
    """Bar chart comparing metrics grouped by a single axis (norm_policy, positional, or pooling).

    Can use either training metrics from runs_df or inference metrics from inference_results.
    If inference_results is provided, uses inference metrics; otherwise uses training metrics.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information, must contain axis_col and run_dir columns
    axis_col : str
        Column name to group by (e.g., 'norm_policy', 'positional', 'pooling')
    metric_col : str
        Column name for metric value (for training) or metric name (for inference)
    inference_results : dict[str, dict[str, Any]] | None
        Optional inference results dict: {run_id: {metrics...}}
        If provided, extracts metric_col from inference metrics instead of runs_df
    inference_figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    fname : str
        Base filename for saved figure
    title : str | None
        Optional custom title (default: "Metric by {axis_col}")
    """
    if axis_col not in runs_df.columns:
        logger.warning(f"Column '{axis_col}' not found in runs_df, skipping plot")
        return

    # Group by axis
    if inference_results is not None:
        # Use inference metrics
        # Map run_dir to run_id for lookup
        from thesis_ml.utils.paths import get_run_id

        axis_groups = {}
        for _idx, row in runs_df.iterrows():
            run_dir = row.get("run_dir")
            if pd.isna(run_dir):
                continue
            run_id = get_run_id(Path(str(run_dir)))
            axis_value = row.get(axis_col)

            if run_id in inference_results:
                metric_value = inference_results[run_id].get(metric_col)
                if metric_value is not None:
                    if axis_value not in axis_groups:
                        axis_groups[axis_value] = []
                    axis_groups[axis_value].append(metric_value)

        # Compute mean and std for each axis value
        axis_values = sorted(axis_groups.keys())
        means = [np.mean(axis_groups[v]) for v in axis_values]
        stds = [np.std(axis_groups[v]) if len(axis_groups[v]) > 1 else 0.0 for v in axis_values]
        labels = [str(v) for v in axis_values]
    else:
        # Use training metrics from runs_df
        grouped = runs_df.groupby(axis_col)[metric_col].agg(["mean", "std", "count"])
        axis_values = grouped.index.tolist()
        means = grouped["mean"].tolist()
        stds = grouped["std"].fillna(0.0).tolist()
        labels = [str(v) for v in axis_values]

    if not axis_values:
        logger.warning(f"No data found for axis '{axis_col}', skipping plot")
        return

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(axis_values))
    width = 0.6

    ax.bar(x, means, width, yerr=stds, capsize=5, alpha=0.7, color="steelblue", edgecolor="black")

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds, strict=False)):
        ax.text(i, mean + std + 0.01, f"{mean:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel(axis_col.replace("_", " ").title(), fontsize=14)
    ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=14)
    ax.set_title(title or f"{metric_col.replace('_', ' ').title()} by {axis_col.replace('_', ' ').title()}", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, inference_figs_dir, fname, fig_cfg)
    plt.close(fig)


def plot_grid_metrics(
    runs_df: pd.DataFrame,
    row_col: str,
    col_col: str,
    metric_col: str,
    title: str,
    inference_results: dict[str, dict[str, Any]] | None,
    inference_figs_dir: Path,
    fname: str,
    fig_cfg: dict[str, Any],
) -> None:
    """Grid heatmap for classification metrics across 2D parameter combinations.

    Can use either training metrics from runs_df or inference metrics from inference_results.
    If inference_results is provided, uses inference metrics; otherwise uses training metrics.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information, must contain row_col, col_col, and run_dir columns
    row_col : str
        Column name for row axis (e.g., 'norm_policy')
    col_col : str
        Column name for column axis (e.g., 'positional')
    metric_col : str
        Column name for metric value (for training) or metric name (for inference)
    title : str
        Plot title
    inference_results : dict[str, dict[str, Any]] | None
        Optional inference results dict: {run_id: {metrics...}}
        If provided, extracts metric_col from inference metrics instead of runs_df
    inference_figs_dir : Path
        Directory to save figures
    fname : str
        Base filename for saved figure
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    """
    from ..plots.grids import plot_grid_heatmap

    # Prepare data for grid heatmap
    if inference_results is not None:
        # Use inference metrics - need to merge into runs_df temporarily
        from thesis_ml.utils.paths import get_run_id

        temp_df = runs_df.copy()
        metric_values = []

        for _idx, row in temp_df.iterrows():
            run_dir = row.get("run_dir")
            if pd.isna(run_dir):
                metric_values.append(None)
                continue
            run_id = get_run_id(Path(str(run_dir)))
            if run_id in inference_results:
                metric_value = inference_results[run_id].get(metric_col)
                metric_values.append(metric_value)
            else:
                metric_values.append(None)

        temp_df[metric_col] = metric_values
        plot_df = temp_df
    else:
        # Use training metrics from runs_df
        plot_df = runs_df

    # Use existing grid heatmap function
    plot_grid_heatmap(plot_df, row_col, col_col, metric_col, title, inference_figs_dir, fname, fig_cfg)


# ============================================================================
# DataFrame-based plotting functions (new API)
# ============================================================================


def plot_metric_vs_hparam(
    df: pd.DataFrame,
    x_col: str,
    group_col: str | None,
    metric_col: str,
    style: str = "line",
    show_individual: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot metric vs hyperparameter with optional grouping.

    Handles: AUROC vs model size; AUROC vs PE type; AUROC vs norm policy.
    Style: Mean line (thick, linewidth=2-3) with individual runs behind (alpha=0.15-0.3).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns x_col, group_col (if provided), metric_col, and run_id/run_dir
    x_col : str
        Column name for x-axis (e.g., 'model_size')
    group_col : str | None
        Column name to group by (e.g., 'positional', 'norm_policy'). If None, no grouping.
    metric_col : str
        Column name for metric value (e.g., 'auroc', 'accuracy')
    style : str
        Plot style: "line" or "bar" (default: "line")
    show_individual : bool
        If True, show individual runs behind mean (default: True)
    figsize : tuple[float, float]
        Figure size (default: (10, 8))

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    # Validate columns
    required_cols = [x_col, metric_col]
    if group_col:
        required_cols.append(group_col)
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get run identifier column
    run_id_col = "run_id" if "run_id" in df.columns else "run_dir"

    # Filter out NaN values
    plot_df = df.dropna(subset=required_cols).copy()

    if len(plot_df) == 0:
        raise ValueError("No valid data after filtering NaN values")

    fig, ax = plt.subplots(figsize=figsize)

    # Get distinct colors for groups
    if group_col:
        groups = sorted(plot_df[group_col].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        color_map = {g: colors[i] for i, g in enumerate(groups)}
    else:
        groups = [None]
        color_map = {None: "#1f77b4"}

    for group_val in groups:
        if group_col:
            group_df = plot_df[plot_df[group_col] == group_val]
            label_prefix = f"{group_val} "
        else:
            group_df = plot_df
            label_prefix = ""

        if len(group_df) == 0:
            continue

        color = color_map[group_val]

        # Plot individual runs if requested
        if show_individual:
            for _run_id, run_data in group_df.groupby(run_id_col):
                x_vals = run_data[x_col].values
                y_vals = run_data[metric_col].values
                if style == "line":
                    ax.plot(
                        x_vals,
                        y_vals,
                        color=color,
                        alpha=0.2,
                        linewidth=1,
                        zorder=1,
                    )
                else:  # bar
                    ax.scatter(
                        x_vals,
                        y_vals,
                        color=color,
                        alpha=0.2,
                        s=30,
                        zorder=1,
                    )

        # Compute and plot mean
        if style == "line":
            # For line plots, need to handle x-axis ordering
            x_unique = sorted(group_df[x_col].unique())
            y_means = []
            y_stds = []
            for x_val in x_unique:
                subset = group_df[group_df[x_col] == x_val]
                y_means.append(subset[metric_col].mean())
                y_stds.append(subset[metric_col].std() if len(subset) > 1 else 0.0)

            ax.plot(
                x_unique,
                y_means,
                color=color,
                linewidth=2.5,
                marker="o",
                markersize=8,
                label=f"{label_prefix}mean (n={len(group_df)})",
                zorder=2,
            )
            # Add error bars
            if len(group_df) > 1:
                ax.fill_between(
                    x_unique,
                    np.array(y_means) - np.array(y_stds),
                    np.array(y_means) + np.array(y_stds),
                    color=color,
                    alpha=0.2,
                    zorder=1,
                )
        else:  # bar
            grouped = group_df.groupby(x_col)[metric_col].agg(["mean", "std", "count"])
            x_pos = np.arange(len(grouped))
            means = grouped["mean"].values
            stds = grouped["std"].fillna(0.0).values

            width = 0.6 / len(groups) if group_col else 0.6
            offset = groups.index(group_val) * width if group_col else 0

            ax.bar(
                x_pos + offset,
                means,
                width,
                yerr=stds if len(group_df) > 1 else None,
                capsize=5,
                color=color,
                alpha=0.7,
                edgecolor="black",
                label=f"{label_prefix}mean (n={len(group_df)})",
                zorder=2,
            )

    # Formatting
    ax.set_xlabel(f"{x_col.replace('_', ' ').title()} [parameters]" if "size" in x_col.lower() else f"{x_col.replace('_', ' ').title()}", fontsize=14)
    ax.set_ylabel(f"{metric_col.replace('_', ' ').title()} [unitless]", fontsize=14)
    ax.set_title(f"{metric_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    if style == "bar" and group_col:
        # Set x-axis ticks for bar plots
        grouped_all = plot_df.groupby(x_col)[metric_col].mean()
        ax.set_xticks(np.arange(len(grouped_all)))
        ax.set_xticklabels(grouped_all.index, rotation=45, ha="right", fontsize=12)

    plt.tight_layout()
    return fig, ax


def plot_metric_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    metric_col: str,
    annotate: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot 2D heatmap of metric across parameter combinations.

    Handles: model size × PE; model size × norm policy; etc.
    Style: Color-coded cells with numeric values displayed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns row_col, col_col, metric_col
    row_col : str
        Column name for row axis (e.g., 'model_size')
    col_col : str
        Column name for column axis (e.g., 'positional', 'norm_policy')
    metric_col : str
        Column name for metric value
    annotate : bool
        If True, display numeric values on cells (default: True)
    figsize : tuple[float, float]
        Figure size (default: (10, 8))

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    # Validate columns
    required_cols = [row_col, col_col, metric_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter out NaN values
    plot_df = df.dropna(subset=required_cols).copy()

    if len(plot_df) == 0:
        raise ValueError("No valid data after filtering NaN values")

    # Create pivot table (mean aggregation)
    pivot = plot_df.pivot_table(
        index=row_col,
        columns=col_col,
        values=metric_col,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", interpolation="nearest")

    # Annotate cells with numeric values
    if annotate:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    # Choose text color based on cell value
                    text_color = "white" if val < pivot.values[~np.isnan(pivot.values)].mean() else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=11,
                        fontweight="bold",
                    )

    # Set ticks and labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=12)
    ax.set_xlabel(f"{col_col.replace('_', ' ').title()}", fontsize=14)
    ax.set_ylabel(f"{row_col.replace('_', ' ').title()}", fontsize=14)
    ax.set_title(f"{metric_col.replace('_', ' ').title()} Heatmap", fontsize=16)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{metric_col.replace('_', ' ').title()} [unitless]", fontsize=12)

    plt.tight_layout()
    return fig, ax


def plot_epoch_curves_grouped(
    df: pd.DataFrame,
    group_col: str,
    y_col: str,
    x_col: str = "epoch",
    show_individual: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot epoch curves grouped by a parameter.

    Handles: val loss / val AUROC vs epoch, grouped by size / PE / norm.
    Style: Mean line (thick, linewidth=2-3) with individual runs behind (alpha=0.15-0.3).

    Parameters
    ----------
    df : pd.DataFrame
        Per-epoch DataFrame with columns x_col, group_col, y_col, and run_id/run_dir
    x_col : str
        Column name for x-axis (default: "epoch")
    group_col : str
        Column name to group by (e.g., 'model_size', 'positional', 'norm_policy')
    y_col : str
        Column name for y-axis metric (e.g., 'val_loss', 'val_auroc')
    show_individual : bool
        If True, show individual runs behind mean (default: True)
    figsize : tuple[float, float]
        Figure size (default: (10, 8))

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    # Validate columns
    required_cols = [x_col, group_col, y_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get run identifier column
    run_id_col = "run_id" if "run_id" in df.columns else "run_dir"
    if run_id_col not in df.columns:
        raise ValueError("DataFrame must contain 'run_id' or 'run_dir' column")

    # Filter out NaN values
    plot_df = df.dropna(subset=required_cols).copy()

    if len(plot_df) == 0:
        raise ValueError("No valid data after filtering NaN values")

    fig, ax = plt.subplots(figsize=figsize)

    # Get distinct colors for groups
    groups = sorted(plot_df[group_col].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    color_map = {g: colors[i] for i, g in enumerate(groups)}

    for group_val in groups:
        group_df = plot_df[plot_df[group_col] == group_val]

        if len(group_df) == 0:
            continue

        color = color_map[group_val]

        # Plot individual runs if requested
        if show_individual:
            for _run_id, run_data in group_df.groupby(run_id_col):
                x_vals = run_data[x_col].values
                y_vals = run_data[y_col].values
                # Sort by x_col for proper line plotting
                sort_idx = np.argsort(x_vals)
                ax.plot(
                    x_vals[sort_idx],
                    y_vals[sort_idx],
                    color=color,
                    alpha=0.2,
                    linewidth=1,
                    zorder=1,
                )

        # Compute mean curve
        # Group by x_col and compute mean/std
        x_unique = sorted(group_df[x_col].unique())
        y_means = []
        y_stds = []

        for x_val in x_unique:
            subset = group_df[group_df[x_col] == x_val]
            y_means.append(subset[y_col].mean())
            y_stds.append(subset[y_col].std() if len(subset) > 1 else 0.0)

        # Plot mean line
        ax.plot(
            x_unique,
            y_means,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=6,
            label=f"{group_val} (n={len(group_df[run_id_col].unique())})",
            zorder=2,
        )

        # Add error band
        if len(group_df) > 1:
            ax.fill_between(
                x_unique,
                np.array(y_means) - np.array(y_stds),
                np.array(y_means) + np.array(y_stds),
                color=color,
                alpha=0.15,
                zorder=1,
            )

    # Formatting
    ax.set_xlabel(f"{x_col.replace('_', ' ').title()} [#]", fontsize=14)
    ax.set_ylabel(f"{y_col.replace('_', ' ').title()} [unitless]", fontsize=14)
    ax.set_title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_grouped_roc_curves(
    df: pd.DataFrame,
    group_col: str,
    n_points: int = 500,
    show_individual: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot ROC curves grouped by a parameter.

    Handles: ROC grouped by PE, by norm policy, or top models.
    Style: Mean curve (thick, linewidth=2-3) with individual curves behind (alpha=0.15-0.3).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns group_col, run_id, and 'fpr'/'tpr' columns (or 'roc_curve' dict column)
    group_col : str
        Column name to group by (e.g., 'positional', 'norm_policy')
    n_points : int
        Number of points for downsampling ROC curves (default: 500)
    show_individual : bool
        If True, show individual curves behind mean (default: True)
    figsize : tuple[float, float]
        Figure size (default: (10, 8))

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    # Validate columns
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame")

    # Check for ROC curve data
    has_fpr_tpr = "fpr" in df.columns and "tpr" in df.columns
    has_roc_dict = "roc_curve" in df.columns

    if not (has_fpr_tpr or has_roc_dict):
        raise ValueError("DataFrame must contain either ('fpr', 'tpr') columns or 'roc_curve' dict column")

    fig, ax = plt.subplots(figsize=figsize)

    # Get distinct colors for groups
    groups = sorted(df[group_col].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    color_map = {g: colors[i] for i, g in enumerate(groups)}

    for group_val in groups:
        group_df = df[df[group_col] == group_val]

        if len(group_df) == 0:
            continue

        color = color_map[group_val]
        all_fprs = []
        all_tprs = []

        # Extract ROC curves
        for _, row in group_df.iterrows():
            if has_roc_dict:
                roc_curve = row["roc_curve"]
                if isinstance(roc_curve, dict):
                    fpr = np.array(roc_curve.get("fpr", []))
                    tpr = np.array(roc_curve.get("tpr", []))
                else:
                    continue
            else:
                fpr = np.array(row["fpr"])
                tpr = np.array(row["tpr"])

            if len(fpr) == 0 or len(tpr) == 0:
                continue

            # Downsample if needed
            if len(fpr) > n_points:
                indices = np.linspace(0, len(fpr) - 1, n_points, dtype=int)
                fpr = fpr[indices]
                tpr = tpr[indices]

            all_fprs.append(fpr)
            all_tprs.append(tpr)

            # Plot individual curve if requested
            if show_individual:
                ax.plot(
                    fpr,
                    tpr,
                    color=color,
                    alpha=0.2,
                    linewidth=1,
                    zorder=1,
                )

        if len(all_fprs) == 0:
            continue

        # Compute mean curve
        # Interpolate all curves to common FPR points
        fpr_common = np.linspace(0, 1, n_points)
        tpr_interp = []

        for fpr, tpr in zip(all_fprs, all_tprs, strict=False):
            # Remove duplicates in FPR (keep last)
            unique_idx = np.unique(fpr, return_index=True)[1]
            fpr_unique = fpr[unique_idx]
            tpr_unique = tpr[unique_idx]
            # Interpolate
            tpr_interp.append(np.interp(fpr_common, fpr_unique, tpr_unique))

        tpr_mean = np.mean(tpr_interp, axis=0)
        tpr_std = np.std(tpr_interp, axis=0) if len(tpr_interp) > 1 else np.zeros_like(tpr_mean)

        # Plot mean curve
        ax.plot(
            fpr_common,
            tpr_mean,
            color=color,
            linewidth=2.5,
            label=f"{group_val} (n={len(all_fprs)})",
            zorder=2,
        )

        # Add error band
        if len(all_fprs) > 1:
            ax.fill_between(
                fpr_common,
                tpr_mean - tpr_std,
                tpr_mean + tpr_std,
                color=color,
                alpha=0.15,
                zorder=1,
            )

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (0.5)", zorder=0)

    ax.set_xlabel("False Positive Rate [unitless]", fontsize=14)
    ax.set_ylabel("True Positive Rate [unitless]", fontsize=14)
    ax.set_title("ROC Curves", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    return fig, ax


def plot_grouped_bars(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    show_points: bool = True,
    error_bars: bool = True,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot grouped bar chart comparing metrics.

    Handles: AUROC vs PE; AUROC vs norm; AUROC vs mask; etc.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns group_col, metric_col
    group_col : str
        Column name to group by (e.g., 'positional', 'norm_policy')
    metric_col : str
        Column name for metric value
    show_points : bool
        If True, show individual data points (default: True)
    error_bars : bool
        If True, show error bars (std) (default: True)
    figsize : tuple[float, float]
        Figure size (default: (10, 6))

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    # Validate columns
    required_cols = [group_col, metric_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter out NaN values
    plot_df = df.dropna(subset=required_cols).copy()

    if len(plot_df) == 0:
        raise ValueError("No valid data after filtering NaN values")

    # Group and aggregate
    grouped = plot_df.groupby(group_col)[metric_col].agg(["mean", "std", "count"])

    groups = grouped.index.tolist()
    means = grouped["mean"].values
    stds = grouped["std"].fillna(0.0).values
    counts = grouped["count"].values

    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(groups))
    width = 0.6

    # Plot bars
    ax.bar(
        x_pos,
        means,
        width,
        yerr=stds if error_bars and len(plot_df) > 1 else None,
        capsize=5,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        zorder=2,
    )

    # Show individual points if requested
    if show_points:
        for i, group_val in enumerate(groups):
            group_data = plot_df[plot_df[group_col] == group_val][metric_col].values
            x_jitter = np.random.normal(i, 0.05, len(group_data))
            ax.scatter(
                x_jitter,
                group_data,
                color="black",
                alpha=0.3,
                s=20,
                zorder=3,
            )

    # Add value labels on bars
    for i, (mean, std, count) in enumerate(zip(means, stds, counts, strict=False)):
        label_y = mean + std + 0.01 if error_bars else mean + 0.01
        ax.text(
            i,
            label_y,
            f"{mean:.3f}\n(n={int(count)})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel(f"{group_col.replace('_', ' ').title()}", fontsize=14)
    ax.set_ylabel(f"{metric_col.replace('_', ' ').title()} [unitless]", fontsize=14)
    ax.set_title(f"{metric_col.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}", fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(g) for g in groups], rotation=45, ha="right", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_score_distributions_df(
    df: pd.DataFrame,
    layout: str = "single",
    group_col: str | None = None,
    signal_class_idx: int = 1,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes] | list[tuple[plt.Figure, plt.Axes]]:
    """Plot score distributions from DataFrame (DataFrame-based API).

    Extends existing function to support grid layout.
    Grid style: Clean, small subplots arranged in grid.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'run_id' (or 'run_dir'), 'per_event_scores', 'per_event_labels'
        OR flattened columns: 'scores', 'labels' per run
    layout : str
        Layout style: "single" or "grid" (default: "single")
    group_col : str | None
        Optional column to group by for grid layout (default: None)
    signal_class_idx : int
        Which class index represents signal (default: 1)
    figsize : tuple[float, float] | None
        Figure size (default: None, auto-determined)

    Returns
    -------
    tuple[plt.Figure, plt.Axes] | list[tuple[plt.Figure, plt.Axes]]
        Single figure/axes for "single" layout, or list for "grid" layout
    """

    # Get run identifier
    run_id_col = "run_id" if "run_id" in df.columns else "run_dir"
    if run_id_col not in df.columns:
        raise ValueError("DataFrame must contain 'run_id' or 'run_dir' column")

    # Check for score/label columns
    has_per_event = "per_event_scores" in df.columns and "per_event_labels" in df.columns
    has_flat = "scores" in df.columns and "labels" in df.columns

    if not (has_per_event or has_flat):
        raise ValueError("DataFrame must contain ('per_event_scores', 'per_event_labels') or ('scores', 'labels') columns")

    if layout == "single":
        # Single plot - plot all runs on one figure
        if figsize is None:
            figsize = (10, 6)

        fig, ax = plt.subplots(figsize=figsize)

        for run_id, run_data in df.groupby(run_id_col):
            if has_per_event:
                scores = np.array(run_data["per_event_scores"].iloc[0])
                labels = np.array(run_data["per_event_labels"].iloc[0])
            else:
                scores = np.array(run_data["scores"].values)
                labels = np.array(run_data["labels"].values)

            if len(scores) == 0 or len(labels) == 0:
                continue

            signal_scores = scores[labels == signal_class_idx]
            background_scores = scores[labels != signal_class_idx]

            if len(signal_scores) == 0 or len(background_scores) == 0:
                continue

            # Get class names
            unique_labels = np.unique(labels)
            background_class_idx = unique_labels[unique_labels != signal_class_idx][0] if len(unique_labels) > 1 else None
            class_names = _get_class_names(n_classes=2, signal_class_idx=signal_class_idx, background_class_idx=background_class_idx)
            signal_name = class_names[signal_class_idx]
            background_name = class_names[background_class_idx] if background_class_idx is not None else class_names[1 - signal_class_idx]

            # Compute bins
            score_min = min(scores.min(), background_scores.min(), signal_scores.min())
            score_max = max(scores.max(), background_scores.max(), signal_scores.max())
            bins = np.linspace(score_min, score_max, 50)

            # Compute normalized histograms
            bg_counts, bg_edges = np.histogram(background_scores, bins=bins)
            sig_counts, sig_edges = np.histogram(signal_scores, bins=bins)

            bg_normalized = bg_counts / len(background_scores)
            sig_normalized = sig_counts / len(signal_scores)

            # Plot
            ax.step(
                bg_edges[:-1],
                bg_normalized,
                where="post",
                label=f"{run_id} - {background_name}",
                linewidth=1.5,
                alpha=0.6,
            )
            ax.step(
                sig_edges[:-1],
                sig_normalized,
                where="post",
                label=f"{run_id} - {signal_name}",
                linewidth=1.5,
                alpha=0.6,
            )

        ax.set_xlabel("Classifier Score (p(signal | event)) [probability]", fontsize=14)
        ax.set_ylabel("bin count / N [unitless]", fontsize=14)
        ax.set_title("Score Distributions", fontsize=16)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig, ax

    else:  # grid layout
        # Determine grid size
        if group_col:
            groups = sorted(df[group_col].dropna().unique())
            n_plots = len(groups)
        else:
            groups = df[run_id_col].unique()
            n_plots = len(groups)

        if n_plots == 0:
            raise ValueError("No valid groups found for grid layout")

        # Calculate grid dimensions
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (4 * n_cols, 3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        plot_idx = 0
        for group_val in groups:
            if plot_idx >= len(axes_flat):
                break

            ax = axes_flat[plot_idx]

            group_df = df[df[group_col] == group_val] if group_col else df[df[run_id_col] == group_val].head(1)

            if len(group_df) == 0:
                plot_idx += 1
                continue

            # Get first run's data
            run_data = group_df.iloc[0]

            if has_per_event:
                scores = np.array(run_data["per_event_scores"])
                labels = np.array(run_data["per_event_labels"])
            else:
                # For flat format, need to get all rows for this run
                run_df = df[df[group_col] == group_val] if group_col else df[df[run_id_col] == run_data[run_id_col]]
                scores = np.array(run_df["scores"].values)
                labels = np.array(run_df["labels"].values)

            if len(scores) == 0 or len(labels) == 0:
                plot_idx += 1
                continue

            signal_scores = scores[labels == signal_class_idx]
            background_scores = scores[labels != signal_class_idx]

            if len(signal_scores) == 0 or len(background_scores) == 0:
                plot_idx += 1
                continue

            # Get class names
            unique_labels = np.unique(labels)
            background_class_idx = unique_labels[unique_labels != signal_class_idx][0] if len(unique_labels) > 1 else None
            class_names = _get_class_names(n_classes=2, signal_class_idx=signal_class_idx, background_class_idx=background_class_idx)
            signal_name = class_names[signal_class_idx]
            background_name = class_names[background_class_idx] if background_class_idx is not None else class_names[1 - signal_class_idx]

            # Compute bins
            score_min = min(scores.min(), background_scores.min(), signal_scores.min())
            score_max = max(scores.max(), background_scores.max(), signal_scores.max())
            bins = np.linspace(score_min, score_max, 30)  # Fewer bins for small subplots

            # Compute normalized histograms
            bg_counts, bg_edges = np.histogram(background_scores, bins=bins)
            sig_counts, sig_edges = np.histogram(signal_scores, bins=bins)

            bg_normalized = bg_counts / len(background_scores)
            sig_normalized = sig_counts / len(signal_scores)

            # Plot
            ax.step(
                bg_edges[:-1],
                bg_normalized,
                where="post",
                label=background_name,
                color="#1f77b4",
                linewidth=1.5,
                alpha=0.8,
            )
            ax.step(
                sig_edges[:-1],
                sig_normalized,
                where="post",
                label=signal_name,
                color="#ff7f0e",
                linewidth=1.5,
                alpha=0.8,
            )

            # Title
            title = str(group_val) if group_col else str(run_data[run_id_col])
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Score [probability]", fontsize=9)
            ax.set_ylabel("Density [unitless]", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes_flat)):
            axes_flat[i].axis("off")

        plt.tight_layout()
        return fig, axes


def plot_roc_curves_grouped_by(
    runs_df: pd.DataFrame,
    inference_results: dict[str, dict[str, Any]],
    group_col: str,
    inference_figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-roc_curves_grouped",
    title: str | None = None,
    show_individual: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Plot ROC curves grouped by a parameter (PE type, norm policy, etc.).

    Shows mean ROC curve per group with optional individual curves behind.
    Similar to notebook-style grouped ROC plots.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information, must contain group_col and run_dir
    inference_results : dict[str, dict[str, Any]]
        Inference results dict: {run_id: {metrics...}}
    group_col : str
        Column name to group by (e.g., 'positional', 'norm_policy')
    inference_figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    fname : str
        Base filename for saved figure
    title : str | None
        Plot title (auto-generated if None)
    show_individual : bool
        If True, show individual curves behind mean (default: True)
    figsize : tuple[float, float]
        Figure size (default: (10, 8))
    """
    from thesis_ml.utils.paths import get_run_id

    if group_col not in runs_df.columns:
        logger.warning(f"Column '{group_col}' not found in runs_df, skipping ROC grouped plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Get unique group values and assign colors
    groups = sorted(runs_df[group_col].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    color_map = {g: colors[i] for i, g in enumerate(groups)}

    n_points = 500  # Number of points for interpolation

    for group_val in groups:
        group_runs = runs_df[runs_df[group_col] == group_val]
        color = color_map[group_val]

        all_fprs = []
        all_tprs = []
        all_aurocs = []

        for _, row in group_runs.iterrows():
            run_dir = row.get("run_dir")
            if pd.isna(run_dir):
                continue
            run_id = get_run_id(Path(str(run_dir)))

            if run_id not in inference_results:
                continue

            metrics = inference_results[run_id]
            roc_curves = metrics.get("roc_curves", {})
            auroc = metrics.get("auroc", 0.0)

            if not roc_curves:
                continue

            # For binary classification, use single curve
            if len(roc_curves) == 1:
                class_idx = list(roc_curves.keys())[0]
                curve = roc_curves[class_idx]
            else:
                # Use first class for multi-class
                class_idx = list(roc_curves.keys())[0]
                curve = roc_curves[class_idx]

            fpr = np.array(curve.get("fpr", []))
            tpr = np.array(curve.get("tpr", []))

            if len(fpr) == 0 or len(tpr) == 0:
                continue

            all_fprs.append(fpr)
            all_tprs.append(tpr)
            all_aurocs.append(auroc)

            # Plot individual curve if requested
            if show_individual:
                ax.plot(fpr, tpr, color=color, alpha=0.2, linewidth=1, zorder=1)

        if len(all_fprs) == 0:
            continue

        # Compute mean curve by interpolating to common FPR points
        fpr_common = np.linspace(0, 1, n_points)
        tpr_interp = []

        for fpr, tpr in zip(all_fprs, all_tprs, strict=False):
            # Remove duplicates in FPR (keep last)
            unique_idx = np.unique(fpr, return_index=True)[1]
            fpr_unique = fpr[unique_idx]
            tpr_unique = tpr[unique_idx]
            # Interpolate
            tpr_interp.append(np.interp(fpr_common, fpr_unique, tpr_unique))

        tpr_mean = np.mean(tpr_interp, axis=0)
        tpr_std = np.std(tpr_interp, axis=0) if len(tpr_interp) > 1 else np.zeros_like(tpr_mean)
        auroc_mean = np.mean(all_aurocs) if all_aurocs else 0.0

        # Plot mean curve
        ax.plot(
            fpr_common,
            tpr_mean,
            color=color,
            linewidth=2.5,
            label=f"{group_val} (n={len(all_fprs)}, AUROC={auroc_mean:.3f})",
            zorder=2,
        )

        # Add error band
        if show_individual and len(all_fprs) > 1:
            ax.fill_between(
                fpr_common,
                tpr_mean - tpr_std,
                tpr_mean + tpr_std,
                color=color,
                alpha=0.15,
                zorder=1,
            )

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (0.5)", zorder=0)

    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(title or f"ROC Curves by {group_col.replace('_', ' ').title()}", fontsize=16)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    save_figure(fig, inference_figs_dir, fname, fig_cfg)
    plt.close(fig)


def plot_auroc_bar_by_group(
    runs_df: pd.DataFrame,
    inference_results: dict[str, dict[str, Any]],
    group_col: str,
    inference_figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-auroc_by_group",
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """Plot AUROC bar chart grouped by a parameter.

    Shows mean AUROC per group with error bars and individual points.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information, must contain group_col and run_dir
    inference_results : dict[str, dict[str, Any]]
        Inference results dict: {run_id: {metrics...}}
    group_col : str
        Column name to group by (e.g., 'positional', 'norm_policy')
    inference_figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    fname : str
        Base filename for saved figure
    title : str | None
        Plot title (auto-generated if None)
    figsize : tuple[float, float]
        Figure size (default: (10, 6))
    """
    from thesis_ml.utils.paths import get_run_id

    if group_col not in runs_df.columns:
        logger.warning(f"Column '{group_col}' not found in runs_df, skipping AUROC bar plot")
        return

    # Collect AUROC values by group
    group_aurocs: dict[str, list[float]] = {}

    for _, row in runs_df.iterrows():
        run_dir = row.get("run_dir")
        group_val = row.get(group_col)

        if pd.isna(run_dir) or pd.isna(group_val):
            continue

        run_id = get_run_id(Path(str(run_dir)))
        if run_id not in inference_results:
            continue

        auroc = inference_results[run_id].get("auroc")
        if auroc is not None:
            if group_val not in group_aurocs:
                group_aurocs[group_val] = []
            group_aurocs[group_val].append(auroc)

    if not group_aurocs:
        logger.warning("No AUROC data found for grouping, skipping plot")
        return

    # Compute statistics
    groups = sorted(group_aurocs.keys())
    means = [np.mean(group_aurocs[g]) for g in groups]
    stds = [np.std(group_aurocs[g]) if len(group_aurocs[g]) > 1 else 0.0 for g in groups]
    counts = [len(group_aurocs[g]) for g in groups]

    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(groups))
    width = 0.6

    # Plot bars
    ax.bar(
        x_pos,
        means,
        width,
        yerr=stds if any(s > 0 for s in stds) else None,
        capsize=5,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        zorder=2,
    )

    # Show individual points
    for i, group_val in enumerate(groups):
        group_data = group_aurocs[group_val]
        x_jitter = np.random.normal(i, 0.05, len(group_data))
        ax.scatter(x_jitter, group_data, color="black", alpha=0.3, s=20, zorder=3)

    # Add value labels on bars
    for i, (mean, std, count) in enumerate(zip(means, stds, counts, strict=False)):
        label_y = mean + std + 0.01 if std > 0 else mean + 0.01
        ax.text(i, label_y, f"{mean:.3f}\n(n={count})", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=14)
    ax.set_ylabel("AUROC", fontsize=14)
    ax.set_title(title or f"AUROC by {group_col.replace('_', ' ').title()}", fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(g) for g in groups], rotation=45, ha="right", fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, inference_figs_dir, fname, fig_cfg)
    plt.close(fig)
