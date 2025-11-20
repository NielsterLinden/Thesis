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


def _get_class_names(signal_class_idx: int, background_class_idx: int | None = None) -> tuple[str, str]:
    """Get class names for signal and background.

    Default mapping based on ProcessIDs:
    - ProcessID 1 → "4t" (signal)
    - ProcessID 2 → "ttH" (background)

    Parameters
    ----------
    signal_class_idx : int
        Class index for signal (0-indexed)
    background_class_idx : int | None
        Class index for background (0-indexed), if None uses the other class

    Returns
    -------
    tuple[str, str]
        (signal_name, background_name)
    """
    # Default mapping: ProcessID 1 → "4t", ProcessID 2 → "ttH"
    # Note: signal_class_idx is 0-indexed, but ProcessIDs are 1-indexed
    # For binary classification with selected_labels=[1, 2]:
    #   - signal_class_idx=1 maps to ProcessID 2 → "ttH"
    #   - signal_class_idx=0 maps to ProcessID 1 → "4t"

    # Common mapping: 0 → "4t", 1 → "ttH"
    class_name_map = {0: "4t", 1: "ttH"}

    signal_name = class_name_map.get(signal_class_idx, f"Signal-{signal_class_idx}")

    if background_class_idx is None:
        # Use the other class
        background_class_idx = 1 - signal_class_idx if signal_class_idx < 2 else 0

    background_name = class_name_map.get(background_class_idx, f"Background-{background_class_idx}")

    return signal_name, background_name


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

        fig, ax = plt.subplots(figsize=(8, 6))
        # Set colorbar limits to [0, 1] for normalized confusion matrix
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label("Normalized Count", fontsize=12)

        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion Matrix: {run_id}",
        )
        ax.set_xlabel("Predicted", fontsize=14)
        ax.set_ylabel("True", fontsize=14)
        ax.set_title(f"Confusion Matrix: {run_id}", fontsize=16)

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

        signal_name, background_name = _get_class_names(signal_class_idx, background_class_idx)

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
