"""Classification plotting functions (matching anomaly detection style)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from thesis_ml.monitoring.io_utils import save_figure


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

    colors = plt.cm.tab10(np.linspace(0, 1, len(inference_results)))

    for i, (run_id, metrics) in enumerate(inference_results.items()):
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

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
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
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion Matrix: {run_id}",
        )

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
    run_ids = sorted(inference_results.keys())
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

    ax.set_xlabel("Model (Run ID)")
    ax.set_ylabel("Score")
    ax.set_title("Classification Performance Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.legend()
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
            label="Background",
            color="#1f77b4",  # Blue
            linewidth=2,
            alpha=0.8,
        )
        ax.step(
            sig_edges[:-1],
            sig_normalized,
            where="post",
            label="Signal",
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
        ax.set_xlabel("Classifier Score (p(signal | event))")
        ax.set_ylabel("bin count / N")
        ax.set_title(f"Score Distributions: {run_id}")
        ax.legend()
        ax.grid(alpha=0.3)

        # Set x-axis limits to tightly cover the distributions
        ax.set_xlim([score_min, score_max])

        plt.tight_layout()
        save_figure(fig, inference_figs_dir, f"{fname}_{run_id}", fig_cfg)
        plt.close(fig)
