"""Anomaly detection plotting functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from thesis_ml.monitoring.io_utils import save_figure
from thesis_ml.reports.plots.style import apply_thesis_style, figure_size, CATEGORICAL_COLORS

apply_thesis_style()


def plot_reconstruction_error_distributions(
    inference_results: dict[str, dict[str, dict[str, Any]]],
    inference_figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-reconstruction_error_distributions",
) -> None:
    """Plot histogram overlays of reconstruction errors: baseline vs corruption strategies.

    Parameters
    ----------
    inference_results : dict[str, dict[str, dict[str, Any]]]
        Nested dict: {run_id: {strategy_name: {metrics...}}}
    inference_figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    fname : str
        Base filename for saved figure
    """
    strategies = set()
    for model_results in inference_results.values():
        strategies.update(model_results.keys())

    if "baseline" not in strategies:
        return

    strategies.remove("baseline")
    strategies = sorted(strategies)

    for run_id, model_results in inference_results.items():
        if "baseline" not in model_results:
            continue

        baseline_mse = model_results["baseline"].get("mse_mean", 0.0)

        fig, ax = plt.subplots(figsize=figure_size("full"))

        ax.axvline(baseline_mse, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Baseline (mean)")

        colors = (CATEGORICAL_COLORS * (len(strategies) // len(CATEGORICAL_COLORS) + 1))[:len(strategies)]
        for i, strategy_name in enumerate(strategies):
            if strategy_name not in model_results:
                continue

            strategy_mse = model_results[strategy_name].get("mse_mean", 0.0)
            strategy_std = model_results[strategy_name].get("mse_std", 0.0)

            ax.barh(i, strategy_mse, xerr=strategy_std, color=colors[i], alpha=0.7, label=strategy_name)

        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies)
        ax.set_xlabel("MSE (Mean ± Std)")
        ax.legend()

        save_figure(fig, inference_figs_dir, f"{fname}_{run_id}", fig_cfg)
        plt.close(fig)


def plot_model_comparison(
    inference_results: dict[str, dict[str, dict[str, Any]]],
    inference_figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-model_comparison_bars",
) -> None:
    """Bar chart comparing models on different corruption types.

    Parameters
    ----------
    inference_results : dict[str, dict[str, dict[str, Any]]]
        Nested dict: {run_id: {strategy_name: {metrics...}}}
    inference_figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    fname : str
        Base filename for saved figure
    """
    strategies = set()
    for model_results in inference_results.values():
        strategies.update(model_results.keys())

    if "baseline" in strategies:
        strategies.remove("baseline")
    strategies = sorted(strategies)

    if not strategies:
        return

    run_ids = sorted(inference_results.keys())
    x = np.arange(len(run_ids))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=figure_size("full"))

    colors = (CATEGORICAL_COLORS * (len(strategies) // len(CATEGORICAL_COLORS) + 1))[:len(strategies)]

    for i, strategy_name in enumerate(strategies):
        mse_values = []
        for run_id in run_ids:
            if strategy_name in inference_results[run_id]:
                mse_values.append(inference_results[run_id][strategy_name].get("mse_mean", 0.0))
            else:
                mse_values.append(0.0)

        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(x + offset, mse_values, width, label=strategy_name, color=colors[i], alpha=0.7)

    ax.set_xlabel("Model (Run ID)")
    ax.set_ylabel("MSE (Mean)")
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    save_figure(fig, inference_figs_dir, fname, fig_cfg)
    plt.close(fig)


def plot_auroc_comparison(
    inference_results: dict[str, dict[str, dict[str, Any]]],
    inference_figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-auroc_comparison",
) -> None:
    """Bar chart showing AUROC per model/strategy.

    Parameters
    ----------
    inference_results : dict[str, dict[str, dict[str, Any]]]
        Nested dict: {run_id: {strategy_name: {metrics...}}}
    inference_figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    fname : str
        Base filename for saved figure
    """
    strategies = set()
    for model_results in inference_results.values():
        strategies.update(model_results.keys())

    if "baseline" in strategies:
        strategies.remove("baseline")
    strategies = sorted(strategies)

    if not strategies:
        return

    run_ids = sorted(inference_results.keys())
    x = np.arange(len(run_ids))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=figure_size("full"))

    colors = (CATEGORICAL_COLORS * (len(strategies) // len(CATEGORICAL_COLORS) + 1))[:len(strategies)]

    for i, strategy_name in enumerate(strategies):
        auroc_values = []
        for run_id in run_ids:
            if strategy_name in inference_results[run_id]:
                auroc = inference_results[run_id][strategy_name].get("auroc")
                auroc_values.append(auroc if auroc is not None else 0.0)
            else:
                auroc_values.append(0.0)

        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(x + offset, auroc_values, width, label=strategy_name, color=colors[i], alpha=0.7)

    ax.set_xlabel("Model (Run ID)")
    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Random (0.5)")
    ax.set_ylim([0, 1])
    ax.legend()

    plt.tight_layout()
    save_figure(fig, inference_figs_dir, fname, fig_cfg)
    plt.close(fig)
