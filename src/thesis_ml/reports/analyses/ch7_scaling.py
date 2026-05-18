"""Chapter 7 report: Model Scaling and Efficiency.

Covers two experiments:

Exp 8A — Scaling curves:
  - H01: classifier.model.dim   ∈ {32, 32, 64, 128, 192}
  - H02: classifier.model.depth ∈ {3, 5, 6, 8, 12}
  - H10: model size label (derived) ∈ {d32_L3, d32_L5, d64_L6, d128_L8, d192_L12}

Exp 8B — Efficiency / Pareto:
  AUROC vs. FLOPs / latency / throughput / peak memory.
  Same 45 runs as 8A — no additional data needed.

Exp 8C — Axis transfer: BLOCKED. Not staged here.

Entry point: C — all metrics sourced from the canonical cleaned analysis CSV
(thesis_results/04_cleaned_backfilled_analysis_ready.csv). No inference or
training is required.

Naming note: W&B groups and run configs use "ch8_scaling" (chapter numbering
shifted during writing). This module uses "ch7" to match the thesis text.

Plots generated
---------------
Exp 8A (scaling curves):
  - auroc_vs_model_size
  - flops_vs_model_size
  - wallclock_vs_model_size
  - best_epoch_vs_model_size

Exp 8B (efficiency / Pareto):
  - pareto_auroc_vs_flops
  - pareto_auroc_vs_latency
  - pareto_auroc_vs_throughput
  - pareto_auroc_vs_memory

Usage
-----
thesis-report --config-name thesis_experiments_reports/ch7_scaling \\
  inputs.csv_path=thesis_results/04_cleaned_backfilled_analysis_ready.csv \\
  env.output_root=/data/atlas/users/nterlind/outputs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from thesis_ml.monitoring.io_utils import save_figure
from thesis_ml.reports.plots.style import (
    AXIS_GROUP_COLORS,
    CATEGORICAL_COLORS,
    apply_thesis_style,
    axis_color,
    figure_size,
)
from thesis_ml.reports.utils.io import ensure_report_dirs, resolve_report_output_dir

logger = logging.getLogger(__name__)

apply_thesis_style()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Ordered size labels (smallest → largest)
_SIZE_ORDER: list[str] = ["d32_L3", "d32_L5", "d64_L6", "d128_L8", "d192_L12"]

# Approximate parameter counts for x-axis labels
_SIZE_PARAMS: dict[str, int] = {
    "d32_L3":  26_000,
    "d32_L5":  43_000,
    "d64_L6":  202_000,
    "d128_L8": 1_060_000,
    "d192_L12": 3_570_000,
}

# Human-readable size labels for plot annotations
_SIZE_LABELS: dict[str, str] = {
    "d32_L3":   "d32 L3\n(26k)",
    "d32_L5":   "d32 L5\n(43k)",
    "d64_L6":   "d64 L6\n(202k)",
    "d128_L8":  "d128 L8\n(1.1M)",
    "d192_L12": "d192 L12\n(3.6M)",
}

# Task display names derived from W&B group suffix
_TASK_LABELS: dict[str, str] = {
    "4t_vs_bg":   "4t vs. bg",
    "4t_vs_ttH":  "4t vs. ttH",
    "multiclass":  "Multiclass",
}

# One colour per task (three categorical colors)
_TASK_COLORS: dict[str, str] = {
    "4t_vs_bg":   CATEGORICAL_COLORS[0],   # D — dark blue
    "4t_vs_ttH":  CATEGORICAL_COLORS[1],   # T — sky blue
    "multiclass":  CATEGORICAL_COLORS[2],   # E — green
}

# Column name aliases (canonical CSV)
_COL_SIZE = "config/axes/H10_Model Size Label"
_COL_AUROC = "eval_v2/test_auroc"
_COL_FLOPS = "eval_v2/flops_per_event_analytic"
_COL_LATENCY_B512 = "eval_v2/inference_latency_ms_b512_mean"
_COL_THROUGHPUT = "eval_v2/throughput_samples_per_s_b512"
_COL_MEMORY = "eval_v2/peak_memory_mib_inference_b512"
_COL_WALLCLOCK = "eval_v2/checkpoint_size_mb"   # placeholder — see _get_wallclock_col
_COL_EPOCH = "eval_v2/checkpoint_epoch"
_COL_GROUP = "meta_run/group"


def _get_wallclock_col(df: pd.DataFrame) -> str | None:
    """Return the wall-clock column name if present."""
    for candidate in [
        "eval_v2/runtime_seconds",
        "eval_v2/training_wallclock_s",
        "eval_v2/wallclock_s",
        "eval_v2/train_wallclock_s",
        "training_wallclock_s",
        "wallclock_s",
    ]:
        if candidate in df.columns:
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_ch7_data(cfg: DictConfig) -> pd.DataFrame:
    """Load ch8_scaling rows from the canonical CSV.

    Derives a ``task`` column from the W&B group name suffix
    (e.g. ``exp_..._4t_vs_bg`` → ``"4t_vs_bg"``).
    """
    csv_path = Path(cfg.inputs.csv_path)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    prefix = cfg.inputs.get("group_prefix", "ch8_scaling")
    mask = df[_COL_GROUP].str.contains(prefix, na=False)
    df = df[mask].copy()

    if df.empty:
        raise RuntimeError(f"No rows matching group_prefix={prefix!r} in {csv_path}")

    logger.info("Loaded %d ch7 rows from %s", len(df), csv_path)

    # Derive task column from group name suffix
    def _task_from_group(g: str) -> str:
        for suffix in ["4t_vs_bg", "4t_vs_ttH", "multiclass"]:
            if suffix in g:
                return suffix
        return g.split("_")[-1]

    df["task"] = df[_COL_GROUP].apply(_task_from_group)

    # Coerce numeric metric columns (some CSVs store floats as strings)
    _numeric_cols = [
        _COL_AUROC, _COL_FLOPS, _COL_LATENCY_B512, _COL_THROUGHPUT,
        _COL_MEMORY, _COL_EPOCH,
        "eval_v2/runtime_seconds",
    ]
    for col in _numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure size is in expected order
    df[_COL_SIZE] = pd.Categorical(df[_COL_SIZE], categories=_SIZE_ORDER, ordered=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Exp 8A helpers
# ─────────────────────────────────────────────────────────────────────────────


def _agg_by_size_task(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Return mean ± std grouped by (size, task)."""
    return (
        df.dropna(subset=[_COL_SIZE, value_col])
        .groupby([_COL_SIZE, "task"], observed=True)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def _agg_by_size(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Return mean ± std grouped by size only (all tasks + seeds)."""
    return (
        df.dropna(subset=[_COL_SIZE, value_col])
        .groupby(_COL_SIZE, observed=True)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Exp 8A plots
# ─────────────────────────────────────────────────────────────────────────────


def _plot_auroc_vs_model_size(
    df: pd.DataFrame,
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-auroc_vs_model_size",
) -> None:
    """AUROC by model size — one line per task, with seed/task-aggregated overlay.

    Left panel: per-task mean ± std across 3 seeds.
    Right panel: aggregated (all 3 tasks × 3 seeds) mean ± std.
    """
    x_pos = np.arange(len(_SIZE_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(figure_size("full")[0], figure_size("full")[1]))

    # ── Left: per-task ──
    ax = axes[0]
    for task in ["4t_vs_bg", "4t_vs_ttH", "multiclass"]:
        agg = _agg_by_size_task(df[df["task"] == task], _COL_AUROC)
        agg = agg.set_index(_COL_SIZE).reindex(_SIZE_ORDER)
        means = agg["mean"].values
        stds = agg["std"].fillna(0).values
        color = _TASK_COLORS[task]
        ax.plot(x_pos, means, "o-", color=color, label=_TASK_LABELS[task], linewidth=1.5)
        ax.fill_between(x_pos, means - stds, means + stds, color=color, alpha=0.15)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(_SIZE_ORDER, rotation=20, ha="right")
    ax.set_xlabel("Model size")
    ax.set_ylabel("Test AUROC")
    ax.legend(title="Task")

    # ── Right: aggregated ──
    ax = axes[1]
    agg_all = _agg_by_size(df, _COL_AUROC).set_index(_COL_SIZE).reindex(_SIZE_ORDER)
    means = agg_all["mean"].values
    stds = agg_all["std"].fillna(0).values
    color = axis_color("H")
    ax.plot(x_pos, means, "o-", color=color, linewidth=2.0)
    ax.fill_between(x_pos, means - stds, means + stds, color=color, alpha=0.18)
    # Individual seed dots
    for i, size in enumerate(_SIZE_ORDER):
        pts = df[df[_COL_SIZE] == size][_COL_AUROC].dropna().values
        ax.scatter(np.full(len(pts), i), pts, color=color, s=14, alpha=0.45, zorder=4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(_SIZE_ORDER, rotation=20, ha="right")
    ax.set_xlabel("Model size")
    ax.set_ylabel("Test AUROC (aggregated)")

    fig.tight_layout()
    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
    logger.info("Saved %s", fname)


def _plot_flops_vs_model_size(
    df: pd.DataFrame,
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-flops_vs_model_size",
) -> None:
    """FLOPs/event (analytic) vs. model size — log-log axes."""
    agg = _agg_by_size(df, _COL_FLOPS).set_index(_COL_SIZE).reindex(_SIZE_ORDER)
    params = [_SIZE_PARAMS[s] for s in _SIZE_ORDER]

    fig, ax = plt.subplots(figsize=figure_size("half"))
    ax.loglog(params, agg["mean"].values, "o-", color=axis_color("H"), linewidth=1.8)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("FLOPs / event (analytic)")
    _annotate_sizes(ax, params, agg["mean"].values)

    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
    logger.info("Saved %s", fname)


def _plot_wallclock_vs_model_size(
    df: pd.DataFrame,
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-wallclock_vs_model_size",
) -> None:
    """Training wall-clock (s) vs. model size — log y-axis."""
    wc_col = _get_wallclock_col(df)
    if wc_col is None:
        logger.warning("No wall-clock column found in CSV — skipping wallclock plot")
        return

    x_pos = np.arange(len(_SIZE_ORDER))
    agg = _agg_by_size(df, wc_col).set_index(_COL_SIZE).reindex(_SIZE_ORDER)
    means = agg["mean"].values
    stds = agg["std"].fillna(0).values

    fig, ax = plt.subplots(figsize=figure_size("half"))
    color = axis_color("H")
    ax.errorbar(x_pos, means, yerr=stds, fmt="o-", color=color, capsize=3, linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_SIZE_ORDER, rotation=20, ha="right")
    ax.set_xlabel("Model size")
    ax.set_ylabel("Training wall-clock (s)")

    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
    logger.info("Saved %s", fname)


def _plot_best_epoch_vs_model_size(
    df: pd.DataFrame,
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-best_epoch_vs_model_size",
) -> None:
    """Best checkpoint epoch vs. model size — shows convergence speed."""
    x_pos = np.arange(len(_SIZE_ORDER))
    agg = _agg_by_size(df, _COL_EPOCH).set_index(_COL_SIZE).reindex(_SIZE_ORDER)
    means = agg["mean"].values
    stds = agg["std"].fillna(0).values

    fig, ax = plt.subplots(figsize=figure_size("half"))
    color = axis_color("H")
    ax.errorbar(x_pos, means, yerr=stds, fmt="o-", color=color, capsize=3, linewidth=1.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_SIZE_ORDER, rotation=20, ha="right")
    ax.set_xlabel("Model size")
    ax.set_ylabel("Best checkpoint epoch")

    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
    logger.info("Saved %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Exp 8B plots — Pareto / efficiency
# ─────────────────────────────────────────────────────────────────────────────


def _annotate_sizes(
    ax: plt.Axes,
    xs: list | np.ndarray,
    ys: list | np.ndarray,
    sizes: list[str] | None = None,
    offset: tuple[float, float] = (0, 6),
) -> None:
    """Add size-label text annotations above each point."""
    if sizes is None:
        sizes = _SIZE_ORDER
    for x, y, s in zip(xs, ys, sizes):
        ax.annotate(
            s,
            (x, y),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            fontsize=7,
        )


def _plot_pareto(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    fname: str,
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    x_log: bool = False,
) -> None:
    """Scatter of mean AUROC vs. mean efficiency metric, one point per size.

    Points are coloured by task; aggregated (all-task) mean is drawn as a
    larger grey marker with the size label annotated.
    """
    fig, ax = plt.subplots(figsize=figure_size("two_thirds"))

    # Per-task scatter (small dots)
    for task in ["4t_vs_bg", "4t_vs_ttH", "multiclass"]:
        sub = df[df["task"] == task]
        agg = _agg_by_size_task(sub, _COL_AUROC).rename(columns={"mean": "auroc_mean"})
        agg_x = _agg_by_size_task(sub, x_col).rename(columns={"mean": "x_mean"})
        merged = pd.merge(agg[[_COL_SIZE, "auroc_mean"]], agg_x[[_COL_SIZE, "x_mean"]], on=_COL_SIZE)
        merged = merged.set_index(_COL_SIZE).reindex(_SIZE_ORDER).dropna()
        ax.scatter(
            merged["x_mean"].values,
            merged["auroc_mean"].values,
            color=_TASK_COLORS[task],
            label=_TASK_LABELS[task],
            s=30,
            alpha=0.7,
            zorder=3,
        )

    # Aggregated (all-task) mean — larger grey markers + labels
    agg_auroc = _agg_by_size(df, _COL_AUROC).set_index(_COL_SIZE).reindex(_SIZE_ORDER)
    agg_x_all = _agg_by_size(df, x_col).set_index(_COL_SIZE).reindex(_SIZE_ORDER)
    xs = agg_x_all["mean"].values
    ys = agg_auroc["mean"].values
    ax.scatter(xs, ys, color=axis_color("H"), s=60, zorder=5, label="Aggregated mean")
    _annotate_sizes(ax, xs, ys)

    if x_log:
        ax.set_xscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Test AUROC")
    ax.legend(fontsize=7)

    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
    logger.info("Saved %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Report entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_report(cfg: DictConfig) -> None:
    """Chapter 7 report: Model Scaling and Efficiency.

    Reads from the canonical cleaned analysis CSV (Entry Point C).
    No inference or training required.

    Run via::

        thesis-report --config-name thesis_experiments_reports/ch7_scaling \\
            inputs.csv_path=thesis_results/04_cleaned_backfilled_analysis_ready.csv \\
            env.output_root=/data/atlas/users/nterlind/outputs
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # ── Load data ──────────────────────────────────────────────────────────────
    df = _load_ch7_data(cfg)

    # ── Resolve output dirs ────────────────────────────────────────────────────
    report_name = cfg.get("report_name", "ch7_scaling")
    output_root = Path(cfg.env.output_root)
    from thesis_ml.utils.paths import get_report_id

    report_id = get_report_id(report_name)
    report_dir = resolve_report_output_dir(report_id, report_name, output_root)
    training_dir, _inf_dir, figs_dir, _inf_figs_dir = ensure_report_dirs(report_dir)

    logger.info("Report output: %s", report_dir)

    # Save filtered data for traceability
    df.to_csv(training_dir / "ch7_scaling_data.csv", index=False)

    fig_cfg = {"fig_format": str(cfg.outputs.get("fig_format", "pdf")), "dpi": int(cfg.outputs.get("dpi", 300))}
    wanted: set[str] = set(cfg.outputs.get("which_figures") or [])

    # ── Exp 8A — Scaling curves ───────────────────────────────────────────────
    if "auroc_vs_model_size" in wanted:
        try:
            _plot_auroc_vs_model_size(df, figs_dir, fig_cfg)
        except Exception as e:
            logger.warning("Error generating auroc_vs_model_size: %s", e)

    if "flops_vs_model_size" in wanted:
        try:
            _plot_flops_vs_model_size(df, figs_dir, fig_cfg)
        except Exception as e:
            logger.warning("Error generating flops_vs_model_size: %s", e)

    if "wallclock_vs_model_size" in wanted:
        try:
            _plot_wallclock_vs_model_size(df, figs_dir, fig_cfg)
        except Exception as e:
            logger.warning("Error generating wallclock_vs_model_size: %s", e)

    if "best_epoch_vs_model_size" in wanted:
        try:
            _plot_best_epoch_vs_model_size(df, figs_dir, fig_cfg)
        except Exception as e:
            logger.warning("Error generating best_epoch_vs_model_size: %s", e)

    # ── Exp 8B — Pareto / efficiency ──────────────────────────────────────────
    if "pareto_auroc_vs_flops" in wanted:
        try:
            _plot_pareto(
                df,
                x_col=_COL_FLOPS,
                x_label="FLOPs / event (analytic)",
                fname="figure-pareto_auroc_vs_flops",
                figs_dir=figs_dir,
                fig_cfg=fig_cfg,
                x_log=True,
            )
        except Exception as e:
            logger.warning("Error generating pareto_auroc_vs_flops: %s", e)

    if "pareto_auroc_vs_latency" in wanted:
        try:
            _plot_pareto(
                df,
                x_col=_COL_LATENCY_B512,
                x_label="Inference latency, b=512 (ms)",
                fname="figure-pareto_auroc_vs_latency",
                figs_dir=figs_dir,
                fig_cfg=fig_cfg,
            )
        except Exception as e:
            logger.warning("Error generating pareto_auroc_vs_latency: %s", e)

    if "pareto_auroc_vs_throughput" in wanted:
        try:
            _plot_pareto(
                df,
                x_col=_COL_THROUGHPUT,
                x_label="Throughput, b=512 (samples/s)",
                fname="figure-pareto_auroc_vs_throughput",
                figs_dir=figs_dir,
                fig_cfg=fig_cfg,
            )
        except Exception as e:
            logger.warning("Error generating pareto_auroc_vs_throughput: %s", e)

    if "pareto_auroc_vs_memory" in wanted:
        try:
            _plot_pareto(
                df,
                x_col=_COL_MEMORY,
                x_label="Peak inference memory, b=512 (MiB)",
                fname="figure-pareto_auroc_vs_memory",
                figs_dir=figs_dir,
                fig_cfg=fig_cfg,
            )
        except Exception as e:
            logger.warning("Error generating pareto_auroc_vs_memory: %s", e)

    logger.info("Chapter 7 report complete: %s", report_dir)
