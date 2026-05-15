"""Chapter 6 report: Attention Mechanisms.

Covers two experiments:

Exp 6A — Attention type and normalization:
  - A3:  classifier.model.attention.type  = standard | differential
  - A4:  classifier.model.attention.norm  = none | layernorm | rmsnorm
  - B1:  classifier.model.attention_biases = none | lorentz_scalar

Exp 6B — Differential attention bias mode (prerequisite: A3 = differential):
  - A3-a: classifier.model.attention.diff_bias_mode = none | shared | split

All axes are defined in docs/AXES_REFERENCE_V2.md.

The report reads axis values directly from the runs' ``resolved_config.yaml``
files.  It does **not** re-run inference — metrics are sourced from the pre-scored
``test_scores.pt`` already present in each run directory.  Interpretability
artifacts (``interpretability/attention_epoch_*.pt``) are used for the lambda-
evolution and per-layer attention-entropy plots.

Plots generated (training phase, no extra inference needed)
-----------------------------------------------------------
- auroc_seedspread_by_attn_type        (A3: standard vs differential)
- auroc_seedspread_by_attn_norm        (A4: none / layernorm / rmsnorm)
- auroc_seedspread_by_bias             (B1: none vs lorentz_scalar)
- auroc_seedspread_by_diff_bias_mode   (A3-a: none / shared / split — diff only)
- auroc_heatmap_attn_type_x_norm       (A3 × A4 interaction)
- auroc_heatmap_attn_type_x_bias       (A3 × B1 interaction)
- auroc_heatmap_norm_x_bias            (A4 × B1 interaction — for std and diff separately)
- per_class_auroc_by_attn_type         (per-class AUROC bar chart — A3)
- val_auroc_by_attn_type               (validation AUROC learning curves — A3)
- val_auroc_by_attn_norm               (validation AUROC learning curves — A4)
- lambda_evolution                     (diff-attention λ per layer across epochs)
- attention_entropy_by_layer           (mean entropy of attention weights — std vs diff)

Usage
-----
thesis-report --config-name thesis_experiments_reports/ch6_attention_mechanisms \\
  'inputs.sweep_dirs=[\\
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job000,\\
    ...]' \\
  env.output_root=/data/atlas/users/nterlind/outputs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from thesis_ml.facts.readers import _extract_value_from_composed_cfg, _read_cfg
from thesis_ml.monitoring.io_utils import save_figure
from thesis_ml.reports.plots.classification import (
    plot_auroc_seedspread_by_group,
    plot_roc_curves,
)
from thesis_ml.reports.plots.curves import plot_val_auroc_grouped_by
from thesis_ml.reports.plots.grids import plot_grid_heatmap
from thesis_ml.reports.plots.style import (
    CATEGORICAL_COLORS,
    apply_thesis_style,
    axis_color,
    figure_size,
)
from thesis_ml.reports.utils.io import finalize_report, get_fig_config, setup_report_environment
from thesis_ml.utils.paths import get_run_id

logger = logging.getLogger(__name__)

apply_thesis_style()

# ─────────────────────────────────────────────────────────────────────────────
# Axis ↔ config-key mapping
# ─────────────────────────────────────────────────────────────────────────────

_CH6_AXES: list[tuple[str, str]] = [
    ("attention_type",    "classifier.model.attention.type"),
    ("attention_norm",    "classifier.model.attention.norm"),
    ("diff_bias_mode",    "classifier.model.attention.diff_bias_mode"),
    ("attention_biases",  "classifier.model.attention_biases"),
]

# Human-readable labels for axis values
_ATTN_TYPE_LABELS: dict[str, str] = {
    "standard":     "Standard",
    "differential": "Differential",
}
_NORM_LABELS: dict[str, str] = {
    "none":      "None",
    "layernorm": "LayerNorm",
    "rmsnorm":   "RMSNorm",
}
_DIFF_BIAS_LABELS: dict[str, str] = {
    "none":   "None",
    "shared": "Shared",
    "split":  "Split",
}
_BIAS_LABELS: dict[str, str] = {
    "none":           "No bias",
    "lorentz_scalar": "Lorentz scalar",
}


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction
# ─────────────────────────────────────────────────────────────────────────────


def _extract_ch6_metadata(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract Chapter 6 axis values from each run's resolved config.

    Columns already present in ``runs_df`` are not overwritten so that Hydra
    override values take priority.

    Returns a DataFrame with additional columns:
        attention_type, attention_norm, diff_bias_mode, attention_biases
    """
    df = runs_df.copy()
    axes_to_read = [(col, path) for col, path in _CH6_AXES if col not in df.columns]

    if axes_to_read:
        extracted: dict[str, list] = {col: [] for col, _ in axes_to_read}
        for run_dir in df["run_dir"]:
            try:
                cfg, _ = _read_cfg(Path(run_dir))
            except Exception as e:
                logger.warning("Failed to read cfg for %s: %s", run_dir, e)
                for col, _ in axes_to_read:
                    extracted[col].append(None)
                continue

            for col, path in axes_to_read:
                try:
                    val = _extract_value_from_composed_cfg(cfg, path)
                    extracted[col].append(val)
                except Exception:
                    extracted[col].append(None)

        for col, values in extracted.items():
            df[col] = values

    logger.info(
        "Ch6 axes — attention_type: %s | attention_norm: %s | diff_bias_mode: %s | bias: %s",
        sorted(df["attention_type"].dropna().unique().tolist()),
        sorted(df["attention_norm"].dropna().unique().tolist()),
        sorted(df["diff_bias_mode"].dropna().unique().tolist()),
        sorted(df["attention_biases"].dropna().unique().tolist()),
    )
    return df


def _load_test_auroc(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Load per-run test AUROC from test_scores.pt files.

    Adds columns: ``auroc``, ``per_class_auroc`` (dict str->float).
    Falls back gracefully if test_scores.pt is absent for a run.
    """
    df = runs_df.copy()
    aurocs: list[float | None] = []
    per_class: list[dict | None] = []

    for run_dir in df["run_dir"]:
        scores_path = Path(run_dir) / "test_scores.pt"
        if not scores_path.exists():
            logger.warning("test_scores.pt missing for %s", run_dir)
            aurocs.append(None)
            per_class.append(None)
            continue
        try:
            scores = torch.load(scores_path, map_location="cpu", weights_only=False)
            auroc_val = None
            pc = None
            if isinstance(scores, dict):
                # Try pre-computed keys first
                if "auroc" in scores or "test_auroc" in scores:
                    auroc_val = float(scores.get("auroc", scores.get("test_auroc", np.nan)))
                    pc_raw = scores.get("per_class_auroc", scores.get("per_class_auroc_json"))
                    if pc_raw is not None:
                        if isinstance(pc_raw, str):
                            pc = json.loads(pc_raw)
                        elif isinstance(pc_raw, dict):
                            pc = {str(k): float(v) for k, v in pc_raw.items()}
                elif "probs" in scores and "labels" in scores:
                    # Compute AUROC from raw probs + labels
                    from sklearn.metrics import roc_auc_score
                    probs = scores["probs"].numpy()
                    labels = scores["labels"].numpy()
                    n_classes = probs.shape[1]
                    auroc_val = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
                    pc = {
                        str(c): float(roc_auc_score((labels == c).astype(int), probs[:, c]))
                        for c in range(n_classes)
                    }
            aurocs.append(auroc_val)
            per_class.append(pc)
        except Exception as e:
            logger.warning("Failed to load test_scores.pt from %s: %s", run_dir, e)
            aurocs.append(None)
            per_class.append(None)

    df["auroc"] = aurocs
    df["per_class_auroc"] = per_class
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-class AUROC bar chart
# ─────────────────────────────────────────────────────────────────────────────

#: 5-class names matching the G3 classification task "4t | ttH | ttW | ttWW | ttZ"
_CLASS_NAMES: list[str] = ["4t", "ttH", "ttW", "ttWW", "ttZ"]


def _plot_per_class_auroc_by_attn_type(
    runs_df: pd.DataFrame,
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-per_class_auroc_by_attn_type",
) -> None:
    """Grouped bar chart of per-class one-vs-rest AUROC by attention type.

    Two bar groups (standard, differential), one bar per class.
    Error bars show std across seeds (n=3 per group).
    """
    subset = runs_df.dropna(subset=["attention_type", "per_class_auroc"]).copy()
    if subset.empty:
        logger.warning("No per_class_auroc data available — skipping per_class_auroc plot")
        return

    attn_types = ["standard", "differential"]
    colors = [axis_color("baseline"), axis_color("A")]
    n_classes = len(_CLASS_NAMES)
    x = np.arange(n_classes)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=figure_size("full"))

    for i, (atype, color) in enumerate(zip(attn_types, colors, strict=False)):
        group = subset[subset["attention_type"] == atype]
        if group.empty:
            continue
        # Collect per-class AUROC across runs
        class_aucs: dict[int, list[float]] = {c: [] for c in range(n_classes)}
        for _, row in group.iterrows():
            pc = row["per_class_auroc"]
            if not isinstance(pc, dict):
                continue
            for c in range(n_classes):
                val = pc.get(str(c), pc.get(c))
                if val is not None:
                    class_aucs[c].append(float(val))

        means = [np.mean(class_aucs[c]) if class_aucs[c] else np.nan for c in range(n_classes)]
        stds = [np.std(class_aucs[c]) if len(class_aucs[c]) > 1 else 0.0 for c in range(n_classes)]

        offset = (i - 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=3,
            color=color,
            alpha=0.85,
            label=_ATTN_TYPE_LABELS.get(atype, atype),
            zorder=3,
        )
        # Seed dots
        for c in range(n_classes):
            dots = class_aucs[c]
            if dots:
                ax.scatter(
                    np.full(len(dots), x[c] + offset),
                    dots,
                    color=color,
                    s=20,
                    alpha=0.5,
                    zorder=4,
                )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(_CLASS_NAMES)
    ax.set_xlabel("Class")
    ax.set_ylabel("One-vs-rest AUROC")
    ax.set_ylim(bottom=0.5)
    ax.legend(title="Attention type")

    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
    logger.info("Saved %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Lambda evolution
# ─────────────────────────────────────────────────────────────────────────────


def _plot_lambda_evolution(
    runs_df: pd.DataFrame,
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-lambda_evolution",
) -> None:
    """Plot differential-attention λ values per layer across training epochs.

    One sub-panel per transformer layer.  Each curve is one run (seed); the
    mean across seeds is drawn with a heavier line.  Epoch checkpoints are
    ``attention_epoch_0.pt``, ``attention_epoch_10.pt``, ``attention_epoch_final.pt``,
    ``attention_epoch_best.pt`` (best used only when final not present).

    Only differential-attention runs are used.
    """
    diff_runs = runs_df[runs_df["attention_type"] == "differential"].copy()
    if diff_runs.empty:
        logger.warning("No differential-attention runs found — skipping lambda_evolution plot")
        return

    # Epoch labels in ascending order; best is excluded to avoid double-counting
    _EPOCH_ORDER = [0, 10, 49]  # 50-epoch training → epoch 49 = "final"
    _PT_NAMES: dict[int, list[str]] = {
        0:  ["attention_epoch_0.pt"],
        10: ["attention_epoch_10.pt"],
        49: ["attention_epoch_final.pt", "attention_epoch_best.pt"],
    }

    # Collect λ traces: {layer_idx: {run_dir: {epoch: float}}}
    n_layers = 6
    traces: dict[int, dict[str, dict[int, float]]] = {li: {} for li in range(n_layers)}

    for _, row in diff_runs.iterrows():
        run_dir = Path(row["run_dir"])
        interp_dir = run_dir / "interpretability"
        if not interp_dir.exists():
            continue
        run_id = str(run_dir)
        for ep in _EPOCH_ORDER:
            pt_loaded = None
            for pt_name in _PT_NAMES[ep]:
                pt_path = interp_dir / pt_name
                if pt_path.exists():
                    try:
                        pt_loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", pt_path, e)
                    break
            if pt_loaded is None:
                continue
            layers_data = pt_loaded.get("layers", {})
            for li in range(n_layers):
                lk = f"layer_{li}"
                lam = layers_data.get(lk, {}).get("lambda")
                if lam is None:
                    continue
                lam_val = lam.item() if hasattr(lam, "item") else float(lam)
                traces[li].setdefault(run_id, {})[ep] = lam_val

    # Plot: n_layers sub-panels in a 2×3 grid
    n_cols, n_rows = 3, 2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * figure_size("half")[0], n_rows * figure_size("half")[1]),
        sharex=True, sharey=True,
    )
    color = axis_color("A")

    for li in range(n_layers):
        ax = axes[li // n_cols][li % n_cols]
        layer_traces = traces[li]
        all_curves: list[list[float]] = []
        for run_id, ep_dict in layer_traces.items():
            eps_sorted = sorted(ep_dict)
            vals = [ep_dict[e] for e in eps_sorted]
            ax.plot(eps_sorted, vals, color=color, alpha=0.2, linewidth=0.8)
            all_curves.append(vals)

        if all_curves:
            min_len = min(len(c) for c in all_curves)
            arr = np.array([c[:min_len] for c in all_curves])
            eps_sorted = sorted(list(layer_traces.values())[0])[:min_len]
            ax.plot(eps_sorted, arr.mean(axis=0), color=color, linewidth=2.5, label="mean")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("λ")
        # Tag each sub-panel without ax.set_title (rule: no set_title)
        ax.text(0.05, 0.92, f"Layer {li}", transform=ax.transAxes, ha="left", va="top")
        ax.set_xlim(left=0)

    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
    logger.info("Saved %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Attention entropy
# ─────────────────────────────────────────────────────────────────────────────


def _attention_entropy(weights: np.ndarray, eps: float = 1e-9) -> float:
    """Compute mean Shannon entropy of attention weight distributions.

    Parameters
    ----------
    weights : np.ndarray
        Shape [n_events, n_heads, seq_len, seq_len]. Values are softmax probs.

    Returns
    -------
    float
        Mean entropy averaged over events, heads, and query positions.
    """
    # Clip for numerical safety before log
    w = np.clip(weights, eps, 1.0)
    # H = -sum(p log p) over key positions
    h = -(w * np.log(w)).sum(axis=-1)  # [n_events, n_heads, seq_len]
    return float(h.mean())


def _plot_attention_entropy_by_layer(
    runs_df: pd.DataFrame,
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-attention_entropy_by_layer",
) -> None:
    """Bar chart of mean attention entropy per layer, grouped by attention type.

    Uses the ``attention_epoch_best.pt`` artifact.  Entropy is averaged over
    events, heads, and query positions.  For differential attention, the
    ``combined`` weights are used.
    """
    attn_types = ["standard", "differential"]
    colors = [axis_color("baseline"), axis_color("A")]
    n_layers = 6

    # {attn_type: {layer_idx: [entropy per run]}}
    entropy_data: dict[str, dict[int, list[float]]] = {
        at: {li: [] for li in range(n_layers)} for at in attn_types
    }

    for _, row in runs_df.iterrows():
        atype = row.get("attention_type")
        if atype not in attn_types:
            continue
        run_dir = Path(row["run_dir"])
        pt_path = run_dir / "interpretability" / "attention_epoch_best.pt"
        if not pt_path.exists():
            continue
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning("Failed to load %s: %s", pt_path, e)
            continue

        layers_data = data.get("layers", {})
        for li in range(n_layers):
            lk = f"layer_{li}"
            ld = layers_data.get(lk, {})
            if atype == "differential":
                w = ld.get("combined")
            else:
                w = ld.get("weights")
            if w is None:
                continue
            w_np = w.numpy() if hasattr(w, "numpy") else np.array(w)
            ent = _attention_entropy(w_np)
            entropy_data[atype][li].append(ent)

    x = np.arange(n_layers)
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=figure_size("full"))

    for i, (atype, color) in enumerate(zip(attn_types, colors, strict=False)):
        means = [np.mean(entropy_data[atype][li]) if entropy_data[atype][li] else np.nan for li in range(n_layers)]
        stds = [np.std(entropy_data[atype][li]) if len(entropy_data[atype][li]) > 1 else 0.0 for li in range(n_layers)]
        offset = (i - 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=3,
            color=color,
            alpha=0.85,
            label=_ATTN_TYPE_LABELS.get(atype, atype),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{li}" for li in range(n_layers)])
    ax.set_xlabel("Encoder layer")
    ax.set_ylabel("Mean attention entropy (nats)")
    ax.legend(title="Attention type")

    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
    logger.info("Saved %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# AUROC interaction heatmaps
# ─────────────────────────────────────────────────────────────────────────────


def _build_auroc_df(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten runs into a per-run DataFrame with AUROC and all axis columns."""
    axis_cols = ["attention_type", "attention_norm", "diff_bias_mode", "attention_biases"]
    rows = []
    for _, row in runs_df.iterrows():
        run_dir = row.get("run_dir")
        if pd.isna(run_dir):
            continue
        auroc = row.get("auroc")
        if auroc is None or (isinstance(auroc, float) and np.isnan(auroc)):
            continue
        entry: dict = {"run_id": get_run_id(Path(str(run_dir))), "auroc": auroc}
        for col in axis_cols:
            entry[col] = row.get(col)
        rows.append(entry)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Report entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_report(cfg: DictConfig) -> None:
    """Chapter 6 report: Attention Mechanisms.

    Generates plots for:
    - A3  (attention type: standard vs differential)
    - A4  (attention internal normalisation: none / layernorm / rmsnorm)
    - B1  (bias activation: none vs lorentz_scalar) — interaction with A3
    - A3-a (differential bias mode: none / shared / split)

    Plus interpretability plots:
    - Lambda evolution across training epochs (differential runs only)
    - Per-layer attention entropy comparison (standard vs differential)

    Run via::

        thesis-report --config-name thesis_experiments_reports/ch6_attention_mechanisms \\
            'inputs.run_dirs=[...]' \\
            env.output_root=/data/atlas/users/nterlind/outputs
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Handle sweep_dirs (list of individual run dirs as sweep_dir is null)
    if cfg.inputs.get("sweep_dirs") and len(cfg.inputs.sweep_dirs) > 0:
        from omegaconf import OmegaConf

        from thesis_ml.facts.readers import discover_runs

        all_paths: list[str] = []
        for sd in cfg.inputs.sweep_dirs:
            all_paths.extend(str(p) for p in discover_runs(sweep_dir=str(sd), run_dirs=None))
        logger.info("sweep_dirs: collected %d runs from %d sweep dirs", len(all_paths), len(cfg.inputs.sweep_dirs))
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["inputs"]["sweep_dir"] = None
        cfg_dict["inputs"]["run_dirs"] = all_paths
        cfg = OmegaConf.create(cfg_dict)

    (
        training_dir,
        inference_dir,
        training_figs_dir,
        inference_figs_dir,
        runs_df,
        per_epoch,
        order,
        report_dir,
        report_id,
        report_name,
    ) = setup_report_environment(cfg)

    logger.info("Loaded %d runs for Chapter 6 report", len(runs_df))

    # ── Extract axis metadata ──────────────────────────────────────────────────
    runs_df = _extract_ch6_metadata(runs_df)

    # ── Load test AUROC from test_scores.pt ────────────────────────────────────
    runs_df = _load_test_auroc(runs_df)
    n_with_auroc = runs_df["auroc"].notna().sum()
    logger.info("%d / %d runs have test AUROC from test_scores.pt", n_with_auroc, len(runs_df))

    runs_df.to_csv(training_dir / "summary.csv", index=False)

    fig_cfg = get_fig_config(cfg)
    wanted = set(cfg.outputs.which_figures or [])

    auroc_df = _build_auroc_df(runs_df)

    # ── Training curves ────────────────────────────────────────────────────────
    training_groups = [
        ("attention_type",   "val_auroc_by_attn_type", "Validation AUROC by Attention Type (A3)"),
        ("attention_norm",   "val_auroc_by_attn_norm", "Validation AUROC by Internal Normalization (A4)"),
        ("attention_biases", "val_auroc_by_bias",      "Validation AUROC by Bias Activation (B1)"),
    ]
    for group_col, key, title in training_groups:
        if key not in wanted:
            continue
        subset = runs_df[runs_df[group_col].notna()]
        if subset.empty:
            logger.warning("No runs with '%s' set — skipping %s", group_col, key)
            continue
        try:
            plot_val_auroc_grouped_by(
                subset,
                per_epoch,
                group_col=group_col,
                figs_dir=training_figs_dir,
                fig_cfg=fig_cfg,
                fname=f"figure-{key}",
                title=title,
            )
        except Exception as e:
            logger.warning("Error generating %s: %s", key, e)

    # ── AUROC seed-spread bar charts ───────────────────────────────────────────
    seedspread_groups = [
        ("attention_type",   "auroc_seedspread_by_attn_type",      "AUROC per Seed by Attention Type (A3)"),
        ("attention_norm",   "auroc_seedspread_by_attn_norm",      "AUROC per Seed by Internal Normalization (A4)"),
        ("attention_biases", "auroc_seedspread_by_bias",           "AUROC per Seed by Bias Activation (B1)"),
        ("diff_bias_mode",   "auroc_seedspread_by_diff_bias_mode", "AUROC per Seed by Differential Bias Mode (A3-a)"),
    ]
    for group_col, key, title in seedspread_groups:
        if key not in wanted:
            continue
        subset_df = auroc_df.dropna(subset=[group_col])
        if subset_df.empty:
            logger.warning("No AUROC data for '%s' — skipping %s", group_col, key)
            continue
        # Restrict A3-a plot to differential runs only
        if group_col == "diff_bias_mode":
            subset_df = subset_df[subset_df["attention_type"] == "differential"]
        try:
            plot_auroc_seedspread_by_group(
                subset_df,
                group_col=group_col,
                inference_figs_dir=training_figs_dir,
                fig_cfg=fig_cfg,
                fname=f"figure-{key}",
                title=title,
            )
        except Exception as e:
            logger.warning("Error generating %s: %s", key, e)

    # ── 2D AUROC interaction heatmaps ──────────────────────────────────────────
    heatmap_specs: list[tuple[str, str, str, str]] = [
        ("attention_type", "attention_norm", "auroc", "auroc_heatmap_attn_type_x_norm"),
        ("attention_type", "attention_biases", "auroc", "auroc_heatmap_attn_type_x_bias"),
        ("attention_norm", "attention_biases", "auroc", "auroc_heatmap_norm_x_bias"),
    ]
    for row_col, col_col, val_col, key in heatmap_specs:
        if key not in wanted:
            continue
        subset_df = auroc_df.dropna(subset=[row_col, col_col, val_col])
        if subset_df.empty:
            logger.warning("No data for heatmap %s — skipping", key)
            continue
        agg_df = subset_df.groupby([row_col, col_col], as_index=False)[val_col].mean()
        try:
            plot_grid_heatmap(
                agg_df,
                row_col=row_col,
                col_col=col_col,
                value_col=val_col,
                title=f"Mean AUROC: {row_col} × {col_col}",
                figs_dir=training_figs_dir,
                fname=f"figure-{key}",
                fig_cfg=fig_cfg,
            )
        except Exception as e:
            logger.warning("Error generating %s: %s", key, e)

    # ── Per-class AUROC by attention type ─────────────────────────────────────
    if "per_class_auroc_by_attn_type" in wanted:
        try:
            _plot_per_class_auroc_by_attn_type(
                runs_df,
                figs_dir=training_figs_dir,
                fig_cfg=fig_cfg,
            )
        except Exception as e:
            logger.warning("Error generating per_class_auroc_by_attn_type: %s", e)

    # ── Interpretability: lambda evolution ────────────────────────────────────
    if "lambda_evolution" in wanted:
        try:
            _plot_lambda_evolution(
                runs_df,
                figs_dir=training_figs_dir,
                fig_cfg=fig_cfg,
            )
        except Exception as e:
            logger.warning("Error generating lambda_evolution: %s", e)

    # ── Interpretability: attention entropy by layer ───────────────────────────
    if "attention_entropy_by_layer" in wanted:
        try:
            _plot_attention_entropy_by_layer(
                runs_df,
                figs_dir=training_figs_dir,
                fig_cfg=fig_cfg,
            )
        except Exception as e:
            logger.warning("Error generating attention_entropy_by_layer: %s", e)

    # ── Finalize ───────────────────────────────────────────────────────────────
    finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)
    logger.info("Chapter 6 report complete: %s", report_dir)
