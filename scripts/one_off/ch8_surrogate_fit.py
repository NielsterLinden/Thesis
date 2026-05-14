"""ch8_surrogate_fit.py — Phase C: XGBoost surrogate + SHAP for §8.3.

Fits an XGBoost surrogate predicting eval_v2/test_auroc from architectural
and training axes, runs SHAP attribution, and generates publication-ready
figures for the thesis.

Figures generated (all PDF, 300 DPI)
--------------------------------------
  surrogate_cv_scatter.pdf          OOF predicted vs actual AUROC scatter
  surrogate_shap_family_bar.pdf     Normalised mean |SHAP| per axis family
  surrogate_shap_top5_beeswarm.pdf  Strip plot: SHAP values for top-5 axes
  surrogate_shap_dependence_top3.pdf Per-level SHAP bars for top-3 axes

Source of truth: thesis_results/ch8_streamlined/05_ch8_streamlined_primary.csv
Output: /data/atlas/users/nterlind/outputs/reports/report_ch8_final_after_failed_run_removal/
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_ml.reports.plots.style import (
    CATEGORICAL_COLORS,
    _CATEGORICAL_ORDER,
    apply_thesis_style,
    axis_color,
    figure_size,
)
from thesis_ml.reports.ch8_surrogate import (
    aggregate_shap_to_families,
    build_feature_matrix,
    compute_shap,
    fit_surrogate,
    make_groups,
    _axis_col_from_feature,
    _axis_family_from_feature,
)

apply_thesis_style()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/project/atlas/users/nterlind/Thesis-Code")
PRIMARY_CSV = REPO_ROOT / "thesis_results/ch8_streamlined/05_ch8_streamlined_primary.csv"
OUT_BASE = Path("/data/atlas/users/nterlind/outputs/reports/report_ch8_patching_G1_2")
OUT_FIG = OUT_BASE / "figures"
OUT_SUR = OUT_BASE / "surrogate"

_CFG_LOG = {"fig_format": "pdf", "dpi": 300}

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

# Families with thesis axis_color entries
_KNOWN_FAMILIES = set(_CATEGORICAL_ORDER)  # D, T, E, B, A, F, C, H

# Extra families not in the thesis axis palette — use tab10
_EXTRA_FAMILIES = ["G", "K", "L", "M", "P", "R", "S", "T"]
_extra_colors_raw = plt.cm.tab10(np.linspace(0, 0.9, len(_EXTRA_FAMILIES)))
_EXTRA_COLOR_MAP: dict[str, str] = {
    fam: _extra_colors_raw[i] for i, fam in enumerate(_EXTRA_FAMILIES)
}


def _family_color(fam: str):
    """Consistent colour per family letter."""
    if fam in _KNOWN_FAMILIES:
        return axis_color(fam)
    return _EXTRA_COLOR_MAP.get(fam, "gray")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, fname: str) -> Path:
    from thesis_ml.monitoring.io_utils import save_figure
    path = save_figure(fig, OUT_FIG, fname, _CFG_LOG)
    logger.info("Saved: %s", path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 1 — OOF scatter
# ---------------------------------------------------------------------------


def plot_cv_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cv_metrics: dict,
) -> plt.Figure:
    """OOF predicted vs actual AUROC scatter with diagonal and annotations."""
    fig, ax = plt.subplots(figsize=figure_size("full", aspect=1.0))

    ax.scatter(
        y_true, y_pred,
        s=12, alpha=0.5,
        color=axis_color("D"),
        rasterized=True,
        zorder=3,
    )

    # Diagonal y = x
    lims = [min(y_true.min(), y_pred.min()) - 0.002,
            max(y_true.max(), y_pred.max()) + 0.002]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    rho = cv_metrics["spearman_mean"]
    rho_std = cv_metrics["spearman_std"]
    r2 = cv_metrics["r2_mean"]
    r2_std = cv_metrics["r2_std"]

    ax.text(
        0.05, 0.95,
        rf"Spearman $\rho$ = {rho:.3f} $\pm$ {rho_std:.3f}"
        "\n"
        rf"$R^2$ = {r2:.3f} $\pm$ {r2_std:.3f}",
        transform=ax.transAxes,
        va="top", ha="left",
    )

    ax.set_xlabel("Actual AUROC")
    ax.set_ylabel("Surrogate predicted AUROC")

    return fig


# ---------------------------------------------------------------------------
# Figure 2 — SHAP family bar
# ---------------------------------------------------------------------------


def plot_shap_family_bar(family_importances: dict[str, float]) -> plt.Figure:
    """Horizontal bar chart of normalised mean |SHAP| per axis family."""
    # Sort descending
    sorted_items = sorted(family_importances.items(), key=lambda x: x[1], reverse=True)
    families = [it[0] for it in sorted_items]
    values = [it[1] for it in sorted_items]
    colors = [_family_color(f) for f in families]

    fig, ax = plt.subplots(figsize=figure_size("full"))
    y_pos = np.arange(len(families))
    bars = ax.barh(y_pos, values, color=colors, height=0.6, zorder=3)

    # Label bars with percentage
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center", ha="left",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(families)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised mean |SHAP| (fraction of total)")
    ax.set_xlim(0, max(values) * 1.25)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    return fig


# ---------------------------------------------------------------------------
# Figure 3 — SHAP top-5 strip / beeswarm (matplotlib)
# ---------------------------------------------------------------------------


def plot_shap_top5_beeswarm(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    top5_feat_cols: list[str],
) -> plt.Figure:
    """Strip plot: x=SHAP value, y=axis (parent col), colour=feature value (0/1).

    Uses a manual matplotlib strip rather than shap.plots to keep thesis style.
    Points are jittered on the y-axis.  Colour: feature_value=1 → axis_color("recommended"),
    feature_value=0 → "lightgray".
    """
    feature_names = list(X_sample.columns)
    feat_idx = {f: i for i, f in enumerate(feature_names)}

    # For each top-5 parent axis col, gather all one-hot child features
    # and accumulate their SHAP contributions (sum over children per sample)
    parent_to_children: dict[str, list[str]] = {}
    for f in feature_names:
        parent = _axis_col_from_feature(f)
        if parent in top5_feat_cols:
            parent_to_children.setdefault(parent, []).append(f)

    # Short label: strip 'config/axes/' prefix
    def _short(col: str) -> str:
        return col.replace("config/axes/", "")

    n = len(top5_feat_cols)
    fig, ax = plt.subplots(figsize=figure_size("full"))

    rng = np.random.default_rng(0)

    for rank, parent in enumerate(top5_feat_cols):
        children = parent_to_children.get(parent, [])
        if not children:
            continue
        # Aggregate SHAP over all one-hot children: sum
        child_indices = [feat_idx[c] for c in children if c in feat_idx]
        sv_sum = shap_values[:, child_indices].sum(axis=1)  # (n_samples,)

        # Feature value: use the child with value=1 to determine "active"
        # Combine: 1 if any child == 1 (one-hot), else 0
        child_vals = X_sample[children].values.sum(axis=1)  # 0 or 1 per row
        child_vals_clipped = np.clip(child_vals, 0, 1)

        jitter = rng.uniform(-0.25, 0.25, size=len(sv_sum))
        y_vals = (n - 1 - rank) + jitter

        colors = [
            axis_color("recommended") if v else "lightgray"
            for v in child_vals_clipped
        ]
        ax.scatter(sv_sum, y_vals, c=colors, s=8, alpha=0.5, rasterized=True, zorder=3)

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([_short(c) for c in reversed(top5_feat_cols)])
    ax.set_xlabel("SHAP value (impact on predicted AUROC)")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=axis_color("recommended"),
               markersize=6, label="Feature active"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray",
               markersize=6, label="Feature inactive"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# Figure 4 — SHAP dependence: per-level mean SHAP for top-3 parent axes
# ---------------------------------------------------------------------------


def plot_shap_dependence_top3(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    top3_feat_cols: list[str],
) -> plt.Figure:
    """For top-3 parent axes, bar chart of mean SHAP per one-hot level."""
    feature_names = list(X_sample.columns)
    feat_idx = {f: i for i, f in enumerate(feature_names)}

    def _short(col: str) -> str:
        return col.replace("config/axes/", "")

    def _level_label(feat: str) -> str:
        """Extract the level value from a one-hot feature name."""
        m = re.search(r"__LEVEL__(.+)$", feat)
        return m.group(1) if m else feat

    n = len(top3_feat_cols)
    fig, axes = plt.subplots(1, n, figsize=(n * figure_size("half")[0], figure_size("half")[1]))
    if n == 1:
        axes = [axes]

    for ax, parent in zip(axes, top3_feat_cols):
        fam = _axis_family_from_feature(parent)
        color = _family_color(fam)

        # Find all one-hot children for this parent
        children = [
            f for f in feature_names
            if _axis_col_from_feature(f) == parent
        ]
        if not children:
            ax.set_visible(False)
            continue

        levels = [_level_label(c) for c in children]
        mean_shap = [
            float(shap_values[:, feat_idx[c]].mean()) if c in feat_idx else 0.0
            for c in children
        ]

        # Sort by mean SHAP descending
        order = np.argsort(mean_shap)[::-1]
        levels = [levels[i] for i in order]
        mean_shap_sorted = [mean_shap[i] for i in order]

        colors = [color if v >= 0 else "lightcoral" for v in mean_shap_sorted]

        y_pos = np.arange(len(levels))
        ax.barh(y_pos, mean_shap_sorted, color=colors, height=0.6, zorder=3)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(levels)
        ax.invert_yaxis()
        ax.set_xlabel("Mean SHAP value")
        ax.set_ylabel(_short(parent))

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    # Ensure output directories exist
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_SUR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading primary CSV: %s", PRIMARY_CSV)
    df = pd.read_csv(PRIMARY_CSV)
    logger.info("Loaded %d rows, %d cols", *df.shape)

    axis_cols = [c for c in df.columns if c.startswith("config/axes/")]
    logger.info("Found %d axis columns", len(axis_cols))

    # ------------------------------------------------------------------
    # 2. Build feature matrix
    # ------------------------------------------------------------------
    X, feature_names = build_feature_matrix(df, axis_cols)
    logger.info("Feature matrix shape: %s", X.shape)

    # ------------------------------------------------------------------
    # 3. Groups for GroupKFold
    # ------------------------------------------------------------------
    groups = make_groups(df, axis_cols)
    n_groups = len(np.unique(groups))
    logger.info("Unique groups (excluding seed): %d", n_groups)

    # ------------------------------------------------------------------
    # 4. Target
    # ------------------------------------------------------------------
    y = df["eval_v2/test_auroc"].values
    logger.info("Target: mean=%.4f std=%.4f min=%.4f max=%.4f",
                y.mean(), y.std(), y.min(), y.max())

    # ------------------------------------------------------------------
    # 5. Fit surrogate
    # ------------------------------------------------------------------
    logger.info("Fitting surrogate (GroupKFold CV, 5 folds)...")
    model, oof_preds, cv_metrics = fit_surrogate(X, y, groups)
    logger.info(
        "CV: Spearman = %.3f ± %.3f | R2 = %.3f ± %.3f",
        cv_metrics["spearman_mean"], cv_metrics["spearman_std"],
        cv_metrics["r2_mean"], cv_metrics["r2_std"],
    )

    # ------------------------------------------------------------------
    # 6. Save CV metrics
    # ------------------------------------------------------------------
    metrics_path = OUT_SUR / "cv_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(cv_metrics, fh, indent=2)
    logger.info("Saved CV metrics: %s", metrics_path)

    # ------------------------------------------------------------------
    # 7. Threshold check
    # ------------------------------------------------------------------
    if cv_metrics["spearman_mean"] < 0.3:
        warnings.warn(
            f"Spearman mean ({cv_metrics['spearman_mean']:.3f}) < 0.3 threshold. "
            "Surrogate fit is poor; SHAP interpretations may be unreliable.",
            stacklevel=2,
        )
        logger.warning("Spearman threshold NOT met (%.3f < 0.3)", cv_metrics["spearman_mean"])
    else:
        logger.info("Spearman threshold met (%.3f >= 0.3)", cv_metrics["spearman_mean"])

    # ------------------------------------------------------------------
    # 8. SHAP
    # ------------------------------------------------------------------
    logger.info("Computing SHAP values (max_rows=800)...")
    shap_values, X_sample = compute_shap(model, X, max_rows=800)
    logger.info("SHAP computed: shap_values shape = %s", shap_values.shape)

    # ------------------------------------------------------------------
    # 9. Aggregate to families
    # ------------------------------------------------------------------
    family_importances = aggregate_shap_to_families(shap_values, list(X_sample.columns))
    sorted_families = sorted(family_importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("Family importances (top 10):")
    for fam, val in sorted_families[:10]:
        logger.info("  %s: %.1f%%", fam, val * 100)

    # ------------------------------------------------------------------
    # 10. Top-5 individual axis columns by mean |SHAP|
    # ------------------------------------------------------------------
    feat_names_sample = list(X_sample.columns)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Aggregate to parent axis column (not one-hot child)
    parent_shap: dict[str, float] = {}
    for feat, val in zip(feat_names_sample, mean_abs_shap):
        parent = _axis_col_from_feature(feat)
        parent_shap[parent] = parent_shap.get(parent, 0.0) + float(val)

    sorted_parents = sorted(parent_shap.items(), key=lambda x: x[1], reverse=True)
    top5_parent_cols = [p for p, _ in sorted_parents[:5]]
    top3_parent_cols = [p for p, _ in sorted_parents[:3]]

    logger.info("Top-5 parent axis columns by mean |SHAP|:")
    for col, val in sorted_parents[:5]:
        logger.info("  %s: %.4f", col, val)

    # ------------------------------------------------------------------
    # 11. Save surrogate artefacts
    # ------------------------------------------------------------------
    np.save(str(OUT_SUR / "shap_values.npy"), shap_values)
    X_sample.to_csv(OUT_SUR / "X_shap_sample.csv", index=False)
    model.get_booster().save_model(str(OUT_SUR / "surrogate_xgb.json"))
    logger.info("Surrogate artefacts saved to %s", OUT_SUR)

    # ------------------------------------------------------------------
    # 12. Figures
    # ------------------------------------------------------------------

    # Figure 1: OOF scatter
    logger.info("Plotting OOF scatter...")
    fig1 = plot_cv_scatter(y, oof_preds, cv_metrics)
    _save(fig1, "surrogate_cv_scatter")

    # Figure 2: SHAP family bar
    logger.info("Plotting SHAP family bar...")
    fig2 = plot_shap_family_bar(family_importances)
    _save(fig2, "surrogate_shap_family_bar")

    # Figure 3: SHAP top-5 strip/beeswarm
    logger.info("Plotting SHAP top-5 beeswarm (strip plot)...")
    fig3 = plot_shap_top5_beeswarm(shap_values, X_sample, top5_parent_cols)
    _save(fig3, "surrogate_shap_top5_beeswarm")

    # Figure 4: SHAP dependence top-3
    logger.info("Plotting SHAP dependence top-3...")
    fig4 = plot_shap_dependence_top3(shap_values, X_sample, top3_parent_cols)
    _save(fig4, "surrogate_shap_dependence_top3")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase C complete.")
    logger.info("Figures: %s", OUT_FIG)
    logger.info("Surrogate artefacts: %s", OUT_SUR)
    logger.info(
        "CV: Spearman = %.3f ± %.3f | R2 = %.3f ± %.3f",
        cv_metrics["spearman_mean"], cv_metrics["spearman_std"],
        cv_metrics["r2_mean"], cv_metrics["r2_std"],
    )
    logger.info("Top-5 parent axes by mean |SHAP|:")
    for col, val in sorted_parents[:5]:
        logger.info("  %s: %.4f", col.replace("config/axes/", ""), val)
    logger.info("Top-5 families by norm. |SHAP|:")
    for fam, val in sorted_families[:5]:
        logger.info("  %s: %.1f%%", fam, val * 100)


if __name__ == "__main__":
    main()
