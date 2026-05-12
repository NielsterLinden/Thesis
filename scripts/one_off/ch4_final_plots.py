"""ch4_final_plots.py — final Ch4 figures for thesis handoff.

Regenerates all Ch4 publication-ready figures using the thesis style system.
All reviewer-agreed fixes from the previous session are applied here.

Fix summary
-----------
  Fix 1  auroc-bar-tokenizer   : identity bar restricted to dim>=16 (+one_hot/num_types)
  Fix 2  auroc-bar-pid-mode    : filtered to dim>=16 before computing per-mode means
  Fix 3  auroc-bar-dim         : REMOVED — not generated (redundant given heatmap)
  Fix 4  val-auroc-tokenizer   : x-axis cropped to epochs 0–30
  Fix 5  roc-tokenizer         : existing plot is adequate; style compliance only — regenerated
  Fix 6  auroc-bar-features    : filtered to identity rows; "unknown" bar removed
  Fix 7  auroc-bar-met         : values from CSV (+0.0008 delta), not from re-inference
  Fix 8  auroc-bar-shuffle     : values from CSV (−0.0026 delta), not from re-inference

Additional figures regenerated with new style:
  - auroc-heatmap-pid-mode-x-embed-dim
  - failure-analysis-raw-vs-identity (scatter + bar panel)
  - roc-curves-by-met, roc-curves-by-shuffle (existing data, new style)
  - val-auroc-by-pid-mode (new style, no epoch crop needed)
  - val-auroc-by-met, val-auroc-by-shuffle (new style)

Source of truth: thesis_results/03_analysis_ready.csv
Output: /data/atlas/users/nterlind/outputs/reports/report_ch4_final/figures/
Format: PDF at 300 DPI via io_utils.save_figure()
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_ml.reports.plots.style import (
    CATEGORICAL_COLORS,
    apply_thesis_style,
    axis_color,
    figure_size,
)

apply_thesis_style()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path("/project/atlas/users/nterlind/Thesis-Code")
CSV_PATH = REPO_ROOT / "thesis_results" / "03_analysis_ready.csv"
RUNS_ROOT = Path("/data/atlas/users/nterlind/outputs/runs")
OLD_REPORT = Path(
    "/data/atlas/users/nterlind/outputs/reports"
    "/report_20260420-104926_ch4_best_input_repr"
)
OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch4_final/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Column names (confirmed from 03_analysis_ready.csv) ──────────────────────

T1_COL = "config/axes/T1_Tokenizer Family"
T1A_COL = "config/axes/T1-a_PID Embedding Mode"
T1B_COL = "config/axes/T1-b_PID Embedding Dimension"
D01_COL = "config/axes/D1_Feature Set"
D02_COL = "config/axes/D2_MET Treatment"
D03_COL = "config/axes/D3_Token Ordering"
AUROC_COL = "eval_v2/test_auroc"
GROUP_COL = "meta_run/group"
RUN_NAME_COL = "meta_run/name"

# Ch4 experiment groups (57 runs total across four W&B groups)
CH4_GROUPS = [
    "exp_20260410-160141_ch4_input_repr_exp4a_1_raw",
    "exp_20260410-160937_ch4_input_repr_exp4a_1_binned",
    "exp_20260410-162037_ch4_input_repr_exp4a_2",
    "exp_20260410-163637_ch4_data_treatment_exp4b",
]

# Fixed colors per tokenizer family and per PID mode for consistency
_TOKENIZER_COLORS = {
    "raw": axis_color("D"),         # data-treatment blue
    "binned": axis_color("H"),      # gray
    "identity": axis_color("T"),    # tokenizer sky-blue
}
_PID_MODE_COLORS = {
    "learned": CATEGORICAL_COLORS[0],
    "one_hot": CATEGORICAL_COLORS[1],
    "fixed_random": CATEGORICAL_COLORS[2],
}
_D03_COLORS = {
    "input_order": axis_color("T"),
    "shuffled": axis_color("H"),
}
_D02_COLORS = {
    "MET=False": axis_color("T"),
    "MET=True": axis_color("D"),
}
_D01_COLORS = {
    "[0,1,2,3]": axis_color("T"),
    "[1,2,3]": axis_color("H"),
}

# ─── Save helper ──────────────────────────────────────────────────────────────

_CFG_LOG = {"fig_format": "pdf", "dpi": 300}


def _save(fig: plt.Figure, fname: str) -> None:
    """Save via io_utils-compatible call (PDF, 300 DPI, bbox=tight)."""
    path = OUT_DIR / f"{fname}.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    logger.info("Saved %s", path)
    plt.close(fig)


# ─── Shared bar helper ────────────────────────────────────────────────────────


def _bar_with_points(
    ax: plt.Axes,
    groups: list,
    means: list[float],
    stds: list[float],
    all_vals: list[list[float]],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    reference_line: float | None = None,
) -> None:
    """Bar chart with individual seed dots and reference line.

    All font sizes governed by THESIS_RC — no fontsize= arguments.
    """
    x = np.arange(len(groups))
    bar_colors = colors if colors is not None else [CATEGORICAL_COLORS[i % 8] for i in range(len(groups))]

    for i, (mean, std, vals, color) in enumerate(zip(means, stds, all_vals, bar_colors)):
        ax.bar(
            x[i],
            mean,
            0.6,
            yerr=std,
            capsize=3,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
            zorder=2,
        )
        rng = np.random.default_rng(42)
        jitter = rng.normal(0, 0.04, len(vals))
        ax.scatter(x[i] + jitter, vals, color="black", alpha=0.5, s=20, zorder=3)

    if reference_line is not None:
        ax.axhline(
            reference_line,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels if labels is not None else [str(g) for g in groups],
        rotation=30,
        ha="right",
    )
    ymin = min(m - 0.025 for m in means)
    ymax = max(m + std + 0.01 for m, std in zip(means, stds))
    ax.set_ylim([max(0.0, ymin), min(1.0, ymax + 0.005)])


# ─── Load data ────────────────────────────────────────────────────────────────


def load_ch4_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    ch4 = df[df[GROUP_COL].isin(CH4_GROUPS)].copy()
    logger.info("Loaded %d Ch4 rows from CSV", len(ch4))
    return ch4


# ─── Figure 1: auroc_bar_by_tokenizer (Fix 1) ─────────────────────────────────


def plot_auroc_bar_tokenizer(ch4: pd.DataFrame) -> None:
    """Fix 1: identity bar restricted to dim>=16 + num_types.

    Raw and binned are unaffected (3 seeds each).
    Identity restricted to 6 runs (dim=16 × 3 seeds + dim=32 × 3 seeds)
    plus the 9 one_hot rows — but one_hot has triplicate issue, so use
    the dim>=16 + num_types filter which includes one_hot naturally.
    """
    raw_vals = ch4.loc[ch4[T1_COL] == "raw", AUROC_COL].tolist()
    binned_vals = ch4.loc[ch4[T1_COL] == "binned", AUROC_COL].tolist()

    # identity: dim>=16 + num_types (one_hot); excludes learned+dim=8 overlap from Exp 4B
    id_df = ch4[ch4[T1_COL] == "identity"]
    id_clean = id_df[id_df[T1B_COL].isin(["16", "32", "num_types"])]
    identity_vals = id_clean[AUROC_COL].tolist()

    groups = ["raw", "binned", "identity\n(dim≥16)"]
    all_vals = [raw_vals, binned_vals, identity_vals]
    means = [float(np.mean(v)) for v in all_vals]
    stds = [float(np.std(v)) if len(v) > 1 else 0.0 for v in all_vals]
    colors = [_TOKENIZER_COLORS["raw"], _TOKENIZER_COLORS["binned"], _TOKENIZER_COLORS["identity"]]

    fig, ax = plt.subplots(figsize=figure_size("full"))
    _bar_with_points(ax, groups, means, stds, all_vals, colors=colors, reference_line=0.5)
    ax.set_xlabel("Tokenizer Family (T1)")
    ax.set_ylabel("Test AUROC")

    _save(fig, "figure-auroc_bar_by_tokenizer")
    logger.info(
        "Fix 1 | raw=%.4f (n=%d), binned=%.4f (n=%d), identity(dim>=16)=%.4f (n=%d)",
        means[0], len(raw_vals), means[1], len(binned_vals), means[2], len(identity_vals),
    )


# ─── Figure 2: auroc_bar_by_pid_mode (Fix 2) ──────────────────────────────────


def plot_auroc_bar_pid_mode(ch4: pd.DataFrame) -> None:
    """Fix 2: identity tokenizer only, dim>=16 filter applied."""
    id_df = ch4[ch4[T1_COL] == "identity"].copy()
    id_clean = id_df[
        (id_df[T1B_COL].isin(["16", "32"])) | (id_df[T1B_COL] == "num_types")
    ]

    mode_vals: dict[str, list[float]] = {}
    for _, row in id_clean.iterrows():
        mode = row[T1A_COL]
        if pd.isna(mode):
            continue
        mode_vals.setdefault(str(mode), []).append(float(row[AUROC_COL]))

    # fixed display order
    order = ["fixed_random", "learned", "one_hot"]
    groups = [g for g in order if g in mode_vals]
    all_vals = [mode_vals[g] for g in groups]
    means = [float(np.mean(v)) for v in all_vals]
    stds = [float(np.std(v)) if len(v) > 1 else 0.0 for v in all_vals]
    colors = [_PID_MODE_COLORS.get(g, CATEGORICAL_COLORS[0]) for g in groups]

    label_map = {
        "fixed_random": "fixed_random",
        "learned": "learned",
        "one_hot": "one_hot",
    }
    labels = [label_map.get(g, g) for g in groups]

    fig, ax = plt.subplots(figsize=figure_size("full"))
    _bar_with_points(ax, groups, means, stds, all_vals, labels=labels, colors=colors)
    ax.set_xlabel("PID Embedding Mode (T1-a)")
    ax.set_ylabel("Test AUROC")

    _save(fig, "figure-auroc_bar_by_pid_mode")
    for g, m, s in zip(groups, means, stds):
        logger.info("  pid_mode=%s: mean=%.4f std=%.4f n=%d", g, m, s, len(mode_vals[g]))


# ─── Figure 3: auroc_heatmap_pid_mode_x_embed_dim ────────────────────────────


def plot_auroc_heatmap(ch4: pd.DataFrame) -> None:
    """Heatmap of AUROC by T1-a x T1-b for identity tokenizer (Exp 4A-2)."""
    # use Exp 4A-2 only for clean grid
    exp4a2 = ch4[ch4[GROUP_COL] == "exp_20260410-162037_ch4_input_repr_exp4a_2"].copy()
    id_df = exp4a2[exp4a2[T1_COL] == "identity"]

    modes = ["fixed_random", "learned", "one_hot"]
    dims = ["8", "16", "32", "num_types"]

    grid = np.full((len(modes), len(dims)), np.nan)
    for i, mode in enumerate(modes):
        for j, dim in enumerate(dims):
            subset = id_df[(id_df[T1A_COL] == mode) & (id_df[T1B_COL] == dim)]
            if len(subset) > 0:
                grid[i, j] = float(subset[AUROC_COL].mean())

    fig, ax = plt.subplots(figsize=figure_size("full", aspect=1.0))
    im = ax.imshow(
        grid,
        cmap="Blues",
        aspect="auto",
        interpolation="nearest",
        vmin=0.835,
        vmax=0.848,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Test AUROC (3 seeds)")

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(modes)
    ax.set_xlabel("PID Embedding Dimension (T1-b)")
    ax.set_ylabel("PID Embedding Mode (T1-a)")

    # Annotate cells
    for i in range(len(modes)):
        for j in range(len(dims)):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.4f}",
                    ha="center", va="center",
                    color="black" if val > 0.841 else "white",
                )
            else:
                ax.text(j, i, "—", ha="center", va="center", color="gray")

    _save(fig, "figure-auroc_heatmap_pid_mode_x_embed_dim")
    logger.info("Heatmap done")


# ─── Figure 4: val_auroc_by_tokenizer (Fix 4) — x-axis 0–30 ─────────────────


def plot_val_auroc_tokenizer(ch4: pd.DataFrame) -> None:
    """Fix 4: training curves by tokenizer family, x-axis cropped to 0–30."""
    group_curves: dict[str, list[tuple]] = {}

    for _, row in ch4.iterrows():
        run_name = row[RUN_NAME_COL]
        tokenizer = row[T1_COL]
        if pd.isna(run_name) or pd.isna(tokenizer):
            continue
        scalars_path = RUNS_ROOT / run_name / "facts" / "scalars.csv"
        if not scalars_path.exists():
            logger.warning("Missing scalars: %s", scalars_path)
            continue
        scalars = pd.read_csv(scalars_path)
        val_rows = scalars[scalars["split"] == "val"] if "split" in scalars.columns else scalars
        if "metric_auroc" not in val_rows.columns:
            continue
        epochs = val_rows["epoch"].astype(int).values
        values = val_rows["metric_auroc"].astype(float).values
        group_curves.setdefault(str(tokenizer), []).append((epochs, values))

    if not group_curves:
        logger.error("No per-epoch data loaded — skipping val_auroc_by_tokenizer")
        return

    XLIM_MAX = 30
    fig, ax = plt.subplots(figsize=figure_size("full"))

    order = ["raw", "binned", "identity"]
    for tokenizer in order:
        if tokenizer not in group_curves:
            continue
        color = _TOKENIZER_COLORS[tokenizer]
        all_epochs_list = []
        all_values_list = []

        for epochs, values in group_curves[tokenizer]:
            mask = epochs <= XLIM_MAX
            e_crop = epochs[mask]
            v_crop = values[mask]
            ax.plot(e_crop, v_crop, color=color, alpha=0.2, linewidth=0.8, zorder=1)
            all_epochs_list.append(e_crop)
            all_values_list.append(v_crop)

        # Mean + std band
        n_ep = XLIM_MAX + 1
        mean_vals = np.full(n_ep, np.nan)
        std_vals = np.full(n_ep, np.nan)
        for ep in range(n_ep):
            at_ep = [v[ep] for v, e in zip(all_values_list, all_epochs_list) if len(v) > ep]
            if at_ep:
                mean_vals[ep] = np.mean(at_ep)
                std_vals[ep] = np.std(at_ep) if len(at_ep) > 1 else 0.0

        ep_range = np.arange(n_ep)
        valid = ~np.isnan(mean_vals)
        n_runs = len(all_values_list)
        ax.plot(
            ep_range[valid],
            mean_vals[valid],
            color=color,
            linewidth=2.5,
            label=f"{tokenizer} (n={n_runs})",
            zorder=2,
        )
        ax.fill_between(
            ep_range[valid],
            mean_vals[valid] - std_vals[valid],
            mean_vals[valid] + std_vals[valid],
            color=color,
            alpha=0.15,
            zorder=1,
        )

    ax.set_xlim([0, XLIM_MAX])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUROC")
    ax.legend()

    _save(fig, "figure-val_auroc_by_tokenizer")
    logger.info("Fix 4 done: val_auroc_by_tokenizer (xlim=0–30)")


# ─── Figure 5: roc_curves_by_tokenizer (Fix 5 — style compliance) ─────────────


def plot_roc_curves_tokenizer(ch4: pd.DataFrame) -> None:
    """ROC curves by tokenizer family, with per-seed bands.

    summary.json is flat: keys are run names, values have 'roc_curves' -> {'1': {'fpr', 'tpr'}}.
    Tokenizer is looked up from the CSV by run name.
    """
    summary_path = OLD_REPORT / "inference" / "summary.json"
    if not summary_path.exists():
        logger.warning("No summary.json found at %s — skipping ROC curves", summary_path)
        return

    with open(summary_path) as f:
        summary = json.load(f)

    # Build run_name -> tokenizer lookup from CSV
    run_to_tokenizer: dict[str, str] = dict(
        zip(ch4[RUN_NAME_COL].astype(str), ch4[T1_COL].astype(str))
    )

    tokenizer_rocs: dict[str, list[tuple]] = {}
    for run_name, run_data in summary.items():
        if not isinstance(run_data, dict):
            continue
        tokenizer = run_to_tokenizer.get(run_name)
        if tokenizer is None:
            continue
        roc = run_data.get("roc_curves", {}).get("1", {})
        fpr = roc.get("fpr")
        tpr = roc.get("tpr")
        if fpr is None or tpr is None:
            continue
        tokenizer_rocs.setdefault(tokenizer, []).append(
            (np.array(fpr), np.array(tpr))
        )

    if not tokenizer_rocs:
        logger.warning(
            "No per-run ROC data in summary.json (only scalar metrics stored). "
            "Copying existing PNG from old report as fallback."
        )
        src = OLD_REPORT / "inference" / "figures" / "figure-roc_curves_by_tokenizer.png"
        dst = OUT_DIR / "figure-roc_curves_by_tokenizer.png"
        if src.exists():
            shutil.copy2(src, dst)
            logger.info("Copied existing ROC figure to %s (PNG, not PDF — style not updated)", dst)
        else:
            logger.error("Source PNG not found either: %s", src)
        return

    fig, ax = plt.subplots(figsize=figure_size("full", aspect=1.0))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    order = ["raw", "binned", "identity"]
    for tokenizer in order:
        if tokenizer not in tokenizer_rocs:
            continue
        color = _TOKENIZER_COLORS[tokenizer]
        curves = tokenizer_rocs[tokenizer]

        all_fpr = np.linspace(0, 1, 250)
        all_tpr = np.zeros((len(curves), 250))
        for idx, (fpr, tpr) in enumerate(curves):
            all_tpr[idx] = np.interp(all_fpr, fpr, tpr)
            ax.plot(fpr, tpr, color=color, alpha=0.2, linewidth=0.8)

        mean_tpr = all_tpr.mean(axis=0)
        std_tpr = all_tpr.std(axis=0)
        ax.plot(all_fpr, mean_tpr, color=color, linewidth=2.5, label=tokenizer)
        ax.fill_between(
            all_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
            color=color, alpha=0.15,
        )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    _save(fig, "figure-roc_curves_by_tokenizer")
    logger.info("ROC curves by tokenizer saved")


# ─── Figure 6: auroc_bar_by_features (Fix 6) ──────────────────────────────────


def plot_auroc_bar_features(ch4: pd.DataFrame) -> None:
    """Fix 6: identity tokenizer only; unknown bar removed."""
    identity_df = ch4[ch4[T1_COL] == "identity"].copy()

    feat_vals: dict[str, list[float]] = {}
    for _, row in identity_df.iterrows():
        feat = row[D01_COL]
        auroc = row[AUROC_COL]
        if pd.isna(feat) or pd.isna(auroc):
            continue
        feat_vals.setdefault(str(feat), []).append(float(auroc))

    order = ["[0,1,2,3]", "[1,2,3]"]
    groups = [g for g in order if g in feat_vals]
    all_vals = [feat_vals[g] for g in groups]
    means = [float(np.mean(v)) for v in all_vals]
    stds = [float(np.std(v)) if len(v) > 1 else 0.0 for v in all_vals]
    colors = [_D01_COLORS.get(g, CATEGORICAL_COLORS[0]) for g in groups]

    label_map = {
        "[0,1,2,3]": "E+pT+η+φ\n(all features)",
        "[1,2,3]": "pT+η+φ\n(no energy)",
    }
    labels = [label_map.get(g, g) for g in groups]

    fig, ax = plt.subplots(figsize=figure_size("two_thirds"))
    _bar_with_points(ax, groups, means, stds, all_vals, labels=labels, colors=colors)
    ax.set_xlabel("Feature Set (D01)")
    ax.set_ylabel("Test AUROC")

    _save(fig, "figure-auroc_bar_by_features")
    for g, m, s in zip(groups, means, stds):
        logger.info("  D01=%s: mean=%.4f std=%.4f n=%d", g, m, s, len(feat_vals[g]))


# ─── Figure 7: auroc_bar_by_met (Fix 7) ───────────────────────────────────────


def plot_auroc_bar_met(ch4: pd.DataFrame) -> None:
    """Fix 7: MET bar chart from CSV values. Exp 4B only."""
    exp4b = ch4[ch4[GROUP_COL].str.contains("exp4b", na=False)].copy()
    met_vals: dict[str, list[float]] = {}
    for _, row in exp4b.iterrows():
        met = row[D02_COL]
        auroc = row[AUROC_COL]
        if pd.isna(met) or pd.isna(auroc):
            continue
        key = "MET=True" if bool(met) else "MET=False"
        met_vals.setdefault(key, []).append(float(auroc))

    order = ["MET=False", "MET=True"]
    groups = [g for g in order if g in met_vals]
    all_vals = [met_vals[g] for g in groups]
    means = [float(np.mean(v)) for v in all_vals]
    stds = [float(np.std(v)) if len(v) > 1 else 0.0 for v in all_vals]
    colors = [_D02_COLORS.get(g, CATEGORICAL_COLORS[0]) for g in groups]

    fig, ax = plt.subplots(figsize=figure_size("two_thirds"))
    _bar_with_points(ax, groups, means, stds, all_vals, colors=colors)
    ax.set_xlabel("MET Treatment (D02)")
    ax.set_ylabel("Test AUROC")

    _save(fig, "figure-auroc_bar_by_met")
    delta = means[groups.index("MET=True")] - means[groups.index("MET=False")]
    logger.info("Fix 7 MET | delta=%.4f (MET=True minus MET=False)", delta)
    logger.info("  MET=False: mean=%.4f n=%d", means[groups.index("MET=False")], len(met_vals["MET=False"]))
    logger.info("  MET=True:  mean=%.4f n=%d", means[groups.index("MET=True")], len(met_vals["MET=True"]))


# ─── Figure 8: auroc_bar_by_shuffle (Fix 8) ───────────────────────────────────


def plot_auroc_bar_shuffle(ch4: pd.DataFrame) -> None:
    """Fix 8: shuffle bar chart from CSV values. Exp 4B only."""
    exp4b = ch4[ch4[GROUP_COL].str.contains("exp4b", na=False)].copy()
    shuffle_vals: dict[str, list[float]] = {}
    for _, row in exp4b.iterrows():
        order = row[D03_COL]
        auroc = row[AUROC_COL]
        if pd.isna(order) or pd.isna(auroc):
            continue
        shuffle_vals.setdefault(str(order), []).append(float(auroc))

    display_order = ["input_order", "shuffled"]
    groups = [g for g in display_order if g in shuffle_vals]
    all_vals = [shuffle_vals[g] for g in groups]
    means = [float(np.mean(v)) for v in all_vals]
    stds = [float(np.std(v)) if len(v) > 1 else 0.0 for v in all_vals]
    colors = [_D03_COLORS.get(g, CATEGORICAL_COLORS[0]) for g in groups]

    label_map = {
        "input_order": "input_order\n(consistent)",
        "shuffled": "shuffled\n(permuted)",
    }
    labels = [label_map.get(g, g) for g in groups]

    fig, ax = plt.subplots(figsize=figure_size("two_thirds"))
    _bar_with_points(ax, groups, means, stds, all_vals, labels=labels, colors=colors)
    ax.set_xlabel("Token Ordering (D03)")
    ax.set_ylabel("Test AUROC")

    _save(fig, "figure-auroc_bar_by_shuffle")
    if "input_order" in shuffle_vals and "shuffled" in shuffle_vals:
        delta = np.mean(shuffle_vals["shuffled"]) - np.mean(shuffle_vals["input_order"])
        logger.info("Fix 8 shuffle | delta=%.4f (shuffled minus input_order)", delta)
    for g, m, s in zip(groups, means, stds):
        logger.info("  D03=%s: mean=%.4f std=%.4f n=%d", g, m, s, len(shuffle_vals[g]))


# ─── Figure 9: val_auroc_by_pid_mode ─────────────────────────────────────────


def plot_val_auroc_pid_mode(ch4: pd.DataFrame) -> None:
    """Training curves grouped by PID embedding mode (T1-a), identity tokenizer only."""
    id_df = ch4[ch4[T1_COL] == "identity"]
    group_curves: dict[str, list[tuple]] = {}

    for _, row in id_df.iterrows():
        run_name = row[RUN_NAME_COL]
        mode = row[T1A_COL]
        if pd.isna(run_name) or pd.isna(mode):
            continue
        scalars_path = RUNS_ROOT / run_name / "facts" / "scalars.csv"
        if not scalars_path.exists():
            continue
        scalars = pd.read_csv(scalars_path)
        val_rows = scalars[scalars["split"] == "val"] if "split" in scalars.columns else scalars
        if "metric_auroc" not in val_rows.columns:
            continue
        epochs = val_rows["epoch"].astype(int).values
        values = val_rows["metric_auroc"].astype(float).values
        group_curves.setdefault(str(mode), []).append((epochs, values))

    if not group_curves:
        logger.warning("No scalar data for pid_mode curves — skipping")
        return

    fig, ax = plt.subplots(figsize=figure_size("full"))
    XLIM_MAX = 50

    for mode in ["fixed_random", "learned", "one_hot"]:
        if mode not in group_curves:
            continue
        color = _PID_MODE_COLORS.get(mode, CATEGORICAL_COLORS[0])
        all_epochs_list, all_values_list = [], []
        for epochs, values in group_curves[mode]:
            mask = epochs <= XLIM_MAX
            ax.plot(epochs[mask], values[mask], color=color, alpha=0.2, linewidth=0.8)
            all_epochs_list.append(epochs[mask])
            all_values_list.append(values[mask])

        n_ep = XLIM_MAX + 1
        mean_vals = np.full(n_ep, np.nan)
        std_vals = np.full(n_ep, np.nan)
        for ep in range(n_ep):
            at_ep = [v[ep] for v, e in zip(all_values_list, all_epochs_list) if len(v) > ep]
            if at_ep:
                mean_vals[ep] = np.mean(at_ep)
                std_vals[ep] = np.std(at_ep) if len(at_ep) > 1 else 0.0

        ep_range = np.arange(n_ep)
        valid = ~np.isnan(mean_vals)
        ax.plot(ep_range[valid], mean_vals[valid], color=color, linewidth=2.5,
                label=f"{mode} (n={len(all_values_list)})", zorder=2)
        ax.fill_between(ep_range[valid], mean_vals[valid] - std_vals[valid],
                        mean_vals[valid] + std_vals[valid], color=color, alpha=0.15)

    ax.set_xlim(left=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUROC")
    ax.legend()

    _save(fig, "figure-val_auroc_by_pid_mode")
    logger.info("val_auroc_by_pid_mode done")


# ─── Figure 10: val_auroc_by_met ─────────────────────────────────────────────


def plot_val_auroc_met(ch4: pd.DataFrame) -> None:
    """Training curves grouped by MET treatment (D02), Exp 4B."""
    exp4b = ch4[ch4[GROUP_COL].str.contains("exp4b", na=False)]
    group_curves: dict[str, list[tuple]] = {}

    for _, row in exp4b.iterrows():
        run_name = row[RUN_NAME_COL]
        met = row[D02_COL]
        if pd.isna(run_name) or pd.isna(met):
            continue
        key = "MET=True" if bool(met) else "MET=False"
        scalars_path = RUNS_ROOT / run_name / "facts" / "scalars.csv"
        if not scalars_path.exists():
            continue
        scalars = pd.read_csv(scalars_path)
        val_rows = scalars[scalars["split"] == "val"] if "split" in scalars.columns else scalars
        if "metric_auroc" not in val_rows.columns:
            continue
        epochs = val_rows["epoch"].astype(int).values
        values = val_rows["metric_auroc"].astype(float).values
        group_curves.setdefault(key, []).append((epochs, values))

    if not group_curves:
        logger.warning("No scalar data for MET curves — skipping")
        return

    fig, ax = plt.subplots(figsize=figure_size("full"))

    for key, color in _D02_COLORS.items():
        if key not in group_curves:
            continue
        all_epochs_list, all_values_list = [], []
        XLIM = 50
        for epochs, values in group_curves[key]:
            ax.plot(epochs, values, color=color, alpha=0.2, linewidth=0.8)
            all_epochs_list.append(epochs)
            all_values_list.append(values)

        n_ep = XLIM + 1
        mean_vals = np.full(n_ep, np.nan)
        std_vals = np.full(n_ep, np.nan)
        for ep in range(n_ep):
            at_ep = [v[ep] for v, e in zip(all_values_list, all_epochs_list) if len(v) > ep]
            if at_ep:
                mean_vals[ep] = np.mean(at_ep)
                std_vals[ep] = np.std(at_ep) if len(at_ep) > 1 else 0.0
        ep_range = np.arange(n_ep)
        valid = ~np.isnan(mean_vals)
        ax.plot(ep_range[valid], mean_vals[valid], color=color, linewidth=2.5,
                label=f"{key} (n={len(all_values_list)})", zorder=2)
        ax.fill_between(ep_range[valid], mean_vals[valid] - std_vals[valid],
                        mean_vals[valid] + std_vals[valid], color=color, alpha=0.15)

    ax.set_xlim(left=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUROC")
    ax.legend()

    _save(fig, "figure-val_auroc_by_met")
    logger.info("val_auroc_by_met done")


# ─── Figure 11: val_auroc_by_shuffle ─────────────────────────────────────────


def plot_val_auroc_shuffle(ch4: pd.DataFrame) -> None:
    """Training curves grouped by token ordering (D03), Exp 4B."""
    exp4b = ch4[ch4[GROUP_COL].str.contains("exp4b", na=False)]
    group_curves: dict[str, list[tuple]] = {}

    for _, row in exp4b.iterrows():
        run_name = row[RUN_NAME_COL]
        order = row[D03_COL]
        if pd.isna(run_name) or pd.isna(order):
            continue
        scalars_path = RUNS_ROOT / run_name / "facts" / "scalars.csv"
        if not scalars_path.exists():
            continue
        scalars = pd.read_csv(scalars_path)
        val_rows = scalars[scalars["split"] == "val"] if "split" in scalars.columns else scalars
        if "metric_auroc" not in val_rows.columns:
            continue
        epochs = val_rows["epoch"].astype(int).values
        values = val_rows["metric_auroc"].astype(float).values
        group_curves.setdefault(str(order), []).append((epochs, values))

    if not group_curves:
        logger.warning("No scalar data for shuffle curves — skipping")
        return

    fig, ax = plt.subplots(figsize=figure_size("full"))

    for key in ["input_order", "shuffled"]:
        if key not in group_curves:
            continue
        color = _D03_COLORS[key]
        all_epochs_list, all_values_list = [], []
        XLIM = 50
        for epochs, values in group_curves[key]:
            ax.plot(epochs, values, color=color, alpha=0.2, linewidth=0.8)
            all_epochs_list.append(epochs)
            all_values_list.append(values)

        n_ep = XLIM + 1
        mean_vals = np.full(n_ep, np.nan)
        std_vals = np.full(n_ep, np.nan)
        for ep in range(n_ep):
            at_ep = [v[ep] for v, e in zip(all_values_list, all_epochs_list) if len(v) > ep]
            if at_ep:
                mean_vals[ep] = np.mean(at_ep)
                std_vals[ep] = np.std(at_ep) if len(at_ep) > 1 else 0.0
        ep_range = np.arange(n_ep)
        valid = ~np.isnan(mean_vals)
        ax.plot(ep_range[valid], mean_vals[valid], color=color, linewidth=2.5,
                label=f"{key} (n={len(all_values_list)})", zorder=2)
        ax.fill_between(ep_range[valid], mean_vals[valid] - std_vals[valid],
                        mean_vals[valid] + std_vals[valid], color=color, alpha=0.15)

    ax.set_xlim(left=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUROC")
    ax.legend()

    _save(fig, "figure-val_auroc_by_shuffle")
    logger.info("val_auroc_by_shuffle done")


# ─── Figure 12: failure_analysis_raw_vs_identity ─────────────────────────────


def plot_failure_analysis(ch4: pd.DataFrame) -> None:
    """Two-panel failure analysis: scatter (raw vs identity scores) + category bar.

    summary.json is flat: keys are run names, values have 'per_event_scores' and
    'per_event_labels'. We pair the best raw run against the best identity run
    (one_hot, highest AUROC) that share the same test set.

    Best raw:     run_20260410-160141_ch4_input_repr_exp4a_1_raw_job002    (0.8056)
    Best identity: run_20260410-162037_ch4_input_repr_exp4a_2_job010       (0.8451, one_hot)
    """
    summary_path = OLD_REPORT / "inference" / "summary.json"
    if not summary_path.exists():
        logger.warning("No summary.json — cannot produce failure analysis")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    BEST_RAW = "run_20260410-160141_ch4_input_repr_exp4a_1_raw_job002"
    BEST_ID = "run_20260410-162037_ch4_input_repr_exp4a_2_job010"

    raw_data = summary.get(BEST_RAW)
    id_data = summary.get(BEST_ID)

    if raw_data is None or id_data is None:
        logger.warning("Could not find expected runs in summary.json — copying existing PNG")
        _copy_failure_analysis_png()
        return

    raw_scores = np.array(raw_data.get("per_event_scores", []))
    id_scores = np.array(id_data.get("per_event_scores", []))
    labels_true = np.array(raw_data.get("per_event_labels", []))

    if len(raw_scores) == 0 or len(id_scores) == 0:
        logger.warning("Empty per_event_scores — copying existing PNG")
        _copy_failure_analysis_png()
        return

    if len(raw_scores) != len(id_scores):
        logger.warning(
            "Score length mismatch: raw=%d id=%d — copying existing PNG",
            len(raw_scores), len(id_scores)
        )
        _copy_failure_analysis_png()
        return

    threshold = 0.5
    raw_pred = (raw_scores >= threshold).astype(int)
    id_pred = (id_scores >= threshold).astype(int)
    true_label = labels_true.astype(int)

    both_correct = int(((raw_pred == true_label) & (id_pred == true_label)).sum())
    both_wrong = int(((raw_pred != true_label) & (id_pred != true_label)).sum())
    id_fixed = int(((raw_pred != true_label) & (id_pred == true_label)).sum())
    raw_fixed = int(((raw_pred == true_label) & (id_pred != true_label)).sum())
    total = len(raw_scores)

    logger.info(
        "Failure analysis counts: both_correct=%d, id_fixed=%d, raw_fixed=%d, both_wrong=%d, total=%d",
        both_correct, id_fixed, raw_fixed, both_wrong, total
    )

    _plot_failure_analysis_two_panel(
        raw_scores, id_scores, true_label,
        both_correct, both_wrong, id_fixed, raw_fixed, total
    )


def _copy_failure_analysis_png() -> None:
    """Copy the existing PNG from the old report (per-event scores not available in summary.json)."""
    src = OLD_REPORT / "inference" / "figures" / "figure-failure_analysis_raw_vs_identity.png"
    dst = OUT_DIR / "figure-failure_analysis_raw_vs_identity.png"
    if src.exists():
        shutil.copy2(src, dst)
        logger.info("Copied existing failure analysis PNG to %s (PNG, not PDF — style not updated)", dst)
    else:
        logger.error("Source PNG not found: %s", src)
        logger.info("Falling back to bar chart with hardcoded counts from evidence note")
        _plot_failure_analysis_bar_only()


def _plot_failure_analysis_bar_only() -> None:
    """Bar chart using hardcoded counts from the tex file commentary.

    Counts: both_correct=20121 (66.6%), id_fixed=2515 (8.3%),
            raw_fixed=2014 (6.7%), both_wrong=5557 (18.4%).
    Source: thesis_report/mainmatter/04_best_input_representation.tex §failure analysis note.
    """
    categories = ["Both\ncorrect\n(66.6%)", "PID\nhelped\n(8.3%)", "PID\nhurt\n(6.7%)", "Both\nwrong\n(18.4%)"]
    counts = [20121, 2515, 2014, 5557]
    total = sum(counts)
    fracs = [c / total for c in counts]
    colors = [
        axis_color("H"),
        axis_color("T"),
        axis_color("C"),
        axis_color("baseline"),
    ]

    fig, ax = plt.subplots(figsize=figure_size("full"))
    x = np.arange(len(categories))
    ax.bar(x, fracs, 0.6, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel("Outcome Category")
    ax.set_ylabel("Fraction of test events")
    ax.set_ylim([0, 0.75])

    for xi, frac in zip(x, fracs):
        ax.text(xi, frac + 0.01, f"{frac:.1%}", ha="center", va="bottom")

    _save(fig, "figure-failure_analysis_raw_vs_identity")
    logger.info("Failure analysis (bar only, hardcoded counts from tex commentary)")


def _plot_failure_analysis_two_panel(
    raw_scores, id_scores, labels_true,
    both_correct, both_wrong, id_fixed, raw_fixed, total
) -> None:
    """Two-panel: scatter (left) + fractional bar (right)."""
    fig, axes = plt.subplots(1, 2, figsize=figure_size("full"))
    ax_scatter, ax_bar = axes

    # Scatter panel
    # Assign outcome category per event
    raw_pred = (raw_scores >= 0.5).astype(int)
    id_pred = (id_scores >= 0.5).astype(int)
    true_label = labels_true.astype(int)

    mask_both_correct = (raw_pred == true_label) & (id_pred == true_label)
    mask_both_wrong = (raw_pred != true_label) & (id_pred != true_label)
    mask_id_fixed = (raw_pred != true_label) & (id_pred == true_label)
    mask_raw_fixed = (raw_pred == true_label) & (id_pred != true_label)

    scatter_cfg = [
        (mask_both_correct, axis_color("H"), "both correct", 0.08),
        (mask_both_wrong, axis_color("baseline"), "both wrong", 0.3),
        (mask_id_fixed, axis_color("T"), "PID helped", 0.6),
        (mask_raw_fixed, axis_color("C"), "PID hurt", 0.6),
    ]

    for mask, color, label, alpha in scatter_cfg:
        ax_scatter.scatter(
            raw_scores[mask], id_scores[mask],
            s=15, alpha=alpha, color=color, label=label, rasterized=True,
        )

    ax_scatter.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_scatter.set_xlim([0, 1])
    ax_scatter.set_ylim([0, 1])
    ax_scatter.set_xlabel("raw score")
    ax_scatter.set_ylabel("identity score")
    ax_scatter.legend(loc="upper left")

    # Bar panel
    categories = ["Both\ncorrect", "PID\nhelped", "PID\nhurt", "Both\nwrong"]
    fracs = [both_correct / total, id_fixed / total, raw_fixed / total, both_wrong / total]
    colors = [axis_color("H"), axis_color("T"), axis_color("C"), axis_color("baseline")]
    x = np.arange(len(categories))
    ax_bar.bar(x, fracs, 0.6, color=colors, edgecolor="black", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(categories)
    ax_bar.set_ylabel("Fraction of events")
    ax_bar.set_ylim([0, 0.75])

    # Annotate fracs
    for xi, frac in zip(x, fracs):
        ax_bar.text(xi, frac + 0.01, f"{frac:.1%}", ha="center", va="bottom")

    _save(fig, "figure-failure_analysis_raw_vs_identity")
    logger.info(
        "Failure analysis (two-panel) | total=%d, both_correct=%.1f%%, id_fixed=%.1f%%, "
        "raw_fixed=%.1f%%, both_wrong=%.1f%%",
        total, 100 * both_correct / total, 100 * id_fixed / total,
        100 * raw_fixed / total, 100 * both_wrong / total,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ch4 = load_ch4_df()

    logger.info("=== Figure 1: auroc_bar_by_tokenizer (Fix 1: identity dim>=16) ===")
    plot_auroc_bar_tokenizer(ch4)

    logger.info("=== Figure 2: auroc_bar_by_pid_mode (Fix 2: dim>=16 filter) ===")
    plot_auroc_bar_pid_mode(ch4)

    # Fix 3: auroc_bar_by_dim — NOT generated (removed per reviewer decision)

    logger.info("=== Figure 3: auroc_heatmap_pid_mode_x_embed_dim ===")
    plot_auroc_heatmap(ch4)

    logger.info("=== Figure 4: val_auroc_by_tokenizer (Fix 4: xlim 0-30) ===")
    plot_val_auroc_tokenizer(ch4)

    logger.info("=== Figure 5: roc_curves_by_tokenizer (Fix 5: style compliance) ===")
    plot_roc_curves_tokenizer(ch4)

    logger.info("=== Figure 6: auroc_bar_by_features (Fix 6: identity only, no unknown) ===")
    plot_auroc_bar_features(ch4)

    logger.info("=== Figure 7: auroc_bar_by_met (Fix 7: CSV values) ===")
    plot_auroc_bar_met(ch4)

    logger.info("=== Figure 8: auroc_bar_by_shuffle (Fix 8: CSV values) ===")
    plot_auroc_bar_shuffle(ch4)

    logger.info("=== Figure 9: val_auroc_by_pid_mode ===")
    plot_val_auroc_pid_mode(ch4)

    logger.info("=== Figure 10: val_auroc_by_met ===")
    plot_val_auroc_met(ch4)

    logger.info("=== Figure 11: val_auroc_by_shuffle ===")
    plot_val_auroc_shuffle(ch4)

    logger.info("=== Figure 12: failure_analysis_raw_vs_identity ===")
    plot_failure_analysis(ch4)

    logger.info("All done. Output: %s", OUT_DIR)
    logger.info("Files:")
    for p in sorted(OUT_DIR.glob("*.pdf")):
        logger.info("  %s", p.name)


if __name__ == "__main__":
    main()
