"""ch8_final_plots.py — Phase B final Ch8 figures for thesis handoff.

Generates all Ch8 publication-ready figures using the thesis style system.

Figures generated
-----------------
  §8.1 Database audit
    1. audit_run_counts_by_spec.pdf    — bar chart of N runs per eval_v2/spec_version
    2. audit_missingness_heatmap.pdf   — heatmap: axis col × G3 cohort, colour = frac NaN
    3. audit_cramer_v_heatmap.pdf      — Cramér's V between axis families (primary cohort)

  §8.2 Layer 1 marginals (primary cohort only)
    4. marginals_ranked_range_bar.pdf  — horizontal bar: AUROC range per axis, ranked
    5. marginals_top5_boxdot.pdf       — box+dot plots for top-5 axes + R5_Seed noise floor
    6. marginals_seed_noise_floor.pdf  — histogram of within-group AUROC std (seed noise)

Source of truth: thesis_results/ch8_streamlined/05_ch8_streamlined_primary.csv
Missingness data: thesis_results/04_cleaned_backfilled_analysis_ready.csv
Output: /data/atlas/users/nterlind/outputs/reports/report_ch8_final/figures/
Format: PDF at 300 DPI via io_utils.save_figure()
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from thesis_ml.reports.plots.style import (
    CATEGORICAL_COLORS,
    _CATEGORICAL_ORDER,
    apply_thesis_style,
    axis_color,
    figure_size,
)

apply_thesis_style()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path("/project/atlas/users/nterlind/Thesis-Code")

PRIMARY_CSV = REPO_ROOT / "thesis_results/ch8_streamlined/05_ch8_streamlined_primary.csv"
RAW_04_CSV = REPO_ROOT / "thesis_results/04_cleaned_backfilled_analysis_ready.csv"

OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch8_patching_G1_2/figures")

# ─── Constants ────────────────────────────────────────────────────────────────

AUROC_COL = "eval_v2/test_auroc"
SPEC_COL = "eval_v2/spec_version"
G3_COL = "config/axes/G3_Classification Task"

_CFG_LOG = {"fig_format": "pdf", "dpi": 300}

# Families known to axis_color (from _CATEGORICAL_ORDER)
_KNOWN_FAMILIES = set(_CATEGORICAL_ORDER)

# Fallback tab10 palette for families not in _CATEGORICAL_ORDER (G, K, L, M, P, R, S)
_EXTRA_FAMILIES = ["G", "K", "L", "M", "P", "R", "S"]
_EXTRA_COLORS = plt.cm.tab10(np.linspace(0, 1, len(_EXTRA_FAMILIES)))
_EXTRA_COLOR_MAP = {fam: _EXTRA_COLORS[i] for i, fam in enumerate(_EXTRA_FAMILIES)}


def _family_color(fam: str):
    """Return a consistent colour for a given family letter."""
    if fam in _KNOWN_FAMILIES:
        return axis_color(fam)
    return _EXTRA_COLOR_MAP.get(fam, "gray")


def _axis_family(col: str) -> str:
    """Extract the uppercase letter family prefix from an axis column name.

    e.g. 'config/axes/A1_Normalization Policy' -> 'A'
         'config/axes/B1-G1_Global-Conditioned Mode' -> 'B'
    """
    short = col.replace("config/axes/", "")
    prefix = short.split("_")[0]
    m = re.match(r"^([A-Z]+)", prefix)
    return m.group(1) if m else "?"


# ─── Save helper ──────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, fname: str) -> Path:
    """Save via io_utils-compatible call (PDF, 300 DPI, bbox=tight)."""
    from thesis_ml.monitoring.io_utils import save_figure

    path = save_figure(fig, OUT_DIR, fname, _CFG_LOG)
    logger.info("Saved %s", path)
    plt.close(fig)
    return path


# ─── Load data ────────────────────────────────────────────────────────────────


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (primary_df, raw04_df)."""
    primary = pd.read_csv(PRIMARY_CSV)
    raw04 = pd.read_csv(RAW_04_CSV)
    logger.info("Primary CSV: %d rows", len(primary))
    logger.info("Raw 04 CSV: %d rows", len(raw04))
    return primary, raw04


# ─── Figure 1: audit_run_counts_by_spec ───────────────────────────────────────


def plot_audit_run_counts_by_spec(df: pd.DataFrame) -> None:
    """Bar chart of N runs per eval_v2/spec_version value."""
    counts = df[SPEC_COL].value_counts().sort_values(ascending=False)
    labels = counts.index.tolist()
    values = counts.values.tolist()

    # Truncate long hash strings for display
    display_labels = []
    for lbl in labels:
        if len(str(lbl)) > 20:
            display_labels.append(str(lbl)[:12] + "...")
        else:
            display_labels.append(str(lbl))

    fig, ax = plt.subplots(figsize=figure_size("full"))
    x = np.arange(len(labels))
    colors = [CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)] for i in range(len(labels))]
    ax.bar(x, values, 0.6, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=30, ha="right")
    ax.set_xlabel("Eval spec version")
    ax.set_ylabel("Number of runs")

    for xi, v in zip(x, values):
        ax.text(xi, v + 2, str(v), ha="center", va="bottom")

    _save(fig, "audit_run_counts_by_spec")
    logger.info("Spec version counts: %s", dict(zip(labels, values)))


# ─── Figure 2: audit_missingness_heatmap ──────────────────────────────────────


def plot_audit_missingness_heatmap(raw04: pd.DataFrame) -> None:
    """Heatmap: rows = axis column short names, columns = G3 cohort, colour = frac NaN.

    Uses raw 04 CSV before inactive encoding — actual NaN values.
    Only shows config/axes/* columns that have any NaN in at least one cohort.
    """
    axis_cols = [c for c in raw04.columns if c.startswith("config/axes/")]

    cohorts = sorted(raw04[G3_COL].dropna().unique())

    # Build matrix: rows = axis cols with any NaN, cols = cohorts
    rows = []
    row_labels = []
    for col in axis_cols:
        frac_by_cohort = []
        for coh in cohorts:
            sub = raw04[raw04[G3_COL] == coh]
            frac_by_cohort.append(sub[col].isna().mean())
        if any(f > 0 for f in frac_by_cohort):
            rows.append(frac_by_cohort)
            row_labels.append(col.replace("config/axes/", ""))

    if not rows:
        logger.warning("No NaN found in axis cols — skipping missingness heatmap")
        return

    mat = np.array(rows)  # shape: (n_cols_with_nan, n_cohorts)

    # Sort by max frac NaN descending
    order = np.argsort(-mat.max(axis=1))
    mat = mat[order]
    row_labels = [row_labels[i] for i in order]

    # Shorten cohort labels
    cohort_display = []
    for c in cohorts:
        if len(c) > 22:
            cohort_display.append(c[:20] + "...")
        else:
            cohort_display.append(c)

    n_rows, n_cols = mat.shape
    fig_h = max(6, n_rows * 0.18 + 1.5)
    fig, ax = plt.subplots(figsize=(figure_size("full")[0], fig_h))

    im = ax.imshow(mat, cmap="viridis", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction NaN (raw 04 CSV)")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(cohort_display, rotation=20, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("G3 cohort")
    ax.set_ylabel("Axis column")

    # Annotate cells if small enough
    if n_rows <= 30 and n_cols <= 5:
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if mat[i, j] < 0.5 else "black",
                )

    _save(fig, "audit_missingness_heatmap")
    logger.info("Missingness heatmap: %d axis cols with any NaN, %d cohorts", n_rows, n_cols)


# ─── Figure 3: audit_cramer_v_heatmap ─────────────────────────────────────────


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V between two categorical series (NaN rows dropped pairwise)."""
    mask = x.notna() & y.notna()
    x = x[mask].astype(str)
    y = y[mask].astype(str)
    if len(x) < 5:
        return np.nan
    try:
        ct = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(ct, correction=False)
        n = ct.values.sum()
        k = min(ct.shape) - 1
        if k == 0 or n == 0:
            return np.nan
        return float(np.sqrt(chi2 / (n * k)))
    except Exception:
        return np.nan


def plot_audit_cramer_v_heatmap(df: pd.DataFrame) -> None:
    """Cramér's V between axis families in the primary cohort.

    One cell per pair of axis families; value is the mean pairwise Cramér's V
    across all column pairs belonging to those two families.
    Families are identified by the leading uppercase letter(s).
    Only families with at least 2 non-inactive unique values are included.
    """
    axis_cols = [c for c in df.columns if c.startswith("config/axes/")]

    # Assign each col to a family; keep only cols with >=2 non-inactive unique vals
    family_to_cols: dict[str, list[str]] = {}
    for col in axis_cols:
        fam = _axis_family(col)
        if df[col].dtype == object:
            unique_non_inactive = df[col][df[col] != "inactive"].dropna().unique()
            if len(unique_non_inactive) < 2:
                continue
        family_to_cols.setdefault(fam, []).append(col)

    # Only keep families with at least 1 qualifying col
    families = sorted(family_to_cols.keys())
    n = len(families)
    logger.info("Cramér's V: %d families with active columns", n)

    # Build pairwise matrix
    mat = np.full((n, n), np.nan)
    for i, fam_i in enumerate(families):
        for j, fam_j in enumerate(families):
            cols_i = family_to_cols[fam_i]
            cols_j = family_to_cols[fam_j]
            values = []
            for ci in cols_i:
                for cj in cols_j:
                    if ci == cj:
                        values.append(1.0)
                    else:
                        v = _cramers_v(df[ci], df[cj])
                        if not np.isnan(v):
                            values.append(v)
            if values:
                mat[i, j] = float(np.nanmean(values))

    fig, ax = plt.subplots(figsize=figure_size("full", aspect=1.0))
    im = ax.imshow(mat, cmap="viridis", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Cramér's V")

    ax.set_xticks(range(n))
    ax.set_xticklabels(families, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(families)
    ax.set_xlabel("Axis family")
    ax.set_ylabel("Axis family")

    # Annotate cells (always annotate — n families is small)
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="white" if val < 0.5 else "black",
                )
            else:
                ax.text(j, i, "—", ha="center", va="center", color="gray")

    _save(fig, "audit_cramer_v_heatmap")
    logger.info("Cramér's V heatmap done (%d×%d families)", n, n)


# ─── Figure 4: marginals_ranked_range_bar ─────────────────────────────────────


def _compute_axis_ranges(
    df: pd.DataFrame,
    axis_cols: list[str],
    min_rows_per_level: int = 5,
) -> tuple[dict[str, float], dict[str, str], list[str]]:
    """Compute AUROC mean-range per axis column.

    Returns
    -------
    ranges : dict  col -> max_mean - min_mean
    col_to_family : dict  col -> family prefix
    dropped : list  cols dropped by min_rows_per_level gate
    """
    ranges: dict[str, float] = {}
    col_to_family: dict[str, str] = {}
    dropped: list[str] = []

    for col in axis_cols:
        fam = _axis_family(col)
        col_to_family[col] = fam

        if df[col].dtype != object:
            # Numeric axis — skip for marginal analysis (categorical only)
            dropped.append(col)
            continue

        levels = [lv for lv in df[col].unique() if lv != "inactive" and not pd.isna(lv)]
        if len(levels) < 2:
            dropped.append(col)
            continue

        means = []
        valid_levels = []
        for lv in levels:
            sub = df[(df[col] == lv) & df[AUROC_COL].notna()]
            if len(sub) < min_rows_per_level:
                continue
            means.append(sub[AUROC_COL].mean())
            valid_levels.append(lv)

        if len(valid_levels) < 2:
            dropped.append(col)
            continue

        ranges[col] = float(max(means) - min(means))

    return ranges, col_to_family, dropped


def plot_marginals_ranked_range_bar(
    df: pd.DataFrame,
    seed_noise_floor: float | None = None,
) -> list[str]:
    """Horizontal bar chart: AUROC range per axis (ranked), coloured by family.

    Returns the list of dropped axis column names (< min_rows_per_level gate).
    """
    axis_cols = [c for c in df.columns if c.startswith("config/axes/")]
    ranges, col_to_family, dropped = _compute_axis_ranges(df, axis_cols, min_rows_per_level=5)

    if not ranges:
        logger.error("No axis cols passed the range gate — skipping ranked range bar")
        return dropped

    # Sort by range descending
    sorted_cols = sorted(ranges.items(), key=lambda x: x[1], reverse=True)
    cols_sorted = [c for c, _ in sorted_cols]
    vals_sorted = [v for _, v in sorted_cols]
    families_sorted = [col_to_family[c] for c in cols_sorted]

    # Short display labels
    short_labels = [c.replace("config/axes/", "").split("_")[0] for c in cols_sorted]

    colors = [_family_color(fam) for fam in families_sorted]

    fig_h = max(5, len(cols_sorted) * 0.22 + 1.5)
    fig, ax = plt.subplots(figsize=(figure_size("full")[0], fig_h))

    y = np.arange(len(cols_sorted))
    ax.barh(y, vals_sorted, 0.7, color=colors, edgecolor="black", linewidth=0.5)

    if seed_noise_floor is not None:
        ax.axvline(
            seed_noise_floor,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            label=f"Seed noise floor ({seed_noise_floor:.4f})",
        )
        ax.legend()

    ax.set_yticks(y)
    ax.set_yticklabels(short_labels)
    ax.set_xlabel("AUROC range (max mean − min mean)")
    ax.set_ylabel("Axis")
    ax.invert_yaxis()

    # Add family legend patches
    unique_fams = sorted(set(families_sorted))
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=_family_color(f), label=f) for f in unique_fams]
    ax.legend(handles=handles, title="Family", loc="lower right", fontsize="small")

    _save(fig, "marginals_ranked_range_bar")
    logger.info("Ranked range bar: %d axes plotted, %d dropped", len(cols_sorted), len(dropped))
    return dropped


# ─── Figure 5: marginals_top5_boxdot ──────────────────────────────────────────


def plot_marginals_top5_boxdot(df: pd.DataFrame) -> list[str]:
    """Box+dot plots for the 5 axes with largest AUROC range, plus R5_Seed panel.

    Returns the list of top-5 axis column names (short names).
    """
    axis_cols = [c for c in df.columns if c.startswith("config/axes/")]
    seed_col_list = [c for c in axis_cols if "R5_Seed" in c]
    seed_col = seed_col_list[0] if seed_col_list else None

    ranges, col_to_family, dropped = _compute_axis_ranges(df, axis_cols, min_rows_per_level=5)

    if not ranges:
        logger.error("No axis cols for top-5 boxdot — skipping")
        return []

    sorted_cols = sorted(ranges.items(), key=lambda x: x[1], reverse=True)
    top5_cols = [c for c, _ in sorted_cols[:5]]
    top5_names = [c.replace("config/axes/", "") for c in top5_cols]

    # Include R5_Seed as final panel
    panels = top5_cols.copy()
    panel_labels = top5_names.copy()
    if seed_col and seed_col not in top5_cols:
        panels.append(seed_col)
        panel_labels.append(seed_col.replace("config/axes/", ""))

    n_panels = len(panels)
    fig_w = n_panels * figure_size("half")[0]
    fig_h = figure_size("half")[1] * 2.5
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
    if n_panels == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, col, label in zip(axes, panels, panel_labels):
        fam = col_to_family.get(col, _axis_family(col))
        color = _family_color(fam)

        if df[col].dtype != object:
            ax.set_xlabel(label.split("_")[0])
            ax.text(0.5, 0.5, "numeric\n(no levels)", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        levels = [lv for lv in df[col].unique() if lv != "inactive" and not pd.isna(lv)]
        # Sort levels by median AUROC
        level_data = []
        for lv in levels:
            sub = df[(df[col] == lv) & df[AUROC_COL].notna()]
            if len(sub) >= 1:
                level_data.append((lv, sub[AUROC_COL].values))

        if not level_data:
            continue

        level_data.sort(key=lambda x: np.median(x[1]))
        sorted_levels = [d[0] for d in level_data]
        sorted_vals = [d[1] for d in level_data]

        # Box plot
        bp = ax.boxplot(
            sorted_vals,
            positions=range(len(sorted_levels)),
            widths=0.5,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": color, "alpha": 0.6},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"marker": ""},
        )

        # Jittered dots
        for xi, vals in enumerate(sorted_vals):
            jitter = rng.normal(0, 0.06, len(vals))
            ax.scatter(
                np.full(len(vals), xi) + jitter,
                vals,
                color=color,
                alpha=0.5,
                s=15,
                zorder=3,
            )

        ax.set_xticks(range(len(sorted_levels)))
        ax.set_xticklabels(sorted_levels, rotation=40, ha="right")
        ax.set_xlabel(label.split("_")[0])
        ax.set_ylabel("Test AUROC")

    _save(fig, "marginals_top5_boxdot")
    logger.info("Top-5 boxdot done. Top axes: %s", top5_names)
    return top5_names


# ─── Figure 6: marginals_seed_noise_floor ─────────────────────────────────────


def plot_marginals_seed_noise_floor(df: pd.DataFrame) -> float:
    """Histogram of within-fingerprint AUROC std (seed noise floor).

    Fingerprint = hash of all axis cols except R5_Seed.
    Only groups with >=2 seeds are included.

    Returns
    -------
    float
        Median within-group std (the noise floor value).
    """
    axis_cols = [c for c in df.columns if c.startswith("config/axes/")]
    axis_cols_no_seed = [c for c in axis_cols if "R5_Seed" not in c]

    df = df[df[AUROC_COL].notna()].copy()
    df["_fp"] = df[axis_cols_no_seed].apply(lambda r: hash(tuple(r)), axis=1)

    group_stds = []
    for fp, grp in df.groupby("_fp"):
        if len(grp) >= 2:
            group_stds.append(float(grp[AUROC_COL].std()))

    if not group_stds:
        logger.error("No fingerprint groups with >=2 seeds found — skipping seed noise floor")
        return float("nan")

    group_stds = np.array(group_stds)
    noise_floor = float(np.median(group_stds))

    logger.info("Seed noise floor (median within-group std): %.6f", noise_floor)
    print(f"\n[SEED NOISE FLOOR] Median within-group AUROC std = {noise_floor:.6f}")
    print(f"[SEED NOISE FLOOR] N groups with >=2 seeds = {len(group_stds)}")
    print(f"[SEED NOISE FLOOR] Mean within-group std = {group_stds.mean():.6f}")

    fig, ax = plt.subplots(figsize=figure_size("full"))
    ax.hist(
        group_stds,
        bins=40,
        color=CATEGORICAL_COLORS[0],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )
    ax.axvline(
        noise_floor,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.8,
        label=f"Median = {noise_floor:.4f}",
    )
    ax.set_xlabel("Within-group AUROC std (seed noise)")
    ax.set_ylabel("Number of fingerprint groups")
    ax.legend()

    _save(fig, "marginals_seed_noise_floor")
    return noise_floor


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    primary, raw04 = load_data()

    # ── §8.1 Database audit ───────────────────────────────────────────────────

    logger.info("=== Figure 1: audit_run_counts_by_spec ===")
    plot_audit_run_counts_by_spec(primary)

    logger.info("=== Figure 2: audit_missingness_heatmap ===")
    plot_audit_missingness_heatmap(raw04)

    logger.info("=== Figure 3: audit_cramer_v_heatmap ===")
    plot_audit_cramer_v_heatmap(primary)

    # ── §8.2 Layer 1 marginals ────────────────────────────────────────────────

    logger.info("=== Figure 6: marginals_seed_noise_floor (compute first for range bar) ===")
    noise_floor = plot_marginals_seed_noise_floor(primary)

    logger.info("=== Figure 4: marginals_ranked_range_bar ===")
    dropped = plot_marginals_ranked_range_bar(primary, seed_noise_floor=noise_floor)
    if dropped:
        logger.info("Axes dropped by >=5-rows-per-level gate (%d total):", len(dropped))
        for col in dropped:
            logger.info("  %s", col.replace("config/axes/", ""))

    logger.info("=== Figure 5: marginals_top5_boxdot ===")
    top5 = plot_marginals_top5_boxdot(primary)

    # ── Summary ───────────────────────────────────────────────────────────────

    logger.info("All done. Output: %s", OUT_DIR)
    logger.info("Files written:")
    for p in sorted(OUT_DIR.glob("*.pdf")):
        logger.info("  %s", p)

    print("\n=== Phase B summary ===")
    print(f"Output directory: {OUT_DIR}")
    print(f"Seed noise floor (median within-group std): {noise_floor:.6f}")
    print(f"Top-5 axes by AUROC range: {top5}")
    if dropped:
        print(f"Axes dropped (< 5 rows per level): {len(dropped)}")
        for col in sorted(dropped):
            print(f"  {col.replace('config/axes/', '')}")


if __name__ == "__main__":
    main()
