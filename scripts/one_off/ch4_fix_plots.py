"""ch4_fix_plots.py — regenerate corrected Ch4 figures from the frozen CSV.

Addresses reviewer-flagged issues:

  fig:auroc-bar-tokenizer   Fix 1: restrict identity bar to dim>=16 (+one_hot/num_types)
  fig:auroc-bar-pid-mode    Fix 2: filter to dim>=16 before computing per-mode means
  fig:auroc-bar-dim         Fix 3: removed from which_figures in report config (no replot needed)
  fig:val-auroc-tokenizer   Fix 4: x-axis cropped to epochs 0-30
  fig:roc-tokenizer         Fix 5: caption-level note only (bands already present)
  fig:auroc-bar-features    Fix 6: filter to identity tokenizer rows only (drop 'unknown')
  fig:auroc-bar-met         Fix 7: use CSV values (existing inference summary was wrong)
  fig:auroc-bar-shuffle     Fix 7: use CSV values (existing inference summary was wrong)

Source of truth: thesis_results/03_analysis_ready.csv
Run dir for per-epoch scalars: /data/atlas/users/nterlind/outputs/runs/

Output: /data/atlas/users/nterlind/outputs/reports/report_ch4_fixed_plots/figures/
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path("/project/atlas/Users/nterlind/Thesis-Code")
CSV_PATH = REPO_ROOT / "thesis_results" / "03_analysis_ready.csv"
RUNS_ROOT = Path("/data/atlas/users/nterlind/outputs/runs")
OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch4_fixed_plots/figures")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Column names (from 03_analysis_ready.csv) ────────────────────────────────

T1_COL = "config/axes/T1_Tokenizer Family"
T1A_COL = "config/axes/T1-a_PID Embedding Mode"
T1B_COL = "config/axes/T1-b_PID Embedding Dimension"
D01_COL = "config/axes/D1_Feature Set"
D02_COL = "config/axes/D2_MET Treatment"
D03_COL = "config/axes/D3_Token Ordering"
AUROC_COL = "eval_v2/test_auroc"
GROUP_COL = "meta_run/group"
RUN_NAME_COL = "meta_run/name"

# Ch4 experiment groups (exactly the 57 runs loaded by the original report)
CH4_GROUPS = [
    "exp_20260410-160141_ch4_input_repr_exp4a_1_raw",
    "exp_20260410-160937_ch4_input_repr_exp4a_1_binned",
    "exp_20260410-162037_ch4_input_repr_exp4a_2",
    "exp_20260410-163637_ch4_data_treatment_exp4b",
]

FIG_FORMAT = "png"
DPI = 150


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, fname: str) -> None:
    path = OUT_DIR / f"{fname}.{FIG_FORMAT}"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    logger.info("Saved %s", path)
    plt.close(fig)


def _bar_with_points(
    ax: plt.Axes,
    groups: list,
    means: list[float],
    stds: list[float],
    all_vals: list[list[float]],
    labels: list[str] | None = None,
    color: str = "steelblue",
) -> None:
    """Draw a bar chart with individual seed dots."""
    x = np.arange(len(groups))
    ax.bar(
        x,
        means,
        0.6,
        yerr=stds,
        capsize=5,
        alpha=0.7,
        color=color,
        edgecolor="black",
        zorder=2,
    )
    rng = np.random.default_rng(42)
    for i, (mean, std, vals) in enumerate(zip(means, stds, all_vals, strict=False)):
        jitter = rng.normal(0, 0.05, len(vals))
        ax.scatter(x[i] + jitter, vals, color="black", alpha=0.4, s=25, zorder=3)
        ax.text(
            i,
            mean + std + 0.004,
            f"{mean:.4f}\n(n={len(vals)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels or [str(g) for g in groups], rotation=45, ha="right", fontsize=11)
    ax.set_ylim([max(0.0, min(m - 0.03 for m in means)), min(1.0, max(m + 0.05 for m in means))])
    ax.grid(axis="y", alpha=0.3)


# ─── Load data ────────────────────────────────────────────────────────────────


def load_ch4_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    ch4 = df[df[GROUP_COL].isin(CH4_GROUPS)].copy()
    logger.info("Loaded %d Ch4 rows from CSV", len(ch4))
    return ch4


# ─── Fix 1: auroc_bar_by_tokenizer — identity bar restricted to dim>=16 ────────


def fix1_auroc_bar_tokenizer(ch4: pd.DataFrame) -> None:
    """fig:auroc-bar-tokenizer.

    Identity bar uses only dim>=16 runs (plus one_hot/num_types) to avoid
    the learned+dim=8 outlier from Exp 4B dominating the identity mean.
    Raw and binned are unaffected.
    """
    # raw and binned: all rows
    raw_vals = ch4.loc[ch4[T1_COL] == "raw", AUROC_COL].tolist()
    binned_vals = ch4.loc[ch4[T1_COL] == "binned", AUROC_COL].tolist()

    # identity: dim>=16 + num_types only
    id_df = ch4[ch4[T1_COL] == "identity"]
    id_clean = id_df[id_df[T1B_COL].isin(["16", "32", "num_types"])]
    identity_vals = id_clean[AUROC_COL].tolist()

    groups = ["raw", "binned", "identity\n(dim≥16)"]
    all_vals = [raw_vals, binned_vals, identity_vals]
    means = [float(np.mean(v)) for v in all_vals]
    stds = [float(np.std(v)) if len(v) > 1 else 0.0 for v in all_vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    _bar_with_points(ax, groups, means, stds, all_vals)
    ax.set_xlabel("Tokenizer Family (T1)", fontsize=13)
    ax.set_ylabel("Test AUROC", fontsize=13)
    ax.set_title("AUROC by Tokenizer Family\n(identity restricted to dim≥16)", fontsize=13)

    _save(fig, "figure-auroc_bar_by_tokenizer_fixed")
    logger.info("Fix 1 done: raw=%.4f, binned=%.4f, identity(dim>=16)=%.4f", means[0], means[1], means[2])


# ─── Fix 2: auroc_bar_by_pid_mode — filter to dim>=16 ─────────────────────────


def fix2_auroc_bar_pid_mode(ch4: pd.DataFrame) -> None:
    """fig:auroc-bar-pid-mode.

    Only identity tokenizer rows at dim>=16 (or num_types for one_hot).
    This removes the learned+dim=8 Exp-4B rows that inflated/deflated the mode means.
    """
    id_df = ch4[ch4[T1_COL] == "identity"].copy()
    # one_hot only has num_types; for learned+fixed_random restrict to dim>=16
    id_clean = id_df[
        (id_df[T1B_COL].isin(["16", "32"])) | (id_df[T1B_COL] == "num_types")
    ]

    mode_vals: dict[str, list[float]] = {}
    for _, row in id_clean.iterrows():
        mode = row[T1A_COL]
        if pd.isna(mode):
            continue
        mode_vals.setdefault(mode, []).append(float(row[AUROC_COL]))

    groups = sorted(mode_vals.keys())
    all_vals = [mode_vals[g] for g in groups]
    means = [float(np.mean(v)) for v in all_vals]
    stds = [float(np.std(v)) if len(v) > 1 else 0.0 for v in all_vals]

    # Pretty labels
    label_map = {"fixed_random": "fixed_random", "learned": "learned", "one_hot": "one_hot"}
    labels = [label_map.get(g, g) for g in groups]

    fig, ax = plt.subplots(figsize=(7, 5))
    _bar_with_points(ax, groups, means, stds, all_vals, labels=labels)
    ax.set_xlabel("PID Embedding Mode (T1-a)", fontsize=13)
    ax.set_ylabel("Test AUROC", fontsize=13)
    ax.set_title("AUROC by PID Embedding Mode\n(identity tokenizer, dim≥16 only)", fontsize=13)

    _save(fig, "figure-auroc_bar_by_pid_mode_fixed")
    for g, m, s in zip(groups, means, stds, strict=False):
        logger.info("  pid_mode=%s: mean=%.4f, std=%.4f, n=%d", g, m, s, len(mode_vals[g]))


# ─── Fix 4: val_auroc_by_tokenizer — x-axis 0-30 ────────────────────────────


def fix4_val_auroc_tokenizer_xlim30(ch4: pd.DataFrame) -> None:
    """fig:val-auroc-tokenizer with x-axis cropped to epochs 0-30."""
    # Load per-epoch scalars for ch4 runs
    # Group by tokenizer_name: raw, binned, identity
    group_epochs: dict[str, list[tuple]] = {}  # tokenizer -> list of (epochs, values) arrays

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
            logger.warning("No metric_auroc in %s", scalars_path)
            continue

        epochs = val_rows["epoch"].astype(int).values
        values = val_rows["metric_auroc"].astype(float).values
        group_epochs.setdefault(tokenizer, []).append((epochs, values))

    if not group_epochs:
        logger.error("No per-epoch data loaded — skipping fix 4")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    groups = sorted(group_epochs.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    color_map = {g: colors[i] for i, g in enumerate(groups)}

    XLIM_MAX = 30

    for group_val in groups:
        color = color_map[group_val]
        all_epochs_list = []
        all_values_list = []

        for epochs, values in group_epochs[group_val]:
            mask = epochs <= XLIM_MAX
            e_crop = epochs[mask]
            v_crop = values[mask]
            ax.plot(e_crop, v_crop, color=color, alpha=0.2, linewidth=1, zorder=1)
            all_epochs_list.append(e_crop)
            all_values_list.append(v_crop)

        # Mean curve
        max_ep = XLIM_MAX + 1
        mean_vals = np.full(max_ep, np.nan)
        std_vals = np.full(max_ep, np.nan)
        for ep in range(max_ep):
            at_ep = [v[ep] for v, e in zip(all_values_list, all_epochs_list, strict=False) if len(v) > ep]
            if at_ep:
                mean_vals[ep] = np.mean(at_ep)
                std_vals[ep] = np.std(at_ep) if len(at_ep) > 1 else 0.0

        ep_range = np.arange(max_ep)
        valid = ~np.isnan(mean_vals)
        n_runs = len(all_values_list)
        ax.plot(
            ep_range[valid],
            mean_vals[valid],
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=4,
            label=f"{group_val} (n={n_runs})",
            zorder=2,
        )
        if np.any(std_vals[valid] > 0):
            ax.fill_between(
                ep_range[valid],
                mean_vals[valid] - std_vals[valid],
                mean_vals[valid] + std_vals[valid],
                color=color,
                alpha=0.15,
                zorder=1,
            )

    ax.set_xlim([0, XLIM_MAX])
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Validation AUROC", fontsize=13)
    ax.set_title("Validation AUROC by Tokenizer Family (epochs 0–30)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    _save(fig, "figure-val_auroc_by_tokenizer_fixed")
    logger.info("Fix 4 done")


# ─── Fix 6: auroc_bar_by_features — identity only ─────────────────────────────


def fix6_auroc_bar_features(ch4: pd.DataFrame) -> None:
    """fig:auroc-bar-features filtered to identity tokenizer rows only.

    Removes the 'unknown' bar that appeared because raw/binned runs had
    cont_features=null and were assigned 'unknown' feature_set_label.
    """
    identity_df = ch4[ch4[T1_COL] == "identity"].copy()

    feat_vals: dict[str, list[float]] = {}
    for _, row in identity_df.iterrows():
        feat = row[D01_COL]
        auroc = row[AUROC_COL]
        if pd.isna(feat) or pd.isna(auroc):
            continue
        feat_vals.setdefault(str(feat), []).append(float(auroc))

    label_map = {
        "[0,1,2,3]": "E+pT+η+ϕ\n(all features)",
        "[1,2,3]": "pT+η+ϕ\n(no energy)",
    }
    groups = sorted(feat_vals.keys())
    all_vals = [feat_vals[g] for g in groups]
    means = [float(np.mean(v)) for v in all_vals]
    stds = [float(np.std(v)) if len(v) > 1 else 0.0 for v in all_vals]
    labels = [label_map.get(g, g) for g in groups]

    fig, ax = plt.subplots(figsize=(6, 5))
    _bar_with_points(ax, groups, means, stds, all_vals, labels=labels)
    ax.set_xlabel("Feature Set (D01)", fontsize=13)
    ax.set_ylabel("Test AUROC", fontsize=13)
    ax.set_title("AUROC by Feature Set\n(identity tokenizer only)", fontsize=13)

    _save(fig, "figure-auroc_bar_by_features_fixed")
    for g, m, s in zip(groups, means, stds, strict=False):
        logger.info("  features=%s: mean=%.4f, std=%.4f, n=%d", g, m, s, len(feat_vals[g]))


# ─── Fix 7: auroc_bar_by_met and auroc_bar_by_shuffle — correct CSV values ────


def fix7_auroc_bar_met(ch4: pd.DataFrame) -> None:
    """fig:auroc-bar-met using CSV test_auroc (correct values).

    The original inference summary showed MET=True ~0.81 due to a data loading
    issue during re-run inference. The CSV eval_v2/test_auroc shows the correct
    values (+0.0008 delta, within seed-spread noise).
    """
    exp4b = ch4[ch4[GROUP_COL].str.contains("exp4b", na=False)].copy()
    met_vals: dict[str, list[float]] = {}
    for _, row in exp4b.iterrows():
        met = row[D02_COL]
        auroc = row[AUROC_COL]
        if pd.isna(met) or pd.isna(auroc):
            continue
        key = "MET=True" if bool(met) else "MET=False"
        met_vals.setdefault(key, []).append(float(auroc))

    groups = sorted(met_vals.keys())
    all_vals = [met_vals[g] for g in groups]
    means = [float(np.mean(v)) for v in all_vals]
    stds = [float(np.std(v)) if len(v) > 1 else 0.0 for v in all_vals]

    fig, ax = plt.subplots(figsize=(6, 5))
    _bar_with_points(ax, groups, means, stds, all_vals)
    ax.set_xlabel("MET Treatment (D02)", fontsize=13)
    ax.set_ylabel("Test AUROC", fontsize=13)
    ax.set_title("AUROC by MET Inclusion\n(source: eval_v2/test_auroc from CSV)", fontsize=13)

    _save(fig, "figure-auroc_bar_by_met_fixed")
    delta = means[groups.index("MET=True")] - means[groups.index("MET=False")]
    logger.info("Fix 7 MET done: delta=%.4f (positive = MET helps)", delta)


def fix7_auroc_bar_shuffle(ch4: pd.DataFrame) -> None:
    """fig:auroc-bar-shuffle using CSV test_auroc (correct values).

    The original inference summary showed shuffle=True ~0.81 due to the same
    data loading issue. CSV shows the correct delta of -0.0026.
    """
    exp4b = ch4[ch4[GROUP_COL].str.contains("exp4b", na=False)].copy()
    shuffle_vals: dict[str, list[float]] = {}
    for _, row in exp4b.iterrows():
        shuffle = row[D03_COL]
        auroc = row[AUROC_COL]
        if pd.isna(shuffle) or pd.isna(auroc):
            continue
        key = str(shuffle)
        shuffle_vals.setdefault(key, []).append(float(auroc))

    groups = sorted(shuffle_vals.keys())
    all_vals = [shuffle_vals[g] for g in groups]
    means = [float(np.mean(v)) for v in all_vals]
    stds = [float(np.std(v)) if len(v) > 1 else 0.0 for v in all_vals]

    label_map = {"input_order": "input_order\n(consistent)", "shuffled": "shuffled\n(permuted)"}
    labels = [label_map.get(g, g) for g in groups]

    fig, ax = plt.subplots(figsize=(6, 5))
    _bar_with_points(ax, groups, means, stds, all_vals, labels=labels)
    ax.set_xlabel("Token Ordering (D03)", fontsize=13)
    ax.set_ylabel("Test AUROC", fontsize=13)
    ax.set_title("AUROC by Token Ordering\n(source: eval_v2/test_auroc from CSV)", fontsize=13)

    _save(fig, "figure-auroc_bar_by_shuffle_fixed")
    if "input_order" in shuffle_vals and "shuffled" in shuffle_vals:
        delta = np.mean(shuffle_vals["shuffled"]) - np.mean(shuffle_vals["input_order"])
        logger.info("Fix 7 shuffle done: delta=%.4f (shuffled - input_order)", delta)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ch4 = load_ch4_df()

    logger.info("--- Fix 1: auroc_bar_by_tokenizer (identity dim>=16) ---")
    fix1_auroc_bar_tokenizer(ch4)

    logger.info("--- Fix 2: auroc_bar_by_pid_mode (dim>=16 filter) ---")
    fix2_auroc_bar_pid_mode(ch4)

    # Fix 3: auroc_bar_by_dim -- no replot needed; handled by removing from report config

    logger.info("--- Fix 4: val_auroc_by_tokenizer (xlim 0-30) ---")
    fix4_val_auroc_tokenizer_xlim30(ch4)

    # Fix 5: roc_curves_by_tokenizer -- caption-level only; existing bands already present

    logger.info("--- Fix 6: auroc_bar_by_features (identity only) ---")
    fix6_auroc_bar_features(ch4)

    logger.info("--- Fix 7a: auroc_bar_by_met (CSV values) ---")
    fix7_auroc_bar_met(ch4)

    logger.info("--- Fix 7b: auroc_bar_by_shuffle (CSV values) ---")
    fix7_auroc_bar_shuffle(ch4)

    logger.info("All done. Output dir: %s", OUT_DIR)


if __name__ == "__main__":
    main()
