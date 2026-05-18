"""Chapter 9 analysis: best model physics reach.

Compares the best-performing model (MIA, 5 seeds from mia_od_followup_seeds)
against the simple physics baseline (9 seeds) on the main binary task
(ttH+ttW+ttWW+ttZ | 4t).  Optuna best-trial AUROC is shown as a reference
annotation only (no curve available).

Produces four plots
-------------------
1. roc_comparison.pdf          — ROC curves, mean ± std band per model
2. sig_eff_vs_bkg_rej.pdf      — signal efficiency vs. background rejection
3. score_distributions.pdf     — normalised discriminant histograms
4. significance_vs_threshold.pdf — illustrative Z vs. score cut

Data sources
------------
All MIA and baseline data come from the canonical CSV:
  thesis_results/04_cleaned_backfilled_analysis_ready.csv

  MIA  : group = exp_20260511-113827_mia_od_followup_seeds  (5 seeds)
  Baseline: group = exp_20260306-190512_4tbg_physics_baseline (9 seeds)

Optuna best-trial AUROC:
  agent_reference/wandb_export_2026-05-17T22_02_28.058+02_00.csv

Physics significance assumptions (illustrative, stated on plot)
---------------------------------------------------------------
  sigma_4t   = 12.0 fb   (ATLAS arXiv:2303.15061)
  sigma_bkg  = 800  fb   (order-of-magnitude, ttH+ttW+ttWW+ttZ combined)
  luminosity = 300  fb^-1 (HL-LHC benchmark)
  N_MC_sig / N_MC_bkg inferred from score histograms (50 bins, sum = event count)
  syst_bkg   = 0.10      (10% flat systematic on background)

Usage
-----
    python src/thesis_ml/reports/analyses/ch9_best_model.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import auc

from thesis_ml.reports.plots.style import (
    CATEGORICAL_COLORS,
    apply_thesis_style,
    axis_color,
    figure_size,
)

apply_thesis_style()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO = Path("/project/atlas/users/nterlind/Thesis-Code")
_DATA = Path("/data/atlas/users/nterlind")

_MAIN_CSV     = _REPO / "thesis_results" / "04_cleaned_backfilled_analysis_ready.csv"
_OPTUNA_CSV   = _REPO / "agent_reference" / "wandb_export_2026-05-17T22_02_28.058+02_00.csv"
_OUT_DIR      = _DATA / "outputs" / "plots" / "ch9"

_MIA_GROUP      = "exp_20260511-113827_mia_od_followup_seeds"
_BASELINE_GROUP = "exp_20260306-190512_4tbg_physics_baseline"

# Physics assumptions
_SIGMA_SIG_FB = 12.0    # fb  (4t signal, ATLAS arXiv:2303.15061)
_SIGMA_BKG_FB = 800.0   # fb  (combined background, rough estimate)
_LUMINOSITY   = 300.0   # fb^-1
_SYST_BKG     = 0.10    # 10% flat systematic on B

_WORKING_POINTS = [50, 100, 1000]

_COLOR_MIA      = axis_color("recommended")
_COLOR_BASELINE = axis_color("baseline")
_COLOR_OPTUNA   = CATEGORICAL_COLORS[2]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json(s) -> np.ndarray | None:
    if pd.isna(s):
        return None
    try:
        return np.asarray(json.loads(str(s)), dtype=float)
    except (json.JSONDecodeError, ValueError):
        return None


def _save(fig: plt.Figure, name: str) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUT_DIR / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    logger.info("Saved %s", path)
    plt.close(fig)


def _load_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = df[df["meta_run/group"] == group].copy()
    if rows.empty:
        raise RuntimeError(f"Group '{group}' not found in CSV.")
    logger.info("Loaded %d rows for group '%s'", len(rows), group)
    return rows


def _mean_roc(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr_grid, mean_tpr, std_tpr) interpolated across all seeds."""
    fprs, tprs = [], []
    for _, row in rows.iterrows():
        fpr = _parse_json(row.get("eval_v2/roc_fpr"))
        tpr = _parse_json(row.get("eval_v2/roc_tpr"))
        if fpr is None or tpr is None:
            logger.warning("Skipping row — missing ROC: %s", row.get("meta_run/name"))
            continue
        fprs.append(fpr)
        tprs.append(tpr)

    if not fprs:
        raise RuntimeError("No ROC curves could be parsed.")

    all_min = min(f[f > 0].min() for f in fprs if (f > 0).any())
    grid = np.concatenate([[0.0], np.logspace(np.log10(max(all_min, 1e-4)), 0, 500)])

    tpr_mat = np.zeros((len(fprs), len(grid)))
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        f = interp1d(fpr, tpr, kind="linear", bounds_error=False, fill_value=(0.0, 1.0))
        tpr_mat[i] = f(grid)

    return grid, tpr_mat.mean(axis=0), tpr_mat.std(axis=0)


def _mean_hist(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return mean signal and background score histograms across seeds."""
    sigs, bgs = [], []
    for _, row in rows.iterrows():
        s = _parse_json(row.get("eval_v2/score_hist_signal"))
        b = _parse_json(row.get("eval_v2/score_hist_background"))
        if s is not None and b is not None:
            sigs.append(s)
            bgs.append(b)
    if not sigs:
        raise RuntimeError("No score histograms found.")
    return np.stack(sigs).mean(axis=0), np.stack(bgs).mean(axis=0)


def _scalar_stats(rows: pd.DataFrame, col: str) -> tuple[float, float]:
    return float(rows[col].mean()), float(rows[col].std())


def _eps_s(rows: pd.DataFrame) -> dict[int, tuple[float, float]]:
    return {
        inv_b: _scalar_stats(rows, f"eval_v2/eps_S_at_invB_{inv_b}")
        for inv_b in _WORKING_POINTS
        if f"eval_v2/eps_S_at_invB_{inv_b}" in rows.columns
    }


def _find_tpr_at_invb(fpr_grid: np.ndarray, tpr_mean: np.ndarray, inv_b: int) -> float:
    idx = np.searchsorted(fpr_grid, 1.0 / inv_b)
    return float(tpr_mean[min(idx, len(tpr_mean) - 1)])


# ---------------------------------------------------------------------------
# Plot 1: ROC comparison
# ---------------------------------------------------------------------------


def plot_roc(mia_roc, bl_roc, mia_auroc, bl_auroc, optuna_auroc, mia_eps):
    fpr_m, tpr_m, std_m = mia_roc
    fpr_b, tpr_b, std_b = bl_roc
    auroc_m, auroc_m_std = mia_auroc
    auroc_b, auroc_b_std = bl_auroc

    fig, ax = plt.subplots(figsize=figure_size("full", aspect=1.0))
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=0.8, alpha=0.5)

    ax.fill_between(fpr_b, tpr_b - std_b, tpr_b + std_b, color=_COLOR_BASELINE, alpha=0.15)
    ax.plot(fpr_b, tpr_b, color=_COLOR_BASELINE, lw=2.5,
            label=f"Physics baseline  AUROC = {auroc_b:.4f} $\\pm$ {auroc_b_std:.4f}")

    ax.fill_between(fpr_m, tpr_m - std_m, tpr_m + std_m, color=_COLOR_MIA, alpha=0.15)
    ax.plot(fpr_m, tpr_m, color=_COLOR_MIA, lw=2.5,
            label=f"MIA (best model)  AUROC = {auroc_m:.4f} $\\pm$ {auroc_m_std:.4f}")

    ax.plot([], [], color=_COLOR_OPTUNA, lw=2.0, ls=":",
            label=f"Optuna best trial  AUROC = {optuna_auroc:.4f} (no curve)")

    for inv_b in _WORKING_POINTS:
        tpr_wp = _find_tpr_at_invb(fpr_m, tpr_m, inv_b)
        ax.plot(1.0 / inv_b, tpr_wp, "o", color=_COLOR_MIA, ms=6, zorder=5)
        ax.annotate(f"1/B={inv_b}", xy=(1.0 / inv_b, tpr_wp), xytext=(6, -10),
                    textcoords="offset points", fontsize=8)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    _save(fig, "roc_comparison")


# ---------------------------------------------------------------------------
# Plot 2: Signal efficiency vs. background rejection
# ---------------------------------------------------------------------------


def plot_sig_eff(mia_roc, bl_roc, mia_eps, bl_eps):
    fpr_m, tpr_m, std_m = mia_roc
    fpr_b, tpr_b, std_b = bl_roc

    min_fpr = 1e-4
    mask_m = fpr_m > min_fpr
    mask_b = fpr_b > min_fpr

    fig, ax = plt.subplots(figsize=figure_size("full"))

    ax.fill_between(1.0 / fpr_b[mask_b], tpr_b[mask_b] - std_b[mask_b],
                    tpr_b[mask_b] + std_b[mask_b], color=_COLOR_BASELINE, alpha=0.15)
    ax.plot(1.0 / fpr_b[mask_b], tpr_b[mask_b], color=_COLOR_BASELINE, lw=2.5,
            label="Physics baseline")

    ax.fill_between(1.0 / fpr_m[mask_m], tpr_m[mask_m] - std_m[mask_m],
                    tpr_m[mask_m] + std_m[mask_m], color=_COLOR_MIA, alpha=0.15)
    ax.plot(1.0 / fpr_m[mask_m], tpr_m[mask_m], color=_COLOR_MIA, lw=2.5,
            label="MIA (best model)")

    for inv_b in _WORKING_POINTS:
        ax.axvline(inv_b, color="gray", ls="--", lw=0.8, alpha=0.5)
        if inv_b in mia_eps:
            eps_mean, _ = mia_eps[inv_b]
            ax.plot(inv_b, eps_mean, "o", color=_COLOR_MIA, ms=7, zorder=5)
            ax.annotate(f"$\\varepsilon_S={eps_mean:.3f}$", xy=(inv_b, eps_mean),
                        xytext=(6, 4), textcoords="offset points", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlim(1, 2000); ax.set_ylim(0, 1)
    ax.set_xlabel("Background rejection factor (1/FPR)")
    ax.set_ylabel("Signal efficiency (TPR)")
    ax.legend(loc="upper right")
    _save(fig, "sig_eff_vs_bkg_rej")


# ---------------------------------------------------------------------------
# Plot 3: Score distributions
# ---------------------------------------------------------------------------


def plot_score_dists(mia_hists, bl_hists):
    sig_m, bg_m = mia_hists
    sig_b, bg_b = bl_hists
    n_bins = len(sig_m)
    edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = edges[1] - edges[0]

    def _norm(h):
        total = h.sum() * w
        return h / total if total > 0 else h

    fig, ax = plt.subplots(figsize=figure_size("full"))

    ax.step(centers, _norm(sig_b),  where="mid", color=_COLOR_BASELINE, lw=1.8, ls="--",
            label="Physics baseline — signal")
    ax.step(centers, _norm(bg_b),   where="mid", color=_COLOR_BASELINE, lw=1.8, ls=":",
            label="Physics baseline — background")
    ax.step(centers, _norm(sig_m),  where="mid", color=_COLOR_MIA, lw=1.8, ls="--",
            label="MIA — signal")
    ax.step(centers, _norm(bg_m),   where="mid", color=_COLOR_MIA, lw=1.8, ls=":",
            label="MIA — background")

    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    ax.set_xlabel("Classifier output score")
    ax.set_ylabel("Normalised density")
    ax.legend()
    _save(fig, "score_distributions")


# ---------------------------------------------------------------------------
# Plot 4: Illustrative significance vs. cut threshold
# ---------------------------------------------------------------------------


def plot_significance(mia_hists, bl_hists):
    sig_m, bg_m = mia_hists
    sig_b, bg_b = bl_hists
    n_bins = len(sig_m)
    centers = np.linspace(0, 1, n_bins + 1)[:-1] + 0.5 / n_bins

    # Infer N_MC from histogram totals (sum of all bins)
    n_mc_sig_m = sig_m.sum()
    n_mc_bg_m  = bg_m.sum()
    n_mc_sig_b = sig_b.sum()
    n_mc_bg_b  = bg_b.sum()

    def _z_curve(sig_hist, bg_hist, n_mc_sig, n_mc_bg):
        sf_sig = _SIGMA_SIG_FB * _LUMINOSITY / n_mc_sig
        sf_bkg = _SIGMA_BKG_FB * _LUMINOSITY / n_mc_bg
        z = np.zeros(n_bins)
        for i in range(n_bins):
            S = sig_hist[i:].sum() * sf_sig
            B = bg_hist[i:].sum() * sf_bkg
            z[i] = S / np.sqrt(B + (_SYST_BKG * B) ** 2) if B > 0 else 0.0
        return z

    z_m = _z_curve(sig_m, bg_m, n_mc_sig_m, n_mc_bg_m)
    z_b = _z_curve(sig_b, bg_b, n_mc_sig_b, n_mc_bg_b)

    fig, ax = plt.subplots(figsize=figure_size("full"))

    ax.plot(centers, z_b, color=_COLOR_BASELINE, lw=2.5, label="Physics baseline")
    ax.plot(centers, z_m, color=_COLOR_MIA,      lw=2.5, label="MIA (best model)")

    for z_ref, lbl in [(2.0, "$Z=2$"), (5.0, "$Z=5$")]:
        ax.axhline(z_ref, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.text(0.02, z_ref + 0.12, lbl, color="gray", va="bottom", fontsize=8)

    note = (
        "Illustrative — MC counts scaled by literature $\\sigma$:\n"
        f"$\\sigma_{{4t}}={_SIGMA_SIG_FB:.0f}$\\,fb, "
        f"$\\sigma_{{\\rm bkg}}={_SIGMA_BKG_FB:.0f}$\\,fb, "
        f"$\\mathcal{{L}}={_LUMINOSITY:.0f}$\\,fb$^{{-1}}$, "
        f"$\\delta_{{\\rm syst}}={int(_SYST_BKG*100)}\\%$.\n"
        "Not a full profile-likelihood fit."
    )
    ax.text(0.98, 0.98, note, transform=ax.transAxes, ha="right", va="top",
            style="italic", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    ax.set_xlabel("Classifier score threshold")
    ax.set_ylabel("Approximate significance $Z$")
    ax.legend()
    _save(fig, "significance_vs_threshold")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(mia_auroc, bl_auroc, mia_eps, bl_eps, optuna_auroc):
    def _eps_str(eps_dict, inv_b):
        if inv_b not in eps_dict:
            return "   N/A"
        m, s = eps_dict[inv_b]
        return f"{m:.4f}±{s:.4f}"

    print()
    print("=" * 82)
    print("CHAPTER 9 SUMMARY TABLE")
    print("=" * 82)
    print(f"{'Model':<28} {'AUROC':>18} {'εS(1/B=50)':>14} {'εS(1/B=100)':>14} {'εS(1/B=1000)':>14}")
    print("-" * 82)
    m_m, s_m = mia_auroc
    m_b, s_b = bl_auroc
    print(f"{'MIA (5 seeds)':<28} {m_m:.4f}±{s_m:.4f}      "
          f"{_eps_str(mia_eps,50):>14} {_eps_str(mia_eps,100):>14} {_eps_str(mia_eps,1000):>14}")
    print(f"{'Physics baseline (9 seeds)':<28} {m_b:.4f}±{s_b:.4f}      "
          f"{_eps_str(bl_eps,50):>14} {_eps_str(bl_eps,100):>14} {_eps_str(bl_eps,1000):>14}")
    print(f"{'Optuna best trial':<28} {optuna_auroc:.4f}               {'—':>14} {'—':>14} {'—':>14}")
    print("=" * 82)
    print()
    m_eps50, _ = mia_eps.get(50, (None, None))
    b_eps50, _ = bl_eps.get(50, (None, None))
    m_eps100, _ = mia_eps.get(100, (None, None))
    b_eps100, _ = bl_eps.get(100, (None, None))
    if m_eps50 and b_eps50:
        print(f"Relative gain εS(1/B=50):  {(m_eps50-b_eps50)/b_eps50*100:+.0f}%")
    if m_eps100 and b_eps100:
        print(f"Relative gain εS(1/B=100): {(m_eps100-b_eps100)/b_eps100*100:+.0f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("Loading main CSV: %s", _MAIN_CSV)
    df = pd.read_csv(_MAIN_CSV)

    mia_rows = _load_group(df, _MIA_GROUP)
    bl_rows  = _load_group(df, _BASELINE_GROUP)

    mia_roc   = _mean_roc(mia_rows)
    bl_roc    = _mean_roc(bl_rows)
    mia_hists = _mean_hist(mia_rows)
    bl_hists  = _mean_hist(bl_rows)

    mia_auroc = _scalar_stats(mia_rows, "eval_v2/test_auroc")
    bl_auroc  = _scalar_stats(bl_rows,  "eval_v2/test_auroc")
    mia_eps   = _eps_s(mia_rows)
    bl_eps    = _eps_s(bl_rows)

    optuna_df = pd.read_csv(_OPTUNA_CSV)
    optuna_auroc = float(optuna_df["eval_v2/test_auroc"].max())
    logger.info("Optuna best AUROC: %.4f", optuna_auroc)

    logger.info("Plotting...")
    plot_roc(mia_roc, bl_roc, mia_auroc, bl_auroc, optuna_auroc, mia_eps)
    plot_sig_eff(mia_roc, bl_roc, mia_eps, bl_eps)
    plot_score_dists(mia_hists, bl_hists)
    plot_significance(mia_hists, bl_hists)

    _print_summary(mia_auroc, bl_auroc, mia_eps, bl_eps, optuna_auroc)
    logger.info("Done. Plots in %s", _OUT_DIR)


if __name__ == "__main__":
    main()
