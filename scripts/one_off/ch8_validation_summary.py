"""ch8_validation_summary.py — §8.6 + §8.7 final figures for thesis handoff.

Generates:
  §8.6 Validation scatter (predicted vs actual AUROC, 6 candidates)
    validation_predicted_vs_actual.pdf
    validation_summary.csv

  §8.7 Optuna distribution
    optuna_auroc_histogram.pdf

All values for §8.6 are hardcoded from the evidence note (Phase H) — no live
W&B queries needed. §8.7 reads the raw Optuna W&B export CSV.

Output: /data/atlas/users/nterlind/outputs/reports/report_ch8_patching_G1_2/
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_ml.reports.plots.style import apply_thesis_style, axis_color, figure_size

apply_thesis_style()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path("/project/atlas/users/nterlind/Thesis-Code")
OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch8_patching_G1_2")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
OPTUNA_CSV = REPO_ROOT / "agent_reference" / "wandb_export_2026-05-17T22_02_28.058+02_00.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ─── §8.6 Data (hardcoded from evidence note Phase H) ─────────────────────────
#
# predicted_auroc: surrogate out-of-fold prediction (cand01–03) or marginal
#                  AUROC estimate at best axis value (cand_m1–m3, all = 0.8416)
# actual_mean / actual_std: mean ± std over 3 seeds, 50 epochs, G3=4t-vs-bg

CANDIDATES = [
    {
        "candidate": "cand01",
        "strategy": "Surrogate",
        "predicted_auroc": 0.8700,
        "actual_mean": 0.8495,
        "actual_std": 0.0007,
    },
    {
        "candidate": "cand02",
        "strategy": "Surrogate",
        "predicted_auroc": 0.8681,
        "actual_mean": 0.8450,
        "actual_std": 0.0008,
    },
    {
        "candidate": "cand03",
        "strategy": "Surrogate",
        "predicted_auroc": 0.8672,
        "actual_mean": 0.8357,
        "actual_std": 0.0014,
    },
    {
        "candidate": "cand_m1",
        "strategy": "Marginal greedy",
        "predicted_auroc": 0.8416,
        "actual_mean": 0.8372,
        "actual_std": 0.0034,
    },
    {
        "candidate": "cand_m2",
        "strategy": "Marginal greedy",
        "predicted_auroc": 0.8416,
        "actual_mean": 0.8247,
        "actual_std": 0.0005,
    },
    {
        "candidate": "cand_m3",
        "strategy": "Marginal greedy",
        "predicted_auroc": 0.8416,
        "actual_mean": 0.8359,
        "actual_std": 0.0039,
    },
]


def _save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    logger.info("saved %s", path)
    plt.close(fig)


# ─── §8.6 Validation CSV ──────────────────────────────────────────────────────


def write_validation_table(df: pd.DataFrame) -> None:
    path = TABLE_DIR / "validation_summary.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    logger.info("saved %s", path)


# ─── §8.6 Predicted vs actual scatter ─────────────────────────────────────────


def plot_validation_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=figure_size("two_thirds", aspect=1.0))

    strategies = df["strategy"].unique()
    markers = {"Surrogate": "o", "Marginal greedy": "s"}
    colors = {
        "Surrogate": axis_color("recommended"),
        "Marginal greedy": axis_color("baseline"),
    }

    for strategy in strategies:
        sub = df[df["strategy"] == strategy]
        ax.errorbar(
            sub["predicted_auroc"],
            sub["actual_mean"],
            yerr=sub["actual_std"],
            fmt=markers[strategy],
            color=colors[strategy],
            capsize=3,
            markersize=6,
            linewidth=1.0,
            label=strategy,
            zorder=3,
        )
        # Label each point with candidate name
        for _, row in sub.iterrows():
            ax.annotate(
                row["candidate"],
                xy=(row["predicted_auroc"], row["actual_mean"]),
                xytext=(4, 2),
                textcoords="offset points",
                fontsize=7,
                color=colors[strategy],
            )

    # Diagonal y = x
    lo = min(df["predicted_auroc"].min(), df["actual_mean"].min()) - 0.005
    hi = max(df["predicted_auroc"].max(), df["actual_mean"].max()) + 0.010
    diag = np.linspace(lo, hi, 100)
    ax.plot(diag, diag, color="0.65", linestyle="--", linewidth=0.9, zorder=1, label="y = x")

    ax.set_xlabel("Predicted AUROC")
    ax.set_ylabel("Actual AUROC (mean ± std, 3 seeds)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(framealpha=0.85)
    ax.set_title("Predicted vs actual AUROC — Ch8 candidates")

    _save(fig, "validation_predicted_vs_actual")


# ─── §8.7 Optuna histogram ────────────────────────────────────────────────────


def plot_optuna_histogram() -> None:
    df = pd.read_csv(OPTUNA_CSV)
    auroc = df["eval_v2/test_auroc"].dropna().values

    best_optuna = float(auroc.max())
    best_cand01 = 0.8495  # surrogate top-1 mean (3 seeds)

    fig, ax = plt.subplots(figsize=figure_size("two_thirds"))

    ax.hist(auroc, bins=25, color=axis_color("H"), alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(
        best_cand01,
        color=axis_color("recommended"),
        linestyle="--",
        linewidth=1.4,
        label=f"cand01 mean = {best_cand01:.4f}",
        zorder=3,
    )
    ax.axvline(
        best_optuna,
        color=axis_color("baseline"),
        linestyle=":",
        linewidth=1.4,
        label=f"Optuna best = {best_optuna:.4f}",
        zorder=3,
    )

    ax.set_xlabel("Test AUROC")
    ax.set_ylabel("Trials")
    ax.set_title(f"Optuna TPE AUROC distribution (N = {len(auroc)})")
    ax.legend()

    _save(fig, "optuna_auroc_histogram")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    df = pd.DataFrame(CANDIDATES)

    # Add derived columns
    df["residual"] = df["predicted_auroc"] - df["actual_mean"]
    df = df.sort_values("actual_mean", ascending=False).reset_index(drop=True)
    df["rank_actual"] = df.index + 1
    df_pred_sorted = df.sort_values("predicted_auroc", ascending=False)
    rank_pred = {row["candidate"]: i + 1 for i, (_, row) in enumerate(df_pred_sorted.iterrows())}
    df["rank_predicted"] = df["candidate"].map(rank_pred)
    col_order = [
        "candidate", "strategy", "predicted_auroc",
        "actual_mean", "actual_std", "residual",
        "rank_predicted", "rank_actual",
    ]
    df = df[col_order]

    write_validation_table(df)
    plot_validation_scatter(df)
    plot_optuna_histogram()

    logger.info("Ch8 validation summary complete.")


if __name__ == "__main__":
    main()
