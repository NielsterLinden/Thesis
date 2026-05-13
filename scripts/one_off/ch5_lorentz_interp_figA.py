"""ch5_lorentz_interp_figA.py — 1D KAN-vs-standard bias curves (Exp 5B).

Two-panel figure:

- Left:  bias(ΔR) from the single-feature config ``['deltaR']``.
- Right: partial-dependence bias(log kT) from the 4-feature config
  ``['log_kt', 'z', 'deltaR', 'log_m2']`` — non-swept features fixed to
  validation-data medians.

Each panel overlays the per-seed (3 seeds, light alpha) and seed-mean (bold)
curves separately for ``standard`` and ``kan``, gate-off cohort. A small
inset histogram marks the empirical p5–p95 range of the swept feature on
the normalized scale so the reader can see where the spline is supported.

Inputs are z-score normalized — the x-axis is the **normalized feature
value**, not the raw physics quantity. See evidence note 5B-interp §1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _ch5_lorentz_interp import (
    empirical_range,
    load_lorentz_bias,
    load_run_index,
    replay_validation_features,
    sweep_1d,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from thesis_ml.reports.plots.style import (
    apply_thesis_style,
    axis_color,
    figure_size,
)

OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch5_lorentz_interp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS1 = ("deltaR",)
FS4 = ("log_kt", "z", "deltaR", "log_m2")
COLOR_STANDARD = "#888888"      # neutral grey
COLOR_KAN = axis_color("B")      # teal (bias axis-group colour)
N_GRID = 200


def _curves(df, features, sweep_feat, x, baseline):
    sub = df[(df["features"] == features) & (~df["sparse_gating"])]
    out: dict[str, list[np.ndarray]] = {"standard": [], "kan": []}
    for _, r in sub.iterrows():
        recon = load_lorentz_bias(r["run_dir"])
        y = sweep_1d(recon, sweep_feat, x, baseline=baseline)
        out[recon.mlp_type].append(y)
    return {k: np.stack(v, axis=0) for k, v in out.items()}


def main() -> None:
    apply_thesis_style()

    print("[figA] replaying validation features…")
    feats = replay_validation_features(n_events=3000, seed=0)
    medians = {k: float(np.median(v)) for k, v in feats.items()}

    df = load_run_index()

    # --- Left panel: bias(ΔR) — single-feature config ---
    dR_lo, dR_hi = empirical_range(feats["deltaR"], 1, 99)
    x_dR = np.linspace(dR_lo, dR_hi, N_GRID)
    curves_L = _curves(df, FS1, "deltaR", x_dR, baseline={})

    # --- Right panel: bias(log kT) — 4-feature config, others at median ---
    lkt_lo, lkt_hi = empirical_range(feats["log_kt"], 1, 99)
    x_lkt = np.linspace(lkt_lo, lkt_hi, N_GRID)
    baseline_R = {k: medians[k] for k in ("z", "deltaR", "log_m2")}
    curves_R = _curves(df, FS4, "log_kt", x_lkt, baseline=baseline_R)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=figure_size("full", aspect=2.2), constrained_layout=True)
    for ax, x, curves, xlabel in (
        (axes[0], x_dR, curves_L, r"$\Delta R$ (normalized input)"),
        (axes[1], x_lkt, curves_R, r"$\log k_\mathrm{T}$ (normalized input)"),
    ):
        for mlp, color in (("standard", COLOR_STANDARD), ("kan", COLOR_KAN)):
            ys = curves[mlp]                            # [S, N]
            for y in ys:
                ax.plot(x, y, color=color, alpha=0.25, linewidth=0.8)
            ax.plot(x, ys.mean(axis=0), color=color, linewidth=1.7, label=mlp)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("End-to-end attention bias (a.u.)")
        ax.legend(frameon=False, loc="best")

    # Annotate baseline policy on right panel
    axes[1].text(
        0.98, 0.02,
        "non-swept features at validation median",
        transform=axes[1].transAxes, ha="right", va="bottom",
        fontsize=7, color="#444",
    )

    out_pdf = OUT_DIR / "figure-1d_bias_kan_vs_standard.pdf"
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"[figA] wrote {out_pdf}")

    # Companion CSV: per-grid-point seed-mean for both panels
    import pandas as pd
    rows = []
    for x_arr, key, curves in (
        (x_dR, "deltaR_single", curves_L),
        (x_lkt, "log_kt_partial4f", curves_R),
    ):
        for mlp, ys in curves.items():
            mean = ys.mean(axis=0)
            std = ys.std(axis=0)
            for xi, m, s in zip(x_arr, mean, std):
                rows.append(dict(panel=key, mlp_type=mlp, x=float(xi),
                                 bias_mean=float(m), bias_std=float(s)))
    pd.DataFrame(rows).to_csv(OUT_DIR / "exp5b_1d_bias_curves.csv", index=False)
    print(f"[figA] wrote {OUT_DIR/'exp5b_1d_bias_curves.csv'}")


if __name__ == "__main__":
    main()
