"""ch5_lorentz_interp_figB.py — 2D bias(ΔR, log kT) surface (Exp 5B).

Two-panel heatmap. Best-AUROC checkpoint (seed 42) from the 4-feature
config ``['log_kt', 'z', 'deltaR', 'log_m2']``, gate=off, for each MLP
type. Non-swept features fixed to validation medians. A validation-data
2D histogram is overlaid as light contours so the reader sees where the
bias surface is supported.

Axes: normalized inputs (z-score of E, pT, η, φ). See evidence note §1.
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
    sweep_2d,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from thesis_ml.reports.plots.style import apply_thesis_style, figure_size

OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch5_lorentz_interp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS4 = ("log_kt", "z", "deltaR", "log_m2")
N_GRID = 80


def _pick_best(df, features, mlp_type):
    sub = df[(df["features"] == features) & (~df["sparse_gating"]) & (df["mlp_type"] == mlp_type)]
    return sub.sort_values("test_auroc", ascending=False).iloc[0]


def main() -> None:
    apply_thesis_style()

    feats = replay_validation_features(n_events=3000, seed=0)
    medians = {k: float(np.median(v)) for k, v in feats.items()}

    dR_lo, dR_hi = empirical_range(feats["deltaR"], 1, 99)
    lkt_lo, lkt_hi = empirical_range(feats["log_kt"], 1, 99)
    x_dR = np.linspace(dR_lo, dR_hi, N_GRID)
    x_lkt = np.linspace(lkt_lo, lkt_hi, N_GRID)

    df = load_run_index()
    best_std = _pick_best(df, FS4, "standard")
    best_kan = _pick_best(df, FS4, "kan")

    surfaces = {}
    for label, row in (("standard", best_std), ("kan", best_kan)):
        recon = load_lorentz_bias(row["run_dir"])
        Z = sweep_2d(recon, "deltaR", "log_kt", x_dR, x_lkt,
                     baseline={"z": medians["z"], "log_m2": medians["log_m2"]})
        surfaces[label] = (Z, row, recon)

    # Common symmetric colour scale
    zmax = max(np.abs(Z).max() for Z, _, _ in surfaces.values())

    fig, axes = plt.subplots(1, 2, figsize=figure_size("full", aspect=2.0), constrained_layout=True)
    for ax, label in zip(axes, ("standard", "kan")):
        Z, row, recon = surfaces[label]
        im = ax.imshow(
            Z,
            origin="lower",
            extent=(x_dR.min(), x_dR.max(), x_lkt.min(), x_lkt.max()),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-zmax,
            vmax=zmax,
        )
        # Validation data 2D histogram contour overlay
        H, xe, ye = np.histogram2d(
            feats["deltaR"], feats["log_kt"],
            bins=40,
            range=[[x_dR.min(), x_dR.max()], [x_lkt.min(), x_lkt.max()]],
        )
        H = H.T
        xc = 0.5 * (xe[1:] + xe[:-1])
        yc = 0.5 * (ye[1:] + ye[:-1])
        levels = np.percentile(H[H > 0], [50, 75, 90])
        ax.contour(xc, yc, H, levels=levels, colors="black", linewidths=0.5, alpha=0.6)

        ax.set_xlabel(r"$\Delta R$ (normalized input)")
        ax.set_ylabel(r"$\log k_\mathrm{T}$ (normalized input)")
        ax.set_title(
            f"{label}  (seed {int(row['seed'])}, AUROC={row['test_auroc']:.4f}, "
            f"$\\tanh g$={recon.gate_tanh:.2f})",
            fontsize=9,
        )

    fig.colorbar(im, ax=axes, shrink=0.85, label="bias (a.u.)")
    out_pdf = OUT_DIR / "figure-2d_bias_surface_deltaR_logkt.pdf"
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"[figB] wrote {out_pdf}")


if __name__ == "__main__":
    main()
