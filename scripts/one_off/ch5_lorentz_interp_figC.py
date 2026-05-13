"""ch5_lorentz_interp_figC.py — sparse feature-gate heatmap + tanh(g) strip.

Two-panel diagnostic figure:

- Left:  sigmoid(feature_gates) heatmap for every sparse-on KAN checkpoint
  (15 runs = 5 feature sets × 3 seeds), rows = (feature-set, seed),
  columns = the seven possible Lorentz scalars, blank where the feature
  is not in the set.
- Right: tanh(per-module gate) strip plot across all 60 5B runs,
  hue = MLP type, x = feature-set size.  Shows whether the bias module
  was actually used after training.

Confirms (or refutes) "did the sparse gate prune anything?" and "did the
bias stay non-trivially active across all configurations?".
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _ch5_lorentz_interp import load_lorentz_bias, load_run_index

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from thesis_ml.reports.plots.style import (
    apply_thesis_style,
    axis_color,
    figure_size,
)

OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch5_lorentz_interp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_FEATURES = ["m2", "dot", "log_m2", "log_kt", "z", "deltaR", "deltaR_ptw"]
COLOR_STANDARD = "#888888"
COLOR_KAN = axis_color("B")


def _fmt_features(t: tuple[str, ...]) -> str:
    short = {"deltaR": "ΔR", "log_m2": "log m²", "log_kt": "log kT",
             "deltaR_ptw": "ΔR·pT", "m2": "m²"}
    return ",".join(short.get(f, f) for f in t)


def main() -> None:
    apply_thesis_style()
    df = load_run_index()

    # ------------------------------------------------------------------
    # Left panel: sparse-on KAN feature-gate heatmap
    # ------------------------------------------------------------------
    sub = df[(df["sparse_gating"]) & (df["mlp_type"] == "kan")].sort_values(
        ["features", "seed"]
    ).reset_index(drop=True)
    H = np.full((len(sub), len(ALL_FEATURES)), np.nan, dtype=float)
    ytick_labels: list[str] = []
    for i, r in sub.iterrows():
        recon = load_lorentz_bias(r["run_dir"])
        for j, fn in enumerate(recon.features):
            col = ALL_FEATURES.index(fn)
            H[i, col] = float(recon.feature_gates_sigmoid[j])
        ytick_labels.append(f"{_fmt_features(r['features'])}  s{int(r['seed'])}")

    # ------------------------------------------------------------------
    # Right panel: tanh(gate) strip across all 60 runs
    # ------------------------------------------------------------------
    gate_rows = []
    for _, r in df.iterrows():
        recon = load_lorentz_bias(r["run_dir"])
        gate_rows.append(dict(
            features=r["features"],
            n_features=len(r["features"]),
            mlp_type=r["mlp_type"],
            sparse_gating=r["sparse_gating"],
            seed=int(r["seed"]),
            tanh_gate=recon.gate_tanh,
        ))
    import pandas as pd
    gdf = pd.DataFrame(gate_rows)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=figure_size("full", aspect=1.6),
        gridspec_kw={"width_ratios": [1.0, 0.9]},
        constrained_layout=True,
    )

    # Left: heatmap
    ax = axes[0]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(H, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(ALL_FEATURES)))
    ax.set_xticklabels(["m²", "dot", "log m²", "log kT", "z", "ΔR", "ΔR·pT"],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(ytick_labels)))
    ax.set_yticklabels(ytick_labels, fontsize=6)
    ax.set_title("σ(feature_gates) · KAN, sparse=on", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85, label="gate weight")
    # Annotate values
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if not np.isnan(H[i, j]):
                ax.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if H[i, j] < 0.6 else "black")

    # Right: tanh(gate) strip
    ax2 = axes[1]
    x_levels = sorted(gdf["n_features"].unique())
    x_to_pos = {n: i for i, n in enumerate(x_levels)}
    rng = np.random.default_rng(0)
    for mlp, color in (("standard", COLOR_STANDARD), ("kan", COLOR_KAN)):
        d = gdf[gdf["mlp_type"] == mlp]
        xs = np.array([x_to_pos[n] for n in d["n_features"]]) + (
            -0.12 if mlp == "standard" else 0.12
        ) + rng.uniform(-0.04, 0.04, size=len(d))
        ax2.scatter(xs, d["tanh_gate"], color=color, alpha=0.7, s=14,
                    edgecolors="none", label=mlp)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax2.set_xticks(list(x_to_pos.values()))
    ax2.set_xticklabels([str(n) for n in x_levels])
    ax2.set_xlabel("Lorentz feature-set size")
    ax2.set_ylabel(r"$\tanh(\mathrm{gate})$ — bias multiplier")
    ax2.set_title("Per-module gate (60 runs)", fontsize=9)
    ax2.legend(frameon=False, loc="best")

    out_pdf = OUT_DIR / "figure-feature_gates_and_module_gate.pdf"
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"[figC] wrote {out_pdf}")

    gdf.to_csv(OUT_DIR / "exp5b_module_gate_values.csv", index=False)
    print(f"[figC] wrote {OUT_DIR/'exp5b_module_gate_values.csv'}")

    fg_rows = []
    for i, r in sub.iterrows():
        for j, fn in enumerate(ALL_FEATURES):
            if not np.isnan(H[i, j]):
                fg_rows.append(dict(
                    features=_fmt_features(r["features"]),
                    seed=int(r["seed"]),
                    feature=fn,
                    feature_gate_sigmoid=float(H[i, j]),
                ))
    pd.DataFrame(fg_rows).to_csv(OUT_DIR / "exp5b_feature_gates.csv", index=False)
    print(f"[figC] wrote {OUT_DIR/'exp5b_feature_gates.csv'}")


if __name__ == "__main__":
    main()
