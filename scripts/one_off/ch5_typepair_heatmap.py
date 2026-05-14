"""ch5_typepair_heatmap.py — Exp 5C learned type-pair table heatmaps (entry point E).

Loads the `table_raw` parameter from each model.pt checkpoint for the five
(init, freeze) cells in Exp 5C. Produces:

  1. figure-typepair_learned_tables.pdf
     A 2×3 panel grid (row = init group, col = conditions):
       Row 0: none (3 seeds), seed-mean
       Row 1: binary free (seed-mean), binary frozen (seed-mean)
       Row 2: fc free (seed-mean), fc frozen (seed-mean)
     Rows 1–2 also show physics-init baseline for comparison.

  2. figure-typepair_diff_from_init.pdf
     Difference of each learned seed-mean table from its initialisation.
     Diverging colormap; only for init ∈ {binary, fixed_coupling}.
     Shows what the network added to / subtracted from the physics prior.

  3. exp5c_typepair_table_stats.csv
     Per-(init, freeze, seed) scalar statistics: Frobenius norm, gate value,
     drift from init (||learned - init||_F).

State dict key: bias_composer.bias_modules.typepair_kinematic.table_raw
Symmetric table used in forward: 0.5*(raw + raw.T) * pad_mask
Gate used in forward: tanh(gate) * bias  -- extracted but not applied to table
  (the table is shown without the scalar gate for interpretability).

Type IDs (0=pad, 1=jet, 2=b-jet, 3=e+, 4=e-, 5=mu+, 6=mu-, 7=photon)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from thesis_ml.reports.plots.style import apply_thesis_style, axis_color, figure_size
from thesis_ml.monitoring.io_utils import save_figure

apply_thesis_style()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch5_typepair/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint paths as listed in the evidence note
RUNS_P1 = [
    f"/data/atlas/users/nterlind/outputs/runs/run_20260511-193832_ch5_typepair_p1_job{i:03d}/model.pt"
    for i in range(3)
]
RUNS_P2 = [
    f"/data/atlas/users/nterlind/outputs/runs/run_20260511-200334_ch5_typepair_p2_job{i:03d}/model.pt"
    for i in range(12)
]

GROUPS: dict[str, list[str]] = {
    "none":          RUNS_P1,
    "binary_free":   RUNS_P2[0:3],
    "binary_frozen": RUNS_P2[3:6],
    "fc_free":       RUNS_P2[6:9],
    "fc_frozen":     RUNS_P2[9:12],
}

# Human-readable labels used on panels
LABELS = {
    "none":          "none (zero init)",
    "binary_free":   "binary, free",
    "binary_frozen": "binary, frozen",
    "fc_free":       "fixed coupling, free",
    "fc_frozen":     "fixed coupling, frozen",
}

TYPE_NAMES = ["pad", "jet", "b-jet", "e+", "e-", "mu+", "mu-", "photon"]
# We show only indices 1-7 (skip padding row/col)
SHOW_IDX = list(range(1, 8))
SHOW_NAMES = [TYPE_NAMES[i] for i in SHOW_IDX]

TABLE_KEY = "bias_composer.bias_modules.typepair_kinematic.table_raw"
GATE_KEY  = "bias_composer.bias_modules.typepair_kinematic.gate"
PAD_KEY   = "bias_composer.bias_modules.typepair_kinematic.pad_mask"

_GS: float = 1.22
_GE: float = 0.31
_GZ: float = 0.758
MASK_VALUE: float = -5.0

# ---------------------------------------------------------------------------
# Build physics-init reference tables  (reproduce _build_sm_tables)
# ---------------------------------------------------------------------------

def _build_ref_tables() -> dict[str, np.ndarray]:
    n = 8
    binary = np.full((n, n), MASK_VALUE)
    coupling = np.full((n, n), MASK_VALUE)

    def _set(i: int, j: int, b: float, c: float) -> None:
        binary[i, j] = b; binary[j, i] = b
        coupling[i, j] = c; coupling[j, i] = c

    _set(1, 1, 1.0, _GS)
    _set(1, 2, 1.0, _GS)
    _set(2, 2, 1.0, _GS)
    _set(1, 7, 1.0, _GE * 0.5)
    _set(2, 7, 1.0, _GE / 3.0)
    _set(3, 7, 1.0, _GE)
    _set(4, 7, 1.0, _GE)
    _set(5, 7, 1.0, _GE)
    _set(6, 7, 1.0, _GE)
    _set(3, 4, 1.0, _GZ)
    _set(5, 6, 1.0, _GZ)
    binary[0, :] = 0.0; binary[:, 0] = 0.0
    coupling[0, :] = 0.0; coupling[:, 0] = 0.0
    zeros = np.zeros((n, n))
    return {"none": zeros, "binary": binary, "fixed_coupling": coupling}

REF = _build_ref_tables()

# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def _load_table(path: str) -> tuple[np.ndarray, float]:
    """Load symmetric table (8x8) and gate from a checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    msd = ckpt["model_state_dict"]
    raw = msd[TABLE_KEY].float()
    pad = msd[PAD_KEY].float()
    sym = (0.5 * (raw + raw.T) * pad).numpy()
    gate = float(torch.tanh(msd[GATE_KEY]).item())
    return sym, gate


def load_group(paths: list[str]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Return (tables [n_seeds, 8, 8], mean_table [8, 8], gates [n_seeds])."""
    tables, gates = [], []
    for p in paths:
        t, g = _load_table(p)
        tables.append(t)
        gates.append(g)
    arr = np.stack(tables, axis=0)
    return arr, arr.mean(axis=0), gates


# ---------------------------------------------------------------------------
# Scalar statistics table
# ---------------------------------------------------------------------------

def build_stats_csv() -> pd.DataFrame:
    records = []
    init_for_group = {
        "none": "none",
        "binary_free": "binary", "binary_frozen": "binary",
        "fc_free": "fixed_coupling", "fc_frozen": "fixed_coupling",
    }
    for gname, paths in GROUPS.items():
        init_key = init_for_group[gname]
        ref_full = REF[init_key]  # shape [8,8]
        for seed_idx, p in enumerate(paths):
            sym, gate = _load_table(p)
            # Use 7x7 sub-matrix (skip pad row/col) for norm
            sym7 = sym[np.ix_(SHOW_IDX, SHOW_IDX)]
            ref7 = ref_full[np.ix_(SHOW_IDX, SHOW_IDX)]
            frob = float(np.linalg.norm(sym7))
            drift = float(np.linalg.norm(sym7 - ref7))
            records.append({
                "group": gname,
                "init": init_key,
                "freeze": "frozen" in gname,
                "seed_idx": seed_idx,
                "gate": gate,
                "frob_norm": frob,
                "drift_from_init": drift,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Heatmap helper
# ---------------------------------------------------------------------------

def _heatmap(ax: plt.Axes, table: np.ndarray, vmin: float, vmax: float,
             cmap: str, label: str) -> plt.cm.ScalarMappable:
    """Draw a 7x7 (rows/cols 1-7) heatmap on ax. Returns the mappable."""
    sub = table[np.ix_(SHOW_IDX, SHOW_IDX)]
    im = ax.imshow(sub, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(SHOW_NAMES)))
    ax.set_yticks(range(len(SHOW_NAMES)))
    ax.set_xticklabels(SHOW_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(SHOW_NAMES)
    # Annotate cells (7x7 <= 15x15 threshold)
    for r in range(len(SHOW_NAMES)):
        for c in range(len(SHOW_NAMES)):
            val = sub[r, c]
            color = "white" if val < (vmin + (vmax - vmin) * 0.5) else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=5)
    ax.set_xlabel("token type j")
    ax.set_ylabel("token type i")
    return im


# ---------------------------------------------------------------------------
# Figure 1: learned seed-mean tables (5 panels)
# ---------------------------------------------------------------------------

def plot_learned_tables(
    group_data: dict[str, tuple[np.ndarray, np.ndarray, list[float]]]
) -> plt.Figure:
    """5-panel row of seed-mean learned tables, one per (init, freeze) cell."""
    n_panels = 5
    panel_w = figure_size("half", aspect=1.0)[0]
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(n_panels * panel_w, panel_w + 0.6),
    )

    # Determine common colour range across all groups (for comparability)
    all_vals = np.concatenate([
        group_data[g][1][np.ix_(SHOW_IDX, SHOW_IDX)].ravel()
        for g in GROUPS
    ])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    for ax, (gname, lbl) in zip(axes, LABELS.items()):
        _, mean_tbl, gates = group_data[gname]
        im = _heatmap(ax, mean_tbl, vmin=vmin, vmax=vmax, cmap="RdBu_r", label=lbl)
        mean_gate = float(np.mean(gates))
        std_gate = float(np.std(gates))
        ax.set_xlabel(f"token type j\ngate={mean_gate:.3f}±{std_gate:.3f}")

    # Single shared colorbar on the right
    fig.subplots_adjust(right=0.88, wspace=0.45)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("table value")

    return fig


# ---------------------------------------------------------------------------
# Figure 2: difference from physics init (binary-free, binary-frozen,
#            fc-free, fc-frozen — 4 panels)
# ---------------------------------------------------------------------------

def plot_diff_from_init(
    group_data: dict[str, tuple[np.ndarray, np.ndarray, list[float]]]
) -> plt.Figure:
    """4-panel row: seed-mean learned table minus its initialization."""
    diff_groups = ["binary_free", "binary_frozen", "fc_free", "fc_frozen"]
    init_map = {
        "binary_free": "binary", "binary_frozen": "binary",
        "fc_free": "fixed_coupling", "fc_frozen": "fixed_coupling",
    }
    n_panels = 4
    panel_w = figure_size("half", aspect=1.0)[0]
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(n_panels * panel_w, panel_w + 0.6),
    )

    diffs = []
    for gname in diff_groups:
        _, mean_tbl, _ = group_data[gname]
        init_key = init_map[gname]
        diff = mean_tbl - REF[init_key]
        diffs.append(diff[np.ix_(SHOW_IDX, SHOW_IDX)])

    all_diff = np.concatenate([d.ravel() for d in diffs])
    abs_max = float(max(abs(all_diff.min()), abs(all_diff.max())))
    vmin, vmax = -abs_max, abs_max

    for ax, gname, diff in zip(axes, diff_groups, diffs):
        lbl = LABELS[gname]
        # Reconstruct mean table for annotation: diff + init
        sub_init = REF[init_map[gname]][np.ix_(SHOW_IDX, SHOW_IDX)]
        im = ax.imshow(diff, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(SHOW_NAMES)))
        ax.set_yticks(range(len(SHOW_NAMES)))
        ax.set_xticklabels(SHOW_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(SHOW_NAMES)
        for r in range(len(SHOW_NAMES)):
            for c in range(len(SHOW_NAMES)):
                val = diff[r, c]
                color = "white" if abs(val) > abs_max * 0.6 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=5)
        ax.set_xlabel("token type j")
        ax.set_ylabel("token type i")

    fig.subplots_adjust(right=0.88, wspace=0.45)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("learned - init")

    return fig


# ---------------------------------------------------------------------------
# Figure 3: physics init reference tables (binary and fixed_coupling)
# ---------------------------------------------------------------------------

def plot_reference_tables() -> plt.Figure:
    """Two-panel figure showing the two physics-init reference tables."""
    panel_w = figure_size("half", aspect=1.0)[0]
    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_w, panel_w + 0.6))

    ref_items = [("binary", "binary init"), ("fixed_coupling", "fixed coupling init")]
    all_vals = np.concatenate([REF[k][np.ix_(SHOW_IDX, SHOW_IDX)].ravel()
                                for k, _ in ref_items])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    for ax, (rkey, rlbl) in zip(axes, ref_items):
        im = _heatmap(ax, REF[rkey], vmin=vmin, vmax=vmax, cmap="RdBu_r", label=rlbl)
        ax.set_xlabel(f"token type j\n({rlbl})")

    fig.subplots_adjust(right=0.88, wspace=0.45)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("init value")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load all groups
    group_data: dict[str, tuple[np.ndarray, np.ndarray, list[float]]] = {}
    for gname, paths in GROUPS.items():
        print(f"Loading {gname} ({len(paths)} seeds)...", flush=True)
        tables, mean_tbl, gates = load_group(paths)
        group_data[gname] = (tables, mean_tbl, gates)
        frob = float(np.linalg.norm(mean_tbl[np.ix_(SHOW_IDX, SHOW_IDX)]))
        print(f"  gate mean={np.mean(gates):.4f}  frob(mean_tbl)={frob:.4f}")

    # Stats CSV
    stats = build_stats_csv()
    csv_path = OUT_DIR / "exp5c_typepair_table_stats.csv"
    stats.to_csv(csv_path, index=False)
    print(f"\nStats written to: {csv_path}")
    print(stats.groupby(["group", "freeze"])[["frob_norm", "drift_from_init", "gate"]].mean().to_string())

    cfg = {"fig_format": "pdf", "dpi": 300}

    # Figure 1: learned tables
    fig1 = plot_learned_tables(group_data)
    p1 = save_figure(fig1, OUT_DIR, "figure-typepair_learned_tables", cfg)
    print(f"\nFigure 1 saved: {p1}")
    plt.close(fig1)

    # Figure 2: diff from init
    fig2 = plot_diff_from_init(group_data)
    p2 = save_figure(fig2, OUT_DIR, "figure-typepair_diff_from_init", cfg)
    print(f"Figure 2 saved: {p2}")
    plt.close(fig2)

    # Figure 3: reference tables
    fig3 = plot_reference_tables()
    p3 = save_figure(fig3, OUT_DIR, "figure-typepair_reference_tables", cfg)
    print(f"Figure 3 saved: {p3}")
    plt.close(fig3)

    print("\nDone.")


if __name__ == "__main__":
    main()
