"""ch5_attention_maps.py — Exp 5A attention-map type-pair visualisations.

Three figures are written to
``/data/atlas/users/nterlind/outputs/reports/report_ch5_attention_maps/``:

1. ``figure-attn_typepair_mean_none_vs_full.pdf``
   Side-by-side 7×7 type-pair heatmaps for the ``none`` and
   ``lorentz_scalar+typepair_kinematic+sm_interaction+global_conditioned``
   (full-combo) models. Each cell shows the attention weight averaged over
   all heads, all layers, all seed runs, and all events in the validation
   batch.  Pad tokens (type 0) are excluded from axes.

2. ``figure-attn_typepair_per_head_none.pdf``
   4 heads × 3 layers grid of 7×7 type-pair heatmaps for the ``none`` model
   (seed 42 representative run). Shows per-head / per-layer attention
   structure.

3. ``figure-attn_typepair_per_head_full.pdf``
   Same layout for the full-combo model (seed 42 representative run).

Sequence layout (D02=True, CLS pooling):
    [CLS(0)] + [particle tokens 1..18] + [MET(19)] + [MET-phi(20)]
    Total = 21 tokens.
For type-pair aggregation only the 18 physical particle tokens are used:
    attention sub-block rows/cols [1:19, 1:19] (0-indexed into the 21×21 map).

Token type IDs: 0=pad 1=jet 2=b-jet 3=e+ 4=e- 5=mu+ 6=mu- 7=photon
Only types 1–7 are shown on axes (pad excluded).

Model loading: ``resolved_config.yaml`` → ``build_from_config`` →
``load_state_dict``.  Attention maps are extracted by calling the encoder
with ``capture_attention=True`` after ``prepare_encoder_inputs``.

Run this script as:
    python scripts/one_off/ch5_attention_maps.py
from the repo root after ``source ~/.bashrc && thesis``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from thesis_ml.architectures.transformer_classifier.base import build_from_config
from thesis_ml.monitoring.io_utils import save_figure
from thesis_ml.reports.plots.style import apply_thesis_style, axis_color, figure_size

apply_thesis_style()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUNS_ROOT = Path("/data/atlas/users/nterlind/outputs/runs")
OUT_DIR = Path("/data/atlas/users/nterlind/outputs/reports/report_ch5_attention_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path("/data/atlas/users/nterlind/datasets/4tops_splitted.h5")

# Exp 5A run directories: job→(attention_biases, seed) as confirmed from
# .hydra/overrides.yaml in each run dir.
#   jobs 000-002: none,           seeds 42/123/314
#   jobs 015-017: all-four-combo, seeds 42/123/314
FAMILY_JOBS: dict[str, list[int]] = {
    "none": [0, 1, 2],
    "full": [15, 16, 17],
}
TS = "20260511-144128"

# Sequence layout constants
T_PHYS = 18          # number of physical particle tokens
CLS_OFFSET = 1       # index of first physical token in the 21-token sequence
# Slice into the 21-token attention map to get the 18×18 physical block:
PHYS_SLICE = slice(CLS_OFFSET, CLS_OFFSET + T_PHYS)

# Type axis: skip pad (0), show 1-7
TYPE_NAMES_ALL = ["pad", "jet", "b-jet", "e+", "e-", "mu+", "mu-", "photon"]
SHOW_TYPES = list(range(1, 8))
SHOW_NAMES = [TYPE_NAMES_ALL[i] for i in SHOW_TYPES]

N_EVENTS_BATCH = 2000   # validation events to average over
N_LAYERS = 3
N_HEADS = 4

# Meta dict used by build_from_config for the 5A models (identity tokenizer,
# raw format, no vocabulary — these values are fixed for all 18 jobs).
MODEL_META = {
    "n_tokens": T_PHYS,
    "token_feat_dim": 4,
    "has_globals": True,
    "n_classes": 2,
    "num_types": 8,
    "vocab_size": None,
}

PLOT_CFG = {"fig_format": "pdf", "dpi": 300}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _run_dir(job_idx: int) -> Path:
    return RUNS_ROOT / f"run_{TS}_ch5_bias_families_job{job_idx:03d}"


def load_model(job_idx: int) -> torch.nn.Module:
    """Build TransformerClassifier from resolved_config.yaml and load weights."""
    run_dir = _run_dir(job_idx)
    cfg_path = run_dir / "resolved_config.yaml"
    ckpt_path = run_dir / "model.pt"

    with open(cfg_path) as f:
        cfg_raw = yaml.safe_load(f)

    # Patch the data path template so build_from_config does not try to
    # resolve env vars (we never call the data loader from here).
    cfg_raw["data"]["path"] = str(DATA_PATH)

    cfg = OmegaConf.create(cfg_raw)
    model = build_from_config(cfg, MODEL_META)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_validation_batch(n_events: int = N_EVENTS_BATCH, seed: int = 0):
    """Load a normalised validation batch.

    Returns
    -------
    tokens_cont : Tensor [N, T_PHYS, 4]
    tokens_id   : Tensor [N, T_PHYS]  (long)
    globals_    : Tensor [N, 2]
    mask        : Tensor [N, T_PHYS]  (bool; True=valid)
    """
    with h5py.File(str(DATA_PATH), "r") as f:
        Xtr = f["X_train"][:200_000]
        Xva = f["X_val"][:]

    T = T_PHYS
    train_cont = torch.tensor(Xtr[:, T + 2:], dtype=torch.float32).view(-1, T, 4)
    val_all = torch.tensor(Xva, dtype=torch.float32)

    # Normalisation: z-score using train stats (matches H5ClassificationDataset)
    mu = train_cont.mean(dim=(0, 1), keepdim=True)   # [1, 1, 4]
    sd = train_cont.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)

    # Subsample
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(val_all), size=min(n_events, len(val_all)), replace=False)

    Xva_sub = val_all[idx]

    tokens_id = Xva_sub[:, :T].long()              # [N, T]
    globals_ = Xva_sub[:, T: T + 2]                # [N, 2]
    cont = Xva_sub[:, T + 2:].view(-1, T, 4)        # [N, T, 4]
    tokens_cont = (cont - mu[0, 0]) / sd[0, 0]     # normalise

    # Mask: True where token type != 0 (not padding)
    mask = tokens_id != 0                           # [N, T]

    return tokens_cont, tokens_id, globals_, mask


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_attention_maps(
    model: torch.nn.Module,
    tokens_cont: torch.Tensor,
    tokens_id: torch.Tensor,
    globals_: torch.Tensor,
    mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Run a forward pass and capture per-layer attention weights.

    Parameters
    ----------
    model : TransformerClassifier
    tokens_cont : [N, T, 4]
    tokens_id   : [N, T]
    globals_    : [N, 2]
    mask        : [N, T]  True=valid

    Returns
    -------
    List of length n_layers, each element [N, H, T_full, T_full] where
    T_full = T_PHYS + 1 (CLS) + 2 (MET, MET-phi) = 21.
    """
    # prepare_encoder_inputs returns (x, mask_full, attn_bias, ...)
    x, mask_full, attn_bias, _, _, _, _ = model.prepare_encoder_inputs(
        tokens_cont, tokens_id, globals_, mask=mask
    )

    # Run encoder with attention capture
    _, layer_weights = model.encoder(
        x, mask=mask_full, attention_bias=attn_bias, capture_attention=True
    )

    # layer_weights: list of [N, H, T_full, T_full] or None
    result = []
    for w in layer_weights:
        if w is None:
            # Should not happen for standard attention, but guard anyway
            T_full = T_PHYS + 1 + 2  # 21
            result.append(torch.zeros(tokens_cont.size(0), N_HEADS, T_full, T_full))
        else:
            result.append(w)  # [N, H, T_full, T_full]
    return result


# ---------------------------------------------------------------------------
# Type-pair aggregation
# ---------------------------------------------------------------------------


def aggregate_typepair(
    attn_layers: list[torch.Tensor],
    tokens_id: torch.Tensor,
) -> np.ndarray:
    """Compute mean attention per (type_i → type_j) pair.

    Averages over all layers, all heads, all events, and all token-position
    pairs whose source and destination match a given type pair.

    Parameters
    ----------
    attn_layers : list of [N, H, T_full, T_full]  (n_layers elements)
    tokens_id   : [N, T_PHYS]  (long)

    Returns
    -------
    table : np.ndarray [8, 8]  indexed as table[type_i, type_j]
             (type 0 = pad, excluded from display but present in array)
    """
    n_types = 8
    # Sum and count arrays for averaging
    attn_sum = np.zeros((n_types, n_types), dtype=np.float64)
    attn_cnt = np.zeros((n_types, n_types), dtype=np.float64)

    # Stack all layers: [L, N, H, T_full, T_full]
    # Then take the physical sub-block [1:19, 1:19]
    stacked = torch.stack(attn_layers, dim=0)   # [L, N, H, T_full, T_full]
    # Slice to physical tokens: rows/cols PHYS_SLICE = slice(1, 19)
    attn_phys = stacked[:, :, :, PHYS_SLICE, PHYS_SLICE]  # [L, N, H, 18, 18]

    L, N, H, T, _ = attn_phys.shape
    # Flatten L, H → one axis for efficiency
    attn_flat = attn_phys.permute(1, 0, 2, 3, 4)  # [N, L, H, T, T]
    attn_flat = attn_flat.contiguous().view(N, L * H, T, T)  # [N, LH, T, T]

    ids = tokens_id.numpy()  # [N, T]

    for n_idx in range(N):
        id_row = ids[n_idx]   # [T]
        for ti in range(T):
            type_i = int(id_row[ti])
            if type_i == 0:
                continue
            for tj in range(T):
                type_j = int(id_row[tj])
                if type_j == 0:
                    continue
                # Mean over LH heads for this (n_idx, ti, tj)
                val = float(attn_flat[n_idx, :, ti, tj].mean().item())
                attn_sum[type_i, type_j] += val
                attn_cnt[type_i, type_j] += 1.0

    # Safe divide
    with np.errstate(invalid="ignore", divide="ignore"):
        table = np.where(attn_cnt > 0, attn_sum / attn_cnt, 0.0)
    return table.astype(np.float32)


def aggregate_typepair_per_head(
    attn_layers: list[torch.Tensor],
    tokens_id: torch.Tensor,
) -> np.ndarray:
    """Same as ``aggregate_typepair`` but returns a [L, H, 8, 8] array.

    One 8×8 type-pair table per (layer, head).
    """
    n_types = 8
    L = len(attn_layers)
    H = attn_layers[0].shape[1]
    attn_sum = np.zeros((L, H, n_types, n_types), dtype=np.float64)
    attn_cnt = np.zeros((L, H, n_types, n_types), dtype=np.float64)

    ids = tokens_id.numpy()  # [N, T]

    for layer_idx, attn_lyr in enumerate(attn_layers):
        # attn_lyr: [N, H, T_full, T_full]
        attn_phys = attn_lyr[:, :, PHYS_SLICE, PHYS_SLICE]  # [N, H, 18, 18]
        N, _, T, _ = attn_phys.shape
        attn_np = attn_phys.numpy()  # [N, H, T, T]

        for n_idx in range(N):
            id_row = ids[n_idx]  # [T]
            for h_idx in range(H):
                for ti in range(T):
                    type_i = int(id_row[ti])
                    if type_i == 0:
                        continue
                    for tj in range(T):
                        type_j = int(id_row[tj])
                        if type_j == 0:
                            continue
                        val = float(attn_np[n_idx, h_idx, ti, tj])
                        attn_sum[layer_idx, h_idx, type_i, type_j] += val
                        attn_cnt[layer_idx, h_idx, type_i, type_j] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        table = np.where(attn_cnt > 0, attn_sum / attn_cnt, 0.0)
    return table.astype(np.float32)


# ---------------------------------------------------------------------------
# Run a family: load models + data, compute type-pair tables
# ---------------------------------------------------------------------------


def compute_family_tables(
    family_name: str,
    jobs: list[int],
    tokens_cont: torch.Tensor,
    tokens_id: torch.Tensor,
    globals_: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """For each job (seed), extract attention maps and compute type-pair tables.

    Returns dict with:
        "mean_table"     : np.ndarray [8, 8] — mean over all seeds/layers/heads/events
        "per_head_seed0" : np.ndarray [L, H, 8, 8] — per-layer/head table for job[0] (seed=42)
    """
    all_mean_tables = []
    per_head_seed0 = None

    for seed_idx, job_idx in enumerate(jobs):
        print(f"  [{family_name}] seed_idx={seed_idx} job={job_idx:03d}: loading model…",
              flush=True)
        model = load_model(job_idx)

        # Process in sub-batches to avoid memory issues
        batch_size = 256
        n_total = tokens_cont.shape[0]
        all_layer_weights: list[list[torch.Tensor]] = [[] for _ in range(N_LAYERS)]

        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            tc = tokens_cont[start:end]
            ti = tokens_id[start:end]
            gl = globals_[start:end]
            mk = mask[start:end]
            layer_weights = extract_attention_maps(model, tc, ti, gl, mk)
            for l_idx, w in enumerate(layer_weights):
                all_layer_weights[l_idx].append(w)

        # Concatenate sub-batches along event dimension
        concat_layers = [torch.cat(all_layer_weights[l], dim=0) for l in range(N_LAYERS)]

        print(f"  [{family_name}] seed_idx={seed_idx}: computing type-pair tables…",
              flush=True)
        mean_tbl = aggregate_typepair(concat_layers, tokens_id)
        all_mean_tables.append(mean_tbl)

        if seed_idx == 0:
            per_head_seed0 = aggregate_typepair_per_head(concat_layers, tokens_id)

    # Average mean tables across seeds
    mean_over_seeds = np.stack(all_mean_tables, axis=0).mean(axis=0)  # [8, 8]

    return {
        "mean_table": mean_over_seeds,
        "per_head_seed0": per_head_seed0,  # [L, H, 8, 8]
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _draw_typepair_heatmap(
    ax: plt.Axes,
    table: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    annotate: bool = True,
    fontsize_annot: float = 5.0,
) -> plt.cm.ScalarMappable:
    """Draw a 7×7 type-pair heatmap (types 1-7, skipping pad) on ax."""
    sub = table[np.ix_(SHOW_TYPES, SHOW_TYPES)]  # [7, 7]
    im = ax.imshow(sub, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(SHOW_NAMES)))
    ax.set_yticks(range(len(SHOW_NAMES)))
    ax.set_xticklabels(SHOW_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(SHOW_NAMES)
    ax.set_xlabel("token type j (key)")
    ax.set_ylabel("token type i (query)")

    if annotate:
        norm_range = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0
        for r in range(len(SHOW_NAMES)):
            for c in range(len(SHOW_NAMES)):
                val = sub[r, c]
                brightness = (val - vmin) / norm_range
                text_color = "white" if brightness < 0.5 else "black"
                ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                        color=text_color, fontsize=fontsize_annot)
    return im


# ---------------------------------------------------------------------------
# Figure 1: mean type-pair heatmap — none vs full-combo
# ---------------------------------------------------------------------------


def plot_mean_comparison(
    none_table: np.ndarray,
    full_table: np.ndarray,
) -> plt.Figure:
    """Two-panel figure: mean type-pair attn for none vs full-combo models."""
    panel_w = figure_size("half", aspect=1.0)[0]
    fig, axes = plt.subplots(
        1, 2,
        figsize=(2 * panel_w + 0.8, panel_w + 0.6),
        constrained_layout=False,
    )

    # Common colour range across both panels
    none_sub = none_table[np.ix_(SHOW_TYPES, SHOW_TYPES)]
    full_sub = full_table[np.ix_(SHOW_TYPES, SHOW_TYPES)]
    all_vals = np.concatenate([none_sub.ravel(), full_sub.ravel()])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    labels = ["none (no bias)", "full combination"]
    tables = [none_table, full_table]

    for ax, tbl, lbl in zip(axes, tables, labels):
        im = _draw_typepair_heatmap(ax, tbl, vmin=vmin, vmax=vmax, cmap="viridis",
                                     annotate=True, fontsize_annot=4.5)

    # Shared colour bar on the right
    fig.subplots_adjust(right=0.85, wspace=0.55)
    cbar_ax = fig.add_axes([0.87, 0.18, 0.02, 0.65])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("mean attention weight")

    # Panel labels below x-axis label
    for ax, lbl in zip(axes, labels):
        ax.set_xlabel(f"token type j (key)\n[{lbl}]")

    return fig


# ---------------------------------------------------------------------------
# Figure 2/3: per-head × per-layer grid (4 heads × 3 layers)
# ---------------------------------------------------------------------------


def plot_per_head_grid(
    per_head_table: np.ndarray,   # [L, H, 8, 8]
    family_label: str,
) -> plt.Figure:
    """4-head × 3-layer grid of type-pair heatmaps for a single model seed."""
    L, H, _, _ = per_head_table.shape
    panel_w = figure_size("half", aspect=1.0)[0]
    fig, axes = plt.subplots(
        L, H,
        figsize=(H * panel_w, L * panel_w + 0.4 * L),
        constrained_layout=False,
    )

    # Common colour range across all (layer, head) panels
    all_vals = []
    for l_idx in range(L):
        for h_idx in range(H):
            sub = per_head_table[l_idx, h_idx][np.ix_(SHOW_TYPES, SHOW_TYPES)]
            all_vals.append(sub.ravel())
    all_vals_np = np.concatenate(all_vals)
    vmin, vmax = float(all_vals_np.min()), float(all_vals_np.max())

    for l_idx in range(L):
        for h_idx in range(H):
            ax = axes[l_idx, h_idx]
            tbl = per_head_table[l_idx, h_idx]
            im = _draw_typepair_heatmap(
                ax, tbl, vmin=vmin, vmax=vmax, cmap="viridis",
                annotate=True, fontsize_annot=3.5,
            )
            # Row label on the leftmost column
            if h_idx == 0:
                ax.set_ylabel(f"Layer {l_idx}\ntoken type i (query)")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            # Column label on the top row
            if l_idx == 0:
                ax.set_xlabel(f"Head {h_idx}\ntoken type j (key)")
            elif l_idx == L - 1:
                ax.set_xlabel("token type j (key)")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

    # Shared colour bar
    fig.subplots_adjust(right=0.87, hspace=0.55, wspace=0.45)
    cbar_ax = fig.add_axes([0.89, 0.12, 0.018, 0.76])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("mean attention weight")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading validation batch…", flush=True)
    tokens_cont, tokens_id, globals_, mask = load_validation_batch(
        n_events=N_EVENTS_BATCH, seed=0
    )
    print(f"  Loaded {tokens_cont.shape[0]} events  "
          f"(tokens_cont={tokens_cont.shape}, tokens_id={tokens_id.shape})",
          flush=True)

    results: dict[str, dict] = {}
    for family, jobs in FAMILY_JOBS.items():
        print(f"\nProcessing family '{family}' (jobs {jobs})…", flush=True)
        results[family] = compute_family_tables(
            family, jobs, tokens_cont, tokens_id, globals_, mask
        )

    # ------------------------------------------------------------------
    # Figure 1: mean type-pair comparison (none vs full)
    # ------------------------------------------------------------------
    print("\nPlotting Figure 1: mean type-pair heatmaps (none vs full)…", flush=True)
    fig1 = plot_mean_comparison(
        results["none"]["mean_table"],
        results["full"]["mean_table"],
    )
    p1 = save_figure(fig1, OUT_DIR, "figure-attn_typepair_mean_none_vs_full", PLOT_CFG)
    plt.close(fig1)
    print(f"  Saved: {p1}")

    # ------------------------------------------------------------------
    # Figure 2: per-head grid, none model (seed 42 = job000)
    # ------------------------------------------------------------------
    print("Plotting Figure 2: per-head grid, none model…", flush=True)
    fig2 = plot_per_head_grid(results["none"]["per_head_seed0"], "none")
    p2 = save_figure(fig2, OUT_DIR, "figure-attn_typepair_per_head_none", PLOT_CFG)
    plt.close(fig2)
    print(f"  Saved: {p2}")

    # ------------------------------------------------------------------
    # Figure 3: per-head grid, full-combo model (seed 42 = job015)
    # ------------------------------------------------------------------
    print("Plotting Figure 3: per-head grid, full-combo model…", flush=True)
    fig3 = plot_per_head_grid(results["full"]["per_head_seed0"], "full")
    p3 = save_figure(fig3, OUT_DIR, "figure-attn_typepair_per_head_full", PLOT_CFG)
    plt.close(fig3)
    print(f"  Saved: {p3}")

    # ------------------------------------------------------------------
    # Companion CSV: per-type-pair mean attention for none and full
    # ------------------------------------------------------------------
    import pandas as pd

    rows = []
    for family_name, res in results.items():
        tbl = res["mean_table"]
        for ti in SHOW_TYPES:
            for tj in SHOW_TYPES:
                rows.append({
                    "family": family_name,
                    "type_i": TYPE_NAMES_ALL[ti],
                    "type_j": TYPE_NAMES_ALL[tj],
                    "mean_attn": float(tbl[ti, tj]),
                })
    csv_path = OUT_DIR / "exp5a_typepair_mean_attn.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nCompanion CSV: {csv_path}")

    print("\nDone. Output directory:", OUT_DIR)


if __name__ == "__main__":
    main()
