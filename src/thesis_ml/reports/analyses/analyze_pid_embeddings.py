"""Report: Analyze PID Embeddings.

Deep-dive analysis of Particle ID embedding representations.

Analyses:
1. **Performance comparison**: accuracy / F1 / AUROC across pid_mode × dim × schedule
2. **Cosine similarity heatmaps**: pairwise cosine similarity between particle type vectors
3. **Orthogonality evolution**: mean off-diagonal cosine over epochs
4. **PCA / tSNE of embedding vectors**: 2D scatter of the 8 particle types
5. **Embedding norm profiles**: per-type L2 norms
6. **Isotropy evolution**: how uniformly the embeddings span the space

Reads:
- Per-epoch training curves from facts/scalars.csv
- PID snapshots from pid_snapshots/epoch_XXXX.pt (saved by training loop)
- Resolved config from .hydra/ or resolved_config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from omegaconf import DictConfig

from thesis_ml.facts.readers import _extract_value_from_composed_cfg, _read_cfg
from thesis_ml.reports.plots.curves import plot_all_train_curves, plot_all_val_curves
from thesis_ml.reports.utils.io import finalize_report, get_fig_config, setup_report_environment

logger = logging.getLogger(__name__)


# Human-readable labels for PID indices (dataset-specific).
# Index 0 is effectively unused/padding in this setup.
PARTICLE_LABELS: list[str] = [
    "pad",  # 0
    "j",  # 1: jet
    "b",  # 2: b-jet
    "e+",  # 3: positron
    "e-",  # 4: electron
    "mu+",  # 5: muon
    "mu-",  # 6: anti-muon
    "g",  # 7: photon
]


def _pid_label(pid: int) -> str:
    """Map PID index to human-readable label."""
    if 0 <= pid < len(PARTICLE_LABELS):
        return PARTICLE_LABELS[pid]
    return str(pid)


# #region agent log
def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    """Lightweight NDJSON logger for debugging analyze_pid_embeddings."""
    try:
        import json
        import time

        log_path = Path(".cursor") / "debug.log"
        payload = {
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "runId": "pid-deepdive",
            "hypothesisId": hypothesis_id,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        # Never let debugging interfere with report generation
        return


# #endregion agent log


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def _extract_pid_metadata(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich runs_df with PID experiment metadata.

    Adds columns: pid_mode, id_embed_dim, pid_schedule, transition_epoch, run_label.
    """
    df = runs_df.copy()

    pid_modes: list[str] = []
    id_embed_dims: list[int] = []
    pid_schedules: list[str] = []
    transition_epochs: list[int | None] = []
    labels: list[str] = []

    for run_dir in df["run_dir"]:
        run_path = Path(run_dir)
        try:
            cfg, _ = _read_cfg(run_path)
        except Exception as e:
            logger.warning("Failed to read cfg for %s: %s", run_dir, e)
            pid_modes.append("unknown")
            id_embed_dims.append(0)
            pid_schedules.append("standard")
            transition_epochs.append(None)
            labels.append("unknown")
            continue

        # Read PID mode and dim explicitly from composed config to be robust
        # against different tokenizer config shapes.
        mode = _extract_value_from_composed_cfg(cfg, "classifier.model.tokenizer.pid_mode") or "learned"
        dim_val = _extract_value_from_composed_cfg(cfg, "classifier.model.tokenizer.id_embed_dim", int)
        dim = dim_val if dim_val is not None else 8

        # Prefer pid_schedule information from the run's overrides.yaml if present,
        # fall back to composed config otherwise.
        sched = "standard"
        trans_ep: int | None = None

        overrides_path = run_path / ".hydra" / "overrides.yaml"
        if overrides_path.exists():
            try:
                for line in overrides_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip().lstrip("-").strip()
                    if line.startswith("classifier.trainer.pid_schedule.mode="):
                        sched = line.split("=", 1)[1].strip()
                    elif line.startswith("classifier.trainer.pid_schedule.transition_epoch="):
                        val = line.split("=", 1)[1].strip()
                        try:
                            trans_ep = int(val)
                        except ValueError:
                            trans_ep = None
            except Exception as e:
                logger.warning("Failed to parse overrides.yaml for %s: %s", run_dir, e)

        if sched == "standard":
            # If not overridden (or parsing failed), try composed config.
            sched_cfg = _extract_value_from_composed_cfg(cfg, "classifier.trainer.pid_schedule")
            if isinstance(sched_cfg, dict):
                sched = sched_cfg.get("mode", "standard")
                trans_ep = sched_cfg.get("transition_epoch")
            else:
                sched = "standard"
                trans_ep = None

        pid_modes.append(mode)
        id_embed_dims.append(int(dim) if dim is not None else 8)
        pid_schedules.append(sched)
        transition_epochs.append(trans_ep)

        # Human-readable label
        if sched != "standard":
            labels.append(f"{mode}_d{dim}_{sched}")
        else:
            labels.append(f"{mode}_d{dim}")

    df["pid_mode"] = pid_modes
    df["id_embed_dim"] = id_embed_dims
    df["pid_schedule"] = pid_schedules
    df["transition_epoch"] = transition_epochs
    df["run_label"] = labels

    unique = df["run_label"].unique()
    logger.info("Found PID configurations: %s", sorted(unique.tolist()))

    return df


# ---------------------------------------------------------------------------
# PID snapshot loading
# ---------------------------------------------------------------------------


def _load_pid_snapshots(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load all pid_snapshots/epoch_XXXX.pt files for a single run.

    Returns list of dicts sorted by epoch, each containing:
      epoch, weight, cosine_matrix, norms, pid_mode, num_types, id_embed_dim
    """
    snap_dir = Path(run_dir) / "pid_snapshots"
    if not snap_dir.is_dir():
        return []
    snapshots = []
    for f in sorted(snap_dir.glob("epoch_*.pt")):
        try:
            snap = torch.load(f, map_location="cpu", weights_only=True)
            snapshots.append(snap)
        except Exception as e:
            logger.warning("Failed to load snapshot %s: %s", f, e)
    return snapshots


# ---------------------------------------------------------------------------
# Plotting: cosine similarity heatmap
# ---------------------------------------------------------------------------


def _plot_cosine_heatmap(
    cosine_matrix: torch.Tensor,
    title: str = "PID Cosine Similarity",
    fig_cfg: dict | None = None,
) -> Figure:
    """Plot a heatmap of the cosine similarity between PID embedding vectors."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Optionally drop padding PID (index 0) so plots focus on physical types.
    full = cosine_matrix.numpy()
    offset = 1 if full.shape[0] == len(PARTICLE_LABELS) else 0
    mat = full[offset:, offset:] if offset == 1 else full

    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = mat.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [_pid_label(i + offset) for i in range(n)]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Particle")
    ax.set_ylabel("Particle")
    ax.set_title(title)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if abs(mat[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plotting: orthogonality & isotropy evolution
# ---------------------------------------------------------------------------


def _plot_pid_evolution(
    snapshots: list[dict],
    run_label: str,
    transition_epoch: int | None = None,
    fig_cfg: dict | None = None,
) -> Figure:
    """Plot orthogonality and isotropy metrics over training epochs."""
    if not snapshots:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No PID snapshots", ha="center", va="center")
        return fig

    epochs = [s["epoch"] for s in snapshots]
    mean_off_diag = []
    max_off_diag = []
    isotropies = []
    mean_norms = []

    for s in snapshots:
        C = s["cosine_matrix"]
        N = C.size(0)
        # Focus orthogonality metrics on physical PIDs (skip padding index 0 if present).
        idx_start = 1 if len(PARTICLE_LABELS) == N else 0
        if idx_start == 1:
            idx = torch.arange(idx_start, N)
            C_phys = C[idx][:, idx]
        else:
            C_phys = C
        N_phys = C_phys.size(0)
        mask = ~torch.eye(N_phys, dtype=torch.bool)
        off = C_phys[mask].abs()
        mean_off_diag.append(off.mean().item())
        max_off_diag.append(off.max().item())

        W = s["weight"].float()
        try:
            S = torch.linalg.svdvals(W)
            isotropies.append((S.min() / S.max()).item() if S.max() > 0 else 0.0)
        except Exception:
            isotropies.append(0.0)

        norms = s["norms"]
        if norms.numel() == len(PARTICLE_LABELS):
            norms = norms[1:]  # drop padding PID
        mean_norms.append(norms.mean().item())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) — Mean off-diagonal cosine similarity
    ax = axes[0, 0]
    ax.plot(epochs, mean_off_diag, "b-", label="mean |cos|", linewidth=1.5)
    ax.plot(epochs, max_off_diag, "r--", alpha=0.6, label="max |cos|", linewidth=1)
    if transition_epoch is not None:
        ax.axvline(transition_epoch, color="green", linestyle=":", linewidth=2, label=f"transition @ {transition_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|Cosine Similarity|")
    ax.set_title("Orthogonality (lower = more orthogonal)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) — Isotropy
    ax = axes[0, 1]
    ax.plot(epochs, isotropies, "g-", linewidth=1.5)
    if transition_epoch is not None:
        ax.axvline(transition_epoch, color="green", linestyle=":", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("σ_min / σ_max")
    ax.set_title("Isotropy (higher = more uniform)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # (1,0) — Mean norms
    ax = axes[1, 0]
    ax.plot(epochs, mean_norms, "m-", linewidth=1.5)
    if transition_epoch is not None:
        ax.axvline(transition_epoch, color="green", linestyle=":", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean L2 Norm")
    ax.set_title("Embedding Vector Norms")
    ax.grid(True, alpha=0.3)

    # (1,1) — Per-type norms at final epoch
    ax = axes[1, 1]
    norms = snapshots[-1]["norms"]
    if norms.numel() == len(PARTICLE_LABELS):
        idx_start = 1
        norms = norms[1:]
    else:
        idx_start = 0
    final_norms = norms.numpy()
    n = len(final_norms)
    ax.bar(range(n), final_norms, color="steelblue", edgecolor="black")
    ax.set_xlabel("Particle")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Per-Particle Norms (final epoch)")
    ax.set_xticks(range(n))
    ax.set_xticklabels([_pid_label(i + idx_start) for i in range(n)])
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"PID Embedding Evolution — {run_label}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Plotting: PCA / tSNE of PID embeddings
# ---------------------------------------------------------------------------


def _plot_pid_pca_tsne(
    snapshots: list[dict],
    run_label: str,
    epochs_to_show: list[int] | None = None,
    fig_cfg: dict | None = None,
) -> Figure:
    """PCA and tSNE scatter of PID embedding vectors at selected epochs."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if not snapshots:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No PID snapshots", ha="center", va="center")
        return fig

    # Select which epochs to visualize
    if epochs_to_show is None:
        # Pick first, quarter, middle, three-quarter, last
        all_epochs = [s["epoch"] for s in snapshots]
        epochs_to_show = (
            all_epochs
            if len(all_epochs) <= 5
            else [
                all_epochs[0],
                all_epochs[len(all_epochs) // 4],
                all_epochs[len(all_epochs) // 2],
                all_epochs[3 * len(all_epochs) // 4],
                all_epochs[-1],
            ]
        )

    # Filter snapshots
    snap_map = {s["epoch"]: s for s in snapshots}
    selected = [(ep, snap_map[ep]) for ep in epochs_to_show if ep in snap_map]

    if not selected:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No matching snapshots", ha="center", va="center")
        return fig

    n_cols = len(selected)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    cmap = plt.cm.tab10
    num_types_full = selected[0][1]["num_types"]
    idx_start = 1 if num_types_full == len(PARTICLE_LABELS) else 0
    pid_indices = list(range(idx_start, num_types_full))
    num_types = len(pid_indices)

    for col_idx, (ep, snap) in enumerate(selected):
        W_full = snap["weight"].numpy()  # [num_types_full, id_embed_dim]
        W = W_full[pid_indices, :]  # [num_types, id_embed_dim]

        # PCA
        ax = axes[0, col_idx]
        if W.shape[1] >= 2:
            pca = PCA(n_components=2)
            W_pca = pca.fit_transform(W)
            for k, pid in enumerate(pid_indices):
                ax.scatter(W_pca[k, 0], W_pca[k, 1], color=cmap(pid), s=100, edgecolors="black", zorder=3)
                ax.annotate(_pid_label(pid), (W_pca[k, 0], W_pca[k, 1]), fontsize=8, ha="center", va="bottom")
            ev = pca.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
            ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
        else:
            ax.bar(range(num_types), W[:, 0], color=[cmap(i) for i in pid_indices])
            ax.set_xlabel("Particle")
            ax.set_ylabel("Value")
        ax.set_title(f"PCA — Epoch {ep}")
        ax.grid(True, alpha=0.3)

        # tSNE
        ax = axes[1, col_idx]
        if W.shape[1] >= 2 and num_types > 2:
            perplexity = min(5, max(1, num_types - 1))
            # Use only arguments supported across sklearn versions
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            W_tsne = tsne.fit_transform(W)
            for k, pid in enumerate(pid_indices):
                ax.scatter(W_tsne[k, 0], W_tsne[k, 1], color=cmap(pid), s=100, edgecolors="black", zorder=3)
                ax.annotate(_pid_label(pid), (W_tsne[k, 0], W_tsne[k, 1]), fontsize=8, ha="center", va="bottom")
            ax.set_xlabel("tSNE-1")
            ax.set_ylabel("tSNE-2")
        else:
            ax.text(0.5, 0.5, "tSNE N/A\n(too few dims/types)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"tSNE — Epoch {ep}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"PID Embedding Space — {run_label}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Plotting: performance comparison
# ---------------------------------------------------------------------------


def _plot_performance_comparison(
    runs_df: pd.DataFrame,
    per_epoch: dict[str, pd.DataFrame],
    figs_dir: Path,
    fig_cfg: dict,
    metric: str = "val_loss",
) -> None:
    """Bar chart of final metric by run_label, plus learning curves colored by pid_mode."""
    df = runs_df.copy()

    # Final metric per run
    final_vals = {}
    for _, row in df.iterrows():
        run_dir = row["run_dir"]
        label = row["run_label"]
        # per_epoch is keyed by run_dir string, see load_runs()
        ep_df = per_epoch.get(str(run_dir), pd.DataFrame())
        if ep_df.empty:
            continue
        if metric in ep_df.columns:
            final_vals[label] = ep_df[metric].iloc[-1]

    if not final_vals:
        logger.warning("No data for performance comparison.")
        return

    # Bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(final_vals) * 0.8), 5))
    labels = sorted(final_vals.keys())
    values = [final_vals[lbl] for lbl in labels]
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Final {metric.replace('_', ' ').title()} by PID Configuration")
    ax.grid(True, alpha=0.3, axis="y")

    # Value annotations
    for bar, val in zip(bars, values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(figs_dir / f"figure-perf_{metric}.{fig_cfg.get('fig_format', 'png')}", dpi=fig_cfg.get("dpi", 150))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main report entry point
# ---------------------------------------------------------------------------


def run_report(cfg: DictConfig) -> None:
    """Analyze PID embeddings: performance, geometry, and evolution.

    Generates:
    - Performance bar charts (val_loss, val_acc, val_f1, val_auroc)
    - Training curves for all runs
    - Per-run: cosine similarity heatmap, evolution plots, PCA/tSNE scatter
    - Cross-run comparison: orthogonality at final epoch
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Setup report environment
    (
        training_dir,
        inference_dir,
        training_figs_dir,
        inference_figs_dir,
        runs_df,
        per_epoch,
        order,
        report_dir,
        report_id,
        report_name,
    ) = setup_report_environment(cfg)

    logger.info("Loaded %d runs for PID analysis", len(runs_df))

    # Enrich with PID metadata
    runs_df = _extract_pid_metadata(runs_df)
    runs_df.to_csv(training_dir / "summary.csv", index=False)

    fig_cfg = get_fig_config(cfg)
    wanted = set(cfg.outputs.which_figures or [])
    fmt = fig_cfg.get("fig_format", "png")
    dpi = fig_cfg.get("dpi", 150)

    # Per-run PID analysis directory
    pid_figs_dir = training_figs_dir / "pid_analysis"
    pid_figs_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # Training curves (all runs)
    # ======================================================================

    if "all_val_curves" in wanted:
        plot_all_val_curves(runs_df, per_epoch, training_figs_dir, fig_cfg, fname="figure-all_val_curves")

    if "all_train_curves" in wanted:
        plot_all_train_curves(runs_df, per_epoch, training_figs_dir, fig_cfg, fname="figure-all_train_curves")

    # ======================================================================
    # Performance comparison
    # ======================================================================

    for metric in ["val_loss", "val_acc", "val_f1", "val_auroc"]:
        if f"perf_{metric}" in wanted or "performance" in wanted:
            _plot_performance_comparison(runs_df, per_epoch, training_figs_dir, fig_cfg, metric=metric)

    # ======================================================================
    # Per-run PID embedding analysis
    # ======================================================================

    for _, row in runs_df.iterrows():
        run_dir = row["run_dir"]
        run_label = row["run_label"]
        transition_epoch = row.get("transition_epoch")

        snapshots = _load_pid_snapshots(run_dir)
        if not snapshots:
            logger.info("No PID snapshots for %s — skipping geometry analysis", run_label)
            continue

        safe_label = run_label.replace("/", "_").replace(" ", "_")
        run_pid_dir = pid_figs_dir / safe_label
        run_pid_dir.mkdir(parents=True, exist_ok=True)

        # Cosine similarity heatmap at final epoch
        if "cosine_heatmap" in wanted or "pid_geometry" in wanted:
            final_snap = snapshots[-1]
            fig = _plot_cosine_heatmap(
                final_snap["cosine_matrix"],
                title=f"PID Cosine Similarity — {run_label} (epoch {final_snap['epoch']})",
                fig_cfg=fig_cfg,
            )
            fig.savefig(run_pid_dir / f"cosine_heatmap.{fmt}", dpi=dpi)
            plt.close(fig)

        # Evolution plot
        if "pid_evolution" in wanted or "pid_geometry" in wanted:
            fig = _plot_pid_evolution(
                snapshots,
                run_label,
                transition_epoch=int(transition_epoch) if transition_epoch is not None else None,
                fig_cfg=fig_cfg,
            )
            fig.savefig(run_pid_dir / f"evolution.{fmt}", dpi=dpi)
            plt.close(fig)

        # PCA / tSNE
        if "pid_pca_tsne" in wanted or "pid_geometry" in wanted:
            fig = _plot_pid_pca_tsne(snapshots, run_label, fig_cfg=fig_cfg)
            fig.savefig(run_pid_dir / f"pca_tsne.{fmt}", dpi=dpi)
            plt.close(fig)

    # ======================================================================
    # Cross-run orthogonality comparison at final epoch
    # ======================================================================

    if "orthogonality_comparison" in wanted or "pid_geometry" in wanted:
        orth_data = []  # (run_label, mean_off_diag, max_off_diag)
        for _, row in runs_df.iterrows():
            snapshots = _load_pid_snapshots(row["run_dir"])
            if not snapshots:
                continue
            C = snapshots[-1]["cosine_matrix"]
            N = C.size(0)
            mask = ~torch.eye(N, dtype=torch.bool)
            off = C[mask].abs()
            orth_data.append((row["run_label"], off.mean().item(), off.max().item()))

        if orth_data:
            labels, means, maxes = zip(*sorted(orth_data, key=lambda x: x[1]), strict=False)
            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
            x = np.arange(len(labels))
            ax.bar(x - 0.15, means, 0.3, label="Mean |cos|", color="steelblue", edgecolor="black")
            ax.bar(x + 0.15, maxes, 0.3, label="Max |cos|", color="salmon", edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Off-diagonal |Cosine Similarity|")
            ax.set_title("PID Orthogonality Comparison (final epoch)")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(training_figs_dir / f"figure-orthogonality_comparison.{fmt}", dpi=dpi)
            plt.close(fig)

    # ======================================================================
    # Finalize
    # ======================================================================

    finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)
    logger.info("PID deepdive report complete: %s", report_dir)
