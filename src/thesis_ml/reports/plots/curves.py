from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from thesis_ml.monitoring.io_utils import save_figure


def plot_loss_vs_time(
    df: pd.DataFrame,
    per_epoch: dict[str, pd.DataFrame],
    order: list[str],
    figs_dir: Path,
    fig_cfg: dict,
    metric: str = "val_loss",
    fname: str = "figure-loss_vs_time",
) -> None:
    """Generic: plot metric vs cumulative time"""
    fig, ax = plt.subplots()
    for run_dir in order:
        if run_dir not in per_epoch:
            continue
        cur = per_epoch[run_dir].copy()
        cur = cur[cur["split"] == "val"] if "split" in cur.columns else cur
        cur["cum_time_s"] = cur["epoch_time_s"].astype(float).cumsum()
        ax.plot(cur["cum_time_s"], cur[metric].astype(float), label=Path(run_dir).name)
    ax.set_xlabel("wall-clock seconds (cumulative)", fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_title(f"{metric} vs time", fontsize=16)
    ax.legend(fontsize=12)
    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)


def _color_for_latent(latent_space: str | None) -> str:
    mapping = {"none": "tab:blue", "linear": "tab:orange", "vq": "tab:green"}
    return mapping.get(str(latent_space).lower() if latent_space else None, "tab:gray")


def _linestyle_for_beta(beta: float | None) -> str:
    if beta is None:
        return "-"
    try:
        b = float(beta)
    except Exception:
        return "-"
    if abs(b) < 1e-12 or b == 0.0:
        return ":"  # dotted for 0
    if abs(b - 1.0) < 1e-9:
        return "-"  # solid for 1
    return "--"  # dashed for others (e.g., 10)


def plot_all_val_curves(
    runs_df: pd.DataFrame,
    per_epoch: dict[str, pd.DataFrame],
    figs_dir: Path,
    fig_cfg: dict,
    fname: str = "figure-all_val_curves",
) -> None:
    """Plot all runs' validation loss vs epoch, color by latent space, style by globals_beta."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Track best values for annotation
    best_values = []

    for _, row in runs_df.iterrows():
        rd = str(row.get("run_dir"))
        ls = row.get("latent_space")
        beta = row.get("globals_beta")
        if rd not in per_epoch:
            continue
        hist = per_epoch[rd]
        cur = hist.copy()
        cur = cur[cur["split"] == "val"] if "split" in cur.columns else cur
        if "val_loss" not in cur.columns:
            continue
        epochs = cur["epoch"].astype(int).values
        vals = cur["val_loss"].astype(float).values

        # Find best (lowest) value and its epoch
        best_idx = np.argmin(vals)
        best_val = vals[best_idx]
        best_epoch = epochs[best_idx]
        best_values.append((best_val, best_epoch, rd))

        ax.plot(
            epochs,
            vals,
            color=_color_for_latent(ls),
            linestyle=_linestyle_for_beta(beta),
            linewidth=1.5,
            alpha=0.9,
        )

    # Add annotation showing overall best value
    if best_values:
        overall_best_val, overall_best_epoch, overall_best_rd = min(best_values, key=lambda x: x[0])
        ax.annotate(
            f"Best: {overall_best_val:.4f} @ epoch {overall_best_epoch}",
            xy=(overall_best_epoch, overall_best_val),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            fontsize=11,
        )

    # Legends: colors for latent, linestyles for betas
    color_handles = [Line2D([0], [0], color=_color_for_latent(k), lw=2, label=str(k)) for k in sorted(set([str(v).lower() for v in runs_df.get("latent_space", []) if pd.notna(v)]))]
    beta_values = sorted(set([float(v) for v in runs_df.get("globals_beta", []) if pd.notna(v)]))
    ls_handles = [Line2D([0], [0], color="black", lw=2, linestyle=_linestyle_for_beta(b), label=f"β={int(b) if b.is_integer() else b}") for b in beta_values]
    if color_handles:
        leg1 = ax.legend(handles=color_handles, title="latent_space", loc="upper right", fontsize=11)
        ax.add_artist(leg1)
    if ls_handles:
        ax.legend(handles=ls_handles, title="globals_beta", loc="upper left", fontsize=11)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Validation Loss", fontsize=14)
    ax.set_title("Validation loss per run", fontsize=16)
    ax.grid(True, alpha=0.2)
    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)


def plot_all_train_curves(
    runs_df: pd.DataFrame,
    figs_dir: Path,
    fig_cfg: dict,
    fname: str = "figure-all_train_curves",
) -> None:
    """Plot all runs' training loss vs epoch, color by latent space, style by globals_beta."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in runs_df.iterrows():
        rd = str(row.get("run_dir"))
        ls = row.get("latent_space")
        beta = row.get("globals_beta")
        scalars_path = Path(rd) / "facts" / "scalars.csv"
        if not scalars_path.exists():
            continue
        try:
            df = pd.read_csv(scalars_path)
        except Exception:
            continue
        cur = df[df.get("split", "train") == "train"].copy()
        if cur.empty or "train_loss" not in cur.columns:
            continue
        epochs = cur["epoch"].astype(int).values
        vals = cur["train_loss"].astype(float).values
        ax.plot(
            epochs,
            vals,
            color=_color_for_latent(ls),
            linestyle=_linestyle_for_beta(beta),
            linewidth=1.5,
            alpha=0.9,
        )

    color_handles = [Line2D([0], [0], color=_color_for_latent(k), lw=2, label=str(k)) for k in sorted(set([str(v).lower() for v in runs_df.get("latent_space", []) if pd.notna(v)]))]
    beta_values = sorted(set([float(v) for v in runs_df.get("globals_beta", []) if pd.notna(v)]))
    ls_handles = [Line2D([0], [0], color="black", lw=2, linestyle=_linestyle_for_beta(b), label=f"β={int(b) if b.is_integer() else b}") for b in beta_values]
    if color_handles:
        leg1 = ax.legend(handles=color_handles, title="latent_space", loc="upper right")
        ax.add_artist(leg1)
    if ls_handles:
        ax.legend(handles=ls_handles, title="globals_beta", loc="upper left")

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Training Loss", fontsize=14)
    ax.set_title("Training loss per run", fontsize=16)
    ax.grid(True, alpha=0.2)
    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
