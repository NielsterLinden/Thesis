"""Representation analysis plotting functions.

This module provides visualization functions for analyzing transformer representations
layer-by-layer, including PCA/t-SNE projections, token similarity matrices, and L2 norms.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from thesis_ml.monitoring.io_utils import save_figure
from thesis_ml.reports.plots.style import apply_thesis_style, figure_size

logger = logging.getLogger(__name__)

apply_thesis_style()


def compute_similarity_matrix(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """Compute cosine similarity between all token pairs.

    Parameters
    ----------
    tensor : torch.Tensor | np.ndarray
        Shape [B, T, D] - batch of token representations

    Returns
    -------
    np.ndarray
        Similarity matrix [T, T] averaged over batch
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    token_vecs = tensor.mean(axis=0)

    norms = np.linalg.norm(token_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    token_vecs_norm = token_vecs / norms

    sim_matrix = token_vecs_norm @ token_vecs_norm.T

    return sim_matrix


def plot_token_similarity_matrix(
    representations: dict[str, torch.Tensor | np.ndarray],
    pe_type: str,
    layer_name: str,
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot token similarity matrix for a specific layer and PE type.

    Parameters
    ----------
    representations : dict[str, torch.Tensor | np.ndarray]
        Dict mapping layer names to representation tensors [B, T, D]
    pe_type : str
        Positional encoding type (e.g., 'sinusoidal', 'learned', 'rotary', 'none')
    layer_name : str
        Layer name to plot (e.g., 'after_pos_enc', 'after_block_5')
    figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    fname : str | None
        Base filename (auto-generated if None)
    figsize : tuple[float, float] | None
        Override figure size. Defaults to figure_size("full", aspect=1.0).
    """
    if figsize is None:
        figsize = figure_size("full", aspect=1.0)

    if layer_name not in representations:
        logger.warning(f"Layer '{layer_name}' not found in representations, skipping similarity plot")
        return

    tensor = representations[layer_name]
    sim_matrix = compute_similarity_matrix(tensor)
    num_tokens = sim_matrix.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Token Index")

    positions = np.arange(num_tokens)
    labels = ["CLS"] + [str(i) for i in range(1, num_tokens)]
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(labels[:num_tokens], rotation=90)
    ax.set_yticklabels(labels[:num_tokens])

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()

    save_figure(fig, figs_dir, fname or f"figure-token_similarity_{pe_type}_{layer_name}", fig_cfg)
    plt.close(fig)


def plot_all_similarity_matrices(
    all_representations: dict[str, dict[str, torch.Tensor | np.ndarray]],
    layer_name: str,
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    fname: str = "figure-token_similarity_comparison",
    figsize: tuple[float, float] = (12.6, 12.6),
) -> None:
    """Plot token similarity matrices for all PE types side by side.

    Parameters
    ----------
    all_representations : dict[str, dict[str, torch.Tensor | np.ndarray]]
        Dict mapping PE type to layer dict to representation tensors
        Format: {pe_type: {layer_name: tensor [B, T, D]}}
    layer_name : str
        Layer name to compare across PE types
    figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    fname : str
        Base filename for saved figure
    figsize : tuple[float, float]
        Figure size for 2×2 panel grid (default: (12.6, 12.6)).
    """
    pe_types_ordered = ["none", "sinusoidal", "learned", "rotary"]
    pe_types_available = [pt for pt in pe_types_ordered if pt in all_representations]

    if not pe_types_available:
        logger.warning("No PE types found in representations, skipping similarity comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, pe_type in enumerate(pe_types_available):
        if idx >= len(axes):
            break

        ax = axes[idx]
        representations = all_representations[pe_type]

        if layer_name in representations:
            tensor = representations[layer_name]
            actual_layer = layer_name
        elif "after_embedding" in representations:
            tensor = representations["after_embedding"]
            actual_layer = "after_embedding"
        else:
            ax.text(0.5, 0.5, f"{pe_type}\nNo data", ha="center", va="center", transform=ax.transAxes)
            continue

        sim_matrix = compute_similarity_matrix(tensor)
        num_tokens = sim_matrix.shape[0]

        im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xlabel("Token Index")
        ax.set_ylabel("Token Index")

        positions = np.arange(num_tokens)
        labels = ["CLS"] + [str(i) for i in range(1, num_tokens)]
        ax.set_xticks(positions)
        ax.set_yticks(positions)
        ax.set_xticklabels(labels[:num_tokens], rotation=90)
        ax.set_yticklabels(labels[:num_tokens])

        plt.colorbar(im, ax=ax, label="Cosine Similarity")

    for idx in range(len(pe_types_available), 4):
        axes[idx].axis("off")

    plt.tight_layout()
    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)


def plot_layer_evolution_pca(
    representations: dict[str, torch.Tensor | np.ndarray],
    labels: torch.Tensor | np.ndarray,
    pe_type: str,
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    layers_to_plot: list[str] | None = None,
    fname: str | None = None,
    figsize: tuple[float, float] | None = None,
    pooling: str = "mean",
) -> None:
    """Plot PCA projections showing how class separation evolves through layers.

    Parameters
    ----------
    representations : dict[str, torch.Tensor | np.ndarray]
        Dict mapping layer names to representation tensors [B, T, D]
    labels : torch.Tensor | np.ndarray
        Class labels [B]
    pe_type : str
        Positional encoding type (used in filename)
    figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    layers_to_plot : list[str] | None
        List of layer names to plot. If None, auto-selects key layers.
    fname : str | None
        Base filename (auto-generated if None)
    figsize : tuple[float, float] | None
        Override figure size. Defaults to n_cols × half-width panels.
    pooling : str
        How to pool tokens: 'mean' or 'cls' (default: 'mean')
    """
    from sklearn.decomposition import PCA

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if layers_to_plot is None:
        num_blocks = len([k for k in representations if "block" in k])
        layers_to_plot = ["after_embedding"]
        if "after_pos_enc" in representations:
            layers_to_plot.append("after_pos_enc")
        if num_blocks > 0:
            layers_to_plot.extend(["after_block_0", f"after_block_{num_blocks//2}", f"after_block_{num_blocks-1}"])

    n_plots = min(6, len(layers_to_plot))
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    if figsize is None:
        half_w, half_h = figure_size("half")
        figsize = (n_cols * half_w, n_rows * half_h)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for idx, layer_name in enumerate(layers_to_plot[:n_plots]):
        if idx >= len(axes):
            break

        ax = axes[idx]

        if layer_name not in representations:
            ax.axis("off")
            continue

        tensor = representations[layer_name]
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()

        if pooling == "cls":
            X = tensor[:, 0, :]
        else:
            X = tensor.mean(axis=1)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        unique_labels = np.unique(labels)
        class_names = {0: "Bkg", 1: "4t"} if len(unique_labels) == 2 else {i: f"Class {i}" for i in unique_labels}

        for class_idx in unique_labels:
            mask = labels == class_idx
            class_name = class_names.get(class_idx, f"Class {class_idx}")
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=class_name, alpha=0.6, s=15, rasterized=True)

        var_explained = pca.explained_variance_ratio_[:2].sum()
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.annotate(f"Var: {var_explained:.1%}", xy=(0.05, 0.95), xycoords="axes fraction", va="top")
        ax.legend()

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    save_figure(fig, figs_dir, fname or f"figure-layer_evolution_pca_{pe_type}", fig_cfg)
    plt.close(fig)


def plot_layer_evolution_tsne(
    representations: dict[str, torch.Tensor | np.ndarray],
    labels: torch.Tensor | np.ndarray,
    pe_type: str,
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    layers_to_plot: list[str] | None = None,
    fname: str | None = None,
    figsize: tuple[float, float] | None = None,
    pooling: str = "mean",
    perplexity: int = 30,
    max_iter: int = 1000,
) -> None:
    """Plot t-SNE projections showing how class separation evolves through layers.

    Parameters
    ----------
    representations : dict[str, torch.Tensor | np.ndarray]
        Dict mapping layer names to representation tensors [B, T, D]
    labels : torch.Tensor | np.ndarray
        Class labels [B]
    pe_type : str
        Positional encoding type (used in filename)
    figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    layers_to_plot : list[str] | None
        List of layer names to plot. If None, auto-selects key layers.
    fname : str | None
        Base filename (auto-generated if None)
    figsize : tuple[float, float] | None
        Override figure size. Defaults to n_cols × half-width panels.
    pooling : str
        How to pool tokens: 'mean' or 'cls' (default: 'mean')
    perplexity : int
        t-SNE perplexity (default: 30)
    max_iter : int
        t-SNE max iterations (default: 1000)
    """
    from sklearn.manifold import TSNE

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if layers_to_plot is None:
        num_blocks = len([k for k in representations if "block" in k])
        layers_to_plot = ["after_embedding"]
        if "after_pos_enc" in representations:
            layers_to_plot.append("after_pos_enc")
        if num_blocks > 0:
            layers_to_plot.extend(["after_block_0", f"after_block_{num_blocks//2}", f"after_block_{num_blocks-1}"])

    n_plots = min(6, len(layers_to_plot))
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    if figsize is None:
        half_w, half_h = figure_size("half")
        figsize = (n_cols * half_w, n_rows * half_h)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for idx, layer_name in enumerate(layers_to_plot[:n_plots]):
        if idx >= len(axes):
            break

        ax = axes[idx]

        if layer_name not in representations:
            ax.axis("off")
            continue

        tensor = representations[layer_name]
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()

        if pooling == "cls":
            X = tensor[:, 0, :]
        else:
            X = tensor.mean(axis=1)

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=max_iter)
        X_tsne = tsne.fit_transform(X)

        unique_labels = np.unique(labels)
        class_names = {0: "Bkg", 1: "4t"} if len(unique_labels) == 2 else {i: f"Class {i}" for i in unique_labels}

        for class_idx in unique_labels:
            mask = labels == class_idx
            class_name = class_names.get(class_idx, f"Class {class_idx}")
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=class_name, alpha=0.6, s=15, rasterized=True)

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend()

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    save_figure(fig, figs_dir, fname or f"figure-layer_evolution_tsne_{pe_type}", fig_cfg)
    plt.close(fig)


def plot_l2_norms_comparison(
    all_representations: dict[str, dict[str, torch.Tensor | np.ndarray]],
    figs_dir: Path,
    fig_cfg: dict[str, Any],
    layers_to_plot: list[str] | None = None,
    fname: str = "figure-l2_norms_comparison",
    figsize: tuple[float, float] = (12.6, 12.6),
) -> None:
    """Plot L2 norms of token representations across layers for all PE types.

    Parameters
    ----------
    all_representations : dict[str, dict[str, torch.Tensor | np.ndarray]]
        Dict mapping PE type to layer dict to representation tensors
        Format: {pe_type: {layer_name: tensor [B, T, D]}}
    figs_dir : Path
        Directory to save figures
    fig_cfg : dict[str, Any]
        Figure configuration (fig_format, dpi)
    layers_to_plot : list[str] | None
        List of layer names to plot. If None, auto-selects key layers.
    fname : str
        Base filename for saved figure
    figsize : tuple[float, float]
        Figure size for 2×2 panel grid (default: (12.6, 12.6)).
    """
    if not all_representations:
        logger.warning("No representations provided, skipping L2 norms plot")
        return

    l2_norms: dict[str, dict[str, np.ndarray]] = {}

    for pe_type, layers_dict in all_representations.items():
        l2_norms[pe_type] = {}
        for layer_name, tensor in layers_dict.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu()
                l2 = torch.norm(tensor, p=2, dim=-1)
                l2_norms[pe_type][layer_name] = l2.mean(dim=0).numpy()
            else:
                l2 = np.linalg.norm(tensor, ord=2, axis=-1)
                l2_norms[pe_type][layer_name] = l2.mean(axis=0)

    if layers_to_plot is None:
        sample_pe = next(iter(l2_norms.keys()))
        num_blocks = len([k for k in l2_norms[sample_pe] if "block" in k])
        layers_to_plot = ["after_embedding"]
        if "after_pos_enc" in l2_norms[sample_pe]:
            layers_to_plot.append("after_pos_enc")
        if num_blocks > 0:
            layers_to_plot.extend(["after_block_0", f"after_block_{num_blocks//2}", f"after_block_{num_blocks-1}"])

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    pe_types_ordered = ["none", "sinusoidal", "learned", "rotary"]
    pe_types_available = [pt for pt in pe_types_ordered if pt in l2_norms]

    for idx, pe_type in enumerate(pe_types_available):
        if idx >= len(axes):
            break

        ax = axes[idx]
        num_tokens = None

        for layer_name in layers_to_plot:
            if layer_name in l2_norms[pe_type]:
                norms = l2_norms[pe_type][layer_name]
                if num_tokens is None:
                    num_tokens = len(norms)
                ax.plot(norms, label=layer_name, marker="o", alpha=0.7, linewidth=2)

        if num_tokens is not None:
            positions = np.arange(num_tokens)
            ax.set_xticks(positions)
            ax.set_xticklabels(["CLS"] + [str(i) for i in range(1, num_tokens)])

        ax.set_xlabel("Token Position")
        ax.set_ylabel("L2 Norm")
        ax.legend(loc="best")

    for idx in range(len(pe_types_available), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    save_figure(fig, figs_dir, fname, fig_cfg)
    plt.close(fig)
