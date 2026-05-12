"""Positional encoding specific visualization functions.

This module provides PE-specific visualization functions for deep dive analysis.
It reuses generic utilities from classification.py where possible.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from thesis_ml.reports.plots.style import apply_thesis_style, figure_size

apply_thesis_style()

# Note: This module focuses on PE-specific analysis.
# For generic classification plots, use functions from classification.py directly.


def plot_pe_matrix_heatmap(
    pe_matrix: np.ndarray | torch.Tensor,
    pe_type: str,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot positional encoding matrix as heatmap.

    Visualizes how each position is encoded across all model dimensions.
    Works for sinusoidal and learned positional encodings.

    Parameters
    ----------
    pe_matrix : np.ndarray | torch.Tensor
        Positional encoding matrix of shape [num_tokens, model_dim]
    pe_type : str
        Type of positional encoding (e.g., 'sinusoidal', 'learned')
    figsize : tuple[float, float] | None
        Override figure size. Defaults to figure_size("full", aspect=1.0).

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    if figsize is None:
        figsize = figure_size("full", aspect=1.0)

    if isinstance(pe_matrix, torch.Tensor):
        pe_matrix = pe_matrix.detach().cpu().numpy()

    num_tokens = pe_matrix.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        pe_matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.abs(pe_matrix).max(),
        vmax=np.abs(pe_matrix).max(),
        interpolation="nearest",
    )
    ax.set_xlabel("Model Dimension")
    ax.set_ylabel("Token Position")

    y_ticks = np.arange(num_tokens)
    y_labels = ["CLS"] + [str(i) for i in range(1, num_tokens)] if num_tokens > 0 else []
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    for y in np.arange(num_tokens + 1) - 0.5:
        ax.axhline(y, color="k", linewidth=0.5, alpha=0.3)

    plt.colorbar(im, ax=ax, label="Encoding Value")
    plt.tight_layout()

    return fig, ax


def plot_pe_per_token_patterns(
    pe_matrix: np.ndarray | torch.Tensor,
    pe_type: str,
    num_tokens_to_plot: int = 5,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot per-token encoding patterns across dimensions.

    Shows how each token's encoding varies across model dimensions.

    Parameters
    ----------
    pe_matrix : np.ndarray | torch.Tensor
        Positional encoding matrix of shape [num_tokens, model_dim]
    pe_type : str
        Type of positional encoding (e.g., 'sinusoidal', 'learned')
    num_tokens_to_plot : int
        Number of tokens to plot (default: 5)
    figsize : tuple[float, float] | None
        Override figure size. Defaults to figure_size("full").

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    if figsize is None:
        figsize = figure_size("full")

    if isinstance(pe_matrix, torch.Tensor):
        pe_matrix = pe_matrix.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    tokens_to_plot = min(num_tokens_to_plot, pe_matrix.shape[0])
    for token_idx in range(tokens_to_plot):
        ax.plot(pe_matrix[token_idx, :], label=f"Token {token_idx}", alpha=0.7, linewidth=2)

    ax.set_xlabel("Model Dimension")
    ax.set_ylabel("Encoding Value")
    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig, ax


def plot_rotary_patterns(
    cos_cache: np.ndarray | torch.Tensor,
    sin_cache: np.ndarray | torch.Tensor,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot Rotary (RoPE) sin/cos patterns.

    Visualizes the sin and cosine patterns used in Rotary Position Embeddings.

    Parameters
    ----------
    cos_cache : np.ndarray | torch.Tensor
        Cosine cache of shape [seq_len, head_dim]
    sin_cache : np.ndarray | torch.Tensor
        Sine cache of shape [seq_len, head_dim]
    figsize : tuple[float, float] | None
        Override figure size. Defaults to figure_size("full") for 1×2 panel.

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Figure and axes objects
    """
    if figsize is None:
        figsize = figure_size("full")

    if isinstance(cos_cache, torch.Tensor):
        cos_cache = cos_cache.squeeze().detach().cpu().numpy()
    if isinstance(sin_cache, torch.Tensor):
        sin_cache = sin_cache.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    num_positions = cos_cache.shape[0]

    im0 = axes[0].imshow(
        cos_cache,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axes[0].set_xlabel("Head Dimension")
    axes[0].set_ylabel("Sequence Position")

    y_ticks = np.arange(num_positions)
    y_labels = ["CLS"] + [str(i) for i in range(1, num_positions)] if num_positions > 0 else []
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_labels)
    for y in np.arange(num_positions + 1) - 0.5:
        axes[0].axhline(y, color="k", linewidth=0.5, alpha=0.3)
    plt.colorbar(im0, ax=axes[0], label="Cosine Value")

    im1 = axes[1].imshow(
        sin_cache,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_xlabel("Head Dimension")
    axes[1].set_ylabel("Sequence Position")
    axes[1].set_yticks(y_ticks)
    axes[1].set_yticklabels(y_labels)
    for y in np.arange(num_positions + 1) - 0.5:
        axes[1].axhline(y, color="k", linewidth=0.5, alpha=0.3)
    plt.colorbar(im1, ax=axes[1], label="Sine Value")

    plt.tight_layout()
    return fig, axes


def plot_layer_l2_norms(
    l2_norms_dict: dict[str, dict[str, np.ndarray]],
    layers_to_plot: list[str] | None = None,
    figsize: tuple[float, float] = (12.6, 12.6),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot L2 norms of token representations across layers.

    Shows how representation magnitudes evolve through the transformer layers
    for different positional encoding types.

    Parameters
    ----------
    l2_norms_dict : dict[str, dict[str, np.ndarray]]
        Dictionary mapping PE type to layer names to L2 norm arrays
        Format: {pe_type: {layer_name: l2_norms_array}}
        where l2_norms_array is shape [num_tokens]
    layers_to_plot : list[str] | None
        List of layer names to plot. If None, auto-selects key layers.
    figsize : tuple[float, float]
        Figure size for 2×2 panel grid (default: (12.6, 12.6)).

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and axes array
    """
    if len(l2_norms_dict) == 0:
        raise ValueError("l2_norms_dict is empty")

    if layers_to_plot is None:
        sample_pe = next(iter(l2_norms_dict.keys()))
        num_blocks = len([k for k in l2_norms_dict[sample_pe] if "block" in k])
        layers_to_plot = ["after_embedding"]
        if "after_pos_enc" in l2_norms_dict[sample_pe]:
            layers_to_plot.append("after_pos_enc")
        layers_to_plot.extend(["after_block_0", f"after_block_{num_blocks//2}", f"after_block_{num_blocks-1}"])

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    pe_types_ordered = ["none", "sinusoidal", "learned", "rotary"]
    pe_types_available = [pt for pt in pe_types_ordered if pt in l2_norms_dict]

    for idx, pe_type in enumerate(pe_types_available):
        if idx >= len(axes):
            break

        ax = axes[idx]
        num_tokens = None

        for layer_name in layers_to_plot:
            if layer_name in l2_norms_dict[pe_type]:
                norms = l2_norms_dict[pe_type][layer_name]
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
    return fig, axes
