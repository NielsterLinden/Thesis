"""Positional encoding specific visualization functions.

This module provides PE-specific visualization functions for deep dive analysis.
It reuses generic utilities from classification.py where possible.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

# Note: This module focuses on PE-specific analysis.
# For generic classification plots, use functions from classification.py directly.


def plot_pe_matrix_heatmap(
    pe_matrix: np.ndarray | torch.Tensor,
    pe_type: str,
    figsize: tuple[float, float] = (8, 6),
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
    figsize : tuple[float, float]
        Figure size (default: (8, 6))

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    if isinstance(pe_matrix, torch.Tensor):
        pe_matrix = pe_matrix.detach().cpu().numpy()

    num_tokens = pe_matrix.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap (no interpolation so each token is a flat band)
    im = ax.imshow(
        pe_matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.abs(pe_matrix).max(),
        vmax=np.abs(pe_matrix).max(),
        interpolation="nearest",
    )
    ax.set_xlabel("Model Dimension", fontsize=14)
    ax.set_ylabel("Token Position", fontsize=14)
    ax.set_title(
        f"{pe_type.capitalize()} Positional Encoding\n({num_tokens} tokens × {pe_matrix.shape[1]} dims)",
        fontsize=16,
        fontweight="bold",
    )

    # Show one tick per token (CLS + physics tokens)
    y_ticks = np.arange(num_tokens)
    y_labels = ["CLS"] + [str(i) for i in range(1, num_tokens)] if num_tokens > 0 else []
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Draw horizontal grid lines between tokens
    for y in np.arange(num_tokens + 1) - 0.5:
        ax.axhline(y, color="k", linewidth=0.5, alpha=0.3)

    plt.colorbar(im, ax=ax, label="Encoding Value", fontsize=12)
    plt.tight_layout()

    return fig, ax


def plot_pe_per_token_patterns(
    pe_matrix: np.ndarray | torch.Tensor,
    pe_type: str,
    num_tokens_to_plot: int = 5,
    figsize: tuple[float, float] = (14, 6),
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
    figsize : tuple[float, float]
        Figure size (default: (14, 6))

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    if isinstance(pe_matrix, torch.Tensor):
        pe_matrix = pe_matrix.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot first N tokens to show pattern
    tokens_to_plot = min(num_tokens_to_plot, pe_matrix.shape[0])
    for token_idx in range(tokens_to_plot):
        ax.plot(pe_matrix[token_idx, :], label=f"Token {token_idx}", alpha=0.7, linewidth=2)

    ax.set_xlabel("Model Dimension", fontsize=14)
    ax.set_ylabel("Encoding Value", fontsize=14)
    ax.set_title(
        f"{pe_type.capitalize()} - Encoding Pattern Across Dimensions",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, ax


def plot_rotary_patterns(
    cos_cache: np.ndarray | torch.Tensor,
    sin_cache: np.ndarray | torch.Tensor,
    figsize: tuple[float, float] = (16, 6),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot Rotary (RoPE) sin/cos patterns.

    Visualizes the sin and cosine patterns used in Rotary Position Embeddings.

    Parameters
    ----------
    cos_cache : np.ndarray | torch.Tensor
        Cosine cache of shape [seq_len, head_dim]
    sin_cache : np.ndarray | torch.Tensor
        Sine cache of shape [seq_len, head_dim]
    figsize : tuple[float, float]
        Figure size (default: (16, 6))

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Figure and axes objects
    """
    if isinstance(cos_cache, torch.Tensor):
        cos_cache = cos_cache.squeeze().detach().cpu().numpy()
    if isinstance(sin_cache, torch.Tensor):
        sin_cache = sin_cache.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    num_positions = cos_cache.shape[0]

    # Plot cos
    im0 = axes[0].imshow(
        cos_cache,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axes[0].set_xlabel("Head Dimension", fontsize=14)
    axes[0].set_ylabel("Sequence Position", fontsize=14)
    axes[0].set_title(
        f"Rotary Cosine Pattern\n({num_positions} positions × {cos_cache.shape[1]} head_dims)",
        fontsize=16,
        fontweight="bold",
    )
    # Y-axis ticks and labels
    y_ticks = np.arange(num_positions)
    y_labels = ["CLS"] + [str(i) for i in range(1, num_positions)] if num_positions > 0 else []
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_labels)
    # Horizontal lines between positions
    for y in np.arange(num_positions + 1) - 0.5:
        axes[0].axhline(y, color="k", linewidth=0.5, alpha=0.3)
    plt.colorbar(im0, ax=axes[0], label="Cosine Value", fontsize=12)

    # Plot sin
    im1 = axes[1].imshow(
        sin_cache,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_xlabel("Head Dimension", fontsize=14)
    axes[1].set_ylabel("Sequence Position", fontsize=14)
    axes[1].set_title(
        f"Rotary Sine Pattern\n({num_positions} positions × {sin_cache.shape[1]} head_dims)",
        fontsize=16,
        fontweight="bold",
    )
    axes[1].set_yticks(y_ticks)
    axes[1].set_yticklabels(y_labels)
    for y in np.arange(num_positions + 1) - 0.5:
        axes[1].axhline(y, color="k", linewidth=0.5, alpha=0.3)
    plt.colorbar(im1, ax=axes[1], label="Sine Value", fontsize=12)

    plt.tight_layout()
    return fig, axes


def plot_layer_l2_norms(
    l2_norms_dict: dict[str, dict[str, np.ndarray]],
    layers_to_plot: list[str] | None = None,
    figsize: tuple[float, float] = (16, 12),
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
        Figure size (default: (16, 12))

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and axes array
    """
    if len(l2_norms_dict) == 0:
        raise ValueError("l2_norms_dict is empty")

    # Auto-select layers if not provided
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

        # X-axis: CLS, 1..N
        if num_tokens is not None:
            positions = np.arange(num_tokens)
            ax.set_xticks(positions)
            ax.set_xticklabels(["CLS"] + [str(i) for i in range(1, num_tokens)])

        ax.set_xlabel("Token Position", fontsize=14)
        ax.set_ylabel("L2 Norm", fontsize=14)
        ax.set_title(
            f"{pe_type.capitalize()} - Token Representation Magnitude",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(pe_types_available), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig, axes
