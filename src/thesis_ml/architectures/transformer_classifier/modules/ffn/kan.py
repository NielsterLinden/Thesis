"""Kolmogorov-Arnold Network feed-forward module for transformer encoder blocks.

Drop-in replacement for :class:`StandardFFN` using learnable B-spline
activations (:class:`KANLinear`) instead of fixed GELU.

Variants
--------
- **hybrid** (default): ``Linear(dim, mlp_dim) → Dropout → KANLinear(mlp_dim, dim) → Dropout``
  KAN replaces the activation + down-projection.
- **bottleneck**: ``Linear(dim, bd) → KANLinear(bd, bd) → Linear(bd, dim) → Dropout``
  KAN operates in a compressed space (parameter-competitive with standard FFN).
- **pure** (experimental): ``KANLinear(dim, mlp_dim) → KANLinear(mlp_dim, dim) → Dropout``
  Full KAN replacement — most expressive but ~8-10x parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from thesis_ml.architectures.transformer_classifier.modules.attention import build_norm
from thesis_ml.architectures.transformer_classifier.modules.kan.kan_linear import (
    KANLinear,
)


class KANFFN(nn.Module):
    """KAN-based feed-forward network.

    Parameters
    ----------
    dim : int
        Model (input / output) dimension.
    mlp_dim : int
        Hidden dimension (used by hybrid and pure variants).
    dropout : float
        Dropout rate.
    norm_policy : str
        ``"pre"`` / ``"post"`` / ``"normformer"``.  NormFormer inserts a
        block-level norm between the up-projection and the KAN layer.
    block_norm_type : str
        ``"layernorm"`` or ``"rmsnorm"`` — used for the NormFormer mid-norm.
    variant : str
        ``"hybrid"`` | ``"bottleneck"`` | ``"pure"``.
    bottleneck_dim : int | None
        Compressed dimension for the bottleneck variant.  ``None`` defaults
        to ``mlp_dim // 4``.
    grid_size : int
        B-spline grid intervals for :class:`KANLinear`.
    spline_order : int
        B-spline polynomial degree.
    grid_range : tuple[float, float]
        Input range for the B-spline grid.
    """

    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        dropout: float = 0.1,
        norm_policy: str = "pre",
        block_norm_type: str = "layernorm",
        variant: str = "hybrid",
        bottleneck_dim: int | None = None,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__()
        self.variant = variant

        kan_kwargs = dict(grid_size=grid_size, spline_order=spline_order, grid_range=grid_range)

        if variant == "hybrid":
            layers: list[nn.Module] = [nn.Linear(dim, mlp_dim)]
            if norm_policy == "normformer":
                layers.append(build_norm(block_norm_type, mlp_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(KANLinear(mlp_dim, dim, **kan_kwargs))
            layers.append(nn.Dropout(dropout))

        elif variant == "bottleneck":
            bd = bottleneck_dim if bottleneck_dim is not None else max(1, mlp_dim // 4)
            layers = [nn.Linear(dim, bd)]
            if norm_policy == "normformer":
                layers.append(build_norm(block_norm_type, bd))
            layers.append(KANLinear(bd, bd, **kan_kwargs))
            layers.append(nn.Linear(bd, dim))
            layers.append(nn.Dropout(dropout))

        elif variant == "pure":
            layers = [KANLinear(dim, mlp_dim, **kan_kwargs)]
            if norm_policy == "normformer":
                layers.append(build_norm(block_norm_type, mlp_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(KANLinear(mlp_dim, dim, **kan_kwargs))
            layers.append(nn.Dropout(dropout))

        else:
            raise ValueError(f"Unknown KANFFN variant: {variant!r}. " "Choose 'hybrid', 'bottleneck', or 'pure'.")

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the KAN FFN.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, T, D]``.
        mask : torch.Tensor | None
            Ignored — accepted for interface compatibility with MoE.
        """
        return self.net(x)
