"""Shared normalization utilities for attention modules and encoder blocks.

This module provides:
- ``RMSNorm``: Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
- ``build_norm``: Factory function that returns either ``nn.LayerNorm`` or
  ``RMSNorm`` based on a string key.

These are used across two independent normalization axes:

- **Axis B (block norm type)**: controls which norm module is used for all
  block-level norms in ``TransformerEncoderBlock`` (``norm1``, ``norm2``,
  ``norm_attn_out``, ``norm_mlp_mid``).
- **Axis C (attention-internal norm)**: controls optional per-head
  normalization inside ``MultiHeadAttention`` and ``DifferentialAttention``.

Both axes call ``build_norm`` so there is a single source of truth for norm
construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm does not re-center activations (no mean
    subtraction).  It normalizes by the root mean square and applies a
    learnable elementwise scale.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def build_norm(norm_type: str, dim: int) -> nn.Module:
    """Return a normalization layer given a type string.

    Parameters
    ----------
    norm_type : str
        ``"layernorm"`` or ``"rmsnorm"``.
    dim : int
        Feature dimension to normalize over.

    Returns
    -------
    nn.Module
        A ``nn.LayerNorm`` or ``RMSNorm`` instance.
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm_type: {norm_type!r}. Expected 'layernorm' or 'rmsnorm'.")
