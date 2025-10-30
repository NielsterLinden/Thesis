from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LinearBottleneck(nn.Module):
    """Learned linear projection bottleneck.

    Applies a learned linear transformation to the encoder output.
    Unlike VQ, this maintains a continuous latent space but adds
    a learnable transformation layer.
    """

    def __init__(self, *, cfg: Any):
        super().__init__()
        latent_dim = int(cfg.phase1.latent_space.latent_dim)
        # Linear projection: latent_dim -> latent_dim
        # This adds a learnable transformation without discretization
        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        """Apply learned linear projection.

        Args:
            z_e: Encoder output [B, T, D]

        Returns:
            Projected latent [B, T, D]
        """
        return self.projection(z_e)
