"""Legacy pairwise bias utilities.

New code should import from
``thesis_ml.architectures.transformer_classifier.modules.biases``.

Kept for backward-compat with old checkpoints and configs that reference
``PairwiseBiasNet`` or ``compute_pairwise_kinematics``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Re-export shared feature helpers so old imports still work
from .biases._features import (
    _FEATURES_NEED_E,
    VALID_FEATURES,
    _extract_kinematics,
    compute_pairwise_feature_set,
)

# Re-export new module so old imports still work
from .biases.lorentz_scalar import LorentzScalarBias

__all__ = [
    "VALID_FEATURES",
    "_FEATURES_NEED_E",
    "_extract_kinematics",
    "compute_pairwise_feature_set",
    "LorentzScalarBias",
    "compute_pairwise_kinematics",
    "PairwiseBiasNet",
]


# ---------------------------------------------------------------------------
# Legacy helpers
# ---------------------------------------------------------------------------


def compute_pairwise_kinematics(
    tokens_cont: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return [B, T, T, 2] with (m2_ij, deltaR_ij).

    Kept for backward compat; prefer ``compute_pairwise_feature_set``.
    """
    out, _ = compute_pairwise_feature_set(tokens_cont, ["m2", "deltaR"], mask=mask)
    return out


class PairwiseBiasNet(nn.Module):
    """Legacy [B, T, T, F] → attention bias.  Prefer LorentzScalarBias."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 8,
        num_heads: int = 1,
        per_head: bool = False,
    ):
        super().__init__()
        self.per_head = per_head
        out_dim = num_heads if per_head else 1
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pairwise_features: torch.Tensor) -> torch.Tensor:
        out = self.mlp(pairwise_features)  # [B, T, T, out_dim]
        if self.per_head:
            return out.permute(0, 3, 1, 2)  # [B, H, T, T]
        return out.squeeze(-1)  # [B, T, T]
