"""Pairwise physics-informed attention bias: kinematics computation and PairwiseBiasNet.

tokens_cont is assumed [B, T, 4] with layout (E, Pt, eta, phi) per token.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def compute_pairwise_kinematics(
    tokens_cont: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute pairwise invariant mass squared and deltaR from continuous token features.

    Parameters
    ----------
    tokens_cont : torch.Tensor
        [B, T, 4] with columns (E, Pt, eta, phi) per token.
    mask : torch.Tensor, optional
        [B, T] True=valid, False=padding. Pairs involving a padded position
        are zeroed in the output so key_padding_mask can still suppress them.

    Returns
    -------
    torch.Tensor | None
        [B, T, T, 2] with (m2_ij, deltaR_ij). Returns None if tokens_cont is not
        of shape [B, T, 4] or on failure.
    """
    if tokens_cont.dim() != 3 or tokens_cont.size(-1) < 4:
        return None
    B, T, _ = tokens_cont.shape
    E = tokens_cont[..., 0]  # [B, T]
    Pt = tokens_cont[..., 1]
    eta = tokens_cont[..., 2]
    phi = tokens_cont[..., 3]

    # Momentum components (px, py, pz)
    px = Pt * torch.cos(phi)
    py = Pt * torch.sin(phi)
    pz = Pt * torch.sinh(eta.clamp(-20.0, 20.0))

    # Pair sums for invariant mass squared: (E_i+E_j)^2 - (p_i+p_j)^2
    E_ij = E.unsqueeze(2) + E.unsqueeze(1)  # [B, T, T]
    px_ij = px.unsqueeze(2) + px.unsqueeze(1)
    py_ij = py.unsqueeze(2) + py.unsqueeze(1)
    pz_ij = pz.unsqueeze(2) + pz.unsqueeze(1)
    m2_ij = (E_ij**2 - px_ij**2 - py_ij**2 - pz_ij**2).clamp(min=0.0)

    # Delta R = sqrt(delta_eta^2 + delta_phi^2), delta_phi in [-pi, pi]
    delta_eta = eta.unsqueeze(2) - eta.unsqueeze(1)
    delta_phi = phi.unsqueeze(2) - phi.unsqueeze(1)
    delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi
    deltaR = torch.sqrt((delta_eta**2 + delta_phi**2).clamp(min=0.0) + 1e-8)

    out = torch.stack([m2_ij, deltaR], dim=-1)  # [B, T, T, 2]

    if mask is not None:
        # Zero out pairs where either i or j is padded
        valid = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, T, T]
        out = out * valid.unsqueeze(-1).to(out.dtype)

    return out


class PairwiseBiasNet(nn.Module):
    """Maps pairwise feature tensor [B, T, T, F] to attention bias [B, T, T] or [B, num_heads, T, T]."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 8,
        num_heads: int = 1,
        per_head: bool = False,
    ):
        """Initialize.

        Parameters
        ----------
        num_features : int
            Last-dim size of pairwise features (e.g. 2 for m2, deltaR).
        hidden_dim : int
            Hidden size of the pointwise MLP over the feature dimension.
        num_heads : int
            Number of attention heads (used only when per_head=True).
        per_head : bool
            If True, output [B, num_heads, T, T]; else [B, T, T].
        """
        super().__init__()
        self.per_head = per_head
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads if per_head else 1),
        )

    def forward(self, pairwise_features: torch.Tensor) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        pairwise_features : torch.Tensor
            [B, T, T, F]

        Returns
        -------
        torch.Tensor
            [B, T, T] or [B, num_heads, T, T] when per_head=True.
        """
        # Pointwise over (T,T): [B, T, T, F] -> [B, T, T, 1] or [B, T, T, H]
        out = self.mlp(pairwise_features)
        out = out.permute(0, 3, 1, 2) if self.per_head else out.squeeze(-1)
        return out
