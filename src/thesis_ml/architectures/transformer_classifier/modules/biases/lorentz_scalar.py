"""LorentzScalarBias — pairwise physics bias from Lorentz-invariant features.

Sources: ParT (Qu et al. 2022), Li et al. (2024), Wu et al. MIParT (2025).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._features import _FEATURES_NEED_E, VALID_FEATURES, compute_pairwise_feature_set


class LorentzScalarBias(nn.Module):
    """Physics-informed pairwise attention bias from Lorentz-scalar features.

    Computes a configurable set of pairwise features from four-momentum data
    and maps them to an additive attention bias via a pointwise MLP.

    Feature hierarchy (Li et al. 2024):
      Full Lorentz scalars  : m2, dot, log_m2          (require E, C=4)
      Partial invariants    : log_kt, z, deltaR,
                              deltaR_ptw                (Pt/eta/phi, C>=3)

    E-dependent features are silently dropped when ``cont_dim < 4``.
    Returns ``None`` if no features remain after filtering.

    Per-module gate (init=0): all biases zero at init.

    Sources: ParT (Qu et al. 2022), Li et al. (2024), Wu et al. MIParT (2025).
    """

    def __init__(
        self,
        features: list[str],
        cont_dim: int,
        hidden_dim: int = 8,
        num_heads: int = 1,
        per_head: bool = False,
        sparse_gating: bool = False,
    ):
        super().__init__()
        unknown = set(features) - VALID_FEATURES
        if unknown:
            raise ValueError(f"Unknown features: {unknown}. Valid: {VALID_FEATURES}")

        has_E = cont_dim >= 4
        self.active_features: list[str] = [f for f in features if has_E or f not in _FEATURES_NEED_E]
        self.per_head = per_head
        self.num_heads = num_heads
        self.sparse_gating = sparse_gating

        F = len(self.active_features)
        if F == 0:
            self._has_mlp = False
        else:
            self._has_mlp = True
            out_dim = num_heads if per_head else 1
            self.mlp = nn.Sequential(
                nn.Linear(F, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
            if sparse_gating:
                self.feature_gates = nn.Parameter(torch.zeros(F))
            else:
                self.feature_gates = None

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        tokens_cont: torch.Tensor | None = None,
        F_ij: torch.Tensor | None = None,
        feature_to_idx: dict[str, int] | None = None,
        mask: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor | None:
        if not self._has_mlp:
            return None
        if F_ij is not None and feature_to_idx is not None:
            idxs = [feature_to_idx[f] for f in self.active_features if f in feature_to_idx]
            feat_tensor = None if len(idxs) != len(self.active_features) else F_ij[..., idxs]
        else:
            feat_tensor = None
        if feat_tensor is None and tokens_cont is not None:
            feat_tensor, _ = compute_pairwise_feature_set(tokens_cont, self.active_features, mask=mask)
        if feat_tensor is None:
            return None
        if self.feature_gates is not None:
            feat_tensor = feat_tensor * torch.sigmoid(self.feature_gates)
        out = self.mlp(feat_tensor)  # [B, T, T, out_dim]
        out = out.permute(0, 3, 1, 2)  # [B, out_dim, T, T]
        return torch.tanh(self.gate) * out
