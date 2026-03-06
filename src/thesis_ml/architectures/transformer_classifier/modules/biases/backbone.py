"""Shared pairwise feature backbone — compute F_ij once for all bias modules.

All modules consume the same F_ij [B,T,T,F], avoiding redundant computation
and enabling "which module uses which features" analyses. Aligns with
MIParT interaction-embedding spirit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._features import (
    _FEATURES_NEED_E,
    VALID_FEATURES,
    compute_pairwise_feature_set,
)

# Canonical feature order for consistent channel indexing
_FEATURE_ORDER: tuple[str, ...] = tuple(sorted(VALID_FEATURES))  # deltaR, deltaR_ptw, dot, log_kt, log_m2, m2, z


class PairwiseFeatureBackbone(nn.Module):
    """Compute shared pairwise feature tensor F_ij [B,T,T,F] once per forward.

    All bias modules that need pairwise kinematics consume F_ij instead of
    recomputing from tokens_cont. Features are computed in a canonical order
    so modules can index by name via feature_to_idx.

    Parameters
    ----------
    cont_dim : int
        Number of continuous features (3 or 4). E-dependent features
        are excluded when cont_dim < 4.
    features : list[str] | str
        Feature names from VALID_FEATURES, or "all" for maximum set.
    """

    def __init__(self, cont_dim: int, features: list[str] | str = "all"):
        super().__init__()
        has_E = cont_dim >= 4
        if features == "all":
            self.active_features = [f for f in _FEATURE_ORDER if has_E or f not in _FEATURES_NEED_E]
        else:
            self.active_features = [f for f in features if f in VALID_FEATURES and (has_E or f not in _FEATURES_NEED_E)]
        self.cont_dim = cont_dim

    def forward(
        self,
        tokens_cont: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, list[str], dict[str, int]]:
        """Compute F_ij and feature index mapping.

        Returns
        -------
        F_ij : torch.Tensor | None
            [B, T, T, F] pairwise features. None if no features.
        active_names : list[str]
            Feature names in channel order.
        feature_to_idx : dict[str, int]
            Maps feature name → channel index for F_ij.
        """
        if not self.active_features:
            return None, [], {}
        F_ij, active = compute_pairwise_feature_set(tokens_cont, self.active_features, mask=mask)
        if F_ij is None:
            return None, [], {}
        feature_to_idx = {name: i for i, name in enumerate(active)}
        return F_ij, active, feature_to_idx
