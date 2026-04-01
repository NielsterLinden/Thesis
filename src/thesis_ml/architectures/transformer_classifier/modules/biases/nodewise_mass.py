"""Nodewise-mass embedding patch (Li et al. 2024, Section III.B).

For each particle i, find its k nearest "friends" by Lorentz dot-product
p_i · p_j and compute the invariant mass of that friend group.  The resulting
per-node scalar (one mass value per k level) is embedded by a small MLP and
added to the token representation *before* the encoder.

This is fully Lorentz-invariant (uses only Lorentz dot products) and acts as a
local neighbourhood feature rather than a pairwise attention bias.

Requires E (cont_dim = 4 / C = 4).  Returns None gracefully when E is absent.

Source: Li et al. (2024), "Does Lorentz-Symmetric Design Boost Network
Performance in Jet Physics?", Section III.B.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._features import _extract_kinematics


class NodewiseMassBias(nn.Module):
    """Nodewise invariant mass patch applied to token embeddings.

    For each of the supplied k-values, builds the friend-group of k particles
    with the largest Lorentz dot-product with the query particle, then computes
    the invariant mass of that group.  The resulting [B, T, len(k_values)] mass
    features are embedded to model-dimension and added as a residual to ``x``.

    The addition happens *before* the transformer encoder.

    Integration::

        x_phys = x_phys + self.nodewise_mass(tokens_cont, x_phys, mask_phys)

    where x_phys is the [B, T, D] slice of x corresponding to physical tokens.
    """

    def __init__(
        self,
        model_dim: int,
        cont_dim: int,
        k_values: list[int] | None = None,
        hidden_dim: int = 64,
    ):
        """Initialise NodewiseMassBias.

        Parameters
        ----------
        model_dim : int
            Transformer model dimension D.  Output is projected to this size.
        cont_dim : int
            Continuous feature dimension.  Module is a no-op when cont_dim < 4.
        k_values : list[int], optional
            Neighbourhood sizes.  Defaults to [2, 4, 8].
        hidden_dim : int
            Hidden dim for the embedding MLP.
        """
        super().__init__()
        self.cont_dim = cont_dim
        self.has_E = cont_dim >= 4
        self.k_values = sorted(k_values or [2, 4, 8])
        n_k = len(self.k_values)

        if self.has_E:
            self.embed_mlp = nn.Sequential(
                nn.Linear(n_k, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, model_dim),
            )
            # Near-zero init: patch starts as identity
            nn.init.zeros_(self.embed_mlp[-1].weight)
            nn.init.zeros_(self.embed_mlp[-1].bias)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lorentz_dot(tokens_cont: torch.Tensor) -> torch.Tensor:
        """Compute p_i · p_j = E_iE_j − px_i·px_j − py_i·py_j − pz_i·pz_j.

        Parameters
        ----------
        tokens_cont : torch.Tensor
            [B, T, 4] with (E, Pt, eta, phi) columns.

        Returns
        -------
        torch.Tensor
            [B, T, T] Lorentz dot-product matrix.
        """
        E, Pt, eta, phi = _extract_kinematics(tokens_cont)
        px = Pt * torch.cos(phi)
        py = Pt * torch.sin(phi)
        pz = Pt * torch.sinh(eta.clamp(-20.0, 20.0))

        E_i = E.unsqueeze(2)
        E_j = E.unsqueeze(1)
        px_i = px.unsqueeze(2)
        px_j = px.unsqueeze(1)
        py_i = py.unsqueeze(2)
        py_j = py.unsqueeze(1)
        pz_i = pz.unsqueeze(2)
        pz_j = pz.unsqueeze(1)

        return E_i * E_j - px_i * px_j - py_i * py_j - pz_i * pz_j  # [B, T, T]

    @staticmethod
    def _group_invariant_mass(
        tokens_cont: torch.Tensor,
        topk_idx: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Invariant mass of the friend group for each token.

        Parameters
        ----------
        tokens_cont : torch.Tensor
            [B, T, 4].
        topk_idx : torch.Tensor
            [B, T, k] — indices of the k nearest friends.
        mask : torch.Tensor, optional
            [B, T] True=valid.

        Returns
        -------
        torch.Tensor
            [B, T] scalar invariant mass (clamped >= 0 before sqrt).
        """
        B, T, k = topk_idx.shape

        # Gather 4-vectors of friends
        # tokens_cont: [B, T, 4] → expand to [B, T, k, 4]
        idx_exp = topk_idx.unsqueeze(-1).expand(B, T, k, 4)  # [B, T, k, 4]
        tc_exp = tokens_cont.unsqueeze(1).expand(B, T, T, 4)  # [B, T, T, 4]
        friends = tc_exp.gather(2, idx_exp)  # [B, T, k, 4]

        # Reconstruct 4-vectors in (E, px, py, pz)
        E_f = friends[..., 0]
        Pt_f = friends[..., 1]
        eta_f = friends[..., 2]
        phi_f = friends[..., 3]
        px_f = Pt_f * torch.cos(phi_f)
        py_f = Pt_f * torch.sin(phi_f)
        pz_f = Pt_f * torch.sinh(eta_f.clamp(-20.0, 20.0))

        # Sum 4-vectors over friend group
        E_sum = E_f.sum(dim=2)
        px_sum = px_f.sum(dim=2)
        py_sum = py_f.sum(dim=2)
        pz_sum = pz_f.sum(dim=2)

        m2 = (E_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2).clamp(min=0.0)

        if mask is not None:
            if mask.size(1) != T:
                mask = mask[:, :T]
            m2 = m2 * mask.to(m2.dtype)

        return torch.sqrt(m2 + 1e-8)  # [B, T]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens_cont: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Compute nodewise mass features.

        Parameters
        ----------
        tokens_cont : torch.Tensor
            [B, T, C] — must have C=4 for this module to activate.
        mask : torch.Tensor, optional
            [B, T] True=valid.  Padding tokens are excluded from friend
            selection.

        Returns
        -------
        torch.Tensor | None
            [B, T, model_dim] residual to add to x[:, cls_offset:cls_offset+T];
            None when cont_dim < 4.
        """
        if not self.has_E or tokens_cont.size(-1) < 4:
            return None

        B, T, _ = tokens_cont.shape
        max_k = max(self.k_values)
        if max_k > T:
            max_k = T  # can't have more friends than tokens

        # Lorentz dot-product matrix [B, T, T]
        dot = self._lorentz_dot(tokens_cont)

        # Mask padding tokens from friend selection
        if mask is not None:
            if mask.size(1) != T:
                mask = mask[:, :T]
            # Padding tokens score -inf (never selected as friends)
            padding_mask = ~mask.unsqueeze(1)  # [B, 1, T]  True=pad
            dot = dot.masked_fill(padding_mask, float("-inf"))

        # Top-max_k friends by dot product
        _, topk_idx_max = dot.topk(min(max_k, T), dim=-1, largest=True)  # [B, T, max_k]

        # Compute invariant mass for each k level
        mass_feats: list[torch.Tensor] = []
        for k in self.k_values:
            k_eff = min(k, T)
            topk_idx_k = topk_idx_max[:, :, :k_eff]  # [B, T, k_eff]
            mass = self._group_invariant_mass(tokens_cont, topk_idx_k, mask)  # [B, T]
            mass_feats.append(mass)

        mass_tensor = torch.stack(mass_feats, dim=-1)  # [B, T, n_k]

        # Embed to model_dim
        return self.embed_mlp(mass_tensor)  # [B, T, model_dim]
