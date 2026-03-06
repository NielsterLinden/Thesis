"""More-Interaction Attention (MIA) encoder blocks (Wu et al. 2025 — MIParT).

Architecture (following Wu et al. Figure 1 and Section II.D):

  1. InteractionEmbedding: pairwise features → U1 [B, T, T, D1=64]
     Computed ONCE; U1 is fixed while MIA blocks process x.

  2. K × MIABlock (MI-Particle Attention Block):
       - MIA:   out = einsum('bijc,bjc->bic', softmax(U1, dim=2), V_proj(x))
       - x ← LN(x + dropout(out_proj(out)))
       - MLP(x) + residual

  3. InteractionReduction: U1 → U2 [B, T, T, D2=8]
     One pointwise linear layer.

  Output: (x_updated [B, T, D], U2 [B, T, T, D2])

  U2 is then passed to the standard TransformerEncoder as its attention_bias
  (after permuting to [B, D2, T, T] and being padded for CLS/MET by BiasComposer).

Source: Wu et al. (2025) "Jet Tagging with More-Interaction Particle Transformer",
        Sections II.B–II.D, Figure 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._features import _FEATURES_NEED_E, compute_pairwise_feature_set

# ---------------------------------------------------------------------------
# Default pairwise features for MIA (following Wu et al. Eq 6–9)
# ---------------------------------------------------------------------------

_MIA_DEFAULT_FEATURES = ["log_kt", "z", "deltaR", "log_m2"]


def _mia_features_for_cont_dim(cont_dim: int) -> list[str]:
    """Return the MIA features that can be computed for the given cont_dim."""
    has_E = cont_dim >= 4
    return [f for f in _MIA_DEFAULT_FEATURES if has_E or f not in _FEATURES_NEED_E]


# ---------------------------------------------------------------------------
# InteractionEmbedding
# ---------------------------------------------------------------------------


class InteractionEmbedding(nn.Module):
    """Maps pairwise feature tensor to high-dimensional interaction embedding U1.

    Three-layer pointwise MLP (applied independently to each (i,j) pair) with
    GELU activations and a final LayerNorm.  Following MIParT, operations are
    pointwise over the T×T spatial grid — no mixing across positions.

    Parameters
    ----------
    in_features : int
        Number of input pairwise features F (depends on cont_dim).
    interaction_dim : int
        Output dimension D1 (default 64 in MIParT).
    hidden_dim : int
        Hidden dim for the projection MLP (default 64).
    """

    def __init__(self, in_features: int, interaction_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, interaction_dim),
        )
        self.ln = nn.LayerNorm(interaction_dim)

    def forward(self, pairwise_feats: torch.Tensor) -> torch.Tensor:
        """[B, T, T, F] → U1 [B, T, T, D1]."""
        return self.ln(self.net(pairwise_feats))


# ---------------------------------------------------------------------------
# InteractionReduction
# ---------------------------------------------------------------------------


class InteractionReduction(nn.Module):
    """Reduce U1 [B, T, T, D1] → U2 [B, T, T, D2] via a single linear layer."""

    def __init__(self, interaction_dim: int = 64, reduction_dim: int = 8):
        super().__init__()
        self.proj = nn.Linear(interaction_dim, reduction_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, U1: torch.Tensor) -> torch.Tensor:
        """[B, T, T, D1] → U2 [B, T, T, D2]."""
        return self.proj(U1)


# ---------------------------------------------------------------------------
# MIABlock (MI-Particle Attention Block)
# ---------------------------------------------------------------------------


class MIABlock(nn.Module):
    """One MI-Particle Attention Block (MIA + MLP).

    MIA operation (per interaction channel c):
      out[b, i, c] = sum_j( softmax(U1[b,i,:,c])[j] * V[b,j,c] )
    where V = V_proj(x) and out_proj maps D_mia → model_dim.

    Residual structure (pre-norm):
      x ← x + dropout(out_proj(MIA(LN(x))))
      x ← x + dropout(FFN(LN(x)))
    """

    def __init__(
        self,
        model_dim: int,
        interaction_dim: int,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ):
        """Initialise MIABlock.

        Parameters
        ----------
        model_dim : int
            Model embedding dimension D.
        interaction_dim : int
            MIA channel dimension D1 (number of MIA "heads").
        ffn_dim : int, optional
            Feed-forward hidden dim.  Defaults to 4 × model_dim.
        dropout : float
            Dropout probability applied after MIA and after FFN.
        """
        super().__init__()
        ffn_dim = ffn_dim or 4 * model_dim

        self.ln_mia = nn.LayerNorm(model_dim)
        self.v_proj = nn.Linear(model_dim, interaction_dim, bias=False)
        self.out_proj = nn.Linear(interaction_dim, model_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # start as identity pass-through

        self.ln_ffn = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, model_dim),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        U1: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            [B, T, D] particle embeddings.
        U1 : torch.Tensor
            [B, T, T, D1] interaction embedding (fixed; not updated here).
        mask : torch.Tensor, optional
            [B, T] True=valid.  Padding source tokens are suppressed in MIA.

        Returns
        -------
        torch.Tensor
            [B, T, D] updated embeddings.
        """
        B, T, D = x.shape

        # --- MIA sub-block ---
        x_norm = self.ln_mia(x)
        V = self.v_proj(x_norm)  # [B, T, D1]

        # Mask padding source tokens from softmax
        attn_logits = U1  # [B, T, T, D1]
        if mask is not None:
            if mask.size(1) != T:
                mask = mask[:, :T]
            # [B, 1, T, 1] broadcast over (query, channel) dims
            src_mask = (~mask).unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1] True=padding
            attn_logits = attn_logits.masked_fill(src_mask, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=2)  # [B, T, T, D1]
        # Channel-wise weighted sum: out[i,c] = sum_j attn[i,j,c] * V[j,c]
        mia_out = torch.einsum("bijc,bjc->bic", attn_weights, V)  # [B, T, D1]
        x = x + self.dropout(self.out_proj(mia_out))  # [B, T, D]

        # --- FFN sub-block ---
        x = x + self.dropout(self.ffn(self.ln_ffn(x)))

        return x


# ---------------------------------------------------------------------------
# MIAEncoder
# ---------------------------------------------------------------------------


class MIAEncoder(nn.Module):
    """Full MIParT pre-encoder pipeline.

    Wraps InteractionEmbedding, K MIABlocks, and InteractionReduction into a
    single module that operates on the physical token slice of x.

    Usage in TransformerClassifier.forward()::

        x_phys = x[:, cls_offset : cls_offset + T]
        x_phys_updated, U2 = self.mia_encoder(x_phys, tokens_cont, mask_phys)
        x[:, cls_offset : cls_offset + T] = x_phys_updated
        # U2 [B, T, T, D2] → pad & pass to standard encoder as attention_bias

    U2 must be permuted to [B, D2, T, T] before being padded by BiasComposer's
    _pad_bias_for_special_tokens (which expects [B, H, T, T]).

    Source: Wu et al. (2025) MIParT, Sections II.B–II.D and Figure 1.
    """

    def __init__(
        self,
        model_dim: int,
        cont_dim: int,
        num_blocks: int = 5,
        interaction_dim: int = 64,
        reduction_dim: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ):
        """Initialise MIAEncoder.

        Parameters
        ----------
        model_dim : int
            Transformer model dimension D.
        cont_dim : int
            Number of continuous features C.
        num_blocks : int
            K — number of MIA blocks (Wu et al. use K=5).
        interaction_dim : int
            D1 — high-dimensional interaction embedding channels.
        reduction_dim : int
            D2 — channels after reduction (= num_heads of standard encoder).
        ffn_dim : int, optional
            MIABlock FFN hidden dim.  Defaults to 4 × model_dim.
        dropout : float
            Dropout in MIABlocks.
        """
        super().__init__()
        self.cont_dim = cont_dim
        self.interaction_dim = interaction_dim
        self.reduction_dim = reduction_dim

        active_feats = _mia_features_for_cont_dim(cont_dim)
        self._active_features = active_feats
        self._has_features = len(active_feats) > 0

        if self._has_features:
            F = len(active_feats)
            self.interaction_embedding = InteractionEmbedding(F, interaction_dim)
            self.mia_blocks = nn.ModuleList([MIABlock(model_dim, interaction_dim, ffn_dim, dropout) for _ in range(num_blocks)])
            self.interaction_reduction = InteractionReduction(interaction_dim, reduction_dim)

        # Per-module gate for U2 output (init=0 → starts as zero bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        tokens_cont: torch.Tensor,
        F_ij: torch.Tensor | None = None,
        feature_to_idx: dict[str, int] | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run MIA pipeline.

        Parameters
        ----------
        x : torch.Tensor
            [B, T, D] physical token embeddings (CLS and MET already excluded).
        tokens_cont : torch.Tensor
            [B, T, C] continuous features for the same T physical tokens.
        F_ij : torch.Tensor, optional
            Shared pairwise features [B, T, T, F] from backbone.
        feature_to_idx : dict, optional
            Maps feature name → channel index in F_ij.
        mask : torch.Tensor, optional
            [B, T] True=valid (physical tokens only).

        Returns
        -------
        x_updated : torch.Tensor
            [B, T, D] — updated particle embeddings.
        U2 : torch.Tensor | None
            [B, D2, T, T] — reduced interaction embedding as attention bias.
        """
        if not self._has_features:
            return x, None

        feat = None
        if F_ij is not None and feature_to_idx is not None:
            idxs = [feature_to_idx[f] for f in self._active_features if f in feature_to_idx]
            if len(idxs) == len(self._active_features):
                feat = F_ij[..., idxs]
        if feat is None and tokens_cont is not None:
            feat, _ = compute_pairwise_feature_set(tokens_cont, self._active_features, mask=mask)
        if feat is None:
            return x, None

        # U1: computed once, fixed across all MIA blocks
        U1 = self.interaction_embedding(feat)  # [B, T, T, D1]

        # K MIA blocks update x; U1 is read-only
        for block in self.mia_blocks:
            x = block(x, U1, mask=mask)

        # Reduce U1 → U2 and apply gate
        U2_raw = self.interaction_reduction(U1)  # [B, T, T, D2]
        U2 = torch.tanh(self.gate) * U2_raw

        # Permute to [B, D2, T, T] to match [B, H, T, T] attention_bias convention
        U2 = U2.permute(0, 3, 1, 2)  # [B, D2, T, T]

        return x, U2
