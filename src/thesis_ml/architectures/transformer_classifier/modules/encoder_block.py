from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from thesis_ml.architectures.transformer_classifier.modules.attention import (
    DifferentialAttention,
    MultiHeadAttention,
    build_norm,
)
from thesis_ml.architectures.transformer_classifier.modules.ffn import build_ffn


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with attention and MLP.

    Normalization is controlled by four independent axes:

    - **Axis A** (``norm_policy``): WHERE norms are placed — ``pre`` / ``post``
      / ``normformer``.
    - **Axis B** (``block_norm_type``): WHICH norm module is used at block
      level — ``layernorm`` / ``rmsnorm``.
    - **Axis C** (``attention_norm``): attention-internal per-head norm —
      ``none`` / ``layernorm`` / ``rmsnorm``.  Passed to the attention module.
    - **Axis D**: NormFormer extras (``head_scales``, ``norm_attn_out``,
      ``norm_mlp_mid``) — activated when ``norm_policy="normformer"``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        norm_policy: str = "post",
        block_norm_type: str = "layernorm",
        rotary_emb: nn.Module | None = None,
        causal_attention: bool = False,
        attention_type: str = "standard",
        attention_norm: str = "none",
        diff_bias_mode: str = "shared",
        layer_idx: int = 0,
        moe_cfg: dict[str, Any] | None = None,
        use_cls_token: bool = True,
        ffn_type: str = "standard",
        kan_cfg: dict[str, Any] | None = None,
    ):
        """Initialize encoder block.

        Parameters
        ----------
        dim : int
            Model dimension.
        num_heads : int
            Number of attention heads.
        mlp_dim : int
            MLP hidden dimension.
        dropout : float
            Dropout rate.
        norm_policy : str
            Axis A — normalization policy: ``"pre"``, ``"post"``, or
            ``"normformer"``.
        block_norm_type : str
            Axis B — block norm type: ``"layernorm"`` or ``"rmsnorm"``.
        rotary_emb : nn.Module | None
            Optional ``RotaryEmbedding``, shared across all encoder blocks.
        causal_attention : bool
            If True, apply causal (lower-triangular) attention mask.
        attention_type : str
            ``"standard"`` or ``"differential"``.
        attention_norm : str
            Axis C — attention-internal norm: ``"none"``, ``"layernorm"``,
            or ``"rmsnorm"``.
        diff_bias_mode : str
            Bias mode for differential attention: ``"none"`` / ``"shared"``
            / ``"split"``.  Ignored for standard attention.
        layer_idx : int
            Zero-based layer index (used for depth-dependent lambda init
            in differential attention).
        moe_cfg : dict | None
            MoE config dict.  ``None`` or ``{"enabled": False}`` selects
            the standard FFN.
        use_cls_token : bool
            Whether the model prepends a CLS token (needed by MoE for
            event-level routing).
        ffn_type : str
            ``"standard"`` | ``"kan"``.  MoE is activated via *moe_cfg*
            and takes priority when enabled.
        kan_cfg : dict | None
            Global KAN hyperparameters (used when ``ffn_type="kan"``).
        """
        super().__init__()
        self.norm_policy = norm_policy
        self.causal_attention = causal_attention

        # NormFormer head-wise scaling (Axis D)
        head_scales = None
        if norm_policy == "normformer":
            head_scales = nn.Parameter(torch.ones(num_heads))

        # Attention module selection
        if attention_type == "differential":
            self.attention = DifferentialAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
                head_scales=head_scales,
                rotary_emb=rotary_emb,
                attention_norm=attention_norm,
                layer_idx=layer_idx,
                diff_bias_mode=diff_bias_mode,
            )
        else:
            self.attention = MultiHeadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
                head_scales=head_scales,
                rotary_emb=rotary_emb,
                attention_norm=attention_norm,
            )

        # FFN (config-driven: StandardFFN, MoEFFN, or KANFFN via build_ffn)
        self.ffn = build_ffn(
            dim=dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            norm_policy=norm_policy,
            block_norm_type=block_norm_type,
            moe_cfg=moe_cfg,
            use_cls_token=use_cls_token,
            ffn_type=ffn_type,
            kan_cfg=kan_cfg,
        )

        # Block-level norms (Axis B)
        self.norm1 = build_norm(block_norm_type, dim)
        self.norm2 = build_norm(block_norm_type, dim)

        # NormFormer extra norm after attention output (Axis B + D)
        if norm_policy == "normformer":
            self.norm_attn_out = build_norm(block_norm_type, dim)
        else:
            self.norm_attn_out = None

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        capture_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``[B, T, D]``.
        mask : torch.Tensor, optional
            ``[B, T]`` (``True=valid``, ``False=padding``).
        attention_bias : Tensor | tuple[Tensor, Tensor] | None
            Additive bias for attention logits.
        capture_attention : bool
            If True, pass ``need_weights=True`` to attention and return
            ``(x_out, attn_weights)`` with weights ``[B, H, T, T]``.

        Returns
        -------
        torch.Tensor | tuple
            Output tensor ``[B, T, D]``, or ``(tensor, attn_weights)`` when
            ``capture_attention`` is True.
        """
        # Convert mask: loader uses True=valid; attention expects True=pad
        key_padding_mask = ~mask if mask is not None else None

        # Causal mask
        attn_mask = None
        if self.causal_attention:
            T = x.size(1)
            attn_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        attn_weights_out: torch.Tensor | None = None

        # ---- Attention block ----
        if self.norm_policy == "pre":
            x_norm = self.norm1(x)
            attn_out, aw = self.attention(
                x_norm,
                x_norm,
                x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=capture_attention,
                attn_mask=attn_mask,
                attention_bias=attention_bias,
            )
            if capture_attention:
                attn_weights_out = aw
            x = x + self.dropout(attn_out)
        elif self.norm_policy == "post":
            attn_out, aw = self.attention(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=capture_attention,
                attn_mask=attn_mask,
                attention_bias=attention_bias,
            )
            if capture_attention:
                attn_weights_out = aw
            x = self.norm1(x + self.dropout(attn_out))
        elif self.norm_policy == "normformer":
            x_norm = self.norm1(x)
            attn_out, aw = self.attention(
                x_norm,
                x_norm,
                x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=capture_attention,
                attn_mask=attn_mask,
                attention_bias=attention_bias,
            )
            if capture_attention:
                attn_weights_out = aw
            attn_out = self.norm_attn_out(attn_out)
            x = x + self.dropout(attn_out)
        else:
            raise ValueError(f"Unknown norm_policy: {self.norm_policy}")

        # ---- FFN block ----
        if self.norm_policy in ("pre", "normformer"):
            x = x + self.ffn(self.norm2(x), mask=mask)
        elif self.norm_policy == "post":
            x = self.norm2(x + self.ffn(x, mask=mask))
        else:
            raise ValueError(f"Unknown norm_policy: {self.norm_policy}")

        if capture_attention:
            return x, attn_weights_out
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        norm_policy: str = "post",
        block_norm_type: str = "layernorm",
        rotary_emb: nn.Module | None = None,
        causal_attention: bool = False,
        attention_type: str = "standard",
        attention_norm: str = "none",
        diff_bias_mode: str = "shared",
        moe_cfg: dict[str, Any] | None = None,
        use_cls_token: bool = True,
        ffn_type: str = "standard",
        kan_cfg: dict[str, Any] | None = None,
    ):
        """Initialize encoder stack.

        Parameters
        ----------
        dim, depth, num_heads, mlp_dim, dropout
            Standard transformer hyper-parameters.
        norm_policy : str
            Axis A — ``"pre"`` / ``"post"`` / ``"normformer"``.
        block_norm_type : str
            Axis B — ``"layernorm"`` / ``"rmsnorm"``.
        rotary_emb : nn.Module | None
            Shared ``RotaryEmbedding``.
        causal_attention : bool
            Causal mask in each block.
        attention_type : str
            ``"standard"`` or ``"differential"``.
        attention_norm : str
            Axis C — ``"none"`` / ``"layernorm"`` / ``"rmsnorm"``.
        diff_bias_mode : str
            Bias mode for differential attention.
        moe_cfg : dict | None
            MoE configuration.  Scope determines which blocks get MoE.
        use_cls_token : bool
            Whether the model uses a CLS token (for event-level routing).
        ffn_type : str
            ``"standard"`` | ``"kan"``.
        kan_cfg : dict | None
            Global KAN hyperparameters.
        """
        super().__init__()
        self.rotary_emb = rotary_emb

        moe_block_indices = _compute_moe_block_indices(depth, moe_cfg)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    norm_policy=norm_policy,
                    block_norm_type=block_norm_type,
                    rotary_emb=rotary_emb,
                    causal_attention=causal_attention,
                    attention_type=attention_type,
                    attention_norm=attention_norm,
                    diff_bias_mode=diff_bias_mode,
                    layer_idx=i,
                    moe_cfg=moe_cfg if i in moe_block_indices else None,
                    use_cls_token=use_cls_token,
                    ffn_type=ffn_type,
                    kan_cfg=kan_cfg,
                )
                for i in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        capture_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Forward pass through all encoder blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``[B, T, D]``.
        mask : torch.Tensor, optional
            ``[B, T]`` (``True=valid``, ``False=padding``).
        attention_bias : Tensor | tuple[Tensor, Tensor] | None
            Additive bias for attention logits; shared across all blocks.
        capture_attention : bool
            If True, return ``(x, list of per-layer attention weights)``.

        Returns
        -------
        torch.Tensor | tuple
            Output ``[B, T, D]``, or ``(x, weights)`` when capturing.
        """
        if capture_attention:
            weights: list[torch.Tensor | None] = []
            for block in self.blocks:
                out = block(x, mask=mask, attention_bias=attention_bias, capture_attention=True)
                if isinstance(out, tuple):
                    x, w = out
                    weights.append(w)
                else:
                    x = out
                    weights.append(None)
            return x, weights
        for block in self.blocks:
            x = block(x, mask=mask, attention_bias=attention_bias)
        return x


def _compute_moe_block_indices(depth: int, moe_cfg: dict[str, Any] | None) -> set[int]:
    """Return the set of block indices that should use MoE FFN.

    Parameters
    ----------
    depth : int
        Total number of encoder blocks.
    moe_cfg : dict | None
        MoE config.  Relevant key: ``scope``.

    Returns
    -------
    set[int]
        Block indices (0-based) that should receive MoE.
    """
    if not moe_cfg or not moe_cfg.get("enabled", False):
        return set()

    scope = moe_cfg.get("scope", "all_blocks")

    if scope == "all_blocks":
        return set(range(depth))

    if scope == "middle_blocks":
        if depth <= 2:
            return set()
        num_middle = max(1, depth // 3)
        start = (depth - num_middle) // 2
        return set(range(start, start + num_middle))

    # head_only or unknown scope: no encoder blocks get MoE
    return set()


def build_transformer_encoder(
    cfg: DictConfig,
    dim: int,
    rotary_emb: nn.Module | None = None,
    moe_cfg: dict[str, Any] | None = None,
    use_cls_token: bool = True,
    ffn_type: str = "standard",
    kan_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    """Build transformer encoder stack from Hydra config.

    Reads:
    - ``classifier.model.depth``, ``heads``, ``mlp_dim``, ``dropout``
    - ``classifier.model.norm.policy`` (Axis A)
    - ``classifier.model.norm.type`` (Axis B, default ``"layernorm"``)
    - ``classifier.model.attention.type`` (default ``"standard"``)
    - ``classifier.model.attention.norm`` (Axis C, default ``"none"``)
    - ``classifier.model.attention.diff_bias_mode`` (default ``"shared"``)
    - ``classifier.model.causal_attention``
    """
    model_cfg = cfg.classifier.model
    depth = model_cfg.depth
    num_heads = model_cfg.heads
    mlp_dim = model_cfg.mlp_dim
    dropout = model_cfg.get("dropout", 0.1)
    norm_policy = model_cfg.norm.get("policy", "post")
    block_norm_type = model_cfg.norm.get("type", "layernorm")
    causal_attention = model_cfg.get("causal_attention", False)

    attn_cfg = model_cfg.get("attention", {})
    attention_type = attn_cfg.get("type", "standard")
    attention_norm = attn_cfg.get("norm", "none")
    diff_bias_mode = attn_cfg.get("diff_bias_mode", "shared")

    return TransformerEncoder(
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        norm_policy=norm_policy,
        block_norm_type=block_norm_type,
        rotary_emb=rotary_emb,
        causal_attention=causal_attention,
        attention_type=attention_type,
        attention_norm=attention_norm,
        diff_bias_mode=diff_bias_mode,
        moe_cfg=moe_cfg,
        use_cls_token=use_cls_token,
        ffn_type=ffn_type,
        kan_cfg=kan_cfg,
    )
