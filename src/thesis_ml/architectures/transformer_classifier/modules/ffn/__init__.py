"""Feed-forward network modules for transformer encoder blocks.

Provides a ``build_ffn`` factory that returns the appropriate FFN variant
based on config:

- ``StandardFFN`` — the default two-layer MLP (with optional NormFormer
  mid-norm).
- ``MoEFFN`` — sparse Mixture-of-Experts replacement (when enabled via
  config).
- ``KANFFN`` — Kolmogorov-Arnold Network replacement (when
  ``ffn_type="kan"``).
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from thesis_ml.architectures.transformer_classifier.modules.ffn.standard import StandardFFN

__all__ = ["StandardFFN", "build_ffn"]


def build_ffn(
    dim: int,
    mlp_dim: int,
    dropout: float = 0.1,
    norm_policy: str = "pre",
    block_norm_type: str = "layernorm",
    moe_cfg: dict[str, Any] | None = None,
    use_cls_token: bool = True,
    ffn_type: str = "standard",
    kan_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    """Build a feed-forward module based on config.

    Parameters
    ----------
    dim : int
        Model dimension.
    mlp_dim : int
        Hidden dimension.
    dropout : float
        Dropout rate.
    norm_policy : str
        ``"pre"`` / ``"post"`` / ``"normformer"``.
    block_norm_type : str
        ``"layernorm"`` / ``"rmsnorm"``.
    moe_cfg : dict | None
        MoE configuration dict.  When ``None`` or ``moe_cfg["enabled"]`` is
        falsy, MoE is not used.
    use_cls_token : bool
        Whether the model uses a CLS token (passed to MoE for event-level
        routing).
    ffn_type : str
        ``"standard"`` | ``"kan"``.  MoE takes priority when enabled.
    kan_cfg : dict | None
        Global KAN hyperparameters.  Only used when ``ffn_type="kan"``.

    Returns
    -------
    nn.Module
        A ``StandardFFN``, ``MoEFFN``, or ``KANFFN`` instance.
    """
    # MoE takes priority (backward compat)
    if moe_cfg and moe_cfg.get("enabled", False):
        from thesis_ml.architectures.transformer_classifier.modules.ffn.moe import MoEFFN

        return MoEFFN(
            dim=dim,
            mlp_dim=mlp_dim,
            num_experts=moe_cfg.get("num_experts", 4),
            top_k=moe_cfg.get("top_k", 1),
            routing_level=moe_cfg.get("routing_level", "token"),
            dropout=dropout,
            norm_policy=norm_policy,
            block_norm_type=block_norm_type,
            use_cls_token=use_cls_token,
            noisy_gating=moe_cfg.get("noisy_gating", False),
        )

    if ffn_type == "kan":
        from thesis_ml.architectures.transformer_classifier.modules.ffn.kan import KANFFN

        _ffn_kan = (kan_cfg or {}).get("ffn", {}) if isinstance(kan_cfg, dict) else {}
        return KANFFN(
            dim=dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            norm_policy=norm_policy,
            block_norm_type=block_norm_type,
            variant=str(_ffn_kan.get("variant", "hybrid")),
            bottleneck_dim=_ffn_kan.get("bottleneck_dim"),
            grid_size=int((kan_cfg or {}).get("grid_size", 5)),
            spline_order=int((kan_cfg or {}).get("spline_order", 3)),
            grid_range=tuple(float(v) for v in (kan_cfg or {}).get("grid_range", [-2.0, 2.0])),
        )

    return StandardFFN(
        dim=dim,
        mlp_dim=mlp_dim,
        dropout=dropout,
        norm_policy=norm_policy,
        block_norm_type=block_norm_type,
    )
