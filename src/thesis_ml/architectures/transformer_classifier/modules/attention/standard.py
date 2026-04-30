"""Standard multi-head attention with optional per-head normalization.

Supports:
- Attention-internal normalization (Axis C): ``none`` / ``layernorm`` /
  ``rmsnorm``, applied per-head after ``attn_weights @ V`` and before head
  merge.
- NormFormer head-wise scaling (Axis D): via ``head_scales`` parameter.
- Rotary positional embeddings (RoPE).
- Additive attention bias (physics-informed pairwise bias).  Accepts both
  single tensors and ``(branch_1, branch_2)`` tuples; tuples are summed
  transparently for backward compatibility.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis_ml.architectures.transformer_classifier.modules.attention._normalization import (
    build_norm,
)


class MultiHeadAttention(nn.Module):
    """Custom multi-head attention with optional per-head normalization and NormFormer scaling.

    Matches ``nn.MultiheadAttention`` behavior exactly when all optional
    features are disabled (``head_scales=None``, ``rotary_emb=None``,
    ``attention_norm="none"``).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        head_scales: nn.Parameter | None = None,
        rotary_emb: nn.Module | None = None,
        attention_norm: str = "none",
    ):
        """Initialize multi-head attention.

        Parameters
        ----------
        embed_dim : int
            Total dimension of the model.
        num_heads : int
            Number of parallel attention heads.
        dropout : float
            Dropout probability for attention weights.
        batch_first : bool
            If True, input/output tensors are ``[B, T, D]``.
        head_scales : nn.Parameter | None
            Optional learnable per-head scaling factors ``[num_heads]``
            (NormFormer, Axis D).
        rotary_emb : nn.Module | None
            Optional ``RotaryEmbedding`` module for rotary positional encoding.
        attention_norm : str
            Attention-internal normalization (Axis C).  One of ``"none"``,
            ``"layernorm"``, ``"rmsnorm"``.  Applied per-head after
            ``attn_weights @ V``, before head merge.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        # Combined Q/K/V projection (matching PyTorch's nn.MultiheadAttention)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # NormFormer head-wise scaling (Axis D)
        self.head_scales = head_scales

        # Rotary positional embeddings
        self.rotary_emb = rotary_emb

        # Attention-internal normalization (Axis C)
        self.attn_norm = build_norm(attention_norm, self.head_dim) if attention_norm != "none" else None

        # Initialize weights (matching PyTorch defaults)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _resolve_bias(
        attention_bias: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor | None:
        """Collapse a possible ``(branch_1, branch_2)`` tuple into a single tensor."""
        if attention_bias is None:
            return None
        if isinstance(attention_bias, tuple):
            b1, b2 = attention_bias
            if b1 is None and b2 is None:
                return None
            if b1 is None:
                return b2
            if b2 is None:
                return b1
            return b1 + b2
        return attention_bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        query : torch.Tensor
            ``[B, T_q, D]`` if ``batch_first=True``, else ``[T_q, B, D]``.
        key : torch.Tensor
            ``[B, T_k, D]`` if ``batch_first=True``, else ``[T_k, B, D]``.
        value : torch.Tensor
            ``[B, T_v, D]`` if ``batch_first=True``, else ``[T_v, B, D]``.
        key_padding_mask : torch.Tensor, optional
            ``[B, T_k]`` where ``True=pad``, ``False=valid``.
        need_weights : bool
            If True, return attention weights.
        attn_mask : torch.Tensor, optional
            Attention mask (not commonly used, kept for API compatibility).
        attention_bias : Tensor | tuple[Tensor, Tensor] | None
            Additive bias added to attention logits before softmax.  Accepts:

            - ``[B, T_q, T_k]``: broadcast across heads.
            - ``[B, num_heads, T_q, T_k]``: per-head.
            - ``(bias_1, bias_2)`` tuple: summed into a single tensor
              (backward-compatible with split-bias pipeline).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            ``(output, attn_weights)`` where output is ``[B, T_q, D]`` and
            ``attn_weights`` is ``[B, H, T_q, T_k]`` if ``need_weights``.
        """
        attention_bias = self._resolve_bias(attention_bias)

        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Project Q, K, V
        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)

        q = F.linear(query, q_weight, q_bias)
        k = F.linear(key, k_weight, k_bias)
        v = F.linear(value, v_weight, v_bias)

        # Reshape to [B, num_heads, T, head_dim]
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Rotary positional embeddings
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)

        # Attention scores: [B, H, T_q, T_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Additive attention bias
        if attention_bias is not None:
            if attention_bias.ndim == 3:
                attention_bias = attention_bias.unsqueeze(1)
            elif attention_bias.ndim == 4:
                if attention_bias.size(1) != self.num_heads:
                    raise ValueError(f"attention_bias has shape {attention_bias.shape}; " f"when 4D, size(1) must be num_heads={self.num_heads}")
            else:
                raise ValueError(f"attention_bias must be 3D or 4D, got ndim={attention_bias.ndim}")
            b_last = attention_bias.shape[-2:]
            a_last = attn_scores.shape[-2:]
            if b_last != a_last:
                raise RuntimeError(
                    f"attention_bias spatial dims {tuple(attention_bias.shape)} last-2={b_last} "
                    f"do not match attention scores {tuple(attn_scores.shape)} last-2={a_last}. "
                    "Often a MET/globals vs model num_met_tokens or per-run dataloader mismatch."
                )
            attn_scores = attn_scores + attention_bias

        # Padding mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf"))

        # Attention mask
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else attn_scores + attn_mask

        # Softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Weighted sum: [B, H, T_q, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # --- Per-head post-processing ---
        # Axis C: attention-internal normalization
        if self.attn_norm is not None:
            attn_output = self.attn_norm(attn_output)

        # Axis D: NormFormer head-wise scaling
        if self.head_scales is not None:
            attn_output = attn_output * self.head_scales.view(1, self.num_heads, 1, 1)

        # Merge heads: [B, T_q, D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, tgt_len, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        attn_weights_out = attn_weights if need_weights else None

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, attn_weights_out
