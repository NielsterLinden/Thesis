"""Differential Attention (Ye et al., 2024).

Computes attention as the *difference* of two softmax maps to cancel
attention noise:

    A = softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d))

Reference: "Differential Transformer", Ye et al. (2024), arXiv:2410.05258.

Key design choices for this codebase:
- **Same ``forward`` signature** as ``MultiHeadAttention`` for drop-in use.
- ``num_heads`` and ``embed_dim`` are kept from config.  Each head internally
  splits Q/K into two sub-heads of ``sub_head_dim = head_dim // 2``.
  This requires ``embed_dim % (2 * num_heads) == 0``.
- **Lambda reparameterization** (paper eq. 2):
  ``lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init``.
- **Lambda scaling** ``(1 - lambda_init)`` is always applied per-head for
  gradient alignment (paper Appendix G).  This is a fixed architectural
  constant, NOT a normalization step.
- **Attention-internal norm** (Axis C) is independently configurable via
  ``attention_norm``: ``none`` / ``layernorm`` / ``rmsnorm``.
- **Bias modes** control how additive attention bias interacts with the two
  branches: ``none`` / ``shared`` / ``split``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis_ml.architectures.transformer_classifier.modules.attention._normalization import (
    build_norm,
)


def _lambda_init_fn(layer_idx: int) -> float:
    """Depth-dependent lambda initialisation (paper default)."""
    return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)


class DifferentialAttention(nn.Module):
    """Multi-head differential attention.

    Drop-in replacement for ``MultiHeadAttention`` with the same ``forward``
    signature.
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
        layer_idx: int = 0,
        diff_bias_mode: str = "shared",
    ):
        """Initialize differential attention.

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
            Optional NormFormer per-head scaling (Axis D).
        rotary_emb : nn.Module | None
            Optional ``RotaryEmbedding`` for rotary positional encoding.
            Must be built with ``head_dim = sub_head_dim`` (half of the
            standard head dimension).
        attention_norm : str
            Attention-internal norm (Axis C): ``"none"`` / ``"layernorm"`` /
            ``"rmsnorm"``.
        layer_idx : int
            Zero-based layer index, used for depth-dependent ``lambda_init``.
        diff_bias_mode : str
            How additive attention bias is handled:

            - ``"none"``: bias is ignored.
            - ``"shared"``: same bias added to both branches.
            - ``"split"``: tuple → ``[0]`` for branch 1, ``[1]`` for branch 2;
              single tensor → branch 1 only.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for differential attention " f"(embed_dim={embed_dim}, num_heads={num_heads})")
        if diff_bias_mode not in ("none", "shared", "split"):
            raise ValueError(f"diff_bias_mode must be 'none', 'shared', or 'split'; " f"got {diff_bias_mode!r}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = head_dim
        self.sub_head_dim = head_dim // 2
        self.scaling = 1.0 / math.sqrt(self.sub_head_dim)
        self.diff_bias_mode = diff_bias_mode

        # Combined Q/K/V projection (same layout as standard attention)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # NormFormer head-wise scaling (Axis D)
        self.head_scales = head_scales

        # Rotary positional embeddings (expects sub_head_dim)
        self.rotary_emb = rotary_emb

        # Lambda reparameterization (paper eq. 2)
        self.lambda_init = _lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.empty(self.sub_head_dim))
        self.lambda_k1 = nn.Parameter(torch.empty(self.sub_head_dim))
        self.lambda_q2 = nn.Parameter(torch.empty(self.sub_head_dim))
        self.lambda_k2 = nn.Parameter(torch.empty(self.sub_head_dim))

        # Attention-internal normalization (Axis C)
        self.attn_norm = build_norm(attention_norm, self.head_dim) if attention_norm != "none" else None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.lambda_q1, mean=0.0, std=0.1)
        nn.init.normal_(self.lambda_k1, mean=0.0, std=0.1)
        nn.init.normal_(self.lambda_q2, mean=0.0, std=0.1)
        nn.init.normal_(self.lambda_k2, mean=0.0, std=0.1)

    def _compute_lambda(self) -> torch.Tensor:
        """Compute the learnable lambda scalar (paper eq. 2)."""
        return torch.exp(torch.dot(self.lambda_q1, self.lambda_k1)) - torch.exp(torch.dot(self.lambda_q2, self.lambda_k2)) + self.lambda_init

    def _resolve_bias_pair(
        self,
        attention_bias: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return ``(bias_1, bias_2)`` based on ``diff_bias_mode``."""
        if self.diff_bias_mode == "none" or attention_bias is None:
            return None, None

        if isinstance(attention_bias, tuple):
            raw_1, raw_2 = attention_bias
        else:
            raw_1 = attention_bias
            raw_2 = None

        if self.diff_bias_mode == "shared":
            # Shared: same bias for both branches (sum tuple if given)
            combined = raw_1
            if raw_2 is not None:
                combined = raw_1 + raw_2 if raw_1 is not None else raw_2
            return combined, combined

        # split: branch 1 gets raw_1, branch 2 gets raw_2
        return raw_1, raw_2

    @staticmethod
    def _prepare_bias(bias: torch.Tensor | None, num_heads: int) -> torch.Tensor | None:
        """Validate and broadcast a single bias tensor to 4D."""
        if bias is None:
            return None
        if bias.ndim == 3:
            return bias.unsqueeze(1)
        if bias.ndim == 4:
            if bias.size(1) != num_heads:
                raise ValueError(f"attention_bias has shape {bias.shape}; " f"when 4D, size(1) must be num_heads={num_heads}")
            return bias
        raise ValueError(f"attention_bias must be 3D or 4D, got ndim={bias.ndim}")

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
        """Forward pass (same signature as ``MultiHeadAttention``).

        Parameters
        ----------
        query, key, value : torch.Tensor
            ``[B, T, D]`` if ``batch_first=True``.
        key_padding_mask : torch.Tensor, optional
            ``[B, T_k]`` where ``True=pad``.
        need_weights : bool
            If True, return differential attention weights ``A1 - lambda*A2``.
        attn_mask : torch.Tensor, optional
            Boolean or additive attention mask.
        attention_bias : Tensor | tuple[Tensor, Tensor] | None
            Additive bias.  Handling depends on ``diff_bias_mode``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            ``(output, attn_weights)``.
        """
        bias_1, bias_2 = self._resolve_bias_pair(attention_bias)

        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Project Q, K, V
        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)

        q = F.linear(query, q_weight, q_bias)  # [B, T_q, D]
        k = F.linear(key, k_weight, k_bias)  # [B, T_k, D]
        v = F.linear(value, v_weight, v_bias)  # [B, T_v, D]

        # Reshape to [B, H, T, head_dim]
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Split Q, K into two sub-heads: [B, H, T, sub_head_dim] each
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        # Apply RoPE to each sub-head pair separately
        if self.rotary_emb is not None:
            q1, k1 = self.rotary_emb(q1, k1)
            q2, k2 = self.rotary_emb(q2, k2)

        # Attention scores for both branches: [B, H, T_q, T_k]
        logits_1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scaling
        logits_2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scaling

        # Prepare and add biases
        bias_1 = self._prepare_bias(bias_1, self.num_heads)
        bias_2 = self._prepare_bias(bias_2, self.num_heads)
        if bias_1 is not None:
            logits_1 = logits_1 + bias_1
        if bias_2 is not None:
            logits_2 = logits_2 + bias_2

        # Padding mask (same for both branches)
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).bool()
            logits_1 = logits_1.masked_fill(pad_mask, float("-inf"))
            logits_2 = logits_2.masked_fill(pad_mask, float("-inf"))

        # Attention mask (same for both branches)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                logits_1 = logits_1.masked_fill(attn_mask, float("-inf"))
                logits_2 = logits_2.masked_fill(attn_mask, float("-inf"))
            else:
                logits_1 = logits_1 + attn_mask
                logits_2 = logits_2 + attn_mask

        # Softmax + dropout for both branches
        a1 = F.softmax(logits_1, dim=-1)
        a2 = F.softmax(logits_2, dim=-1)
        a1 = F.dropout(a1, p=self.dropout, training=self.training)
        a2 = F.dropout(a2, p=self.dropout, training=self.training)

        # Differential combination
        lam = self._compute_lambda()
        diff_weights = a1 - lam * a2  # [B, H, T_q, T_k]

        # Weighted sum with V: [B, H, T_q, head_dim]
        attn_output = torch.matmul(diff_weights, v)

        # --- Per-head post-processing ---
        # Axis C: attention-internal normalization
        if self.attn_norm is not None:
            attn_output = self.attn_norm(attn_output)

        # Lambda scaling (always applied, gradient alignment)
        attn_output = attn_output * (1.0 - self.lambda_init)

        # Axis D: NormFormer head-wise scaling
        if self.head_scales is not None:
            attn_output = attn_output * self.head_scales.view(1, self.num_heads, 1, 1)

        # Merge heads: [B, T_q, D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, tgt_len, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        attn_weights_out = diff_weights if need_weights else None

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, attn_weights_out
