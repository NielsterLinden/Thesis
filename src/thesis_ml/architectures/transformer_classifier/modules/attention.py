from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Custom multi-head attention that supports head-wise scaling for NormFormer.

    Matches nn.MultiheadAttention behavior exactly when head_scales=None and rotary_emb=None.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        head_scales: nn.Parameter | None = None,
        rotary_emb: nn.Module | None = None,
    ):
        """Initialize multi-head attention.

        Parameters
        ----------
        embed_dim : int
            Total dimension of the model
        num_heads : int
            Number of parallel attention heads
        dropout : float
            Dropout probability for attention weights
        batch_first : bool
            If True, input/output tensors are [B, T, D]. If False, [T, B, D]
        head_scales : nn.Parameter | None
            Optional learnable per-head scaling factors [num_heads].
            If None, no scaling applied (matches nn.MultiheadAttention exactly)
        rotary_emb : nn.Module | None
            Optional RotaryEmbedding module for rotary positional encoding.
            If provided, applies RoPE to Q and K after projection.
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

        # Head-wise scaling (for NormFormer)
        self.head_scales = head_scales

        # Rotary positional embeddings (for RoPE)
        self.rotary_emb = rotary_emb

        # Initialize weights (matching PyTorch defaults)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor [B, T_q, D] if batch_first=True, else [T_q, B, D]
        key : torch.Tensor
            Key tensor [B, T_k, D] if batch_first=True, else [T_k, B, D]
        value : torch.Tensor
            Value tensor [B, T_v, D] if batch_first=True, else [T_v, B, D]
        key_padding_mask : torch.Tensor, optional
            Mask [B, T_k] where True=pad, False=valid
        need_weights : bool
            If True, return attention weights
        attn_mask : torch.Tensor, optional
            Attention mask (not commonly used, kept for API compatibility)
        attention_bias : torch.Tensor, optional
            Additive bias added to attention logits before softmax. Expected shapes:
            - [B, T_q, T_k]: broadcast to [B, 1, T_q, T_k] (shared across heads).
            - [B, num_heads, T_q, T_k]: used directly. Applied after (QK^T)/sqrt(d)
            and before key_padding_mask and attn_mask so padding remains suppressed.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            (output, attn_weights) where output is [B, T_q, D] and
            attn_weights is [B, num_heads, T_q, T_k] if need_weights=True, else None
        """
        if not self.batch_first:
            # Convert to batch_first for internal processing
            query = query.transpose(0, 1)  # [T_q, B, D] -> [B, T_q, D]
            key = key.transpose(0, 1)  # [T_k, B, D] -> [B, T_k, D]
            value = value.transpose(0, 1)  # [T_v, B, D] -> [B, T_v, D]

        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Project Q, K, V separately (matching PyTorch's nn.MultiheadAttention)
        # Split weight and bias into three parts: [Q, K, V]
        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)  # Each: [D, D]
        q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)  # Each: [D]

        # Project each separately
        q = F.linear(query, q_weight, q_bias)  # [B, T_q, D]
        k = F.linear(key, k_weight, k_bias)  # [B, T_k, D]
        v = F.linear(value, v_weight, v_bias)  # [B, T_v, D]

        # Reshape to separate heads: [B, T, D] -> [B, T, num_heads, head_dim]
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim)

        # Transpose for attention computation: [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)  # [B, num_heads, T_q, head_dim]
        k = k.transpose(1, 2)  # [B, num_heads, T_k, head_dim]
        v = v.transpose(1, 2)  # [B, num_heads, T_v, head_dim]

        # Apply rotary positional embeddings to Q and K (if enabled)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)

        # Compute attention scores: [B, num_heads, T_q, T_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Add optional attention bias (e.g. physics-informed pairwise bias)
        if attention_bias is not None:
            if attention_bias.ndim == 3:
                # [B, T_q, T_k] -> [B, 1, T_q, T_k]
                attention_bias = attention_bias.unsqueeze(1)
            elif attention_bias.ndim == 4:
                if attention_bias.size(1) != self.num_heads:
                    raise ValueError(f"attention_bias has shape {attention_bias.shape}; when 4D, " f"size(1) must be num_heads={self.num_heads}")
            else:
                raise ValueError(f"attention_bias must be 3D [B, T_q, T_k] or 4D [B, num_heads, T_q, T_k], got ndim={attention_bias.ndim}")
            attn_scores = attn_scores + attention_bias

        # Apply key_padding_mask (True=pad, False=valid)
        if key_padding_mask is not None:
            # key_padding_mask: [B, T_k], need to expand to [B, 1, 1, T_k]
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf"))

        # Apply attn_mask if provided (for future compatibility)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else attn_scores + attn_mask

        # Softmax and dropout on attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values: [B, num_heads, T_q, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Apply head-wise scaling (for NormFormer)
        if self.head_scales is not None:
            # head_scales: [num_heads] -> [1, num_heads, 1, 1] for broadcasting
            attn_output = attn_output * self.head_scales.view(1, self.num_heads, 1, 1)

        # Transpose back: [B, num_heads, T_q, head_dim] -> [B, T_q, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Concatenate heads: [B, T_q, num_heads, head_dim] -> [B, T_q, D]
        attn_output = attn_output.view(batch_size, tgt_len, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        # Return attention weights if requested
        attn_weights_out = attn_weights if need_weights else None

        if not self.batch_first:
            # Convert back if needed
            output = output.transpose(0, 1)  # [B, T_q, D] -> [T_q, B, D]

        return output, attn_weights_out
