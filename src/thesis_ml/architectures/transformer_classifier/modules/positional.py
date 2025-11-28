from __future__ import annotations

import math

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf


class NonePositional(nn.Module):
    """No positional encoding (identity)."""

    def __init__(self, dim: int, max_seq_length: int):
        super().__init__()
        # No parameters needed for identity

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply positional encoding (identity).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, T, D]
        mask : torch.Tensor, optional
            Attention mask [B, T] (True=valid, False=padding). Not used for identity.

        Returns
        -------
        torch.Tensor
            Output tensor [B, T, D] (unchanged)
        """
        return x


class SinusoidalPositional(nn.Module):
    """Sinusoidal positional encoding from "Attention is All You Need"."""

    def __init__(self, dim: int, max_seq_length: int, dim_mask: torch.Tensor | None = None):
        """Initialize sinusoidal positional encoding.

        Parameters
        ----------
        dim : int
            Dimension of positional encoding
        max_seq_length : int
            Maximum sequence length
        dim_mask : torch.Tensor, optional
            Boolean mask [dim] indicating which dimensions to apply PE to.
            If None, applies PE to all dimensions.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length

        # Handle odd D: slice pairs up to the largest even ≤ D
        # We'll create PE for even dimensions, then pad if needed
        dim_even = dim // 2 * 2  # Largest even ≤ dim

        # Create positional encoding table
        pe = torch.zeros(max_seq_length, dim_even)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)  # [max_seq_length, 1]

        # Compute div_term: 10000^(2i/dim) for i in [0, dim_even//2)
        div_term = torch.exp(torch.arange(0, dim_even, 2).float() * (-math.log(10000.0) / dim_even))  # [dim_even//2]

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_seq_length, dim_even//2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_seq_length, dim_even//2]

        # If dim is odd, pad with zeros
        if dim > dim_even:
            pe = torch.cat([pe, torch.zeros(max_seq_length, dim - dim_even)], dim=1)

        # Register as buffer (not a parameter, but part of model state)
        # persistent=False means it won't be saved in state_dict (we'll recompute if needed)
        self.register_buffer("pe", pe, persistent=False)

        # Register dim_mask as buffer if provided
        if dim_mask is not None:
            if dim_mask.shape != (dim,):
                raise ValueError(f"dim_mask must have shape [dim]={dim}, got {dim_mask.shape}")
            if dim_mask.dtype != torch.bool:
                raise ValueError(f"dim_mask must be bool tensor, got {dim_mask.dtype}")
            self.register_buffer("dim_mask", dim_mask)
        else:
            self.dim_mask = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply sinusoidal positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, T, D]
        mask : torch.Tensor, optional
            Attention mask [B, T] (True=valid, False=padding). Not used.

        Returns
        -------
        torch.Tensor
            Output tensor [B, T, D] with positional encoding added
        """
        B, T, D = x.shape
        pe = self.pe[:T].to(x.device)  # [T, D]

        if self.dim_mask is None:
            return x + pe.unsqueeze(0)
        else:
            x = x.clone()  # Avoid in-place modification
            x[..., self.dim_mask] = x[..., self.dim_mask] + pe[..., self.dim_mask].unsqueeze(0)
            return x


class LearnedPositional(nn.Module):
    """Learned positional encoding with trainable parameters."""

    def __init__(self, dim: int, max_seq_length: int, dim_mask: torch.Tensor | None = None):
        """Initialize learned positional encoding.

        Parameters
        ----------
        dim : int
            Dimension of positional encoding
        max_seq_length : int
            Maximum sequence length
        dim_mask : torch.Tensor, optional
            Boolean mask [dim] indicating which dimensions to apply PE to.
            If None, applies PE to all dimensions.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length

        # Learnable positional embeddings, initialized with small random values
        self.pe = nn.Parameter(torch.randn(max_seq_length, dim) * 0.02)

        # Register dim_mask as buffer if provided
        if dim_mask is not None:
            if dim_mask.shape != (dim,):
                raise ValueError(f"dim_mask must have shape [dim]={dim}, got {dim_mask.shape}")
            if dim_mask.dtype != torch.bool:
                raise ValueError(f"dim_mask must be bool tensor, got {dim_mask.dtype}")
            self.register_buffer("dim_mask", dim_mask)
        else:
            self.dim_mask = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply learned positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, T, D]
        mask : torch.Tensor, optional
            Attention mask [B, T] (True=valid, False=padding). Not used.

        Returns
        -------
        torch.Tensor
            Output tensor [B, T, D] with positional encoding added
        """
        B, T, D = x.shape
        pe = self.pe[:T]  # [T, D]

        if self.dim_mask is None:
            return x + pe.unsqueeze(0)
        else:
            x = x.clone()  # Avoid in-place modification
            x[..., self.dim_mask] = x[..., self.dim_mask] + pe[..., self.dim_mask].unsqueeze(0)
            return x


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor [..., head_dim]

    Returns
    -------
    torch.Tensor
        Rotated tensor [..., head_dim]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor [B, heads, T, head_dim]
    k : torch.Tensor
        Key tensor [B, heads, T, head_dim]
    cos : torch.Tensor
        Cosine values [1, 1, T, head_dim]
    sin : torch.Tensor
        Sine values [1, 1, T, head_dim]

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Rotated (q, k) tensors
    """
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for attention.

    Unlike additive positional encodings, RoPE applies rotations to query and key
    vectors inside the attention mechanism. This module computes sin/cos values
    dynamically based on sequence length.

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        """Initialize rotary embedding.

        Parameters
        ----------
        head_dim : int
            Dimension per attention head (must be even)
        base : float
            Base for computing inverse frequencies (default 10000.0)
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        self.head_dim = head_dim
        self.base = base

        # Precompute inverse frequencies: 1 / (base^(2i/d)) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for sin/cos values (will be computed on first forward)
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._seq_len_cached: int = 0

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Update cached sin/cos values if sequence length changed.

        Parameters
        ----------
        seq_len : int
            Current sequence length
        device : torch.device
            Device to create tensors on
        dtype : torch.dtype
            Data type for tensors
        """
        if seq_len <= self._seq_len_cached and self._cos_cached is not None:
            return

        self._seq_len_cached = seq_len

        # Compute position indices [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # Outer product: [seq_len] x [head_dim/2] -> [seq_len, head_dim/2]
        freqs = torch.outer(t, self.inv_freq.to(device))

        # Duplicate for full head_dim: [seq_len, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # Cache with shape [1, 1, seq_len, head_dim] for broadcasting
        self._cos_cached = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)
        self._sin_cached = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor [B, heads, T, head_dim]
        k : torch.Tensor
            Key tensor [B, heads, T, head_dim]
        offset : int
            Position offset (for incremental decoding, default 0)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Rotated (q, k) tensors with same shape as input
        """
        seq_len = q.size(2) + offset

        # Update cache if needed
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)

        # Slice cached values for current sequence
        cos = self._cos_cached[:, :, offset:seq_len, :]
        sin = self._sin_cached[:, :, offset:seq_len, :]

        return apply_rotary_pos_emb(q, k, cos, sin)


def parse_dim_mask(
    config_value: list[int] | list[str] | ListConfig | DictConfig | None,
    dim: int,
    feature_map: dict[str, list[int]],
) -> torch.Tensor | None:
    """Parse dimension mask from config value.

    Parameters
    ----------
    config_value : list[int] | list[str] | ListConfig | DictConfig | None
        Config value: None (all dimensions), list of ints (explicit indices),
        or list of strings (semantic names mapped via feature_map).
        Can be Python list, OmegaConf ListConfig, or DictConfig (will be converted).
    dim : int
        Total dimension size
    feature_map : dict[str, list[int]]
        Mapping from semantic names to dimension indices

    Returns
    -------
    torch.Tensor | None
        Boolean mask [dim] with True for dimensions to apply PE, or None for all dimensions

    Raises
    ------
    ValueError
        If config_value contains invalid names/indices or mixes ints and strings
    """
    if config_value is None:
        return None

    # Convert OmegaConf types to Python native types
    if isinstance(config_value, ListConfig | DictConfig):
        config_value = OmegaConf.to_container(config_value, resolve=True)

    # Handle nested dict structures from config groups with @package _global_
    # Extract list from: {'classifier': {'model': {'positional_dim_mask': [...]}}}
    if isinstance(config_value, dict):
        # Recursively find the first list value
        def find_list(d):
            if isinstance(d, list):
                return d
            if isinstance(d, dict):
                for v in d.values():
                    result = find_list(v)
                    if result is not None:
                        return result
            return None

        found = find_list(config_value)
        if found is not None:
            config_value = found

    if not isinstance(config_value, list):
        raise ValueError(f"positional_dim_mask must be None, list, ListConfig, or DictConfig containing a list, " f"got {type(config_value)}: {config_value}")

    if len(config_value) == 0:
        raise ValueError("positional_dim_mask cannot be empty list")

    # Check if mixing ints and strings
    has_ints = any(isinstance(item, int) for item in config_value)
    has_strings = any(isinstance(item, str) for item in config_value)

    if has_ints and has_strings:
        raise ValueError("positional_dim_mask cannot mix integers and strings. " "Use either all integers (explicit indices) or all strings (semantic names).")

    # Collect dimension indices
    dim_indices: set[int] = set()

    if has_ints:
        # Direct integer indices
        for idx in config_value:
            if not isinstance(idx, int):
                raise ValueError(f"All items must be int when using explicit indices, got {type(idx)}")
            if idx < 0 or idx >= dim:
                raise ValueError(f"Dimension index {idx} out of range [0, {dim})")
            dim_indices.add(idx)
    else:
        # String semantic names
        for name in config_value:
            if not isinstance(name, str):
                raise ValueError(f"All items must be str when using semantic names, got {type(name)}")
            if name not in feature_map:
                raise ValueError(f"Unknown feature name '{name}'. " f"Available names: {list(feature_map.keys())}")
            dim_indices.update(feature_map[name])

    # Create boolean mask
    mask = torch.zeros(dim, dtype=torch.bool)
    mask[list(dim_indices)] = True
    return mask


def get_positional_encoding(name: str, dim: int, max_seq_length: int, dim_mask: torch.Tensor | None = None, **kwargs) -> nn.Module:
    """Get additive positional encoding module by name.

    Parameters
    ----------
    name : str
        Encoding type: "none", "sinusoidal", or "learned"
        Note: "rotary" is handled separately via RotaryEmbedding in attention.
    dim : int
        Model dimension
    max_seq_length : int
        Maximum sequence length
    dim_mask : torch.Tensor, optional
        Boolean mask [dim] indicating which dimensions to apply PE to.
        If None, applies PE to all dimensions.
    **kwargs
        Additional arguments passed to constructor

    Returns
    -------
    nn.Module
        Positional encoding module
    """
    if name == "none":
        return NonePositional(dim, max_seq_length, **kwargs)
    elif name == "sinusoidal":
        return SinusoidalPositional(dim, max_seq_length, dim_mask=dim_mask, **kwargs)
    elif name == "learned":
        return LearnedPositional(dim, max_seq_length, dim_mask=dim_mask, **kwargs)
    elif name == "rotary":
        # Rotary is handled specially - return None here, actual RoPE is in attention
        return NonePositional(dim, max_seq_length, **kwargs)
    else:
        raise ValueError(f"Unknown positional encoding: {name}")
