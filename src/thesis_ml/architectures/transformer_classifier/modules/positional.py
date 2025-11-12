from __future__ import annotations

import math

import torch
import torch.nn as nn


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

    def __init__(self, dim: int, max_seq_length: int):
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply sinusoidal positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, T, D] or [B, T+1, D] if CLS token is prepended
        mask : torch.Tensor, optional
            Attention mask [B, T] or [B, T+1] (True=valid, False=padding). Used to determine sequence length.

        Returns
        -------
        torch.Tensor
            Output tensor [B, T, D] or [B, T+1, D] with positional encoding added
        """
        seq_len = x.size(1)

        # Check if CLS token is prepended (sequence length > max_seq_length)
        # If so, skip positional encoding for the first token (CLS)
        if seq_len > self.max_seq_length:
            # CLS token is prepended: apply PE to tokens 1:seq_len
            # PE indices should be 0:seq_len-1 (for tokens after CLS)
            pos_enc = self.pe[: seq_len - 1].to(x.device)  # [seq_len-1, dim]
            # Apply PE to all tokens except the first (CLS)
            x[:, 1:] = x[:, 1:] + pos_enc.unsqueeze(0)  # [B, seq_len-1, dim]
        else:
            # No CLS token: apply PE to all tokens
            pos_enc = self.pe[:seq_len].to(x.device)  # [seq_len, dim]
            x = x + pos_enc.unsqueeze(0)  # [B, seq_len, dim]

        return x


class RotaryPositional(nn.Module):
    """Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_length: int):
        super().__init__()
        # Stub for now - requires attention redesign
        self.dim = dim
        self.max_seq_length = max_seq_length

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply rotary positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, T, D]
        mask : torch.Tensor, optional
            Attention mask [B, T] (True=valid, False=padding)

        Returns
        -------
        torch.Tensor
            Output tensor [B, T, D] with rotary encoding applied
        """
        raise NotImplementedError("RotaryPositional not yet implemented. " "Requires attention redesign to apply rotations to query/key vectors.")


def get_positional_encoding(name: str, dim: int, max_seq_length: int, **kwargs) -> nn.Module:
    """Get positional encoding module by name.

    Parameters
    ----------
    name : str
        Encoding type: "none", "sinusoidal", or "rotary"
    dim : int
        Model dimension
    max_seq_length : int
        Maximum sequence length
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
        return SinusoidalPositional(dim, max_seq_length, **kwargs)
    elif name == "rotary":
        return RotaryPositional(dim, max_seq_length, **kwargs)
    else:
        raise ValueError(f"Unknown positional encoding: {name}")
