from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        norm_policy: str = "post",
    ):
        """Initialize encoder block.

        Parameters
        ----------
        dim : int
            Model dimension
        num_heads : int
            Number of attention heads
        mlp_dim : int
            MLP hidden dimension
        dropout : float
            Dropout rate
        norm_policy : str
            Normalization policy: "pre" or "post"
        """
        super().__init__()
        self.norm_policy = norm_policy

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # Use [B, T, D] format
        )

        # MLP: Linear(dim → mlp_dim) → GELU → Dropout → Linear(mlp_dim → dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, T, D]
        mask : torch.Tensor, optional
            Attention mask [B, T] (True=valid, False=padding)
            Will be converted to key_padding_mask format

        Returns
        -------
        torch.Tensor
            Output tensor [B, T, D]
        """
        # Convert mask to key_padding_mask format
        # Loader provides mask with True=valid, False=padding
        # Attention expects key_padding_mask with True=pad, False=valid
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # [B, T] (True=pad, False=valid)

        # Attention block
        if self.norm_policy == "pre":
            # Pre-norm: LayerNorm before attention
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
            x = x + self.dropout(attn_out)
        else:
            # Post-norm: LayerNorm after attention
            attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
            x = self.norm1(x + self.dropout(attn_out))

        # MLP block
        if self.norm_policy == "pre":
            # Pre-norm: LayerNorm before MLP
            x_norm = self.norm2(x)
            mlp_out = self.mlp(x_norm)
            x = x + mlp_out
        else:
            # Post-norm: LayerNorm after MLP
            mlp_out = self.mlp(x)
            x = self.norm2(x + mlp_out)

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
    ):
        """Initialize encoder stack.

        Parameters
        ----------
        dim : int
            Model dimension
        depth : int
            Number of encoder blocks
        num_heads : int
            Number of attention heads
        mlp_dim : int
            MLP hidden dimension
        dropout : float
            Dropout rate
        norm_policy : str
            Normalization policy: "pre" or "post"
        """
        super().__init__()

        # Stack encoder blocks
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    norm_policy=norm_policy,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through all encoder blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, T, D]
        mask : torch.Tensor, optional
            Attention mask [B, T] (True=valid, False=padding)

        Returns
        -------
        torch.Tensor
            Output tensor [B, T, D]
        """
        for block in self.blocks:
            x = block(x, mask=mask)
        return x


def build_transformer_encoder(cfg: DictConfig, dim: int) -> nn.Module:
    """Build transformer encoder stack.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with classifier.model.* keys (depth, heads, mlp_dim, dropout, norm.policy)
    dim : int
        Model dimension

    Returns
    -------
    nn.Module
        Transformer encoder stack with specified depth and normalization policy
    """
    depth = cfg.classifier.model.depth
    num_heads = cfg.classifier.model.heads
    mlp_dim = cfg.classifier.model.mlp_dim
    dropout = cfg.classifier.model.get("dropout", 0.1)
    norm_policy = cfg.classifier.model.norm.get("policy", "post")

    return TransformerEncoder(
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        norm_policy=norm_policy,
    )
