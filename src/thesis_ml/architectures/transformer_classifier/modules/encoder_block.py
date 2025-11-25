from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from thesis_ml.architectures.transformer_classifier.modules.attention import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        norm_policy: str = "post",
        rotary_emb: nn.Module | None = None,
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
            Normalization policy: "pre", "post", or "normformer"
        rotary_emb : nn.Module | None
            Optional RotaryEmbedding module for rotary positional encoding.
            Shared across all encoder blocks.
        """
        super().__init__()
        self.norm_policy = norm_policy

        # Head-wise scaling for NormFormer (only used when norm_policy == "normformer")
        head_scales = None
        if norm_policy == "normformer":
            head_scales = nn.Parameter(torch.ones(num_heads))

        # Multi-head self-attention (custom implementation to support head scaling and RoPE)
        self.attention = MultiHeadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # Use [B, T, D] format
            head_scales=head_scales,
            rotary_emb=rotary_emb,
        )

        # MLP structure depends on normalization policy
        if norm_policy == "normformer":
            # For NormFormer: need to insert LayerNorm after first linear
            self.mlp_fc1 = nn.Linear(dim, mlp_dim)
            self.norm_mlp_mid = nn.LayerNorm(mlp_dim)
            self.activation = nn.GELU()
            self.dropout_mlp = nn.Dropout(dropout)
            self.mlp_fc2 = nn.Linear(mlp_dim, dim)
            self.mlp = None  # Not used for NormFormer
        else:
            # For pre/post norm: use Sequential as before
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim),
                nn.Dropout(dropout),
            )
            # Not used for pre/post norm
            self.mlp_fc1 = None
            self.norm_mlp_mid = None
            self.activation = None
            self.dropout_mlp = None
            self.mlp_fc2 = None

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Additional LayerNorm after attention for NormFormer
        if norm_policy == "normformer":
            self.norm_attn_out = nn.LayerNorm(dim)
        else:
            self.norm_attn_out = None

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
        elif self.norm_policy == "post":
            # Post-norm: LayerNorm after attention
            attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
            x = self.norm1(x + self.dropout(attn_out))
        elif self.norm_policy == "normformer":
            # NormFormer: Pre-norm before attention, then norm after attention
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
            attn_out = self.norm_attn_out(attn_out)  # LayerNorm after attention
            x = x + self.dropout(attn_out)
        else:
            raise ValueError(f"Unknown norm_policy: {self.norm_policy}")

        # MLP block
        if self.norm_policy == "pre":
            # Pre-norm: LayerNorm before MLP
            x_norm = self.norm2(x)
            mlp_out = self.mlp(x_norm)
            x = x + mlp_out
        elif self.norm_policy == "post":
            # Post-norm: LayerNorm after MLP
            mlp_out = self.mlp(x)
            x = self.norm2(x + mlp_out)
        elif self.norm_policy == "normformer":
            # NormFormer: Pre-norm before MLP, then norm after first linear
            x_norm = self.norm2(x)
            mlp_hidden = self.mlp_fc1(x_norm)  # First linear
            mlp_hidden = self.norm_mlp_mid(mlp_hidden)  # LayerNorm after first linear
            mlp_hidden = self.activation(mlp_hidden)  # GELU
            mlp_hidden = self.dropout_mlp(mlp_hidden)
            mlp_out = self.mlp_fc2(mlp_hidden)  # Second linear
            mlp_out = self.dropout(mlp_out)
            x = x + mlp_out
        else:
            raise ValueError(f"Unknown norm_policy: {self.norm_policy}")

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
        rotary_emb: nn.Module | None = None,
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
            Normalization policy: "pre", "post", or "normformer"
        rotary_emb : nn.Module | None
            Optional RotaryEmbedding module for rotary positional encoding.
            Shared across all encoder blocks.
        """
        super().__init__()

        # Store rotary embedding (shared across blocks, not owned by encoder)
        self.rotary_emb = rotary_emb

        # Stack encoder blocks
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    norm_policy=norm_policy,
                    rotary_emb=rotary_emb,
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


def build_transformer_encoder(
    cfg: DictConfig,
    dim: int,
    rotary_emb: nn.Module | None = None,
) -> nn.Module:
    """Build transformer encoder stack.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with classifier.model.* keys (depth, heads, mlp_dim, dropout, norm.policy)
    dim : int
        Model dimension
    rotary_emb : nn.Module | None
        Optional RotaryEmbedding module for rotary positional encoding.

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
        rotary_emb=rotary_emb,
    )
