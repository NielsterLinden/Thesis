from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from thesis_ml.architectures.transformer_classifier.modules.tokenizers import get_tokenizer


class InputEmbedding(nn.Module):
    """Input embedding module with tokenizer and projection.

    Handles both raw and binned token formats.
    Optionally prepends CLS token if pooling=cls.
    """

    def __init__(
        self,
        tokenizer: nn.Module,
        tokenizer_output_dim: int,
        model_dim: int,
        use_cls_token: bool = False,
        pos_enc: nn.Module | None = None,
    ):
        """Initialize input embedding.

        Parameters
        ----------
        tokenizer : nn.Module
            Tokenizer module (identity, raw, binned, or pretrained)
        tokenizer_output_dim : int
            Output dimension of tokenizer
        model_dim : int
            Target model dimension
        use_cls_token : bool
            Whether to prepend CLS token (for cls pooling)
        pos_enc : nn.Module, optional
            Positional encoding module to apply before projection.
            Must have dim == tokenizer_output_dim if provided.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.use_cls_token = use_cls_token
        self.is_binned = hasattr(tokenizer, "token_embedding")  # BinnedTokenizer has token_embedding

        # Validate pos_enc dimension
        if pos_enc is not None:
            if not hasattr(pos_enc, "dim"):
                raise ValueError("pos_enc must have 'dim' attribute")
            assert pos_enc.dim == tokenizer_output_dim, f"pos_enc.dim ({pos_enc.dim}) must match tokenizer_output_dim ({tokenizer_output_dim})"
        self.pos_enc = pos_enc

        # Projection layer (only needed if tokenizer output != model_dim)
        if tokenizer_output_dim != model_dim:
            self.projection = nn.Linear(tokenizer_output_dim, model_dim)
        else:
            self.projection = nn.Identity()

        # CLS token (learned parameter)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

    def forward(
        self,
        *args,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        *args
            For raw format: (tokens_cont [B, T, 4], tokens_id [B, T])
            For binned format: (integer_tokens [B, T],)
        mask : torch.Tensor, optional
            Attention mask [B, T] (True=valid, False=padding)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (embeddings [B, T, D] or [B, T+1, D] if CLS, mask [B, T] or [B, T+1] if CLS)
        """
        # Handle raw vs binned format
        if self.is_binned:
            # Binned format: single argument (integer_tokens)
            integer_tokens = args[0]  # [B, T]
            x = self.tokenizer(integer_tokens)  # [B, T, embed_dim]
        else:
            # Raw format: two arguments (tokens_cont, tokens_id)
            tokens_cont, tokens_id = args[0], args[1]
            x = self.tokenizer(tokens_cont, tokens_id)  # [B, T, tokenizer_output_dim]

        # Apply positional encoding BEFORE projection (if provided)
        # This allows selective PE on semantic dimensions (E, Pt, eta, phi, ID)
        if self.pos_enc is not None:
            x = self.pos_enc(x, mask=mask)  # [B, T, tokenizer_output_dim]

        # Project to model dimension
        x = self.projection(x)  # [B, T, model_dim]

        # Prepend CLS token if needed
        if self.use_cls_token:
            # CLS token: [1, 1, model_dim] -> [B, 1, model_dim]
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, model_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, model_dim]

            # Expand mask: prepend True for CLS position
            if mask is not None:
                # mask: [B, T] -> [B, T+1]
                cls_mask = torch.ones(x.size(0), 1, dtype=mask.dtype, device=mask.device)  # [B, 1]
                mask = torch.cat([cls_mask, mask], dim=1)  # [B, T+1]
            else:
                # If no mask provided, create one with all True
                mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)  # [B, T+1]

        # Ensure mask exists (if not provided and no CLS token)
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)  # [B, T]

        return x, mask


def build_input_embedding(cfg: DictConfig, meta: Mapping[str, Any], pos_enc: nn.Module | None = None) -> nn.Module:
    """Build input embedding layer.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with classifier.model.* keys
    meta : Mapping[str, Any]
        Data metadata with n_tokens, token_feat_dim, has_globals, vocab_size, num_types
    pos_enc : nn.Module, optional
        Positional encoding module to apply before projection (for token-space PE)

    Returns
    -------
    nn.Module
        Input embedding module
    """
    model_dim = cfg.classifier.model.dim
    pooling = cfg.classifier.model.get("pooling", "cls")
    use_cls_token = pooling == "cls"

    # Detect format: check if vocab_size exists (binned) or not (raw)
    is_binned = meta.get("vocab_size") is not None

    if is_binned:
        # Binned tokens: use binned tokenizer
        vocab_size = meta["vocab_size"]
        tokenizer = get_tokenizer(
            name="binned",
            vocab_size=vocab_size,
            embed_dim=model_dim,  # Already model_dim, no projection needed
        )
        tokenizer_output_dim = model_dim
    else:
        # Raw tokens: use specified tokenizer
        tokenizer_name = cfg.classifier.model.tokenizer.name
        num_types = meta.get("num_types")
        cont_dim = meta.get("token_feat_dim", 4)
        id_embed_dim = cfg.classifier.model.tokenizer.get("id_embed_dim", 8)

        tokenizer = get_tokenizer(
            name=tokenizer_name,
            num_types=num_types,
            cont_dim=cont_dim,
            id_embed_dim=id_embed_dim,
        )

        # Get tokenizer output dimension
        # Identity: cont_dim + id_embed_dim
        # Raw: cont_dim
        # Binned: embed_dim (but we're not using this path)
        if tokenizer_name == "identity":
            tokenizer_output_dim = cont_dim + id_embed_dim
        elif tokenizer_name == "raw":
            tokenizer_output_dim = cont_dim
        else:
            # For pretrained, we'd need to inspect the model
            # For now, assume it outputs cont_dim + id_embed_dim
            tokenizer_output_dim = cont_dim + id_embed_dim

    return InputEmbedding(
        tokenizer=tokenizer,
        tokenizer_output_dim=tokenizer_output_dim,
        model_dim=model_dim,
        use_cls_token=use_cls_token,
        pos_enc=pos_enc,
    )
