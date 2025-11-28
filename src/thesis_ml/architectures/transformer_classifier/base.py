from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from thesis_ml.architectures.transformer_classifier.modules.embedding import build_input_embedding
from thesis_ml.architectures.transformer_classifier.modules.encoder_block import build_transformer_encoder
from thesis_ml.architectures.transformer_classifier.modules.head import build_classifier_head
from thesis_ml.architectures.transformer_classifier.modules.positional import (
    RotaryEmbedding,
    get_positional_encoding,
    parse_dim_mask,
)
from thesis_ml.architectures.transformer_classifier.modules.tokenizers.tokenizers import get_feature_map


class TransformerClassifier(nn.Module):
    """Transformer-based classifier for sequence classification."""

    def __init__(
        self,
        embedding: nn.Module,
        pos_enc: nn.Module | None,
        encoder: nn.Module,
        head: nn.Module,
        positional_space: str = "model",
    ):
        """Initialize transformer classifier.

        Parameters
        ----------
        embedding : nn.Module
            Input embedding module (handles tokenization and projection)
        pos_enc : nn.Module | None
            Positional encoding module (None for no positional encoding or when positional_space="token")
        encoder : nn.Module
            Transformer encoder stack
        head : nn.Module
            Classifier head (pooling + linear)
        positional_space : str
            Where PE is applied: "model" (after projection, old behavior) or "token" (before projection)
        """
        super().__init__()
        self.embedding = embedding
        self.pos_enc = pos_enc
        self.encoder = encoder
        self.head = head
        self.positional_space = positional_space

    def forward(self, *args, mask: torch.Tensor | None = None) -> torch.Tensor:
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
        torch.Tensor
            Logits [B, n_classes]
        """
        # Handle raw vs binned input format explicitly
        if len(args) == 2:
            # Raw format: (tokens_cont, tokens_id)
            tokens_cont, tokens_id = args
            x, mask = self.embedding(tokens_cont, tokens_id, mask=mask)
        elif len(args) == 1:
            # Binned format: (integer_tokens,)
            integer_tokens = args[0]
            x, mask = self.embedding(integer_tokens, mask=mask)
        else:
            raise ValueError(f"Expected 1 or 2 arguments, got {len(args)}")

        # Add positional encoding (only for model-space PE, old behavior)
        # Token-space PE is already applied in InputEmbedding
        if self.positional_space == "model" and self.pos_enc is not None:
            x = self.pos_enc(x, mask=mask)

        # Encode
        x = self.encoder(x, mask=mask)

        # Classify
        logits = self.head(x, mask=mask)
        return logits


def build_from_config(cfg: DictConfig, meta: Mapping[str, Any]) -> nn.Module:
    """Build transformer classifier from config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with classifier.model.* keys
    meta : Mapping[str, Any]
        Data metadata with keys: n_tokens, token_feat_dim, has_globals, n_classes, vocab_size, num_types

    Returns
    -------
    nn.Module
        Transformer classifier model
    """
    dim = cfg.classifier.model.dim
    num_heads = cfg.classifier.model.heads
    n_classes = meta["n_classes"]
    max_seq_length = meta["n_tokens"]

    # Parse positional_space config (default "model" for backward compatibility)
    positional_space = cfg.classifier.model.get("positional_space", "model")
    if positional_space not in ("model", "token"):
        raise ValueError(f"positional_space must be 'model' or 'token', got '{positional_space}'")

    # Parse positional_dim_mask config
    dim_mask_config = cfg.classifier.model.get("positional_dim_mask", None)

    # Get tokenizer info for feature mapping
    is_binned = meta.get("vocab_size") is not None
    if is_binned:
        tokenizer_name = "binned"
        tokenizer_output_dim = dim  # Binned tokenizer outputs model_dim directly
        id_embed_dim = 8  # Not used for binned, but needed for get_feature_map signature
    else:
        tokenizer_name = cfg.classifier.model.tokenizer.name
        cont_dim = meta.get("token_feat_dim", 4)
        id_embed_dim = cfg.classifier.model.tokenizer.get("id_embed_dim", 8)
        if tokenizer_name == "identity":
            tokenizer_output_dim = cont_dim + id_embed_dim
        elif tokenizer_name == "raw":
            tokenizer_output_dim = cont_dim
        else:
            # For pretrained, assume same as identity
            tokenizer_output_dim = cont_dim + id_embed_dim

    # Get feature map and parse dim_mask
    feature_map = get_feature_map(tokenizer_name, tokenizer_output_dim, id_embed_dim)
    dim_mask = None
    if dim_mask_config is not None:
        if positional_space != "token":
            raise ValueError("positional_dim_mask is only supported when positional_space='token'. " f"Got positional_space='{positional_space}'")
        dim_mask = parse_dim_mask(dim_mask_config, tokenizer_output_dim, feature_map)

    # Positional encoding
    pos_enc_name = cfg.classifier.model.get("positional", "sinusoidal")

    # Handle rotary PE specially: it goes into attention, not as additive PE
    rotary_emb = None
    pos_enc = None
    pos_enc_for_embedding = None

    if pos_enc_name == "rotary":
        # Rotary embedding: applied in attention to Q and K
        head_dim = dim // num_heads
        rotary_base = cfg.classifier.model.get("rotary", {}).get("base", 10000.0)
        rotary_emb = RotaryEmbedding(head_dim=head_dim, base=rotary_base)
    elif pos_enc_name != "none":
        # Additive positional encodings: sinusoidal or learned
        if positional_space == "token":
            # Token-space PE: create with tokenizer_output_dim and pass to embedding
            pos_enc_for_embedding = get_positional_encoding(pos_enc_name, tokenizer_output_dim, max_seq_length, dim_mask=dim_mask)
            # Don't create pos_enc for TransformerClassifier (handled in embedding)
            pos_enc = None
        else:
            # Model-space PE: create with model_dim (old behavior)
            pos_enc = get_positional_encoding(pos_enc_name, dim, max_seq_length)
            # Don't pass to embedding
            pos_enc_for_embedding = None

    # Build components in order
    embedding = build_input_embedding(cfg, meta, pos_enc=pos_enc_for_embedding)

    encoder = build_transformer_encoder(cfg, dim, rotary_emb=rotary_emb)
    head = build_classifier_head(cfg, dim, n_classes)

    # Assemble model
    model = TransformerClassifier(
        embedding=embedding,
        pos_enc=pos_enc,
        encoder=encoder,
        head=head,
        positional_space=positional_space,
    )

    return model
