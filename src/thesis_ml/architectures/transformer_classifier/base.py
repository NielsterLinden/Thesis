from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from thesis_ml.architectures.transformer_classifier.modules.embedding import build_input_embedding
from thesis_ml.architectures.transformer_classifier.modules.encoder_block import build_transformer_encoder
from thesis_ml.architectures.transformer_classifier.modules.head import build_classifier_head
from thesis_ml.architectures.transformer_classifier.modules.positional import get_positional_encoding


class TransformerClassifier(nn.Module):
    """Transformer-based classifier for sequence classification."""

    def __init__(
        self,
        embedding: nn.Module,
        pos_enc: nn.Module | None,
        encoder: nn.Module,
        head: nn.Module,
    ):
        """Initialize transformer classifier.

        Parameters
        ----------
        embedding : nn.Module
            Input embedding module (handles tokenization and projection)
        pos_enc : nn.Module | None
            Positional encoding module (None for no positional encoding)
        encoder : nn.Module
            Transformer encoder stack
        head : nn.Module
            Classifier head (pooling + linear)
        """
        super().__init__()
        self.embedding = embedding
        self.pos_enc = pos_enc
        self.encoder = encoder
        self.head = head

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

        # Add positional encoding (if not rotary)
        if self.pos_enc is not None:
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
    n_classes = meta["n_classes"]
    max_seq_length = meta["n_tokens"]

    # Build components in order
    embedding = build_input_embedding(cfg, meta)

    # Positional encoding
    pos_enc_name = cfg.classifier.model.get("positional", "sinusoidal")
    pos_enc = None if pos_enc_name == "none" else get_positional_encoding(pos_enc_name, dim, max_seq_length)

    encoder = build_transformer_encoder(cfg, dim)
    head = build_classifier_head(cfg, dim, n_classes)

    # Assemble model
    model = TransformerClassifier(
        embedding=embedding,
        pos_enc=pos_enc,
        encoder=encoder,
        head=head,
    )

    return model
