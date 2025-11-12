from __future__ import annotations

import torch
import torch.nn as nn


class IdentityTokenizer(nn.Module):
    """Identity tokenizer: embed particle IDs and concatenate with continuous features.

    Input: tokens_cont [B, T, 4], tokens_id [B, T] (integers)
    Output: [B, T, 4 + id_embed_dim] (typically [B, T, 12] with id_embed_dim=8)

    This keeps MET and MET phi separate (not included in tokenization).
    """

    def __init__(
        self,
        num_types: int,
        cont_dim: int = 4,
        id_embed_dim: int = 8,
    ):
        super().__init__()
        self.cont_dim = cont_dim
        self.id_embed_dim = id_embed_dim
        self.output_dim = cont_dim + id_embed_dim

        # Embedding for particle IDs (similar to autoencoder encoder)
        self.id_embedding = nn.Embedding(num_types, id_embed_dim)

    def forward(
        self,
        tokens_cont: torch.Tensor,  # [B, T, cont_dim]
        tokens_id: torch.Tensor,  # [B, T] (int64)
    ) -> torch.Tensor:
        """Tokenize by embedding IDs and concatenating with continuous features.

        Parameters
        ----------
        tokens_cont : torch.Tensor
            Continuous features [B, T, cont_dim]
        tokens_id : torch.Tensor
            Particle ID integers [B, T]

        Returns
        -------
        torch.Tensor
            Tokenized features [B, T, cont_dim + id_embed_dim]
        """
        # Embed particle IDs
        id_emb = self.id_embedding(tokens_id)  # [B, T, id_embed_dim]

        # Concatenate continuous features with ID embeddings
        tokenized = torch.cat([tokens_cont, id_emb], dim=-1)  # [B, T, cont_dim + id_embed_dim]

        return tokenized
