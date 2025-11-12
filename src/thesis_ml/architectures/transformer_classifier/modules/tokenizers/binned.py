from __future__ import annotations

import torch
import torch.nn as nn


class BinnedTokenizer(nn.Module):
    """Binned tokenizer: directly embed integer tokens from Ambre's binned dataset.

    Input: integer_tokens [B, T] (values 0-885, where 0 = no particle/padding)
    Output: [B, T, embed_dim] (embedded tokens)

    This handles the pre-tokenized integer sequences from tokens_dataset.h5.
    The integers are already binned, so we just need to embed them.
    """

    def __init__(
        self,
        vocab_size: int = 886,  # 0-885 inclusive
        embed_dim: int = 256,  # Will be set to model.dim
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = embed_dim

        # Direct embedding of integer tokens
        # vocab_size = 886 (0-885), where 0 is padding/no particle
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(
        self,
        integer_tokens: torch.Tensor,  # [B, T] (int64, values 0-885)
        tokens_id: torch.Tensor | None = None,  # Ignored for binned tokens
    ) -> torch.Tensor:
        """Embed integer tokens directly.

        Parameters
        ----------
        integer_tokens : torch.Tensor
            Integer token sequence [B, T] with values 0-885
        tokens_id : torch.Tensor, optional
            Ignored (kept for interface consistency)

        Returns
        -------
        torch.Tensor
            Embedded tokens [B, T, embed_dim]
        """
        # Directly embed the integer tokens
        # padding_idx=0 ensures 0 tokens get zero embeddings
        embedded = self.token_embedding(integer_tokens)  # [B, T, embed_dim]
        return embedded
