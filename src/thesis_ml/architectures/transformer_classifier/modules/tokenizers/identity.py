from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Valid PID embedding modes
PID_MODES = ("learned", "one_hot", "fixed_random")


class IdentityTokenizer(nn.Module):
    """Identity tokenizer: embed particle IDs and concatenate with continuous features.

    Input: tokens_cont [B, T, 4], tokens_id [B, T] (integers)
    Output: [B, T, 4 + id_embed_dim] (typically [B, T, 12] with id_embed_dim=8)

    Supports multiple PID embedding modes:
    - "learned": standard trainable nn.Embedding (default)
    - "one_hot": fixed identity matrix, id_embed_dim forced to num_types
    - "fixed_random": random initialization, frozen (not trainable)

    This keeps MET and MET phi separate (not included in tokenization).
    """

    def __init__(
        self,
        num_types: int,
        cont_dim: int = 4,
        id_embed_dim: int = 8,
        pid_mode: str = "learned",
    ):
        super().__init__()
        if pid_mode not in PID_MODES:
            raise ValueError(f"pid_mode must be one of {PID_MODES}, got '{pid_mode}'")

        self.num_types = num_types
        self.cont_dim = cont_dim
        self.pid_mode = pid_mode

        if pid_mode == "one_hot":
            # Force id_embed_dim = num_types for a proper identity basis
            self.id_embed_dim = num_types
            self.id_embedding = nn.Embedding(num_types, num_types)
            # Initialize with identity matrix and freeze
            self.id_embedding.weight.data.copy_(torch.eye(num_types))
            self.id_embedding.weight.requires_grad = False
            if id_embed_dim != num_types:
                logger.info(
                    "[PID] one_hot mode: overriding id_embed_dim %d → %d (num_types)",
                    id_embed_dim,
                    num_types,
                )
        elif pid_mode == "fixed_random":
            self.id_embed_dim = id_embed_dim
            self.id_embedding = nn.Embedding(num_types, id_embed_dim)
            # Keep default init but freeze
            self.id_embedding.weight.requires_grad = False
        else:  # "learned"
            self.id_embed_dim = id_embed_dim
            self.id_embedding = nn.Embedding(num_types, id_embed_dim)

        self.output_dim = cont_dim + self.id_embed_dim
        logger.info(
            "[PID] mode=%s  num_types=%d  id_embed_dim=%d  output_dim=%d  trainable=%s",
            self.pid_mode,
            self.num_types,
            self.id_embed_dim,
            self.output_dim,
            self.id_embedding.weight.requires_grad,
        )

    # ------------------------------------------------------------------
    # Phase-transition helpers (called from training loop)
    # ------------------------------------------------------------------

    def unfreeze_pid(self) -> None:
        """Unfreeze the PID embedding weights (for warmup_fixed schedule)."""
        self.id_embedding.weight.requires_grad = True
        logger.info("[PID] Unfroze id_embedding weights (pid_mode was '%s')", self.pid_mode)

    def reinit_pid(self, mode: str = "normal") -> None:
        """Re-initialize PID embedding weights and unfreeze them.

        Used for the frozen_backbone schedule: after freezing the entire model,
        re-init + unfreeze only the PID embedding so it learns from scratch while
        the backbone stays fixed.

        Parameters
        ----------
        mode : str
            Initialization mode: "normal" (default) or "one_hot_padded"
            (identity matrix zero-padded to id_embed_dim > num_types).
        """
        with torch.no_grad():
            if mode == "one_hot_padded" and self.id_embed_dim >= self.num_types:
                # [num_types, id_embed_dim] — one-hot in first num_types cols, zeros rest
                weight = torch.zeros(self.num_types, self.id_embed_dim)
                weight[:, : self.num_types] = torch.eye(self.num_types)
                self.id_embedding.weight.copy_(weight)
            else:
                nn.init.normal_(self.id_embedding.weight)
        self.id_embedding.weight.requires_grad = True
        logger.info(
            "[PID] Re-initialized id_embedding (mode=%s, shape=%s)",
            mode,
            list(self.id_embedding.weight.shape),
        )

    def get_pid_weight(self) -> torch.Tensor:
        """Return a detached CPU copy of the PID embedding weight matrix.

        Returns
        -------
        torch.Tensor
            Shape [num_types, id_embed_dim]
        """
        return self.id_embedding.weight.detach().cpu().clone()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
