from __future__ import annotations

import torch
import torch.nn as nn


class RawTokenizer(nn.Module):
    """Raw tokenizer: pass through continuous features as-is (Ambre's clean tokens).

    This simply returns the normalized continuous features without any ID embedding.
    Uses the clean tokens that Ambre has already processed.

    Input: tokens_cont [B, T, 4], tokens_id [B, T] (ignored)
    Output: [B, T, 4] (same as input tokens_cont)
    """

    def __init__(self, cont_dim: int = 4):
        super().__init__()
        self.cont_dim = cont_dim
        self.output_dim = cont_dim  # Output is same as input

    def forward(
        self,
        tokens_cont: torch.Tensor,  # [B, T, cont_dim]
        tokens_id: torch.Tensor,  # [B, T] (ignored, but kept for interface consistency)
    ) -> torch.Tensor:
        """Pass through raw continuous features.

        Parameters
        ----------
        tokens_cont : torch.Tensor
            Continuous features [B, T, cont_dim] (already normalized by data loader)
        tokens_id : torch.Tensor
            Particle ID integers [B, T] (not used, but kept for interface)

        Returns
        -------
        torch.Tensor
            Raw continuous features [B, T, cont_dim] (unchanged)
        """
        # Simply return the continuous features as-is
        # These are already normalized by the data loader
        return tokens_cont
