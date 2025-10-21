from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class DiffusionEncoder(nn.Module):
    """Stub encoder for diffusion-style AE.

    TODO: implement noise conditioning and UNet-like structure in Phase 2.
    """

    def __init__(self, *, cfg: Any):
        super().__init__()
        self.latent_dim = int(cfg.ae.latent_dim)

    def forward(self, *, tokens_cont: torch.Tensor, tokens_id: torch.Tensor, globals_vec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DiffusionEncoder is a Phase 2 implementation.")
