from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


class DiffusionDecoder(nn.Module):
    """Stub decoder for diffusion-style AE."""

    def __init__(self, *, cfg: Any):
        super().__init__()

    def forward(self, *, z: torch.Tensor, tokens_cont: torch.Tensor, tokens_id: torch.Tensor, globals_vec: torch.Tensor) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError("DiffusionDecoder is a Phase 2 implementation.")
