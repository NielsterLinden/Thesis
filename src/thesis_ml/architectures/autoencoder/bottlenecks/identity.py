from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class IdentityBottleneck(nn.Module):
    """Pass-through bottleneck for tokenizer=none."""

    def __init__(self, *, cfg: Any):
        super().__init__()

    def forward(self, z_e: torch.Tensor):
        return z_e
