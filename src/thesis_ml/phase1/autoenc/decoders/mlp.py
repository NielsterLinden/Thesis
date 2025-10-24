from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class MLPDecoder(nn.Module):
    """Token-wise MLP decoder with optional globals head."""

    def __init__(self, *, cfg: Any):
        super().__init__()
        latent_dim = int(cfg.phase1.latent_space.latent_dim)
        cont_dim = int(cfg.meta.cont_dim)
        act = _act(str(cfg.phase1.decoder.activation))
        hidden = list(cfg.phase1.decoder.dec_hidden)
        dropout = float(cfg.phase1.decoder.dropout)

        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, cont_dim))
        self.net = nn.Sequential(*layers)

        self.globals_head = bool(cfg.phase1.decoder.globals_head)
        if self.globals_head:
            g_hidden = list(cfg.phase1.decoder.globals_hidden)
            g_layers: list[nn.Module] = []
            prev = latent_dim
            for h in g_hidden:
                g_layers += [nn.Linear(prev, h), act]
                prev = h
            self.g_head = nn.Sequential(*g_layers, nn.Linear(prev, int(cfg.meta.globals)))
            self.globals_beta = float(cfg.phase1.decoder.globals_beta)

    def forward(self, *, z: torch.Tensor, tokens_cont: torch.Tensor, tokens_id: torch.Tensor, globals_vec: torch.Tensor) -> Mapping[str, torch.Tensor]:
        x_hat = self.net(z)
        out = {"x_hat": x_hat, "aux": {}}
        if self.globals_head:
            g_pred = self.g_head(z.mean(dim=1))  # [B, G]
            rec_globals = F.mse_loss(g_pred, globals_vec)
            out["rec_globals"] = rec_globals
            out["aux"]["rec_globals"] = rec_globals * self.globals_beta
        return out
