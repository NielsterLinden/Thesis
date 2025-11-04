from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class MLPEncoder(nn.Module):
    """Token-wise MLP encoder for Phase 1 tokens layout.

    Expects normalized continuous tokens and ids embedding handled in decoder or ignored here.
    For Phase 1, concatenate an ID embedding inside the encoder for simplicity.
    """

    def __init__(self, *, cfg: Any):
        super().__init__()
        latent_dim = int(cfg.phase1.latent_space.latent_dim)
        id_embed_dim = int(cfg.phase1.encoder.id_embed_dim)
        act = _act(str(cfg.phase1.encoder.activation))
        hidden = list(cfg.phase1.encoder.enc_hidden)
        dropout = float(cfg.phase1.encoder.dropout)
        num_types = int(cfg.meta.num_types)
        cont_dim = int(cfg.meta.cont_dim)

        self.id_emb = nn.Embedding(num_types, id_embed_dim)
        in_dim = cont_dim + id_embed_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, *, tokens_cont: torch.Tensor, tokens_id: torch.Tensor, globals_vec: torch.Tensor) -> torch.Tensor:
        emb = self.id_emb(tokens_id)  # [B,T,E]
        x = torch.cat([tokens_cont, emb], dim=-1)
        z_e = self.net(x)
        return z_e
