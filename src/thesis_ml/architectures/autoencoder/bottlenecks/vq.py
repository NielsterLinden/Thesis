from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes: int, code_dim: int, beta: float = 0.25, ema: bool = True, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.n_codes, self.code_dim, self.beta = int(n_codes), int(code_dim), float(beta)
        self.ema, self.decay, self.eps = bool(ema), float(decay), float(eps)
        self.codebook = nn.Parameter(torch.randn(self.n_codes, self.code_dim))
        if self.ema:
            self.register_buffer("ema_count", torch.zeros(self.n_codes))
            self.register_buffer("ema_weight", torch.randn(self.n_codes, self.code_dim))

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        B, T, D = z_e.shape
        flat = z_e.reshape(-1, D)
        cb = self.codebook
        d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ cb.t() + cb.pow(2).sum(1, keepdim=True).t()
        idx = d.argmin(dim=1)
        z_q = cb[idx].view(B, T, D)

        if self.training and self.ema:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).float()
                count = onehot.sum(0)
                weight = onehot.t() @ flat
                self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
                self.ema_weight.mul_(self.decay).add_(weight, alpha=1 - self.decay)
                n = self.ema_count.sum()
                cluster_size = (self.ema_count + self.eps) / (n + self.n_codes * self.eps) * n
                embed = self.ema_weight / cluster_size.unsqueeze(1)
                self.codebook.data.copy_(embed)

        commit = self.beta * F.mse_loss(z_e, z_q.detach())
        codebook = F.mse_loss(z_e.detach(), z_q)
        if self.ema:
            codebook = codebook.detach() * 0.0
        z_q = z_e + (z_q - z_e).detach()

        with torch.no_grad():
            probs = torch.bincount(idx, minlength=self.n_codes).float()
            probs = probs / probs.sum().clamp_min(1.0)
            perplexity = torch.exp(-(probs[probs > 0] * probs[probs > 0].log()).sum())

        aux = {
            "commit": commit,
            "codebook": codebook,
            "perplex": perplexity,
            "indices": idx.view(B, T),
        }
        return z_q, aux


class VQBottleneck(nn.Module):
    def __init__(self, *, cfg: Any):
        super().__init__()
        self.quant = VectorQuantizer(
            n_codes=int(cfg.phase1.latent_space.codebook_size),
            code_dim=int(cfg.phase1.latent_space.latent_dim),
            beta=float(cfg.phase1.latent_space.beta),
            ema=bool(cfg.phase1.latent_space.ema),
            decay=float(cfg.phase1.latent_space.ema_decay),
        )

    def forward(self, z_e: torch.Tensor):
        z_q, aux = self.quant(z_e)
        return {"z_q": z_q, "indices": aux["indices"], "aux": {k: v for k, v in aux.items() if k != "indices"}}
