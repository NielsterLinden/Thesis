from __future__ import annotations

import torch


def diffusion_loss(*args, **kwargs) -> torch.Tensor:
    raise NotImplementedError("Diffusion losses are Phase 2.")


def schedule_adapter(*args, **kwargs):
    raise NotImplementedError("Diffusion schedules are Phase 2.")
