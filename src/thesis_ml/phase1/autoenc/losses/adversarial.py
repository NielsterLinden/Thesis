from __future__ import annotations

import torch


def generator_loss(*args, **kwargs) -> torch.Tensor:
    raise NotImplementedError("Adversarial losses are Phase 2.")


def discriminator_loss(*args, **kwargs) -> torch.Tensor:
    raise NotImplementedError("Adversarial losses are Phase 2.")
