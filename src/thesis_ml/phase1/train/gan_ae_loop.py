from __future__ import annotations

from omegaconf import DictConfig

SUPPORTED_PLOT_FAMILIES = {"losses", "metrics", "recon", "adversarial", "latency"}


def train(cfg: DictConfig):
    # Skeleton only; Phase 2 will implement adversarial training
    # Note: add tiny per-epoch print ("Epoch X of Max done, loss = ...") when implemented
    raise NotImplementedError("GAN AE training loop is a Phase 2 implementation.")
