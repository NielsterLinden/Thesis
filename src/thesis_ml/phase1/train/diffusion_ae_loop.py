from __future__ import annotations

from omegaconf import DictConfig

SUPPORTED_PLOT_FAMILIES = {"losses", "metrics", "diffusion", "recon", "latency"}


def train(cfg: DictConfig):
    # Skeleton only; Phase 2 will implement diffusion objective.
    raise NotImplementedError("Diffusion AE training loop is a Phase 2 implementation.")
