from __future__ import annotations

from omegaconf import DictConfig

# NOTE: When implementing, use build_event_payload() from thesis_ml.utils
# for consistent fact/payload creation (see ae_loop.py for examples).

SUPPORTED_PLOT_FAMILIES = {"losses", "metrics", "diffusion", "recon", "latency"}


def train(cfg: DictConfig):
    # Skeleton only; Phase 2 will implement diffusion objective.
    # Note: add tiny per-epoch print ("Epoch X of Max done, loss = ...") when implemented
    raise NotImplementedError("Diffusion AE training loop is a Phase 2 implementation.")
