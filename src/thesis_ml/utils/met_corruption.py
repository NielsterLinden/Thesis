"""MET corruption utilities for interpretability tests.

Apply at validation time to test whether the model truly uses MET direction
or magnitude. See plan: "Randomized METphi" and "Shuffle MET across events".
"""

from __future__ import annotations

import math
import random

import torch


def randomize_metphi(globals_: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    """Keep MET magnitude, randomize METphi uniformly in [0, 2π].

    If the model uses MET direction, performance should drop.

    Parameters
    ----------
    globals_ : torch.Tensor
        [B, 2] — (MET, METphi).
    seed : int, optional
        For reproducibility.

    Returns
    -------
    torch.Tensor
        [B, 2] — (MET, random_phi).
    """
    out = globals_.clone()
    B = out.size(0)
    if seed is not None:
        rng = random.Random(seed)
        phis = [rng.uniform(0, 2 * math.pi) for _ in range(B)]
    else:
        phis = torch.rand(B, device=out.device, dtype=out.dtype) * (2 * math.pi)
        phis = phis.numpy().tolist() if out.device.type == "cpu" else phis.cpu().numpy().tolist()
    for i in range(B):
        out[i, 1] = phis[i] if isinstance(phis[i], int | float) else float(phis[i])
    return out


def shuffle_met_across_events(globals_: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    """Shuffle MET (and METphi) across events; keep everything else fixed.

    If performance barely changes, MET integration is weak or info is leaked elsewhere.

    Parameters
    ----------
    globals_ : torch.Tensor
        [B, 2] — (MET, METphi).
    seed : int, optional
        For reproducibility.

    Returns
    -------
    torch.Tensor
        [B, 2] — shuffled MET values.
    """
    if seed is not None:
        torch.manual_seed(seed)
    perm = torch.randperm(globals_.size(0), device=globals_.device)
    return globals_[perm].clone()


def corrupt_batch_globals(
    batch: tuple,
    mode: str,
    seed: int | None = None,
) -> tuple:
    """Apply MET corruption to a raw-format batch.

    Parameters
    ----------
    batch : tuple
        (tokens_cont, tokens_id, globals_, mask, label) for raw format.
    mode : str
        "none" | "randomize_metphi" | "shuffle_met".
    seed : int, optional
        For reproducibility.

    Returns
    -------
    tuple
        Batch with corrupted globals_ when mode is not "none".
    """
    if mode == "none" or len(batch) < 3:
        return batch
    tokens_cont, tokens_id, globals_, mask, label = batch
    if globals_ is None:
        return batch
    if mode == "randomize_metphi":
        globals_ = randomize_metphi(globals_, seed=seed)
    elif mode == "shuffle_met":
        globals_ = shuffle_met_across_events(globals_, seed=seed)
    else:
        return batch
    return (tokens_cont, tokens_id, globals_, mask, label)
