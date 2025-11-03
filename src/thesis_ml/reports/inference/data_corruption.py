"""Data corruption strategies for anomaly detection."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CorruptedDataset(Dataset):
    """Wrapper dataset that applies corruption on-the-fly."""

    def __init__(
        self,
        original_dataset: Dataset,
        corruption_fn: callable,
        seed: int | None = None,
    ):
        self.original_dataset = original_dataset
        self.corruption_fn = corruption_fn
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.np_rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        item = self.original_dataset[idx]
        # Apply corruption function (it expects tuple, rng, np_rng)
        corrupted = self.corruption_fn(item, self.rng, self.np_rng)
        return corrupted


def apply_gaussian_noise(
    batch: tuple[torch.Tensor, ...],
    std: float,
    eta_limit: float = 5.0,
) -> tuple[torch.Tensor, ...]:
    """Apply Gaussian noise to continuous tokens with physics constraints.

    Parameters
    ----------
    batch : tuple[torch.Tensor, ...]
        Batch tuple (tokens_cont, tokens_id, globals) or (tokens_cont, tokens_id, globals, mask)
    std : float
        Standard deviation of Gaussian noise
    eta_limit : float
        Maximum absolute value for eta (default: 5.0)

    Returns
    -------
    tuple[torch.Tensor, ...]
        Corrupted batch with same structure as input
    """
    if len(batch) == 3:
        tokens_cont, tokens_id, globals_vec = batch
        mask = None
    elif len(batch) == 4:
        tokens_cont, tokens_id, globals_vec, mask = batch
    else:
        raise ValueError(f"Expected batch tuple of length 3 or 4, got {len(batch)}")

    # Add noise to continuous tokens only
    noise = torch.randn_like(tokens_cont) * std
    tokens_cont_corrupted = tokens_cont + noise

    # Apply physics constraints
    # Assume layout: [pT, eta, phi, ?] for continuous tokens [B, T, 4]
    # pT must be >= 0
    tokens_cont_corrupted[:, :, 0] = torch.clamp(tokens_cont_corrupted[:, :, 0], min=0.0)

    # Wrap phi into [-π, π)
    if tokens_cont_corrupted.shape[-1] >= 3:
        tokens_cont_corrupted[:, :, 2] = (tokens_cont_corrupted[:, :, 2] + np.pi) % (2 * np.pi) - np.pi

    # Clamp eta within detector limits
    if tokens_cont_corrupted.shape[-1] >= 2:
        tokens_cont_corrupted[:, :, 1] = torch.clamp(tokens_cont_corrupted[:, :, 1], min=-eta_limit, max=eta_limit)

    if mask is None:
        return (tokens_cont_corrupted, tokens_id, globals_vec)
    return (tokens_cont_corrupted, tokens_id, globals_vec, mask)


def apply_token_shuffle(
    batch: tuple[torch.Tensor, ...],
    ratio: float,
    rng: random.Random,
) -> tuple[torch.Tensor, ...]:
    """Shuffle token order for a random subset of tokens.

    Parameters
    ----------
    batch : tuple[torch.Tensor, ...]
        Batch tuple (tokens_cont, tokens_id, globals) or (tokens_cont, tokens_id, globals, mask)
    ratio : float
        Fraction of tokens to shuffle (0.0 to 1.0)
    rng : random.Random
        Random number generator for reproducibility

    Returns
    -------
    tuple[torch.Tensor, ...]
        Corrupted batch with shuffled tokens
    """
    if len(batch) == 3:
        tokens_cont, tokens_id, globals_vec = batch
        mask = None
    elif len(batch) == 4:
        tokens_cont, tokens_id, globals_vec, mask = batch
    else:
        raise ValueError(f"Expected batch tuple of length 3 or 4, got {len(batch)}")

    B, T = tokens_cont.shape[:2]
    n_shuffle = max(1, int(T * ratio))

    # Shuffle within each batch
    tokens_cont_shuffled = tokens_cont.clone()
    tokens_id_shuffled = tokens_id.clone()

    for b in range(B):
        # Select random subset of token positions
        positions = list(range(T))
        rng.shuffle(positions)
        selected = positions[:n_shuffle]

        # Shuffle the selected positions
        shuffled_selected = selected.copy()
        rng.shuffle(shuffled_selected)

        # Apply permutation
        tokens_cont_shuffled[b, selected] = tokens_cont[b, shuffled_selected]
        tokens_id_shuffled[b, selected] = tokens_id[b, shuffled_selected]

    if mask is None:
        return (tokens_cont_shuffled, tokens_id_shuffled, globals_vec)
    return (tokens_cont_shuffled, tokens_id_shuffled, globals_vec, mask)


def apply_drop_tokens(
    batch: tuple[torch.Tensor, ...],
    ratio: float,
    rng: random.Random,
) -> tuple[torch.Tensor, ...]:
    """Drop random tokens by masking them out.

    STUB: Not yet implemented.

    Parameters
    ----------
    batch : tuple[torch.Tensor, ...]
        Batch tuple
    ratio : float
        Fraction of tokens to drop
    rng : random.Random
        Random number generator

    Returns
    -------
    tuple[torch.Tensor, ...]
        Batch with dropped tokens masked
    """
    raise NotImplementedError("drop_tokens corruption not yet implemented")


def apply_swap_id_cont_pairs(
    batch: tuple[torch.Tensor, ...],
    ratio: float,
    rng: random.Random,
) -> tuple[torch.Tensor, ...]:
    """Swap continuous/id pairs between tokens.

    STUB: Not yet implemented.

    Parameters
    ----------
    batch : tuple[torch.Tensor, ...]
        Batch tuple
    ratio : float
        Fraction of token pairs to swap
    rng : random.Random
        Random number generator

    Returns
    -------
    tuple[torch.Tensor, ...]
        Batch with swapped pairs
    """
    raise NotImplementedError("swap_id_cont_pairs corruption not yet implemented")


def apply_global_jet_scramble(
    batch: tuple[torch.Tensor, ...],
    rng: random.Random,
) -> tuple[torch.Tensor, ...]:
    """Scramble jet-level structure.

    STUB: Not yet implemented.

    Parameters
    ----------
    batch : tuple[torch.Tensor, ...]
        Batch tuple
    rng : random.Random
        Random number generator

    Returns
    -------
    tuple[torch.Tensor, ...]
        Batch with scrambled jet structure
    """
    raise NotImplementedError("global_jet_scramble corruption not yet implemented")


def create_corrupted_dataloader(
    original_dataloader: DataLoader,
    strategy_config: dict[str, Any],
    seed: int | None = None,
) -> DataLoader:
    """Create a corrupted dataloader from an original dataloader.

    Parameters
    ----------
    original_dataloader : DataLoader
        Original dataloader to corrupt
    strategy_config : dict[str, Any]
        Corruption strategy configuration with keys:
            - name: str (e.g., "gaussian_noise")
            - type: str (e.g., "additive_noise", "permutation")
            - params: dict (strategy-specific parameters)
    seed : int | None
        Random seed for reproducibility

    Returns
    -------
    DataLoader
        New dataloader that applies corruption on-the-fly
    """
    strategy_type = strategy_config["type"]
    params = strategy_config.get("params", {})

    # Create corruption function
    def corruption_fn(item: tuple, rng: random.Random, np_rng: np.random.Generator) -> tuple:
        if strategy_type == "additive_noise":
            std = params.get("std", 0.1)
            return apply_gaussian_noise(item, std=std)
        elif strategy_type == "permutation":
            ratio = params.get("ratio", 0.2)
            return apply_token_shuffle(item, ratio=ratio, rng=rng)
        elif strategy_type == "drop_tokens":
            ratio = params.get("ratio", 0.1)
            return apply_drop_tokens(item, ratio=ratio, rng=rng)
        elif strategy_type == "swap_id_cont_pairs":
            ratio = params.get("ratio", 0.1)
            return apply_swap_id_cont_pairs(item, ratio=ratio, rng=rng)
        elif strategy_type == "global_jet_scramble":
            return apply_global_jet_scramble(item, rng=rng)
        else:
            raise ValueError(f"Unknown corruption strategy type: {strategy_type}")

    # Wrap original dataset
    corrupted_dataset = CorruptedDataset(
        original_dataset=original_dataloader.dataset,
        corruption_fn=corruption_fn,
        seed=seed,
    )

    # Create new dataloader with same settings
    return DataLoader(
        corrupted_dataset,
        batch_size=original_dataloader.batch_size,
        shuffle=original_dataloader.shuffle,
        num_workers=original_dataloader.num_workers,
        pin_memory=original_dataloader.pin_memory,
        drop_last=original_dataloader.drop_last,
    )
