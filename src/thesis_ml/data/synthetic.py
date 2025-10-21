from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass  # dataclass is a class that is used to store data. it is the same as a regular class, but it has a __init__ and __repr__ that is automatically generated.
class SyntheticMeta:
    input_dim: int
    task: str
    num_train: int
    num_val: int


def _make_synthetic_data(n_samples: int, n_features: int, task: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    w = rng.normal(size=(n_features,)).astype(np.float32)
    noise = 0.1 * rng.normal(size=(n_samples,)).astype(np.float32)

    if task == "regression":
        y = X @ w + noise
        y = y.reshape(-1, 1).astype(np.float32)
    elif task == "binary":
        logits = X @ w + noise
        # Note: leave as raw logits for BCEWithLogits
        y = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.float32)
        y = y.reshape(-1, 1)
    else:
        raise ValueError(f"Unknown task: {task}")

    return torch.from_numpy(X), torch.from_numpy(y)


def build_dataloaders(cfg) -> tuple[DataLoader, DataLoader, dict]:
    n_samples = int(cfg.data.n_samples)
    n_features = int(cfg.data.n_features)
    train_frac = float(cfg.data.train_frac)
    seed = int(cfg.data.seed)
    task = str(cfg.data.task)

    X, y = _make_synthetic_data(n_samples, n_features, task, seed)

    num_train = int(n_samples * train_frac)
    X_train, y_train = X[:num_train], y[:num_train]
    X_val, y_val = X[num_train:], y[num_train:]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    batch_size = int(cfg.trainer.batch_size)
    num_workers = int(cfg.trainer.num_workers)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = SyntheticMeta(input_dim=n_features, task=task, num_train=len(train_ds), num_val=len(val_ds))
    return train_loader, val_loader, {"input_dim": meta.input_dim, "task": meta.task}
