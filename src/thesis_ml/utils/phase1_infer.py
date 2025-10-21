from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from thesis_ml.data.h5_loader import make_dataloaders
from thesis_ml.phase1.autoenc.base import build_from_config


def _resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_run(run_dir: str | Path, device: str | None = None) -> tuple[Any, torch.nn.Module, torch.device]:
    """Load composed cfg and model weights from a Phase 1 run directory.

    Parameters
    ----------
    run_dir : str | Path
        Path to a run directory that contains `cfg.yaml` and `model.pt`.
    device : str | None
        Optional device string. If None, selects CUDA when available.

    Returns
    -------
    (cfg, model, device)
        The composed config (as saved), model in eval mode on device, and the device.
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "cfg.yaml"
    weights_path = run_dir / "model.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"cfg.yaml not found at {cfg_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"model.pt not found at {weights_path}")

    cfg = OmegaConf.load(str(cfg_path))
    dev = _resolve_device(device)
    model = build_from_config(cfg).to(dev)
    state = torch.load(str(weights_path), map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return cfg, model, dev


def get_example_batch(cfg: Any, split: str = "val"):
    """Return a single batch (tokens_cont, tokens_id, globals) for quick inference."""
    train_dl, val_dl, test_dl, _meta = make_dataloaders(cfg)
    if split == "train":
        dl = train_dl
    elif split == "test":
        dl = test_dl
    else:
        dl = val_dl
    batch = next(iter(dl))
    return batch
