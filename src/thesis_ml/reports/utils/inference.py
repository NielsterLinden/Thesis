"""Inference utilities for report generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from thesis_ml.data.h5_loader import make_dataloaders
from thesis_ml.utils.paths import resolve_run_dir

from ..inference.forward_pass import create_model_adapter as _create_model_adapter

logger = logging.getLogger(__name__)


def _resolve_device(device: str | None = None) -> torch.device:
    """Resolve device from string or auto-select."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_run(run_id: str, output_root: Path | str, device: str | None = None) -> tuple[Any, torch.nn.Module, torch.device]:
    """Load composed cfg and model weights from a run.

    Parameters
    ----------
    run_id : str
        Run ID (e.g., "run_20251024-152016_baseline_test")
    output_root : Path | str
        Root output directory
    device : str | None
        Optional device string. If None, selects CUDA when available.

    Returns
    -------
    tuple[Any, torch.nn.Module, torch.device]
        (cfg, model, device) - config, model in eval mode, and device
    """
    run_dir = resolve_run_dir(run_id, output_root)

    # Prefer .hydra/config.yaml as canonical record
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    if hydra_cfg_path.exists():
        cfg = OmegaConf.load(str(hydra_cfg_path))
    else:
        # Fallback to cfg.yaml for old runs
        cfg_path = run_dir / "cfg.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing .hydra/config.yaml or cfg.yaml in {run_dir}")
        cfg = OmegaConf.load(str(cfg_path))

    # Prefer best_val.pt, fallback to model.pt
    best_val_path = run_dir / "best_val.pt"
    model_pt_path = run_dir / "model.pt"
    if best_val_path.exists():
        weights_path = best_val_path
    elif model_pt_path.exists():
        weights_path = model_pt_path
    else:
        raise FileNotFoundError(f"Missing best_val.pt or model.pt in {run_dir}")

    dev = _resolve_device(device)

    # Load checkpoint first to extract meta info if needed
    checkpoint = torch.load(str(weights_path), map_location=dev, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    # Detect model type and build accordingly
    if hasattr(cfg, "classifier"):
        # Classifier model
        from thesis_ml.architectures.transformer_classifier.base import build_from_config as build_classifier
        from thesis_ml.data.h5_loader import make_classification_dataloaders
        from thesis_ml.training_loops.transformer_classifier import _gather_meta

        # Populate meta if missing (needed for model building)
        if not hasattr(cfg, "meta") or cfg.meta is None:
            # Create dataloaders temporarily to get meta
            train_dl, val_dl, test_dl, meta = make_classification_dataloaders(cfg)
            _gather_meta(cfg, meta)

        # Ensure cfg.meta exists as a mutable config node
        if not hasattr(cfg, "meta") or cfg.meta is None:
            cfg.meta = OmegaConf.create({})

        # Merge any metadata stored in the checkpoint into cfg.meta (n_tokens, n_classes, etc.)
        ckpt_meta = checkpoint.get("meta") if isinstance(checkpoint, dict) else None
        if isinstance(ckpt_meta, dict):
            for key, value in ckpt_meta.items():
                # Do not overwrite values that are already present in cfg.meta
                if not hasattr(cfg.meta, key) or getattr(cfg.meta, key) is None:
                    setattr(cfg.meta, key, value)

        # At this point we expect at least n_tokens and n_classes to be present.
        # If not, fall back once more to dataloader-based meta extraction.
        if not hasattr(cfg.meta, "n_tokens") or cfg.meta.n_tokens is None:
            train_dl, val_dl, test_dl, meta = make_classification_dataloaders(cfg)
            _gather_meta(cfg, meta)

        if not hasattr(cfg.meta, "n_classes") or cfg.meta.n_classes is None:
            train_dl, val_dl, test_dl, meta = make_classification_dataloaders(cfg)
            _gather_meta(cfg, meta)

        data_n_tokens = int(cfg.meta.n_tokens)

        # IMPORTANT: Extract n_tokens from checkpoint PE shapes to handle training/inference mismatch.
        # The positional encoding size depends on n_tokens at training time, not inference time.

        if "pos_enc.pe" in state_dict:
            pe_shape = state_dict["pos_enc.pe"].shape  # [max_seq_len, dim]
            checkpoint_max_seq_len = pe_shape[0]
            # For model-space PE with CLS token, max_seq_len = n_tokens + 1
            pooling = cfg.classifier.model.get("pooling", "cls")
            positional_space = cfg.classifier.model.get("positional_space", "model")
            checkpoint_n_tokens = checkpoint_max_seq_len - 1 if positional_space == "model" and pooling == "cls" else checkpoint_max_seq_len
            # Override meta n_tokens to match checkpoint
            if checkpoint_n_tokens != data_n_tokens:
                logger.info(f"Adjusting n_tokens from {data_n_tokens} (data) to {checkpoint_n_tokens} (checkpoint PE shape)")
            cfg.meta.n_tokens = checkpoint_n_tokens

        # Also check for embedding pos_enc (token-space PE)
        elif "embedding.pos_enc.pe" in state_dict:
            pe_shape = state_dict["embedding.pos_enc.pe"].shape
            checkpoint_n_tokens = pe_shape[0]
            if checkpoint_n_tokens != data_n_tokens:
                logger.info(f"Adjusting n_tokens from {data_n_tokens} (data) to {checkpoint_n_tokens} (checkpoint embedding PE shape)")
            cfg.meta.n_tokens = checkpoint_n_tokens

        model = build_classifier(cfg, cfg.meta).to(dev)
    elif hasattr(cfg, "phase1"):
        # Autoencoder model
        from thesis_ml.architectures.autoencoder.base import build_from_config
        from thesis_ml.training_loops.autoencoder import _gather_meta

        # Populate meta if missing (needed for model building)
        if not hasattr(cfg, "meta") or cfg.meta is None:
            # Create dataloaders temporarily to get meta
            train_dl, val_dl, test_dl, meta = make_dataloaders(cfg)
            _gather_meta(cfg, meta)

        model = build_from_config(cfg).to(dev)
    else:
        raise ValueError("Cannot determine model type from config (missing 'classifier' or 'phase1' section)")

    # Load state dict (already extracted above)
    model.load_state_dict(state_dict)
    model.eval()
    return cfg, model, dev


def get_example_batch(cfg: Any, split: str = "val"):
    """Return a single batch for quick inference.

    Parameters
    ----------
    cfg : Any
        Config dict/config object
    split : str
        Dataset split ("train", "val", "test")

    Returns
    -------
    tuple
        Batch tuple (tokens_cont, tokens_id, globals) for phase1 models
    """
    train_dl, val_dl, test_dl, _meta = make_dataloaders(cfg)
    if split == "train":
        dl = train_dl
    elif split == "test":
        dl = test_dl
    else:
        dl = val_dl
    batch = next(iter(dl))
    return batch


def load_models_for_runs(
    run_ids: list[str],
    output_root: Path | str,
    device: str | None = None,
) -> list[tuple[str, Any, torch.nn.Module]]:
    """Batch load models from list of run IDs.

    Parameters
    ----------
    run_ids : list[str]
        List of run IDs to load
    output_root : Path | str
        Root output directory
    device : str | None
        Optional device string. If None, selects CUDA when available.

    Returns
    -------
    list[tuple[str, Any, torch.nn.Module]]
        List of (run_id, cfg, model) tuples
    """
    models = []

    for run_id in run_ids:
        cfg, model, _ = load_model_from_run(run_id, output_root, device=device)
        models.append((run_id, cfg, model))

    return models


def create_model_adapter(model: torch.nn.Module) -> torch.nn.Module:
    """Create adapter for model if needed to provide uniform API.

    Parameters
    ----------
    model : torch.nn.Module
        Model to wrap

    Returns
    -------
    torch.nn.Module
        Model with uniform API (possibly wrapped)
    """
    return _create_model_adapter(model)


def run_inference_minimal(
    models: list[tuple[Any, torch.nn.Module]],
    dataset_cfg: DictConfig | dict[str, Any],
    split: str = "val",
) -> dict[str, Any]:
    """Run inference and return minimal aggregated metrics.

    This function computes aggregated metrics (mean, std, percentiles) without
    persisting per-event scores by default.

    Parameters
    ----------
    models : list[tuple[Any, torch.nn.Module]]
        List of (cfg, model) tuples
    dataset_cfg : DictConfig | dict[str, Any]
        Dataset configuration
    split : str
        Dataset split to use

    Returns
    -------
    dict[str, Any]
        Aggregated metrics per model (e.g., {"run_id": {"mse_mean": ..., "mse_std": ...}})
    """
    # TODO: Implement actual inference logic
    # For now, return placeholder structure
    metrics = {}
    for _cfg, _model in models:
        run_id = "unknown"  # Extract from cfg if available
        metrics[run_id] = {
            "mse_mean": 0.0,
            "mse_std": 0.0,
            "mse_p50": 0.0,
            "mse_p95": 0.0,
        }
    return metrics


def persist_inference_artifacts(
    inference_dir: Path,
    metrics: dict[str, Any],
    figures: list[Any] | None = None,
    persist_raw_scores: bool = False,
    per_event_scores: dict[str, Any] | None = None,
) -> None:
    """Persist minimal inference artifacts.

    Parameters
    ----------
    inference_dir : Path
        Path to inference/ subdirectory
    metrics : dict[str, Any]
        Aggregated metrics per model
    figures : list[Any] | None
        List of matplotlib figures to save
    persist_raw_scores : bool
        If True, persist per-event scores (default: False)
    per_event_scores : dict[str, Any] | None
        Per-event scores dict (only used if persist_raw_scores=True)
    """
    inference_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregated metrics
    summary_path = inference_dir / "summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save figures if provided
    if figures:
        figs_dir = inference_dir / "figures"
        figs_dir.mkdir(exist_ok=True)
        # TODO: Save figures using save_figure utility

    # Optionally persist raw scores
    if persist_raw_scores and per_event_scores:
        raw_scores_dir = inference_dir / "raw_scores"
        raw_scores_dir.mkdir(exist_ok=True)
        # TODO: Save per-event scores (parquet or h5 format)
