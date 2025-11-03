"""Anomaly detection orchestrator for comparing baseline vs corrupted data."""

from __future__ import annotations

import logging
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from thesis_ml.data.h5_loader import make_dataloaders
from thesis_ml.utils.seed import set_all_seeds

from .data_corruption import create_corrupted_dataloader
from .forward_pass import create_model_adapter, run_batch_inference
from .metrics import aggregate_metrics, compute_auroc

logger = logging.getLogger(__name__)


def run_anomaly_detection(
    models: list[tuple[str, Any, torch.nn.Module]],
    dataset_cfg: DictConfig | dict[str, Any],
    corruption_strategies: list[dict[str, Any]],
    split: str = "test",
    inference_cfg: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run anomaly detection inference on models with baseline and corrupted data.

    Parameters
    ----------
    models : list[tuple[str, Any, torch.nn.Module]]
        List of (run_id, cfg, model) tuples
    dataset_cfg : DictConfig | dict[str, Any]
        Dataset configuration for creating dataloaders
    corruption_strategies : list[dict[str, Any]]
        List of corruption strategy configs with keys: name, type, params
    split : str
        Dataset split to use ('test' for baseline comparison)
    inference_cfg : dict[str, Any] | None
        Inference configuration with keys:
            - autocast: bool (default: False)
            - batch_size: int (default: 512)
            - seed: int (default: 42)

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Nested dict: {run_id: {strategy_name: {metrics...}}}
        Includes "baseline" key for baseline results
    """
    if inference_cfg is None:
        inference_cfg = {}

    logger.info("Starting anomaly detection inference")
    logger.info(f"Processing {len(models)} models, {len(corruption_strategies)} corruption strategies")

    # Set seeds for reproducibility
    seed = inference_cfg.get("seed", 42)
    set_all_seeds(seed)

    # Pin device once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} (autocast: {inference_cfg.get('autocast', False)})")

    # Create baseline dataloader
    # dataset_cfg should be a full config (with data and phase1 sections)
    # Override batch_size from inference_cfg
    batch_size = inference_cfg.get("batch_size", 512)

    # Create a copy to avoid modifying the original config
    temp_cfg = OmegaConf.create(dataset_cfg) if isinstance(dataset_cfg, dict) else OmegaConf.create(OmegaConf.to_container(dataset_cfg, resolve=True))

    # If testing, reduce dataset size at the source (speeds up HDF5 read and stats)
    max_samples = inference_cfg.get("max_samples", None)
    if max_samples is not None and max_samples > 0:
        if not hasattr(temp_cfg, "data"):
            temp_cfg.data = OmegaConf.create({})
        temp_cfg.data.limit_samples = int(max_samples)
        logger.info(f"Limiting dataset to first {max_samples} samples at load time (testing mode)")

    # Override batch_size (make_dataloaders expects cfg.phase1.trainer.batch_size)
    if hasattr(temp_cfg, "phase1") and hasattr(temp_cfg.phase1, "trainer"):
        temp_cfg.phase1.trainer.batch_size = batch_size
    else:
        # Create phase1.trainer section if it doesn't exist
        if not hasattr(temp_cfg, "phase1"):
            temp_cfg.phase1 = OmegaConf.create({})
        temp_cfg.phase1.trainer = OmegaConf.create({"batch_size": batch_size})

    train_dl, val_dl, test_dl, _meta = make_dataloaders(temp_cfg)

    if split == "train":
        baseline_dl = train_dl
    elif split == "test":
        baseline_dl = test_dl
    else:
        baseline_dl = val_dl

    autocast = inference_cfg.get("autocast", False)

    results = {}

    # Process each model
    for model_idx, (run_id, _cfg, model) in enumerate(models, 1):
        logger.info(f"[{model_idx}/{len(models)}] Processing model: {run_id}")

        # Wrap model with adapter for uniform API
        model_adapter = create_model_adapter(model)
        model_adapter.to(device)

        model_results = {}

        # Run baseline inference
        logger.info(f"  Running baseline inference on {split} split...")
        baseline_results = run_batch_inference(
            model=model_adapter,
            dataloader=baseline_dl,
            device=device,
            autocast=autocast,
        )
        baseline_n_events = len(baseline_results["per_event"])
        logger.info(f"  Baseline inference complete: {baseline_n_events} events processed")

        # Extract per-event data
        baseline_per_event = baseline_results["per_event"]
        baseline_mse = [e["mse"] for e in baseline_per_event]
        baseline_mae = [e["mae"] for e in baseline_per_event]
        baseline_weights = [e["weight"] for e in baseline_per_event]

        # Aggregate baseline metrics
        baseline_metrics = aggregate_metrics(
            {"mse": baseline_mse, "mae": baseline_mae},
            weights=baseline_weights if any(w != 1.0 for w in baseline_weights) else None,
        )
        baseline_metrics["auroc"] = None  # No AUROC for baseline alone
        model_results["baseline"] = baseline_metrics

        # Run inference for each corruption strategy
        for strategy_idx, strategy_config in enumerate(corruption_strategies, 1):
            strategy_name = strategy_config["name"]
            logger.info(f"  [{strategy_idx}/{len(corruption_strategies)}] Processing corruption: {strategy_name}")

            # Create corrupted dataloader
            corrupted_dl = create_corrupted_dataloader(
                original_dataloader=baseline_dl,
                strategy_config=strategy_config,
                seed=seed,
            )

            # Run inference on corrupted data
            corrupted_results = run_batch_inference(
                model=model_adapter,
                dataloader=corrupted_dl,
                device=device,
                autocast=autocast,
            )
            corrupted_n_events = len(corrupted_results["per_event"])
            logger.info(f"    Corruption inference complete: {corrupted_n_events} events processed")

            # Extract per-event data
            corrupted_per_event = corrupted_results["per_event"]
            corrupted_mse = [e["mse"] for e in corrupted_per_event]
            corrupted_mae = [e["mae"] for e in corrupted_per_event]
            corrupted_weights = [e["weight"] for e in corrupted_per_event]

            # Aggregate corrupted metrics
            corrupted_metrics = aggregate_metrics(
                {"mse": corrupted_mse, "mae": corrupted_mae},
                weights=corrupted_weights if any(w != 1.0 for w in corrupted_weights) else None,
            )

            # Compute AUROC between baseline and corrupted
            # Combine weights if needed
            combined_weights = None
            if any(w != 1.0 for w in baseline_weights + corrupted_weights):
                combined_weights = baseline_weights + corrupted_weights

            auroc_mse = compute_auroc(
                baseline_scores=baseline_mse,
                corrupted_scores=corrupted_mse,
                weights=combined_weights,
            )

            corrupted_metrics["auroc"] = auroc_mse
            model_results[strategy_name] = corrupted_metrics
            logger.info(f"    Metrics computed - MSE: {corrupted_metrics.get('mse_mean', 0):.6f}, AUROC: {auroc_mse}")

        results[run_id] = model_results
        logger.info(f"  Completed model {run_id}")

    logger.info(f"Anomaly detection inference complete for {len(models)} models")
    return results
