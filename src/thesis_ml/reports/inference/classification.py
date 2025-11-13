"""Classification inference orchestrator for transformer classifiers."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from thesis_ml.data.h5_loader import make_classification_dataloaders
from thesis_ml.utils.seed import set_all_seeds

from .classification_metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


def run_classification_inference(
    models: list[tuple[str, Any, torch.nn.Module]],
    dataset_cfg: DictConfig | dict[str, Any],
    split: str = "test",
    inference_cfg: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run classification inference on transformer classifier models.

    Parameters
    ----------
    models : list[tuple[str, Any, torch.nn.Module]]
        List of (run_id, cfg, model) tuples
    dataset_cfg : DictConfig | dict[str, Any]
        Dataset configuration for creating dataloaders
    split : str
        Dataset split to use ('test', 'val', 'train')
    inference_cfg : dict[str, Any] | None
        Inference configuration with keys:
            - autocast: bool (default: False)
            - batch_size: int (default: 512)
            - seed: int (default: 42)
            - max_samples: int | None (default: None)

    Returns
    -------
    dict[str, dict[str, Any]]
        Nested dict: {run_id: {metrics...}}
        Metrics include: accuracy, auroc, precision, recall, f1, confusion_matrix, roc_curves, pr_curves
    """
    if inference_cfg is None:
        inference_cfg = {}

    logger.info("Starting classification inference")
    logger.info(f"Processing {len(models)} models")

    # Set seeds for reproducibility
    seed = inference_cfg.get("seed", 42)
    set_all_seeds(seed)

    # Pin device once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} (autocast: {inference_cfg.get('autocast', False)})")

    # Create dataloader
    batch_size = inference_cfg.get("batch_size", 512)

    # Create a copy to avoid modifying the original config
    temp_cfg = OmegaConf.create(dataset_cfg) if isinstance(dataset_cfg, dict) else OmegaConf.create(OmegaConf.to_container(dataset_cfg, resolve=True))

    # Override batch_size (make_classification_dataloaders expects cfg.classifier.trainer.batch_size)
    if hasattr(temp_cfg, "classifier") and hasattr(temp_cfg.classifier, "trainer"):
        temp_cfg.classifier.trainer.batch_size = batch_size
    else:
        # Create classifier.trainer section if it doesn't exist
        if not hasattr(temp_cfg, "classifier"):
            temp_cfg.classifier = OmegaConf.create({})
        temp_cfg.classifier.trainer = OmegaConf.create({"batch_size": batch_size})

    # If testing, reduce dataset size at the source
    max_samples = inference_cfg.get("max_samples", None)
    if max_samples is not None and max_samples > 0:
        if not hasattr(temp_cfg, "data"):
            temp_cfg.data = OmegaConf.create({})
        temp_cfg.data.limit_samples = int(max_samples)
        logger.info(f"Limiting dataset to first {max_samples} samples at load time (testing mode)")

    train_dl, val_dl, test_dl, _meta = make_classification_dataloaders(temp_cfg)

    if split == "train":
        dataloader = train_dl
    elif split == "test":
        dataloader = test_dl
    else:
        dataloader = val_dl

    autocast = inference_cfg.get("autocast", False)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if autocast and device.type == "cuda" else nullcontext()

    results = {}

    # Process each model
    for model_idx, (run_id, model_cfg, model) in enumerate(models, 1):
        logger.info(f"[{model_idx}/{len(models)}] Processing model: {run_id}")

        model.eval()
        model.to(device)

        all_logits = []
        all_labels = []
        all_probs = []

        # Get n_classes from model config meta
        n_classes = model_cfg.meta.n_classes if hasattr(model_cfg, "meta") and hasattr(model_cfg.meta, "n_classes") else _meta.get("n_classes", 2)
        logger.info(f"  Model has {n_classes} classes")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader, 1):
                # Explicit batch unpacking (matching training loop format)
                if len(batch) == 5:  # raw format
                    tokens_cont, tokens_id, globals, mask, label = batch
                    tokens_cont = tokens_cont.to(device)
                    tokens_id = tokens_id.to(device)
                    mask = mask.to(device)
                    label = label.to(device)

                    with autocast_ctx:
                        logits = model(tokens_cont, tokens_id, mask=mask)
                else:  # binned format (4 items)
                    integer_tokens, globals_ints, mask, label = batch
                    integer_tokens = integer_tokens.to(device)
                    mask = mask.to(device)
                    label = label.to(device)

                    with autocast_ctx:
                        logits = model(integer_tokens, mask=mask)

                # Compute probabilities
                probs = torch.softmax(logits, dim=-1)

                # Accumulate
                all_logits.append(logits.cpu())
                all_labels.append(label.cpu())
                all_probs.append(probs.cpu())

                # Log progress every 10% or every 10 batches
                total_batches = len(dataloader)
                if total_batches > 10 and (batch_idx % max(1, total_batches // 10) == 0 or batch_idx % 10 == 0):
                    logger.debug(f"    Processed batch {batch_idx}/{total_batches} ({100 * batch_idx / total_batches:.1f}%)")

        logger.info(f"  Inference complete: {len(all_logits)} batches processed")

        # Compute metrics
        metrics = compute_classification_metrics(all_logits, all_labels, all_probs, n_classes)

        results[run_id] = metrics
        logger.info(f"  Metrics computed - Accuracy: {metrics.get('accuracy', 0):.4f}, AUROC: {metrics.get('auroc', 'N/A')}")

    logger.info(f"Classification inference complete for {len(models)} models")
    return results
