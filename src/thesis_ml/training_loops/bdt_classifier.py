"""BDT Classifier training loop using XGBoost."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from thesis_ml.architectures.bdt.base import build_from_config
from thesis_ml.data.h5_loader import make_classification_dataloaders
from thesis_ml.facts import append_jsonl_event, append_scalars_csv, build_event_payload
from thesis_ml.monitoring.orchestrator import handle_event
from thesis_ml.utils.seed import set_all_seeds
from thesis_ml.utils.wandb_utils import finish_wandb, init_wandb, log_artifact, log_metrics

SUPPORTED_PLOT_FAMILIES = {"losses", "metrics"}


def _flatten_batch(batch: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a batch of token data to feature vectors.

    Parameters
    ----------
    batch : tuple
        Batch from dataloader (raw or binned format)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Flattened features [B, 72] and labels [B]
    """
    if len(batch) == 5:  # raw format
        tokens_cont, tokens_id, globals_, mask, label = batch
        # Flatten continuous features: [B, T, 4] -> [B, T*4]
        features = tokens_cont.view(tokens_cont.size(0), -1).numpy()
        labels = label.numpy()
    else:  # binned format
        integer_tokens, globals_ints, mask, label = batch
        # For binned, we use integer tokens directly as features
        features = integer_tokens.float().numpy()
        labels = label.numpy()

    return features, labels


def _collect_all_data(loader: torch.utils.data.DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Collect all data from a dataloader into numpy arrays.

    Parameters
    ----------
    loader : DataLoader
        PyTorch dataloader

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        All features [N, D] and labels [N]
    """
    all_features = []
    all_labels = []

    for batch in loader:
        features, labels = _flatten_batch(batch)
        all_features.append(features)
        all_labels.append(labels)

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_classes: int) -> dict[str, float | None]:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels [N]
    y_pred : np.ndarray
        Predicted labels [N]
    y_proba : np.ndarray
        Predicted probabilities [N, n_classes]
    n_classes : int
        Number of classes

    Returns
    -------
    dict[str, float | None]
        Dictionary with acc, f1, auroc
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # F1 score
    f1 = f1_score(y_true, y_pred, average="weighted")

    # AUROC
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        auroc = None
    elif n_classes == 2:
        auroc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auroc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")

    return {"acc": accuracy, "f1": f1, "auroc": auroc}


def _save_split_scores(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    outdir: str,
    split: str = "val",
) -> None:
    """Save scores and labels for a split.

    Parameters
    ----------
    y_true : np.ndarray
        True labels [N]
    y_proba : np.ndarray
        Predicted probabilities [N, n_classes]
    outdir : str
        Output directory
    split : str
        Split name (val or test)
    """
    # Convert to tensors for consistency with transformer outputs
    probs_tensor = torch.from_numpy(y_proba).float()
    labels_tensor = torch.from_numpy(y_true).long()

    # Compute logits from probabilities (inverse softmax approximation)
    logits_tensor = torch.log(probs_tensor + 1e-8)

    os.makedirs(outdir, exist_ok=True)
    torch.save(
        {
            "logits": logits_tensor,
            "probs": probs_tensor,
            "labels": labels_tensor,
            "pooled": torch.empty(0, 0),  # No pooled features for BDT
        },
        os.path.join(outdir, f"{split}_scores.pt"),
    )


def _gather_meta(cfg: DictConfig, ds_meta: Mapping[str, Any]) -> None:
    """Attach data-derived meta to cfg for module constructors."""
    prev_struct = OmegaConf.is_struct(cfg)
    try:
        OmegaConf.set_struct(cfg, False)
        cfg.meta = OmegaConf.create(
            {
                "n_tokens": int(ds_meta["n_tokens"]),
                "token_feat_dim": int(ds_meta.get("token_feat_dim", 4)),
                "has_globals": bool(ds_meta.get("has_globals", False)),
                "n_classes": int(ds_meta["n_classes"]),
                "num_types": int(ds_meta.get("num_types", 0)),
                "vocab_size": ds_meta.get("vocab_size"),
            }
        )
    finally:
        OmegaConf.set_struct(cfg, prev_struct)


def train(cfg: DictConfig) -> dict:
    """Train BDT classifier.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration

    Returns
    -------
    dict
        Training results with keys: best_val_loss, best_val_acc, test_loss, test_acc
    """
    # Reproducibility
    set_all_seeds(cfg.classifier.trainer.seed)

    # Data
    train_dl, val_dl, test_dl, meta = make_classification_dataloaders(cfg)
    _gather_meta(cfg, meta)

    # Collect all data (BDT trains on full dataset, not mini-batches)
    print("Collecting training data...")
    X_train, y_train = _collect_all_data(train_dl)
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    print("Collecting validation data...")
    X_val, y_val = _collect_all_data(val_dl)
    print(f"Validation set: {X_val.shape[0]} samples")

    print("Collecting test data...")
    X_test, y_test = _collect_all_data(test_dl)
    print(f"Test set: {X_test.shape[0]} samples")

    # Build model
    model = build_from_config(cfg, meta)

    # Initialize W&B (returns None if disabled or on error - training continues normally)
    # Note: BDT is not a torch.nn.Module so we don't pass model for watching
    wandb_run = init_wandb(cfg, model=None)

    # Log complexity estimate
    complexity = model.get_complexity_estimate()
    print(f"BDT complexity: {complexity['n_estimators']} trees, depth {complexity['max_depth']}")
    print(f"Effective parameters estimate: {complexity['effective_params_estimate']:,}")

    # Logging dir from Hydra
    outdir = None
    if cfg.logging.save_artifacts:
        outdir = os.getcwd()
        os.makedirs(outdir, exist_ok=True)

        # Dump resolved config
        config_path = os.path.join(outdir, "resolved_config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)

    # on_start event
    if outdir and cfg.logging.save_artifacts:
        start_payload = build_event_payload(
            moment="on_start",
            run_dir=outdir,
            cfg=cfg,
            seed=cfg.classifier.trainer.seed,
            class_weights=None,  # BDT handles class weights internally
        )
        append_jsonl_event(str(outdir), start_payload)
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_start", start_payload)

    # Train BDT
    print("\nTraining BDT...")
    training_start = time.time()

    # Early stopping config
    early_stopping_cfg = cfg.classifier.trainer.get("early_stopping", {})
    early_stopping_rounds = early_stopping_cfg.get("patience", 20) if early_stopping_cfg.get("enabled", False) else None

    model.fit(
        X_train,
        y_train,
        # Log both train and validation to obtain per-round curves
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=True,
    )

    training_time = time.time() - training_start
    print(f"Training completed in {training_time:.2f}s")

    # Extract per-iteration metrics from XGBoost (if available)
    evals_result: dict[str, dict[str, list[float]]] | None = None
    try:
        # sklearn API exposes evals_result() directly
        evals_result = model.model.evals_result()
    except Exception:
        try:
            evals_result = model.model.get_booster().evals_result()
        except Exception:
            evals_result = None

    # Prepare histories for plotting like epochs (boosting rounds ~ epochs)
    histories = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "train_auroc": [],
        "val_auroc": [],
    }

    if evals_result is not None:
        # Names are 'validation_0' (train), 'validation_1' (val) by sklearn convention
        train_key = "validation_0"
        val_key = "validation_1" if "validation_1" in evals_result else "validation_0"

        train_loss_curve = evals_result.get(train_key, {}).get("logloss", [])
        val_loss_curve = evals_result.get(val_key, {}).get("logloss", [])
        # AUROC only for binary config (we enabled it in model params)
        train_auc_curve = evals_result.get(train_key, {}).get("auc", [])
        val_auc_curve = evals_result.get(val_key, {}).get("auc", [])

        # Populate histories
        histories["train_loss"] = list(train_loss_curve)
        histories["val_loss"] = list(val_loss_curve)
        histories["train_auroc"] = list(train_auc_curve) if train_auc_curve else []
        histories["val_auroc"] = list(val_auc_curve) if val_auc_curve else []

        # Also append scalars.csv per iteration to mimic epoch logging
        if outdir and cfg.logging.save_artifacts and val_loss_curve:
            for i in range(len(val_loss_curve)):
                append_scalars_csv(
                    str(outdir),
                    epoch=i,
                    split="val",
                    train_loss=float(train_loss_curve[i]) if i < len(train_loss_curve) else None,
                    val_loss=float(val_loss_curve[i]),
                    metrics={
                        "auroc": float(val_auc_curve[i]) if i < len(val_auc_curve) else None,
                    },
                    epoch_time_s=None,
                    throughput=None,
                    max_memory_mib=None,
                )

        # W&B logging for per-round metrics
        for i in range(len(val_loss_curve)):
            log_metrics(
                wandb_run,
                {
                    "epoch": i,  # Use "epoch" for consistency (boosting round)
                    "train/loss": float(train_loss_curve[i]) if i < len(train_loss_curve) else None,
                    "val/loss": float(val_loss_curve[i]),
                    "val/auroc": float(val_auc_curve[i]) if i < len(val_auc_curve) else None,
                },
                step=i,
            )

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    val_metrics = _compute_metrics(y_val, y_val_pred, y_val_proba, meta["n_classes"])

    # Compute validation loss (log loss)
    from sklearn.metrics import log_loss

    val_loss = log_loss(y_val, y_val_proba)

    print("\nValidation Results:")
    print(f"  Loss: {val_loss:.6f}")
    print(f"  Accuracy: {val_metrics['acc']:.4f}")
    print(f"  F1: {val_metrics['f1']:.4f}")
    print(f"  AUROC: {val_metrics['auroc']:.4f}" if val_metrics["auroc"] else "  AUROC: N/A")

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    test_metrics = _compute_metrics(y_test, y_test_pred, y_test_proba, meta["n_classes"])
    test_loss = log_loss(y_test, y_test_proba)

    print("\nTest Results:")
    print(f"  Loss: {test_loss:.6f}")
    print(f"  Accuracy: {test_metrics['acc']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  AUROC: {test_metrics['auroc']:.4f}" if test_metrics["auroc"] else "  AUROC: N/A")

    # Save artifacts
    if outdir:
        # Save model
        model.save(os.path.join(outdir, "model.json"))

        # Save validation scores
        _save_split_scores(y_val, y_val_proba, outdir, split="val")

        # Save test scores
        _save_split_scores(y_test, y_test_proba, outdir, split="test")

        # If we did not manage to get per-iteration curves, at least log a single point
        if not histories["val_loss"]:
            histories = {
                "train_loss": [0.0],
                "val_loss": [val_loss],
                "train_acc": [0.0],
                "val_acc": [val_metrics["acc"]],
                "train_f1": [0.0],
                "val_f1": [val_metrics["f1"]],
                "train_auroc": [0.0],
                "val_auroc": [val_metrics["auroc"]],
            }

        # Emit a final on_epoch_end payload carrying full histories (so reports can see the curves)
        final_epoch = len(histories["val_loss"]) - 1 if histories["val_loss"] else 0
        payload = build_event_payload(
            moment="on_epoch_end",
            run_dir=outdir,
            epoch=final_epoch,
            train_loss=histories["train_loss"][final_epoch] if histories["train_loss"] else 0.0,
            val_loss=histories["val_loss"][final_epoch] if histories["val_loss"] else val_loss,
            metrics={
                "acc": val_metrics["acc"],
                "f1": val_metrics["f1"],
                "auroc": histories["val_auroc"][final_epoch] if histories["val_auroc"] else val_metrics["auroc"],
            },
            histories=histories,
            epoch_time_s=training_time,
            cfg=cfg,
        )
        append_jsonl_event(str(outdir), payload)
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_epoch_end", payload)

        # on_train_end
        payload_end = build_event_payload(
            moment="on_train_end",
            run_dir=outdir,
            total_time_s=training_time,
            histories=histories,
            cfg=cfg,
        )
        append_jsonl_event(str(outdir), payload_end)
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_train_end", payload_end)

        # Upload model artifact to W&B
        model_path = Path(outdir) / "model.json"
        log_artifact(wandb_run, model_path, "model", cfg)

    # Log final test metrics to W&B
    final_epoch = len(histories["val_loss"]) - 1 if histories["val_loss"] else 0
    log_metrics(
        wandb_run,
        {
            "test/loss": float(test_loss),
            "test/acc": float(test_metrics["acc"]),
            "test/f1": float(test_metrics["f1"]),
            "test/auroc": float(test_metrics["auroc"]) if test_metrics["auroc"] is not None else None,
        },
        step=final_epoch + 1,
    )

    # Finish W&B run
    finish_wandb(wandb_run)

    return {
        "best_val_loss": val_loss,
        "best_val_acc": val_metrics["acc"],
        "best_epoch": 0,
        "test_loss": test_loss,
        "test_acc": test_metrics["acc"],
    }
