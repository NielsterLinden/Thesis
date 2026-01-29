"""MLP Classifier training loop."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score, roc_auc_score

from thesis_ml.architectures.mlp_classifier.base import build_from_config
from thesis_ml.data.h5_loader import make_classification_dataloaders
from thesis_ml.facts import append_jsonl_event, append_scalars_csv, build_event_payload, build_meta, write_meta
from thesis_ml.monitoring.orchestrator import handle_event
from thesis_ml.utils import TrainingProgressShower
from thesis_ml.utils.seed import set_all_seeds
from thesis_ml.utils.wandb_utils import finish_wandb, init_wandb, log_artifact, log_metrics

SUPPORTED_PLOT_FAMILIES = {"losses", "metrics"}


def _flatten_batch(batch: tuple, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a batch of token data to feature vectors.

    Parameters
    ----------
    batch : tuple
        Batch from dataloader (raw or binned format)
    device : torch.device
        Device to move tensors to

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Flattened features [B, 72] and labels [B]
    """
    if len(batch) == 5:  # raw format
        tokens_cont, tokens_id, globals_, mask, label = batch
        # Flatten continuous features: [B, T, 4] -> [B, T*4]
        features = tokens_cont.view(tokens_cont.size(0), -1).to(device)
        labels = label.to(device)
    else:  # binned format
        integer_tokens, globals_ints, mask, label = batch
        # For binned, we use integer tokens directly as features
        features = integer_tokens.float().to(device)
        labels = label.to(device)

    return features, labels


def _save_split_scores_and_embeddings(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    outdir: str,
    split: str = "val",
) -> None:
    """Save logits, probabilities, labels, and features for a split."""
    model.eval()

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_features: list[torch.Tensor] = []

    # Hook to capture features before final classifier
    captured: list[torch.Tensor] = []

    def hook_fn(module, inputs, output):
        if inputs and torch.is_tensor(inputs[0]):
            captured.append(inputs[0].detach().cpu())

    hook = None
    if hasattr(model, "classifier"):
        hook = model.classifier.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in loader:
            features, labels = _flatten_batch(batch, device)
            logits = model(features)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.cpu())

            if captured:
                all_features.append(captured[-1])

    if hook is not None:
        hook.remove()

    logits_cat = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, 0)
    labels_cat = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    probs_cat = torch.softmax(logits_cat, dim=-1) if logits_cat.numel() > 0 else logits_cat
    features_cat = torch.cat(all_features, dim=0) if all_features else torch.empty(0, 0)

    os.makedirs(outdir, exist_ok=True)
    torch.save(
        {"logits": logits_cat, "probs": probs_cat, "labels": labels_cat, "pooled": features_cat},
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


def _compute_class_weights(
    train_loader: torch.utils.data.DataLoader,
    n_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute class weights from train loader for imbalanced datasets."""
    class_counts = torch.zeros(n_classes, dtype=torch.long)

    for batch in train_loader:
        if len(batch) == 5:  # raw format
            _, _, _, _, labels = batch
        else:  # binned format
            _, _, _, labels = batch

        for c in range(n_classes):
            class_counts[c] += (labels == c).sum().item()

    total = class_counts.sum().float()
    weights = total / (n_classes * class_counts.float() + 1e-8)
    weights = weights / weights.sum() * n_classes

    return weights.to(device)


def _compute_metrics_epoch(
    all_logits: list[torch.Tensor],
    all_labels: list[torch.Tensor],
    n_classes: int,
) -> dict[str, float | None]:
    """Compute classification metrics at epoch level."""
    logits_cat = torch.cat(all_logits, dim=0).cpu()
    labels_cat = torch.cat(all_labels, dim=0).cpu()

    probs = torch.softmax(logits_cat, dim=-1)
    preds = probs.argmax(dim=-1)

    accuracy = (preds == labels_cat).float().mean().item()
    f1 = f1_score(labels_cat.numpy(), preds.numpy(), average="weighted")

    unique_labels = labels_cat.unique()
    auroc = None if unique_labels.numel() < 2 else roc_auc_score(labels_cat.numpy(), probs[:, 1].numpy()) if n_classes == 2 else roc_auc_score(labels_cat.numpy(), probs.numpy(), multi_class="ovr", average="macro")

    return {"acc": accuracy, "f1": f1, "auroc": auroc}


def _train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    cfg: DictConfig,
    n_classes: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> dict[str, float | None]:
    """Train for one epoch."""
    model.train()

    all_logits = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    use_amp = cfg.classifier.trainer.get("use_amp", False)
    if use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        scaler = None
        autocast_ctx = nullcontext()

    for batch in loader:
        optimizer.zero_grad()

        features, labels = _flatten_batch(batch, device)

        with autocast_ctx:
            logits = model(features)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.classifier.trainer.get("grad_clip", 1.0),
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.classifier.trainer.get("grad_clip", 1.0),
            )
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = _compute_metrics_epoch(all_logits, all_labels, n_classes)

    return {"loss": avg_loss, **metrics}


def _validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    cfg: DictConfig,
    n_classes: int,
) -> dict[str, float | None]:
    """Validate for one epoch."""
    model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    use_amp = cfg.classifier.trainer.get("use_amp", False)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp and device.type == "cuda" else nullcontext()

    with torch.no_grad():
        for batch in loader:
            features, labels = _flatten_batch(batch, device)

            with autocast_ctx:
                logits = model(features)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = _compute_metrics_epoch(all_logits, all_labels, n_classes)

    return {"loss": avg_loss, **metrics}


def train(cfg: DictConfig) -> dict:
    """Train MLP classifier.

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_dl, val_dl, test_dl, meta = make_classification_dataloaders(cfg)
    _gather_meta(cfg, meta)

    # Compute class weights
    class_weights = _compute_class_weights(train_dl, meta["n_classes"], device)

    # Model assembly
    model = build_from_config(cfg, meta).to(device)

    # Initialize W&B (returns None if disabled or on error - training continues normally)
    wandb_run = init_wandb(cfg, model=model)

    # Log parameter count
    param_count = model.count_parameters()
    print(f"MLP parameters: {param_count:,}")

    # AdamW optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.classifier.trainer.lr,
        weight_decay=cfg.classifier.trainer.get("weight_decay", 1e-2),
    )

    # Optional warmup scheduler
    warmup_steps = cfg.classifier.trainer.get("warmup_steps", 0)
    warmup_scheduler = None
    if warmup_steps > 0:
        from torch.optim.lr_scheduler import LinearLR

        warmup_scheduler = LinearLR(opt, start_factor=0.1, total_iters=warmup_steps)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Logging dir from Hydra
    outdir = None
    if cfg.logging.save_artifacts:
        outdir = os.getcwd()
        os.makedirs(outdir, exist_ok=True)

        config_path = os.path.join(outdir, "resolved_config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)

    # on_start
    if outdir and cfg.logging.save_artifacts:
        start_payload = build_event_payload(
            moment="on_start",
            run_dir=outdir,
            cfg=cfg,
            seed=cfg.classifier.trainer.seed,
            class_weights=class_weights.cpu().tolist(),
        )
        append_jsonl_event(str(outdir), start_payload)
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_start", start_payload)

        # Emit facts/meta.json for semantic slicing in W&B
        try:
            meta = build_meta(cfg)
            write_meta(meta, Path(outdir) / "facts" / "meta.json")
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("[meta] Could not write meta.json: %s", e)

    # Training loop
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
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    last_valid_auroc = None
    global_step = 0
    training_start_time = time.time()

    # Early stopping state
    early_stopping_cfg = cfg.classifier.trainer.get("early_stopping", {})
    early_stopping_enabled = early_stopping_cfg.get("enabled", False)
    patience_counter = 0
    best_model_state = None
    early_stopping_triggered = False

    progress = TrainingProgressShower(cfg.classifier.trainer.epochs)

    for epoch in range(cfg.classifier.trainer.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = _train_one_epoch(model, train_dl, opt, criterion, device, cfg, meta["n_classes"], warmup_scheduler)
        histories["train_loss"].append(train_metrics["loss"])
        histories["train_acc"].append(train_metrics["acc"])
        histories["train_f1"].append(train_metrics["f1"])
        if train_metrics["auroc"] is not None:
            last_valid_auroc = train_metrics["auroc"]
        histories["train_auroc"].append(last_valid_auroc if train_metrics["auroc"] is None else train_metrics["auroc"])

        # Validate
        val_metrics = _validate_one_epoch(model, val_dl, criterion, device, cfg, meta["n_classes"])
        histories["val_loss"].append(val_metrics["loss"])
        histories["val_acc"].append(val_metrics["acc"])
        histories["val_f1"].append(val_metrics["f1"])
        if val_metrics["auroc"] is not None:
            last_valid_auroc = val_metrics["auroc"]
        histories["val_auroc"].append(last_valid_auroc if val_metrics["auroc"] is None else val_metrics["auroc"])

        epoch_time = time.time() - epoch_start
        global_step += len(train_dl)

        # Checkpointing
        improved = False
        if val_metrics["loss"] < best_val_loss:
            improvement = best_val_loss - val_metrics["loss"]
            min_delta = early_stopping_cfg.get("min_delta", 0.0)
            if improvement >= min_delta:
                improved = True
                best_val_loss = val_metrics["loss"]
                best_val_acc = val_metrics["acc"]
                best_epoch = epoch
                if early_stopping_enabled:
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if outdir:
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": warmup_scheduler.state_dict() if warmup_scheduler else None,
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_metric": best_val_loss,
                        "meta": {
                            "n_tokens": meta["n_tokens"],
                            "num_types": meta.get("num_types"),
                            "n_classes": meta["n_classes"],
                            "class_weights": class_weights.cpu().tolist(),
                        },
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    }
                    torch.save(checkpoint, os.path.join(outdir, "best_val.pt"))

                    model_path = os.path.join(outdir, "model.pt")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    try:
                        os.symlink("best_val.pt", model_path)
                    except OSError:
                        import shutil

                        shutil.copy2(os.path.join(outdir, "best_val.pt"), model_path)

                    try:
                        _save_split_scores_and_embeddings(model, val_dl, device, outdir, split="val")
                    except Exception as e:
                        import warnings

                        warnings.warn(f"Failed to save validation scores/embeddings: {e}", stacklevel=2)

        # Early stopping check
        if early_stopping_enabled:
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                patience = early_stopping_cfg.get("patience", 10)
                if patience_counter >= patience:
                    early_stopping_triggered = True
                    if outdir:
                        import warnings

                        warnings.warn(
                            f"Early stopping triggered at epoch {epoch}: validation loss did not improve for {patience} epochs. " f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}",
                            stacklevel=2,
                        )
                    break

        # Emit facts
        if outdir:
            val_auroc_for_facts = last_valid_auroc if val_metrics["auroc"] is None else val_metrics["auroc"]

            payload = build_event_payload(
                moment="on_epoch_end",
                run_dir=outdir,
                epoch=epoch,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                metrics={
                    "acc": val_metrics["acc"],
                    "f1": val_metrics["f1"],
                    "auroc": val_auroc_for_facts,
                },
                histories=histories,
                epoch_time_s=epoch_time,
                cfg=cfg,
            )
            append_jsonl_event(str(outdir), payload)
            append_scalars_csv(
                str(outdir),
                epoch=epoch,
                split="val",
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                metrics={
                    "acc": val_metrics["acc"],
                    "f1": val_metrics["f1"],
                    "auroc": val_auroc_for_facts,
                },
                epoch_time_s=epoch_time,
                throughput=None,
                max_memory_mib=None,
            )
            handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_epoch_end", payload)

        # W&B logging (alongside Facts, not replacing)
        val_auroc_for_wandb = last_valid_auroc if val_metrics["auroc"] is None else val_metrics["auroc"]
        log_metrics(
            wandb_run,
            {
                "epoch": epoch,
                "train/loss": float(train_metrics["loss"]),
                "train/acc": float(train_metrics["acc"]),
                "train/f1": float(train_metrics["f1"]),
                "val/loss": float(val_metrics["loss"]),
                "val/acc": float(val_metrics["acc"]),
                "val/f1": float(val_metrics["f1"]),
                "val/auroc": float(val_auroc_for_wandb) if val_auroc_for_wandb is not None else None,
                "perf/epoch_time_s": float(epoch_time),
            },
            step=epoch,
        )

        progress.update(epoch, epoch_time, train_loss=train_metrics["loss"], val_loss=val_metrics["loss"])

    # Restore best model weights if early stopping was enabled and triggered
    if early_stopping_enabled and early_stopping_triggered and best_model_state is not None:
        restore_best_weights = early_stopping_cfg.get("restore_best_weights", True)
        if restore_best_weights:
            model.load_state_dict(best_model_state)
            if outdir:
                import warnings

                warnings.warn(
                    f"Restored best model weights from epoch {best_epoch} (val_loss: {best_val_loss:.6f})",
                    stacklevel=2,
                )

    # Test evaluation
    test_metrics = _validate_one_epoch(model, test_dl, criterion, device, cfg, meta["n_classes"])

    # on_train_end
    if outdir:
        total_time = time.time() - training_start_time

        try:
            _save_split_scores_and_embeddings(model, test_dl, device, outdir, split="test")
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to save test scores/embeddings: {e}", stacklevel=2)

        payload_end = build_event_payload(
            moment="on_train_end",
            run_dir=outdir,
            total_time_s=total_time,
            histories=histories,
            cfg=cfg,
        )
        append_jsonl_event(str(outdir), payload_end)
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_train_end", payload_end)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": warmup_scheduler.state_dict() if warmup_scheduler else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_val_loss,
            "meta": {
                "n_tokens": meta["n_tokens"],
                "num_types": meta.get("num_types"),
                "n_classes": meta["n_classes"],
                "class_weights": class_weights.cpu().tolist(),
            },
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(checkpoint, os.path.join(outdir, "last.pt"))

        # Upload model artifact to W&B
        best_val_artifact_path = Path(outdir) / "best_val.pt"
        log_artifact(wandb_run, best_val_artifact_path, "model", cfg)

    # Log test metrics to W&B
    log_metrics(
        wandb_run,
        {
            "test/loss": float(test_metrics["loss"]),
            "test/acc": float(test_metrics["acc"]),
            "test/f1": float(test_metrics["f1"]),
            "test/auroc": float(test_metrics["auroc"]) if test_metrics["auroc"] is not None else None,
        },
        step=int(cfg.classifier.trainer.epochs),
    )

    # Finish W&B run
    finish_wandb(wandb_run)

    return {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
    }
