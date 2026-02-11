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

from thesis_ml.architectures.transformer_classifier.base import build_from_config
from thesis_ml.data.h5_loader import make_classification_dataloaders
from thesis_ml.facts import append_jsonl_event, append_scalars_csv, build_event_payload, build_meta, write_meta
from thesis_ml.monitoring.orchestrator import handle_event
from thesis_ml.utils import TrainingProgressShower
from thesis_ml.utils.seed import set_all_seeds
from thesis_ml.utils.wandb_utils import finish_wandb, init_wandb, log_artifact, log_metrics

SUPPORTED_PLOT_FAMILIES = {"losses", "metrics"}


def _save_split_scores_and_embeddings(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    outdir: str,
    split: str = "val",
) -> None:
    """Save logits, probabilities, labels, and CLS-pooled embeddings for a split.

    Uses a forward hook on the classifier linear layer to capture pooled embeddings.
    """
    model.eval()

    # Storage
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_pooled: list[torch.Tensor] = []

    # Forward hook to capture Linear input (pooled features)
    captured: list[torch.Tensor] = []

    def hook_fn(module, inputs, output):
        # inputs is a tuple; take first tensor [B, D]
        if inputs and torch.is_tensor(inputs[0]):
            captured.append(inputs[0].detach().cpu())

    hook = None
    if hasattr(model, "head") and hasattr(model.head, "classifier"):
        hook = model.head.classifier.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in loader:
            # Unpack batch (raw vs binned)
            if len(batch) == 5:  # raw format
                tokens_cont, tokens_id, globals, mask, label = batch
                tokens_cont = tokens_cont.to(device)
                tokens_id = tokens_id.to(device)
                globals = globals.to(device)
                mask = mask.to(device)
                label = label.to(device)
                logits = model(tokens_cont, tokens_id, globals, mask=mask)
            else:  # binned format
                integer_tokens, _globals_ints, mask, label = batch
                integer_tokens = integer_tokens.to(device)
                mask = mask.to(device)
                label = label.to(device)
                logits = model(integer_tokens, mask=mask)

            all_logits.append(logits.detach().cpu())
            all_labels.append(label.detach().cpu())

            # Collect pooled features captured by hook (if available)
            if captured:
                all_pooled.append(captured[-1])

    # Remove hook
    if hook is not None:
        hook.remove()

    # Concatenate
    logits_cat = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, 0)
    labels_cat = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    probs_cat = torch.softmax(logits_cat, dim=-1) if logits_cat.numel() > 0 else logits_cat
    pooled_cat = torch.cat(all_pooled, dim=0) if all_pooled else torch.empty(0, 0)

    # Persist
    os.makedirs(outdir, exist_ok=True)
    torch.save({"logits": logits_cat, "probs": probs_cat, "labels": labels_cat, "pooled": pooled_cat}, os.path.join(outdir, f"{split}_scores.pt"))


def _gather_meta(cfg: DictConfig, ds_meta: Mapping[str, Any]) -> None:
    """Attach data-derived meta to cfg for module constructors."""
    # #region agent log
    try:
        import json

        _tf = ds_meta.get("token_feat_dim")
        _nt = ds_meta.get("num_types")
        with open(r"c:\Users\niels\Projects\Thesis-Code\Code\Niels_repo\.cursor\debug.log", "a") as _f:
            _f.write(json.dumps({"location": "transformer_classifier._gather_meta", "message": "ds_meta token_feat_dim and num_types", "data": {"token_feat_dim": _tf, "num_types": _nt, "token_feat_dim_is_none": _tf is None, "num_types_is_none": _nt is None}, "hypothesisId": "H1_H2", "timestamp": __import__("time").time()}) + "\n")
    except Exception:
        pass
    # #endregion
    prev_struct = OmegaConf.is_struct(cfg)
    try:
        OmegaConf.set_struct(cfg, False)
        _token_feat_dim = ds_meta.get("token_feat_dim", 4)
        _num_types = ds_meta.get("num_types", 0)
        cfg.meta = OmegaConf.create(
            {
                "n_tokens": int(ds_meta["n_tokens"]),
                "token_feat_dim": int(_token_feat_dim) if _token_feat_dim is not None else 4,
                "has_globals": bool(ds_meta.get("has_globals", False)),
                "n_classes": int(ds_meta["n_classes"]),
                "num_types": int(_num_types) if _num_types is not None else 0,  # For identity tokenizer
                "vocab_size": ds_meta.get("vocab_size"),  # For binned tokenizer
            }
        )
    finally:
        OmegaConf.set_struct(cfg, prev_struct)


def _compute_class_weights(
    train_loader: torch.utils.data.DataLoader,
    n_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute class weights from train loader for imbalanced datasets.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Training data loader
    n_classes : int
        Number of classes
    device : torch.device
        Device to place weights on

    Returns
    -------
    torch.Tensor
        Class weights [n_classes] on specified device
    """
    # Count class occurrences in train loader
    class_counts = torch.zeros(n_classes, dtype=torch.long)

    for batch in train_loader:
        # Detect format by batch length
        if len(batch) == 5:  # raw format
            _, _, _, _, labels = batch
        else:  # binned format (4 items)
            _, _, _, labels = batch

        # Count occurrences of each class
        for c in range(n_classes):
            class_counts[c] += (labels == c).sum().item()

    # Compute weights: inverse frequency, normalized
    total = class_counts.sum().float()
    weights = total / (n_classes * class_counts.float() + 1e-8)
    weights = weights / weights.sum() * n_classes  # Normalize so sum = n_classes

    return weights.to(device)


def _compute_metrics_epoch(
    all_logits: list[torch.Tensor],
    all_labels: list[torch.Tensor],
    n_classes: int,
) -> dict[str, float | None]:
    """Compute classification metrics at epoch level (accumulated over all batches).

    Parameters
    ----------
    all_logits : list[torch.Tensor]
        List of logit tensors [B, n_classes] from all batches (CPU tensors)
    all_labels : list[torch.Tensor]
        List of label tensors [B] from all batches (CPU tensors)
    n_classes : int
        Number of classes

    Returns
    -------
    dict[str, float | None]
        Dictionary with keys: acc, f1, auroc (auroc may be None if single class)
    """
    # Concatenate all logits and labels
    logits_cat = torch.cat(all_logits, dim=0).cpu()  # [N, n_classes]
    labels_cat = torch.cat(all_labels, dim=0).cpu()  # [N]

    # Compute probabilities and predictions
    probs = torch.softmax(logits_cat, dim=-1)  # [N, n_classes]
    preds = probs.argmax(dim=-1)  # [N]

    # Accuracy
    accuracy = (preds == labels_cat).float().mean().item()

    # F1 score
    f1 = f1_score(labels_cat.numpy(), preds.numpy(), average="weighted")

    # AUROC (guard edge cases)
    unique_labels = labels_cat.unique()
    # Single class in batch/epoch - skip AUROC
    # Binary classification: use positive class probability
    # Multi-class: use one-vs-rest
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
    """Train for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    loader : torch.utils.data.DataLoader
        Training data loader
    optimizer : torch.optim.Optimizer
        Optimizer
    criterion : torch.nn.Module
        Loss function
    device : torch.device
        Device to run on
    cfg : DictConfig
        Configuration
    n_classes : int
        Number of classes
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler (for warmup)

    Returns
    -------
    dict[str, float | None]
        Dictionary with keys: loss, acc, f1, auroc (auroc may be None)
    """
    model.train()

    # Initialize accumulators
    all_logits = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    # AMP setup
    use_amp = cfg.classifier.trainer.get("use_amp", False)
    if use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        scaler = None
        autocast_ctx = nullcontext()

    # Loop batches
    for batch in loader:
        optimizer.zero_grad()

        # Explicit batch unpacking
        if len(batch) == 5:  # raw format
            tokens_cont, tokens_id, globals, mask, label = batch
            tokens_cont = tokens_cont.to(device)
            tokens_id = tokens_id.to(device)
            globals = globals.to(device)
            mask = mask.to(device)
            label = label.to(device)

            with autocast_ctx:
                logits = model(tokens_cont, tokens_id, globals, mask=mask)
        else:  # binned format (4 items)
            integer_tokens, globals_ints, mask, label = batch
            integer_tokens = integer_tokens.to(device)
            mask = mask.to(device)
            label = label.to(device)

            with autocast_ctx:
                logits = model(integer_tokens, mask=mask)

        # Loss
        loss = criterion(logits, label)

        # Backward
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

        # Scheduler step (for warmup)
        if scheduler is not None:
            scheduler.step()

        # Accumulate
        total_loss += loss.item()
        num_batches += 1
        all_logits.append(logits.detach().cpu())
        all_labels.append(label.cpu())

    # Compute metrics
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
    """Validate for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to validate
    loader : torch.utils.data.DataLoader
        Validation data loader
    criterion : torch.nn.Module
        Loss function
    device : torch.device
        Device to run on
    cfg : DictConfig
        Configuration
    n_classes : int
        Number of classes

    Returns
    -------
    dict[str, float | None]
        Dictionary with keys: loss, acc, f1, auroc (auroc may be None)
    """
    model.eval()

    # Initialize accumulators
    all_logits = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    # AMP setup
    use_amp = cfg.classifier.trainer.get("use_amp", False)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp and device.type == "cuda" else nullcontext()

    with torch.no_grad():
        for batch in loader:
            # Explicit batch unpacking
            if len(batch) == 5:  # raw format
                tokens_cont, tokens_id, globals, mask, label = batch
                tokens_cont = tokens_cont.to(device)
                tokens_id = tokens_id.to(device)
                globals = globals.to(device)
                mask = mask.to(device)
                label = label.to(device)

                with autocast_ctx:
                    logits = model(tokens_cont, tokens_id, globals, mask=mask)
            else:  # binned format (4 items)
                integer_tokens, globals_ints, mask, label = batch
                integer_tokens = integer_tokens.to(device)
                mask = mask.to(device)
                label = label.to(device)

                with autocast_ctx:
                    logits = model(integer_tokens, mask=mask)

            # Loss
            loss = criterion(logits, label)

            # Accumulate
            total_loss += loss.item()
            num_batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(label.cpu())

    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = _compute_metrics_epoch(all_logits, all_labels, n_classes)

    return {"loss": avg_loss, **metrics}


def train(cfg: DictConfig) -> dict:
    """Train transformer classifier.

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

    # Basic diagnostics for dry runs: parameter count and input sequence length
    try:
        total_params = sum(p.numel() for p in model.parameters())
        # Peek one batch to infer effective sequence length
        first_batch = next(iter(train_dl))
        if len(first_batch) == 5:  # raw format
            tokens_cont, tokens_id, globals, mask, _ = first_batch
            seq_len = int(tokens_cont.shape[1])
        else:  # binned format
            integer_tokens, globals_ints, mask, _ = first_batch
            seq_len = int(integer_tokens.shape[1])
    except Exception:
        total_params = None
        seq_len = None

    # Initialize W&B (returns None if disabled or on error - training continues normally)
    wandb_run = init_wandb(cfg, model=model)

    # Log basic diagnostics once
    diag_metrics = {}
    if total_params is not None:
        diag_metrics["model/num_parameters"] = float(total_params)
    if seq_len is not None:
        diag_metrics["data/seq_length_tokens"] = float(seq_len)
    if diag_metrics:
        log_metrics(wandb_run, diag_metrics, step=0)

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

    # Unified loss: Always use CrossEntropyLoss with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Logging dir from Hydra
    outdir = None
    if cfg.logging.save_artifacts:
        outdir = os.getcwd()
        os.makedirs(outdir, exist_ok=True)

        # Dump resolved config
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
        # Must happen after Hydra resolution, before training starts
        # Use facts_meta so we do not overwrite dataloader meta (n_tokens, n_classes, etc.)
        try:
            facts_meta = build_meta(cfg)
            write_meta(facts_meta, Path(outdir) / "facts" / "meta.json")
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
    last_valid_auroc = None  # Carry last valid AUROC if single class detected
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
        # Handle None AUROC (carry last valid value)
        if train_metrics["auroc"] is not None:
            last_valid_auroc = train_metrics["auroc"]
        histories["train_auroc"].append(last_valid_auroc if train_metrics["auroc"] is None else train_metrics["auroc"])

        # Validate
        val_metrics = _validate_one_epoch(model, val_dl, criterion, device, cfg, meta["n_classes"])
        histories["val_loss"].append(val_metrics["loss"])
        histories["val_acc"].append(val_metrics["acc"])
        histories["val_f1"].append(val_metrics["f1"])
        # Handle None AUROC (carry last valid value)
        if val_metrics["auroc"] is not None:
            last_valid_auroc = val_metrics["auroc"]
        histories["val_auroc"].append(last_valid_auroc if val_metrics["auroc"] is None else val_metrics["auroc"])

        epoch_time = time.time() - epoch_start
        global_step += len(train_dl)

        # Checkpointing (comprehensive)
        improved = False
        if val_metrics["loss"] < best_val_loss:
            improvement = best_val_loss - val_metrics["loss"]
            min_delta = early_stopping_cfg.get("min_delta", 0.0)
            if improvement >= min_delta:
                improved = True
                best_val_loss = val_metrics["loss"]
                best_val_acc = val_metrics["acc"]
                best_epoch = epoch
                # Save best model state for early stopping restoration
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
                            "tokenizer_name": cfg.classifier.model.tokenizer.name,
                            "tokenizer_version": "1.0",
                        },
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    }
                torch.save(checkpoint, os.path.join(outdir, "best_val.pt"))

                # Create symlink
                model_path = os.path.join(outdir, "model.pt")
                if os.path.exists(model_path):
                    os.remove(model_path)
                try:
                    os.symlink("best_val.pt", model_path)
                except OSError:
                    import shutil

                    shutil.copy2(os.path.join(outdir, "best_val.pt"), model_path)

                # Save per-event scores and CLS embeddings on validation split for downstream analysis
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
            # Use last valid AUROC if current is None
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

        # Log AUROC warning if single class detected
        if outdir and val_metrics["auroc"] is None:
            import warnings

            warnings.warn(
                f"Epoch {epoch}: Single class detected in validation set, skipping AUROC computation.",
                stacklevel=2,
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

        # Save per-event scores and CLS embeddings on test split as well
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

        # Save final checkpoint (comprehensive)
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
                "tokenizer_name": cfg.classifier.model.tokenizer.name,
                "tokenizer_version": "1.0",
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
