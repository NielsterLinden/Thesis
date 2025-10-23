from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from thesis_ml.data import build_dataloaders
from thesis_ml.general.models.mlp import build_model
from thesis_ml.plots.io_utils import append_jsonl_event, append_scalars_csv
from thesis_ml.plots.orchestrator import handle_event
from thesis_ml.utils import TrainingProgressShower, set_all_seeds

SUPPORTED_PLOT_FAMILIES = {"losses", "metrics", "latency"}


def _select_device(cfg) -> torch.device:
    device_pref = str(cfg.trainer.get("device", "auto"))
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


def _ensure_run_dir(cfg) -> Path | None:
    if bool(cfg.logging.save_artifacts):
        run_dir = Path(os.getcwd())  # Hydra sets this when chdir=true
        run_dir.mkdir(parents=True, exist_ok=True)
        # Save composed config for reproducibility
        with (run_dir / "cfg.yaml").open("w", encoding="utf-8") as f:
            OmegaConf.save(config=cfg, f=f)
        return run_dir
    return None


def train(cfg) -> dict:
    set_all_seeds(int(cfg.data.seed))
    device = _select_device(cfg)
    print(f"Using device: {device} (cuda_available={torch.cuda.is_available()})")

    train_loader, val_loader, meta = build_dataloaders(cfg)
    model = build_model(cfg, input_dim=meta["input_dim"], task=meta["task"]).to(device)

    # Optional: initialize Weights & Biases
    wandb_run = None
    if bool(cfg.logging.use_wandb):
        try:
            import wandb

            # Ensure W&B output directory exists
            wandb_dir = Path(str(cfg.logging.wandb.dir)).resolve()
            wandb_dir.mkdir(parents=True, exist_ok=True)

            wandb_run = wandb.init(
                project=str(cfg.logging.wandb.project),
                entity=str(cfg.logging.wandb.entity) or None,
                name=str(cfg.logging.wandb.run_name) or None,
                mode=str(cfg.logging.wandb.mode),
                dir=str(wandb_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            if bool(cfg.logging.wandb.watch_model):
                wandb.watch(
                    model,
                    log="all",
                    log_freq=int(cfg.logging.wandb.log_freq),
                )
        except Exception as e:  # pragma: no cover - logging side-effect only
            print(f"[wandb] disabled due to init error: {e}")
            wandb_run = None

    task = meta["task"]
    if task == "regression":
        criterion = torch.nn.MSELoss()
    elif task == "binary":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown task: {task}")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.trainer.lr))

    run_dir = _ensure_run_dir(cfg)

    train_losses, val_losses = [], []
    history_metrics: dict[str, list[float]] = {}
    history_epoch_time_s: list[float] = []
    history_throughput: list[float] = []

    # on_start event
    if run_dir is not None:
        start_payload = {
            "run_dir": str(run_dir),
            "epoch": None,
            "step": None,
            "metrics": {},
            "epoch_time_s": None,
            "total_time_s": None,
        }
        if bool(cfg.logging.save_artifacts):
            append_jsonl_event(str(run_dir), {"moment": "on_start", **start_payload})
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_start", start_payload)

    total_t0 = time.perf_counter()
    progress = TrainingProgressShower(total_epochs=int(cfg.trainer.epochs), bar_width=30)
    for epoch in range(int(cfg.trainer.epochs)):
        t0 = time.perf_counter()
        model.train()
        epoch_train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * xb.size(0)
        epoch_train_loss /= len(train_loader.dataset)

        model.eval()
        epoch_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                epoch_val_loss += loss.item() * xb.size(0)
                if task == "binary":
                    pred_labels = (preds.sigmoid() > 0.5).float()
                    correct += (pred_labels == yb).sum().item()
                    total += yb.numel()
        epoch_val_loss /= len(val_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        dt = time.perf_counter() - t0
        if task == "binary" and total > 0:
            acc = correct / total
            history_metrics.setdefault("acc", []).append(float(acc))
            progress.update(
                epoch,
                dt,
                train_loss=epoch_train_loss,
                val_loss=epoch_val_loss,
                extras={"acc": float(acc)},
            )
        else:
            progress.update(
                epoch,
                dt,
                train_loss=epoch_train_loss,
                val_loss=epoch_val_loss,
            )

        # update histories
        history_epoch_time_s.append(float(dt))
        if dt > 0 and hasattr(train_loader, "dataset"):
            try:
                n_samples = len(train_loader.dataset)
                history_throughput.append(float(n_samples) / float(dt))
            except Exception:
                history_throughput.append(0.0)

        # Emit on_epoch_end event and write facts
        max_mem = None
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                max_mem = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            except Exception:
                max_mem = None
        payload = {
            "run_dir": str(run_dir) if run_dir is not None else "",
            "epoch": int(epoch),
            "split": "val",
            "train_loss": float(epoch_train_loss),
            "val_loss": float(epoch_val_loss),
            "metrics": {k: v[-1] for k, v in history_metrics.items() if v},
            "epoch_time_s": float(dt),
            "throughput": history_throughput[-1] if history_throughput else None,
            "max_memory_mib": max_mem,
            # histories for families
            "history_train_loss": list(train_losses),
            "history_val_loss": list(val_losses),
            "history_metrics": {k: list(v) for k, v in history_metrics.items()},
            "history_epoch_time_s": list(history_epoch_time_s),
            "history_throughput": list(history_throughput),
        }
        if run_dir is not None and bool(cfg.logging.save_artifacts):
            append_jsonl_event(str(run_dir), {"moment": "on_epoch_end", **payload})
            append_scalars_csv(
                str(run_dir),
                epoch=epoch,
                split="val",
                train_loss=epoch_train_loss,
                val_loss=epoch_val_loss,
                metrics=payload["metrics"],
                epoch_time_s=dt,
                throughput=payload["throughput"],
                max_memory_mib=max_mem,
            )
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_epoch_end", payload)

        # Per-epoch W&B logging
        if wandb_run is not None:
            log = {
                "epoch": epoch + 1,
                "train/loss": float(epoch_train_loss),
                "val/loss": float(epoch_val_loss),
            }
            if task == "binary" and total > 0:
                log["val/acc"] = float(acc)
            try:  # pragma: no cover - logging side-effect only
                import wandb

                wandb.log(log, step=epoch + 1)
            except Exception as e:
                print(f"[wandb] log failed: {e}")

    # Save artifacts
    saved_path: str | None = None
    if run_dir is not None:
        model_path = run_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        total_time = time.perf_counter() - total_t0
        # train_end event
        payload_end = {
            "run_dir": str(run_dir),
            "epoch": int(len(train_losses) - 1) if train_losses else -1,
            "train_loss": float(train_losses[-1]) if train_losses else None,
            "val_loss": float(val_losses[-1]) if val_losses else None,
            "metrics": {k: v[-1] for k, v in history_metrics.items() if v},
            "total_time_s": float(total_time),
            "history_train_loss": list(train_losses),
            "history_val_loss": list(val_losses),
            "history_metrics": {k: list(v) for k, v in history_metrics.items()},
            "history_epoch_time_s": list(history_epoch_time_s),
            "history_throughput": list(history_throughput),
        }
        if bool(cfg.logging.save_artifacts):
            append_jsonl_event(str(run_dir), {"moment": "on_train_end", **payload_end})
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_train_end", payload_end)
        # Upload artifacts to W&B
        if wandb_run is not None and bool(cfg.logging.wandb.log_artifacts):
            try:  # pragma: no cover - logging side-effect only
                import wandb

                art = wandb.Artifact("model", type="model")
                art.add_file(str(model_path))
                wandb.log_artifact(art)
                # Optional future: log figures via external tracker
            except Exception as e:
                print(f"[wandb] artifact log failed: {e}")
        saved_path = str(run_dir.resolve())
    else:
        # Ephemeral: write to temp dir and remove after
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            model_path = tmp_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            # Emit a final event without saving
            payload_end = {
                "run_dir": str(tmp_dir),
                "epoch": int(len(train_losses) - 1) if train_losses else -1,
                "train_loss": float(train_losses[-1]) if train_losses else None,
                "val_loss": float(val_losses[-1]) if val_losses else None,
                "metrics": {k: v[-1] for k, v in history_metrics.items() if v},
                "total_time_s": float(time.perf_counter() - total_t0),
                "history_train_loss": list(train_losses),
                "history_val_loss": list(val_losses),
                "history_metrics": {k: list(v) for k, v in history_metrics.items()},
                "history_epoch_time_s": list(history_epoch_time_s),
                "history_throughput": list(history_throughput),
            }
            handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_train_end", payload_end)
            # Upload artifacts to W&B before temp dir deletes
            if wandb_run is not None and bool(cfg.logging.wandb.log_artifacts):
                try:  # pragma: no cover - logging side-effect only
                    import wandb

                    art = wandb.Artifact("model", type="model")
                    art.add_file(str(model_path))
                    wandb.log_artifact(art)
                except Exception as e:
                    print(f"[wandb] artifact log failed: {e}")
            # directory auto-deletes here

    # Finish W&B run
    if wandb_run is not None:
        try:  # pragma: no cover - logging side-effect only
            wandb_run.finish()
        except Exception as e:
            print(f"[wandb] finish failed: {e}")

    return {
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "task": task,
        "device": str(device),
        "saved_to": saved_path,
    }
