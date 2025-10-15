from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from thesis_ml.data import build_dataloaders
from thesis_ml.models import build_model
from thesis_ml.utils import plot_loss_curve, set_all_seeds


def _select_device(cfg) -> torch.device:
    device_pref = str(cfg.trainer.device)
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


def _ensure_run_dir(cfg) -> Path | None:
    if bool(cfg.logging.save_artifacts):
        root = Path(cfg.logging.output_root)
        run_dir = root / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        # Save composed config for reproducibility
        with (run_dir / "cfg.yaml").open("w", encoding="utf-8") as f:
            OmegaConf.save(config=cfg, f=f)
        return run_dir
    return None


def train(cfg) -> dict:
    set_all_seeds(int(cfg.data.seed))
    device = _select_device(cfg)

    train_loader, val_loader, meta = build_dataloaders(cfg)
    model = build_model(cfg, input_dim=meta["input_dim"], task=meta["task"]).to(device)

    # Optional: initialize Weights & Biases
    wandb_run = None
    if bool(cfg.logging.use_wandb):
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(cfg.logging.wandb.project),
                entity=str(cfg.logging.wandb.entity) or None,
                name=str(cfg.logging.wandb.run_name) or None,
                mode=str(cfg.logging.wandb.mode),
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

    for epoch in range(int(cfg.trainer.epochs)):
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

        if task == "binary" and total > 0:
            acc = correct / total
            print(f"Epoch {epoch+1}: train_loss={epoch_train_loss:.4f} " f"val_loss={epoch_val_loss:.4f} acc={acc:.3f}")
        else:
            print(f"Epoch {epoch+1}: train_loss={epoch_train_loss:.4f} val_loss={epoch_val_loss:.4f}")

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
        if bool(cfg.logging.make_plots) or bool(cfg.logging.show_plots):
            plots_dir = run_dir / cfg.logging.plots_subdir if str(cfg.logging.plots_subdir) else run_dir
            out_path = plots_dir / f"loss.{cfg.logging.fig_format}"
            plot_loss_curve(
                train_losses,
                val_losses,
                show=bool(cfg.logging.show_plots),
                save=bool(cfg.logging.make_plots),
                out_path=out_path,
            )
        # Upload artifacts to W&B
        if wandb_run is not None and bool(cfg.logging.wandb.log_artifacts):
            try:  # pragma: no cover - logging side-effect only
                import wandb

                art = wandb.Artifact("model", type="model")
                art.add_file(str(model_path))
                wandb.log_artifact(art)
                if ("out_path" in locals()) and out_path is not None and out_path.exists():
                    wandb.log({"loss_curve": wandb.Image(str(out_path))})
            except Exception as e:
                print(f"[wandb] artifact log failed: {e}")
        saved_path = str(run_dir.resolve())
    else:
        # Ephemeral: write to temp dir and remove after
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            model_path = tmp_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            if bool(cfg.logging.make_plots) or bool(cfg.logging.show_plots):
                plots_dir = tmp_dir / cfg.logging.plots_subdir if str(cfg.logging.plots_subdir) else tmp_dir
                out_path = plots_dir / f"loss.{cfg.logging.fig_format}"
                plot_loss_curve(
                    train_losses,
                    val_losses,
                    show=bool(cfg.logging.show_plots),
                    save=bool(cfg.logging.make_plots),
                    out_path=out_path,
                )
            # Upload artifacts to W&B before temp dir deletes
            if wandb_run is not None and bool(cfg.logging.wandb.log_artifacts):
                try:  # pragma: no cover - logging side-effect only
                    import wandb

                    art = wandb.Artifact("model", type="model")
                    art.add_file(str(model_path))
                    wandb.log_artifact(art)
                    if ("out_path" in locals()) and out_path is not None and out_path.exists():
                        wandb.log({"loss_curve": wandb.Image(str(out_path))})
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


# Hydra CLI entrypoint kept separate to avoid side-effects in notebooks
def _hydra_main(cfg) -> None:  # pragma: no cover - thin wrapper
    train(cfg)


try:
    import hydra

    @hydra.main(config_path="../../../configs", config_name="config", version_base=None)
    def main(cfg) -> None:  # pragma: no cover - CLI only
        _hydra_main(cfg)

except Exception:  # pragma: no cover - allow import without hydra installed

    def main():  # type: ignore
        raise RuntimeError("Hydra is required for CLI usage. Install hydra-core to run main().")


if __name__ == "__main__":  # when this file is run direcly, this will run the main function
    main()
