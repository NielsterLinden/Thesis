"""Centralized W&B utilities for consistent behavior across all training loops.

This module provides a thin wrapper around W&B that:
- Returns None on failure (never raises) so training can continue without W&B
- Uses resume="allow" to prevent duplicates if re-run
- Defines consistent metric axes for proper epoch-based plotting
- Guards artifact uploads with config settings

The Facts system remains the canonical source of truth. W&B is a parallel
logging layer for visualization and collaboration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _get_run_name_from_cwd() -> str | None:
    """Extract a meaningful run name from the current working directory.

    Hydra sets cwd to the run directory, which typically has a name like:
    run_20260128-111834_experiment_job0

    Returns
    -------
    str | None
        Run name extracted from cwd, or None if not determinable
    """
    import os

    cwd = Path(os.getcwd())
    name = cwd.name

    # If it looks like a Hydra run directory, use it
    if name.startswith("run_") or name.startswith("exp_"):
        return name

    # Otherwise return the directory name as-is
    return name if name else None


def _get_group_from_run_name(run_name: str | None) -> str | None:
    """Extract group name from run directory name.

    For a run named: run_20260128-111834_compare_positional_encodings_job00
    Returns: exp_20260128-111834_compare_positional_encodings

    This allows runs from the same experiment to be grouped together in W&B.

    Parameters
    ----------
    run_name : str | None
        Run name (typically from _get_run_name_from_cwd)

    Returns
    -------
    str | None
        Group name or None if not extractable
    """
    if not run_name or not run_name.startswith("run_"):
        return None

    parts = run_name.split("_")
    if len(parts) < 4:
        return None

    # Format: run_YYYYMMDD-HHMMSS_experimentname_jobNN
    # Find the job suffix
    job_idx = None
    for i, part in enumerate(parts):
        if part.startswith("job"):
            job_idx = i
            break

    if job_idx is None or job_idx < 3:
        return None

    # Extract: timestamp + experiment name
    timestamp = parts[1]  # YYYYMMDD-HHMMSS
    experiment_name = "_".join(parts[2:job_idx])

    return f"exp_{timestamp}_{experiment_name}"


def init_wandb(cfg: DictConfig, model: Any = None) -> Any:
    """Initialize W&B run with safe defaults.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config with logging.use_wandb and logging.wandb settings
    model : Any, optional
        PyTorch model for optional gradient/parameter watching

    Returns
    -------
    wandb.sdk.wandb_run.Run | None
        W&B run object if successful, None if disabled or on error.
        Training should continue normally if None is returned.

    Notes
    -----
    - Never raises exceptions - logs warnings instead
    - Uses resume="allow" to prevent duplicates on re-run
    - Defines metric axes for consistent epoch-based plotting
    """
    if not cfg.logging.use_wandb:
        return None

    try:
        import wandb
        from omegaconf import OmegaConf

        wandb_cfg = cfg.logging.wandb
        wandb_dir = Path(str(wandb_cfg.dir)).resolve()
        wandb_dir.mkdir(parents=True, exist_ok=True)

        # Handle empty strings as None, with fallback to cwd-based name
        entity = str(wandb_cfg.entity) if wandb_cfg.entity else None
        run_name = str(wandb_cfg.run_name) if wandb_cfg.run_name else _get_run_name_from_cwd()

        # Auto-extract group from run name if not explicitly set
        # This groups runs from the same experiment (multirun) together
        group = str(wandb_cfg.group) if wandb_cfg.get("group") else _get_group_from_run_name(run_name)
        tags = list(wandb_cfg.tags) if wandb_cfg.get("tags") else None

        run = wandb.init(
            project=str(wandb_cfg.project),
            entity=entity,
            name=run_name,
            group=group,
            tags=tags,
            mode=str(wandb_cfg.mode),
            dir=str(wandb_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",  # Prevents duplicates on re-run
        )

        # Define metric axes for consistent plotting
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")

        # Optional model watching (gradients/parameters)
        if model is not None and wandb_cfg.get("watch_model", False):
            wandb.watch(
                model,
                log="all",
                log_freq=int(wandb_cfg.get("log_freq", 100)),
            )

        logger.info("[wandb] initialized: project=%s, mode=%s", wandb_cfg.project, wandb_cfg.mode)
        return run

    except Exception as e:
        logger.warning("[wandb] disabled due to init error: %s", e)
        return None


def log_metrics(wandb_run: Any, metrics: dict[str, Any], step: int) -> None:
    """Log metrics to W&B with automatic error handling.

    This is a thin wrapper that silently handles failures - never raises.
    Training loops can call this without try/except boilerplate.

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run | None
        W&B run object from init_wandb(), or None if disabled
    metrics : dict[str, Any]
        Metrics to log (e.g., {"train/loss": 0.5, "val/loss": 0.6})
    step : int
        Step number (typically epoch number)
    """
    if wandb_run is None:
        return

    try:
        import wandb

        wandb.log(metrics, step=step)
    except Exception as e:
        logger.warning("[wandb] log failed: %s", e)


def log_artifact(
    wandb_run: Any,
    path: Path,
    artifact_type: str,
    cfg: DictConfig,
    artifact_name: str | None = None,
) -> None:
    """Upload artifact to W&B if enabled in config.

    Respects cfg.logging.wandb.log_artifacts setting.

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run | None
        W&B run object from init_wandb(), or None if disabled
    path : Path
        Path to the artifact file (e.g., best_val.pt)
    artifact_type : str
        Type of artifact (e.g., "model", "dataset")
    cfg : DictConfig
        Full config to check log_artifacts setting
    artifact_name : str | None
        Custom artifact name. If None, uses path.stem
    """
    if wandb_run is None:
        return

    if not cfg.logging.wandb.get("log_artifacts", True):
        return

    if not path.exists():
        logger.warning("[wandb] artifact not found: %s", path)
        return

    try:
        import wandb

        name = artifact_name if artifact_name else path.stem
        art = wandb.Artifact(name, type=artifact_type)
        art.add_file(str(path))
        wandb.log_artifact(art)
        logger.info("[wandb] uploaded artifact: %s", name)
    except Exception as e:
        logger.warning("[wandb] artifact upload failed: %s", e)


def finish_wandb(wandb_run: Any) -> None:
    """Finish W&B run gracefully.

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run | None
        W&B run object from init_wandb(), or None if disabled
    """
    if wandb_run is None:
        return

    try:
        wandb_run.finish()
        logger.info("[wandb] run finished")
    except Exception as e:
        logger.warning("[wandb] finish failed: %s", e)
