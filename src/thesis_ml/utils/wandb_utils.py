"""Centralized W&B utilities for consistent behavior across all training loops.

This module provides a thin wrapper around W&B that:
- Returns None on failure (never raises) so training can continue without W&B
- Uses resume="allow" to prevent duplicates if re-run
- Defines consistent metric axes for proper epoch-based plotting
- Guards artifact uploads with config settings
- Extracts comprehensive config metadata for maximum WandB divisibility
- Integrates with the metadata schema (facts/meta.json) for semantic slicing

The Facts system remains the canonical source of truth. W&B is a parallel
logging layer for visualization and collaboration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Config Extraction for Maximum WandB Divisibility
# =============================================================================


def _safe_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    """Safely get nested value from config using dot notation.

    Parameters
    ----------
    cfg : dict
        Config dictionary
    path : str
        Dot-separated path (e.g., "classifier.model.dim")
    default : Any
        Default value if path not found

    Returns
    -------
    Any
        Value at path or default
    """
    try:
        val = cfg
        for key in path.split("."):
            if not isinstance(val, dict):
                return default
            val = val.get(key, {})
        return val if val != {} else default
    except Exception:
        return default


def extract_wandb_config(cfg: dict[str, Any], source_location: str = "live") -> dict[str, Any]:
    """Extract ALL config fields into flat WandB-friendly dict.

    Uses '/' as namespace separator for clean WandB UI organization.
    Every extracted value becomes filterable in WandB dashboards.

    This function is used both for:
    - Live training runs (via init_wandb)
    - Retroactive migration (via migrate_runs_to_wandb.py)

    Parameters
    ----------
    cfg : dict
        Full Hydra config dict (as plain dict, not DictConfig)
    source_location : str
        Source of the run ("hpc", "local", "live", etc.)

    Returns
    -------
    dict
        Flat dictionary with all relevant parameters for WandB config
    """
    wc: dict[str, Any] = {}

    # === Model Type (Primary Classification) ===
    loop = _safe_get(cfg, "loop", "")
    wc["model/loop"] = loop if loop else None

    # Determine model type from loop or classifier config
    if loop:
        if "transformer" in loop:
            wc["model/type"] = "transformer"
        elif "mlp" in loop:
            wc["model/type"] = "mlp"
        elif "bdt" in loop:
            wc["model/type"] = "bdt"
        elif "ae" in loop or "autoencoder" in loop:
            wc["model/type"] = "autoencoder"
        else:
            wc["model/type"] = loop
    else:
        # Try to infer from config structure
        if _safe_get(cfg, "phase1"):
            wc["model/type"] = "autoencoder"
        elif _safe_get(cfg, "classifier.model.name"):
            wc["model/type"] = _safe_get(cfg, "classifier.model.name")
        else:
            wc["model/type"] = None

    # === Transformer Architecture ===
    wc["model/dim"] = _safe_get(cfg, "classifier.model.dim")
    wc["model/depth"] = _safe_get(cfg, "classifier.model.depth")
    wc["model/heads"] = _safe_get(cfg, "classifier.model.heads")
    wc["model/mlp_dim"] = _safe_get(cfg, "classifier.model.mlp_dim")
    wc["model/dropout"] = _safe_get(cfg, "classifier.model.dropout")

    # Model size estimate for transformers
    dim = _safe_get(cfg, "classifier.model.dim")
    depth = _safe_get(cfg, "classifier.model.depth")
    if dim is not None and depth is not None:
        try:
            dim = int(dim)
            depth = int(depth)
            mlp_ratio = 4
            block_params = 4 * dim * dim + 2 * dim * dim * mlp_ratio
            wc["model/params_est"] = int(depth * block_params + 10000)
            wc["model/size_label"] = f"{dim}d{depth}L"
        except (ValueError, TypeError):
            pass

    # === MLP Architecture ===
    hidden = _safe_get(cfg, "classifier.model.hidden_sizes")
    if hidden is not None:
        wc["model/hidden_sizes"] = str(hidden) if isinstance(hidden, list) else hidden
        wc["model/n_layers"] = len(hidden) if isinstance(hidden, list) else None
    wc["model/use_batch_norm"] = _safe_get(cfg, "classifier.model.use_batch_norm")
    wc["model/activation"] = _safe_get(cfg, "classifier.model.activation")

    # === BDT Architecture ===
    wc["model/n_estimators"] = _safe_get(cfg, "classifier.model.n_estimators")
    wc["model/max_depth"] = _safe_get(cfg, "classifier.model.max_depth")
    wc["model/bdt_lr"] = _safe_get(cfg, "classifier.model.learning_rate")
    wc["model/subsample"] = _safe_get(cfg, "classifier.model.subsample")
    wc["model/colsample_bytree"] = _safe_get(cfg, "classifier.model.colsample_bytree")

    # === Positional Encoding ===
    wc["pos_enc/type"] = _safe_get(cfg, "classifier.model.positional")
    wc["pos_enc/space"] = _safe_get(cfg, "classifier.model.positional_space")
    dim_mask = _safe_get(cfg, "classifier.model.positional_dim_mask")
    if dim_mask is not None:
        wc["pos_enc/dim_mask"] = str(dim_mask) if isinstance(dim_mask, list) else dim_mask
    wc["pos_enc/rotary_base"] = _safe_get(cfg, "classifier.model.rotary.base")

    # === Normalization ===
    wc["norm/policy"] = _safe_get(cfg, "classifier.model.norm.policy")

    # === Tokenizer ===
    tok = _safe_get(cfg, "classifier.model.tokenizer") or _safe_get(cfg, "classifier.tokenizer") or _safe_get(cfg, "tokenizer")
    if isinstance(tok, dict):
        wc["tokenizer/type"] = tok.get("name")
        wc["tokenizer/id_embed_dim"] = tok.get("id_embed_dim")
        wc["tokenizer/model_type"] = tok.get("model_type")
    elif tok:
        wc["tokenizer/type"] = tok

    # === Pooling ===
    wc["pooling/type"] = _safe_get(cfg, "classifier.model.pooling")

    # === Causal attention ===
    wc["model/causal_attention"] = _safe_get(cfg, "classifier.model.causal_attention")

    # === Training Hyperparameters ===
    wc["training/lr"] = _safe_get(cfg, "classifier.trainer.lr") or _safe_get(cfg, "trainer.lr")
    wc["training/weight_decay"] = _safe_get(cfg, "classifier.trainer.weight_decay") or _safe_get(cfg, "trainer.weight_decay")
    wc["training/batch_size"] = _safe_get(cfg, "classifier.trainer.batch_size") or _safe_get(cfg, "trainer.batch_size")
    wc["training/epochs"] = _safe_get(cfg, "classifier.trainer.epochs") or _safe_get(cfg, "trainer.epochs")
    wc["training/label_smoothing"] = _safe_get(cfg, "classifier.trainer.label_smoothing")
    wc["training/warmup_steps"] = _safe_get(cfg, "classifier.trainer.warmup_steps")
    wc["training/lr_schedule"] = _safe_get(cfg, "classifier.trainer.lr_schedule")
    wc["training/grad_clip"] = _safe_get(cfg, "classifier.trainer.grad_clip")
    wc["training/seed"] = _safe_get(cfg, "trainer.seed") or _safe_get(cfg, "seed")

    # === Early Stopping ===
    wc["early_stop/enabled"] = _safe_get(cfg, "classifier.trainer.early_stopping.enabled")
    wc["early_stop/patience"] = _safe_get(cfg, "classifier.trainer.early_stopping.patience")

    # === Autoencoder (Phase 1) ===
    if _safe_get(cfg, "phase1"):
        enc = _safe_get(cfg, "phase1.encoder")
        wc["ae/encoder"] = enc.get("name") if isinstance(enc, dict) else enc
        dec = _safe_get(cfg, "phase1.decoder")
        wc["ae/decoder"] = dec.get("name") if isinstance(dec, dict) else dec
        lat = _safe_get(cfg, "phase1.latent_space")
        wc["ae/latent"] = lat.get("name") if isinstance(lat, dict) else lat
        wc["ae/latent_dim"] = _safe_get(cfg, "phase1.model.latent_dim")
        wc["ae/codebook_size"] = _safe_get(cfg, "phase1.model.codebook_size")
        wc["ae/globals_beta"] = _safe_get(cfg, "phase1.decoder.globals_beta")
        wc["ae/beta"] = _safe_get(cfg, "phase1.model.beta")

    # === Dataset ===
    wc["data/name"] = _safe_get(cfg, "data.name") or _safe_get(cfg, "data.path")

    # === Metrics ===
    metrics = _safe_get(cfg, "metrics")
    if metrics:
        wc["metrics/list"] = str(metrics) if isinstance(metrics, list) else metrics

    # === Source Info ===
    wc["source/location"] = source_location

    # === Metadata Schema Fields (meta.*) ===
    # These enable reliable slicing and filtering in W&B dashboards
    meta_fields = extract_meta_fields(cfg)
    wc.update(meta_fields)

    # Remove None values for cleaner WandB display
    return {k: v for k, v in wc.items() if v is not None}


def extract_meta_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata schema fields for W&B config.

    Uses the facts.meta module to build canonical metadata, then
    formats it for W&B with meta.* prefix.

    Parameters
    ----------
    cfg : dict
        Full Hydra config dict

    Returns
    -------
    dict
        Dict with meta.* keys for W&B config
    """
    try:
        from thesis_ml.facts.meta import build_meta
    except ImportError:
        logger.warning("[wandb] Could not import facts.meta, skipping meta fields")
        return {}

    try:
        meta = build_meta(cfg)
    except Exception as e:
        logger.warning("[wandb] Could not build meta: %s", e)
        return {}

    wc: dict[str, Any] = {}

    # Canonical fields
    wc["meta.schema_version"] = meta.get("schema_version")
    wc["meta.level"] = meta.get("level")
    wc["meta.goal"] = meta.get("goal")
    wc["meta.model_family"] = meta.get("model_family")
    wc["meta.dataset_name"] = meta.get("dataset_name")
    wc["meta.feature_mode"] = meta.get("feature_mode")

    # Process groups: three representations for different uses
    process_groups = meta.get("process_groups")
    wc["meta.process_groups"] = json.dumps(process_groups) if process_groups else None
    wc["meta.process_groups_key"] = meta.get("process_groups_key")  # Best for W&B grouping
    wc["meta.class_def_str"] = meta.get("class_def_str")  # For display
    wc["meta.row_key"] = meta.get("row_key")
    wc["meta.n_classes"] = meta.get("n_classes")

    # Datatreatment and confidence
    datatreatment = meta.get("datatreatment")
    wc["meta.datatreatment"] = json.dumps(datatreatment) if datatreatment else None
    wc["meta.meta_hash"] = meta.get("meta_hash")
    wc["meta.meta_confidence"] = meta.get("meta_confidence")
    wc["meta.needs_review"] = meta.get("needs_review", False)

    # Flatten key subset of datatreatment for easier W&B filtering
    if datatreatment:
        # Existing convenience key (kept for backwards compatibility)
        wc["data/token_order"] = datatreatment.get("token_order")

        # Explicit meta.* mirrors for core datatreatment fields so you can slice
        # directly in W&B without parsing JSON strings.
        wc["meta.datatreatment_token_order"] = datatreatment.get("token_order")
        wc["meta.datatreatment_tokenization"] = datatreatment.get("tokenization")
        wc["meta.datatreatment_pid_encoding"] = datatreatment.get("pid_encoding")
        wc["meta.datatreatment_id_embed_dim"] = datatreatment.get("id_embed_dim")
        wc["meta.datatreatment_met_rep"] = datatreatment.get("met_rep")
        wc["meta.datatreatment_globals_included"] = datatreatment.get("globals_included")
        wc["meta.datatreatment_feature_set_version"] = datatreatment.get("feature_set_version")
        wc["meta.datatreatment_normalization"] = datatreatment.get("normalization")

    return wc


def extract_meta_tags(cfg: dict[str, Any]) -> list[str]:
    """Extract minimal tags from metadata for W&B.

    Keep tags few and stable - don't explode with numeric values.

    Parameters
    ----------
    cfg : dict
        Full Hydra config dict

    Returns
    -------
    list[str]
        List of tags (e.g., ["level:sim_event", "goal:classification", "family:transformer"])
    """
    try:
        from thesis_ml.facts.meta import build_meta
    except ImportError:
        return []

    try:
        meta = build_meta(cfg)
    except Exception:
        return []

    tags = []

    # Level
    level = meta.get("level")
    if level:
        tags.append(f"level:{level}")

    # Goal
    goal = meta.get("goal")
    if goal:
        tags.append(f"goal:{goal}")

    # Model family
    family = meta.get("model_family")
    if family:
        tags.append(f"family:{family}")

    # Dataset
    dataset = meta.get("dataset_name")
    if dataset:
        tags.append(f"dataset:{dataset}")

    # Needs review flag
    if meta.get("needs_review"):
        tags.append("needs_review")

    return tags


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
    """Initialize W&B run with safe defaults and comprehensive metadata.

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
    - Extracts comprehensive config metadata for maximum WandB divisibility
    """
    if not cfg.logging.use_wandb:
        return None

    try:
        from omegaconf import OmegaConf

        import wandb

        wandb_cfg = cfg.logging.wandb
        wandb_dir = Path(str(wandb_cfg.dir)).resolve()
        wandb_dir.mkdir(parents=True, exist_ok=True)

        # Handle empty strings as None, with fallback to cwd-based name
        entity = str(wandb_cfg.entity) if wandb_cfg.entity else None
        run_name = str(wandb_cfg.run_name) if wandb_cfg.run_name else _get_run_name_from_cwd()

        # Auto-extract group from run name if not explicitly set
        # This groups runs from the same experiment (multirun) together
        group = str(wandb_cfg.group) if wandb_cfg.get("group") else _get_group_from_run_name(run_name)

        # Tags: use explicit tags if provided, otherwise extract from meta
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        tags = list(wandb_cfg.tags) if wandb_cfg.get("tags") else extract_meta_tags(cfg_dict)

        # Extract comprehensive config for maximum WandB divisibility
        # Pass cfg (DictConfig) so build_meta can access nested config (e.g. data.classifier)
        wandb_config = extract_wandb_config(cfg, source_location="live")

        run = wandb.init(
            project=str(wandb_cfg.project),
            entity=entity,
            name=run_name,
            group=group,
            tags=tags,
            mode=str(wandb_cfg.mode),
            dir=str(wandb_dir),
            config=wandb_config,  # Use extracted config for consistent metadata
            resume="allow",  # Prevents duplicates on re-run
        )

        # Define metric axes for consistent plotting
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("perf/*", step_metric="epoch")

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
