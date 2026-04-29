"""Inference utilities for report generation."""

from __future__ import annotations

import json
import logging
import re
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


def load_cfg_from_run_dir(run_dir: Path | str) -> Any:
    """Load Hydra (or legacy) config from a run directory."""
    run_dir = Path(run_dir)
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    if hydra_cfg_path.exists():
        return OmegaConf.load(str(hydra_cfg_path))
    cfg_path = run_dir / "cfg.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing .hydra/config.yaml or cfg.yaml in {run_dir}")
    return OmegaConf.load(str(cfg_path))


def resolve_classifier_weights_path(run_dir: Path | str) -> tuple[Path, str]:
    """Pick classifier checkpoint: best_val.pt, last.pt, model.pt, then highest epoch_*.pt.

    Returns
    -------
    tuple[Path, str]
        (path, kind) where kind is ``best_val``, ``last``, ``model``, or ``epoch``.
    """
    run_dir = Path(run_dir)
    ordered: list[tuple[Path, str]] = [
        (run_dir / "best_val.pt", "best_val"),
        (run_dir / "last.pt", "last"),
        (run_dir / "model.pt", "model"),
    ]
    for path, kind in ordered:
        if path.is_file():
            return path, kind

    best_epoch = -1
    best_path: Path | None = None
    for path in run_dir.glob("epoch_*.pt"):
        m = re.match(r"^epoch_(\d+)\.pt$", path.name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_epoch:
            best_epoch = ep
            best_path = path
    if best_path is not None:
        return best_path, "epoch"

    raise FileNotFoundError(
        f"No classifier weights in {run_dir} (tried best_val.pt, last.pt, model.pt, epoch_*.pt)"
    )


def _maybe_remap_legacy_encoder_mlp(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Map ``encoder.blocks.N.mlp.*`` (legacy checkpoints) to ``encoder.blocks.N.ffn.net.*``."""
    prefix = "encoder.blocks."
    if not any(k.startswith(prefix) and ".mlp." in k for k in state_dict):
        return state_dict
    out: dict[str, Any] = {}
    n_migrated = 0
    for k, v in state_dict.items():
        if k.startswith(prefix) and ".mlp." in k:
            idx = k.index(".mlp.")
            head, tail = k[:idx], k[idx + len(".mlp.") :]
            out[f"{head}.ffn.net.{tail}"] = v
            n_migrated += 1
        else:
            out[k] = v
    logger.info("Remapped %d legacy encoder.blocks.*.mlp.* keys to *.ffn.net.*", n_migrated)
    return out


def _build_classifier_from_checkpoint(
    cfg: Any,
    run_dir: Path,
    checkpoint: dict | Any,
    weights_path: Path,
    dev: torch.device,
) -> torch.nn.Module:
    """Build transformer classifier and load state dict (shared by run_id and run_dir loaders)."""
    from thesis_ml.architectures.transformer_classifier.base import build_from_config as build_classifier
    from thesis_ml.data.h5_loader import make_classification_dataloaders
    from thesis_ml.training_loops.transformer_classifier import _gather_meta

    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    if isinstance(state_dict, dict):
        state_dict = _maybe_remap_legacy_encoder_mlp(state_dict)

    if not hasattr(cfg, "meta") or cfg.meta is None:
        train_dl, val_dl, test_dl, meta = make_classification_dataloaders(cfg)
        _gather_meta(cfg, meta)

    if not hasattr(cfg, "meta") or cfg.meta is None:
        cfg.meta = OmegaConf.create({})

    ckpt_meta = checkpoint.get("meta") if isinstance(checkpoint, dict) else None
    if isinstance(ckpt_meta, dict):
        for key, value in ckpt_meta.items():
            if not hasattr(cfg.meta, key) or getattr(cfg.meta, key) is None:
                setattr(cfg.meta, key, value)

    if not hasattr(cfg.meta, "n_tokens") or cfg.meta.n_tokens is None:
        train_dl, val_dl, test_dl, meta = make_classification_dataloaders(cfg)
        _gather_meta(cfg, meta)

    if not hasattr(cfg.meta, "n_classes") or cfg.meta.n_classes is None:
        train_dl, val_dl, test_dl, meta = make_classification_dataloaders(cfg)
        _gather_meta(cfg, meta)

    data_n_tokens = int(cfg.meta.n_tokens)

    if "pos_enc.pe" in state_dict:
        pe_shape = state_dict["pos_enc.pe"].shape
        checkpoint_max_seq_len = pe_shape[0]
        pooling = cfg.classifier.model.get("pooling", "cls")
        positional_space = cfg.classifier.model.get("positional_space", "model")
        include_met = bool(cfg.classifier.get("globals", {}).get("include_met", False))
        extra_tokens = (1 if pooling == "cls" else 0) + (2 if include_met else 0)
        checkpoint_n_tokens = checkpoint_max_seq_len - extra_tokens if positional_space == "model" else checkpoint_max_seq_len
        if checkpoint_n_tokens != data_n_tokens:
            logger.info(
                "Adjusting n_tokens from %s (data) to %s (checkpoint PE shape)",
                data_n_tokens,
                checkpoint_n_tokens,
            )
        cfg.meta.n_tokens = checkpoint_n_tokens

    elif "embedding.pos_enc.pe" in state_dict:
        pe_shape = state_dict["embedding.pos_enc.pe"].shape
        checkpoint_n_tokens = pe_shape[0]
        if checkpoint_n_tokens != data_n_tokens:
            logger.info(
                "Adjusting n_tokens from %s (data) to %s (checkpoint embedding PE shape)",
                data_n_tokens,
                checkpoint_n_tokens,
            )
        cfg.meta.n_tokens = checkpoint_n_tokens

    pretrained_key = "embedding.tokenizer._index_embedding.weight"
    binned_emb_key = "embedding.tokenizer.token_embedding.weight"
    if pretrained_key in state_dict:
        if not hasattr(cfg.classifier.model, "tokenizer"):
            cfg.classifier.model.tokenizer = OmegaConf.create({})
        current_name = cfg.classifier.model.tokenizer.get("name", "")
        if current_name != "pretrained":
            logger.info(
                "Adjusting tokenizer.name from %r (config) to 'pretrained' (checkpoint has _index_embedding)",
                current_name,
            )
            cfg.classifier.model.tokenizer.name = "pretrained"

        vq_checkpoint_path = None
        if hasattr(cfg.classifier.model.tokenizer, "checkpoint_path") and cfg.classifier.model.tokenizer.checkpoint_path:
            vq_checkpoint_path = Path(cfg.classifier.model.tokenizer.checkpoint_path)
        elif hasattr(cfg.classifier.model.tokenizer, "checkpoint_path_5vec"):
            path_5vec = cfg.classifier.model.tokenizer.get("checkpoint_path_5vec")
            path_4vec = cfg.classifier.model.tokenizer.get("checkpoint_path_4vec")
            if path_5vec:
                vq_checkpoint_path = Path(path_5vec)
            elif path_4vec:
                vq_checkpoint_path = Path(path_4vec)

        if vq_checkpoint_path and vq_checkpoint_path.exists():
            try:
                vq_ckpt = torch.load(vq_checkpoint_path, map_location="cpu", weights_only=True)
                vq_state = vq_ckpt.get("state_dict", vq_ckpt) if isinstance(vq_ckpt, dict) else vq_ckpt
                encoder_key = "encoder.net.0.weight"
                if encoder_key in vq_state:
                    encoder_shape = vq_state[encoder_key].shape
                    if len(encoder_shape) == 2:
                        encoder_input_dim = int(encoder_shape[1])
                        id_embed_dim = cfg.classifier.model.tokenizer.get("id_embed_dim", 8)
                        inferred_cont_dim = encoder_input_dim - id_embed_dim
                        if inferred_cont_dim in (3, 4):
                            current = getattr(cfg.meta, "token_feat_dim", None)
                            if current != inferred_cont_dim:
                                logger.info(
                                    "Adjusting token_feat_dim from %s (config) to %s (VQ encoder input=%d - id_embed=%d)",
                                    current,
                                    inferred_cont_dim,
                                    encoder_input_dim,
                                    id_embed_dim,
                                )
                                cfg.meta.token_feat_dim = inferred_cont_dim
            except Exception as e:
                logger.warning("Could not infer cont_dim from VQ checkpoint %s: %s", vq_checkpoint_path, e)
    elif binned_emb_key in state_dict:
        ckpt_vocab_size = int(state_dict[binned_emb_key].shape[0])
        current = getattr(cfg.meta, "vocab_size", None)
        if current != ckpt_vocab_size:
            logger.info(
                "Adjusting vocab_size from %s (config) to %s (checkpoint token_embedding shape)",
                current,
                ckpt_vocab_size,
            )
        cfg.meta.vocab_size = ckpt_vocab_size

    proj_key = "embedding.projection.weight"
    if proj_key in state_dict:
        proj_shape = state_dict[proj_key].shape
        if len(proj_shape) == 2:
            ckpt_tokenizer_out = int(proj_shape[1])
            id_embed_dim = cfg.classifier.model.tokenizer.get("id_embed_dim", 8)
            tokenizer_name = cfg.classifier.model.tokenizer.get("name", "identity")
            if tokenizer_name == "identity" and ckpt_tokenizer_out > id_embed_dim:
                inferred_cont_dim = ckpt_tokenizer_out - id_embed_dim
                if inferred_cont_dim in (3, 4):
                    current = getattr(cfg.meta, "token_feat_dim", None)
                    if current != inferred_cont_dim:
                        logger.info(
                            "Adjusting token_feat_dim from %s (config) to %s (checkpoint projection shape)",
                            current,
                            inferred_cont_dim,
                        )
                        cfg.meta.token_feat_dim = inferred_cont_dim
            elif tokenizer_name == "raw":
                inferred_cont_dim = ckpt_tokenizer_out
                if inferred_cont_dim in (3, 4):
                    current = getattr(cfg.meta, "token_feat_dim", None)
                    if current != inferred_cont_dim:
                        logger.info(
                            "Adjusting token_feat_dim from %s (config) to %s (checkpoint projection shape)",
                            current,
                            inferred_cont_dim,
                        )
                        cfg.meta.token_feat_dim = inferred_cont_dim

    encoder_key = "embedding.tokenizer._encoder.net.0.weight"
    if encoder_key in state_dict:
        encoder_weight_shape = state_dict[encoder_key].shape
        if len(encoder_weight_shape) == 2:
            encoder_input_dim = int(encoder_weight_shape[1])
            id_embed_dim = cfg.classifier.model.tokenizer.get("id_embed_dim", 8)
            tokenizer_name = cfg.classifier.model.tokenizer.get("name", "identity")
            if tokenizer_name in ("identity", "pretrained") and encoder_input_dim > id_embed_dim:
                inferred_cont_dim = encoder_input_dim - id_embed_dim
                if inferred_cont_dim in (3, 4):
                    current = getattr(cfg.meta, "token_feat_dim", None)
                    if current != inferred_cont_dim:
                        logger.info(
                            "Adjusting token_feat_dim from %s to %s (main checkpoint encoder key, encoder_input=%d - id_embed=%d)",
                            current,
                            inferred_cont_dim,
                            encoder_input_dim,
                            id_embed_dim,
                        )
                        cfg.meta.token_feat_dim = inferred_cont_dim
            elif tokenizer_name == "raw":
                inferred_cont_dim = encoder_input_dim
                if inferred_cont_dim in (3, 4):
                    current = getattr(cfg.meta, "token_feat_dim", None)
                    if current != inferred_cont_dim:
                        logger.info(
                            "Adjusting token_feat_dim from %s to %s (main checkpoint encoder key)",
                            current,
                            inferred_cont_dim,
                        )
                        cfg.meta.token_feat_dim = inferred_cont_dim

    model = build_classifier(cfg, cfg.meta).to(dev)

    from thesis_ml.architectures.transformer_classifier.modules.tokenizers.pretrained import PretrainedTokenizer

    tokenizer = getattr(getattr(model, "embedding", None), "tokenizer", None)
    if isinstance(tokenizer, PretrainedTokenizer) and not tokenizer._loaded:
        try:
            tokenizer._load_model()
            logger.info("Pre-loaded VQ tokenizer submodules for state_dict compatibility")
        except Exception as e:
            logger.warning("Could not pre-load VQ tokenizer (checkpoint missing?): %s", e)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.error("load_state_dict failed for run %s: %s", run_dir, e)
        model_params = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        ckpt_params = {k: tuple(v.shape) for k, v in state_dict.items()}
        mismatches = [
            f"  {k}: model={model_params[k]} vs ckpt={ckpt_params[k]}"
            for k in set(model_params) & set(ckpt_params)
            if model_params[k] != ckpt_params[k]
        ]
        if mismatches:
            logger.error("Shape mismatches:\n%s", "\n".join(mismatches))
        missing_in_model = set(ckpt_params) - set(model_params)
        missing_in_ckpt = set(model_params) - set(ckpt_params)
        if missing_in_model:
            logger.error("Keys in checkpoint but not in model: %s", missing_in_model)
        if missing_in_ckpt:
            logger.error("Keys in model but not in checkpoint: %s", missing_in_ckpt)
        raise
    model.eval()
    return model


def load_classifier_from_run_dir(
    run_dir: Path | str,
    device: str | None = None,
) -> tuple[Any, torch.nn.Module, torch.device, dict[str, Any]]:
    """Load classifier ``cfg``, ``model``, and device from an explicit run directory.

    Uses checkpoint priority: ``best_val.pt`` → ``last.pt`` → ``model.pt`` → highest ``epoch_*.pt``.

    Returns
    -------
    tuple[Any, torch.nn.Module, torch.device, dict[str, Any]]
        ``cfg``, ``model``, ``device``, and metadata
        ``{checkpoint_path, checkpoint_kind, checkpoint_epoch}``.
    """
    run_dir = Path(run_dir)
    cfg = load_cfg_from_run_dir(run_dir)
    if not hasattr(cfg, "classifier"):
        raise ValueError(f"{run_dir} is not a classifier run (no cfg.classifier)")

    weights_path, checkpoint_kind = resolve_classifier_weights_path(run_dir)
    dev = _resolve_device(device)
    checkpoint = torch.load(str(weights_path), map_location=dev, weights_only=False)
    model = _build_classifier_from_checkpoint(cfg, run_dir, checkpoint, weights_path, dev)

    epoch_val = None
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        try:
            epoch_val = int(checkpoint["epoch"])
        except (TypeError, ValueError):
            epoch_val = None

    meta = {
        "checkpoint_path": str(weights_path.resolve()),
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_epoch": epoch_val,
    }
    return cfg, model, dev, meta


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
    cfg = load_cfg_from_run_dir(run_dir)
    dev = _resolve_device(device)

    if hasattr(cfg, "classifier"):
        weights_path, _kind = resolve_classifier_weights_path(run_dir)
        checkpoint = torch.load(str(weights_path), map_location=dev, weights_only=False)
        model = _build_classifier_from_checkpoint(cfg, run_dir, checkpoint, weights_path, dev)
        return cfg, model, dev

    if hasattr(cfg, "phase1"):
        from thesis_ml.architectures.autoencoder.base import build_from_config
        from thesis_ml.training_loops.autoencoder import _gather_meta

        best_val_path = run_dir / "best_val.pt"
        model_pt_path = run_dir / "model.pt"
        if best_val_path.exists():
            weights_path = best_val_path
        elif model_pt_path.exists():
            weights_path = model_pt_path
        else:
            raise FileNotFoundError(f"Missing best_val.pt or model.pt in {run_dir}")

        checkpoint = torch.load(str(weights_path), map_location=dev, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        if not hasattr(cfg, "meta") or cfg.meta is None:
            train_dl, val_dl, test_dl, meta = make_dataloaders(cfg)
            _gather_meta(cfg, meta)

        model = build_from_config(cfg).to(dev)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.error("load_state_dict failed for run %s: %s", run_dir, e)
            raise
        model.eval()
        return cfg, model, dev

    raise ValueError("Cannot determine model type from config (missing 'classifier' or 'phase1' section)")


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
    failed = []

    for run_id in run_ids:
        try:
            cfg, model, _ = load_model_from_run(run_id, output_root, device=device)
            models.append((run_id, cfg, model))
        except Exception as e:
            logger.error("Failed to load model for run %s: %s", run_id, e)
            failed.append((run_id, str(e)))

    if failed:
        logger.warning(
            "%d/%d runs failed to load:\n%s",
            len(failed),
            len(run_ids),
            "\n".join(f"  {rid}: {err}" for rid, err in failed),
        )

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
