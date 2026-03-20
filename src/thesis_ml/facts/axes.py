"""Flat axis metadata for thesis runs (config-only, no model/data imports).

Writes ``facts/axes.json`` and feeds W&B under ``axes/*`` for consistent filtering.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _safe_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Dot-path get for dict or DictConfig (aligned with facts.meta)."""
    try:
        val = cfg
        for key in path.split("."):
            if hasattr(val, "get"):
                val = val.get(key, {})
            elif hasattr(val, key):
                val = getattr(val, key)
            else:
                return default
        return val if val != {} else default
    except Exception:
        return default


def _infer_token_order(cfg: Any) -> str:
    shuffle = _safe_get(cfg, "data.shuffle_tokens")
    if shuffle is True:
        return "shuffled"
    sort_by = _safe_get(cfg, "data.sort_tokens_by")
    if sort_by == "pt":
        return "pt_sorted"
    return "input_order"


def _serialize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool | int | float | str):
        return v
    if isinstance(v, list | tuple):
        return json.dumps(list(v), sort_keys=False)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


def build_axes_metadata(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Build flat axis dict from resolved Hydra config (no model / dataloader)."""
    mc = "classifier.model"
    tok = f"{mc}.tokenizer"
    bc = f"{mc}.bias_config"

    dim = _safe_get(cfg, f"{mc}.dim")
    depth = _safe_get(cfg, f"{mc}.depth")

    model_size_key = None
    if dim is not None and depth is not None:
        try:
            model_size_key = f"d{int(dim)}_L{int(depth)}"
        except (TypeError, ValueError):
            model_size_key = f"d{dim}_L{depth}"

    pooling = _safe_get(cfg, f"{mc}.head.pooling")
    if pooling is None:
        pooling = _safe_get(cfg, f"{mc}.pooling")

    axes: dict[str, Any] = {
        "positional": _serialize_value(_safe_get(cfg, f"{mc}.positional")),
        "positional_space": _serialize_value(_safe_get(cfg, f"{mc}.positional_space")),
        "positional_dim_mask": _serialize_value(_safe_get(cfg, f"{mc}.positional_dim_mask")),
        "tokenizer_name": _serialize_value(_safe_get(cfg, f"{tok}.name")),
        "pid_mode": _serialize_value(_safe_get(cfg, f"{tok}.pid_mode")),
        "id_embed_dim": _serialize_value(_safe_get(cfg, f"{tok}.id_embed_dim")),
        "token_order": _infer_token_order(cfg),
        "include_met": _serialize_value(_safe_get(cfg, "classifier.globals.include_met")),
        "cont_features": _serialize_value(_safe_get(cfg, "data.cont_features")),
        "norm_policy": _serialize_value(_safe_get(cfg, f"{mc}.norm.policy")),
        "norm_type": _serialize_value(_safe_get(cfg, f"{mc}.norm.type")),
        "attention_type": _serialize_value(_safe_get(cfg, f"{mc}.attention.type")),
        "attention_norm": _serialize_value(_safe_get(cfg, f"{mc}.attention.norm")),
        "diff_bias_mode": _serialize_value(_safe_get(cfg, f"{mc}.attention.diff_bias_mode")),
        "ffn_type": _serialize_value(_safe_get(cfg, f"{mc}.ffn.type")),
        "kan_ffn_variant": _serialize_value(_safe_get(cfg, f"{mc}.ffn.kan.variant")),
        "pooling": _serialize_value(pooling),
        "head_type": _serialize_value(_safe_get(cfg, f"{mc}.head.type")),
        "causal_attention": _serialize_value(_safe_get(cfg, f"{mc}.causal_attention")),
        "attention_biases": _serialize_value(_safe_get(cfg, f"{mc}.attention_biases")),
        "sm_mode": _serialize_value(_safe_get(cfg, f"{bc}.sm_interaction.mode")),
        "lorentz_features": _serialize_value(_safe_get(cfg, f"{bc}.lorentz_scalar.features")),
        "lorentz_mlp_type": _serialize_value(_safe_get(cfg, f"{bc}.lorentz_scalar.mlp_type")),
        "lorentz_per_head": _serialize_value(_safe_get(cfg, f"{bc}.lorentz_scalar.per_head")),
        "lorentz_sparse_gating": _serialize_value(_safe_get(cfg, f"{bc}.lorentz_scalar.sparse_gating")),
        "typepair_init": _serialize_value(_safe_get(cfg, f"{bc}.typepair_kinematic.init_from_physics")),
        "typepair_freeze": _serialize_value(_safe_get(cfg, f"{bc}.typepair_kinematic.freeze_table")),
        "typepair_kinematic_gate": _serialize_value(_safe_get(cfg, f"{bc}.typepair_kinematic.kinematic_gate")),
        "global_conditioned_mode": _serialize_value(_safe_get(cfg, f"{bc}.global_conditioned.mode")),
        "nodewise_mass_enabled": _serialize_value(_safe_get(cfg, f"{mc}.nodewise_mass.enabled")),
        "mia_enabled": _serialize_value(_safe_get(cfg, f"{mc}.mia_blocks.enabled")),
        "mia_placement": _serialize_value(_safe_get(cfg, f"{mc}.mia_blocks.placement")),
        "moe_enabled": _serialize_value(_safe_get(cfg, f"{mc}.moe.enabled")),
        "moe_top_k": _serialize_value(_safe_get(cfg, f"{mc}.moe.top_k")),
        "moe_routing_level": _serialize_value(_safe_get(cfg, f"{mc}.moe.routing_level")),
        "moe_scope": _serialize_value(_safe_get(cfg, f"{mc}.moe.scope")),
        "kan_grid_size": _serialize_value(_safe_get(cfg, f"{mc}.kan.grid_size")),
        "kan_spline_order": _serialize_value(_safe_get(cfg, f"{mc}.kan.spline_order")),
        "dim": _serialize_value(dim),
        "depth": _serialize_value(depth),
        "heads": _serialize_value(_safe_get(cfg, f"{mc}.heads")),
        "mlp_dim": _serialize_value(_safe_get(cfg, f"{mc}.mlp_dim")),
        "dropout": _serialize_value(_safe_get(cfg, f"{mc}.dropout")),
        "lr": _serialize_value(_safe_get(cfg, "classifier.trainer.lr")),
        "weight_decay": _serialize_value(_safe_get(cfg, "classifier.trainer.weight_decay")),
        "batch_size": _serialize_value(_safe_get(cfg, "classifier.trainer.batch_size")),
        "epochs": _serialize_value(_safe_get(cfg, "classifier.trainer.epochs")),
        "seed": _serialize_value(_safe_get(cfg, "classifier.trainer.seed")),
        "model_size_key": model_size_key,
        "experiment_name": _serialize_value(_safe_get(cfg, "experiment.name")),
    }

    return axes


def write_axes(axes: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(axes, f, indent=2, ensure_ascii=False)
    logger.info("[axes] wrote %s", path)
