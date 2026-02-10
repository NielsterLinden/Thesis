"""Metadata schema builder for run classification and W&B integration.

This module provides functions to build, canonicalize, and write the `facts/meta.json`
file that enables reliable slicing, filtering, and comparison of runs.

Key design principles:
- `process_groups` is the canonical slicer (replaces brittle "task names")
- Never guess - use null with needs_review=True if uncertain
- Canonical ordering ensures identical tasks hash identically
- Derived fields are cache and can be regenerated

See METADATA_SCHEMA.md for full documentation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Process ID to physics name mapping
PROCESS_ID_NAMES: dict[int, str] = {
    1: "4t",  # tttt
    2: "ttH",
    3: "ttW",
    4: "ttWW",
    5: "ttZ",
}

# Current feature set version - update when data features change
FEATURE_SET_VERSION = "v1"

# Current schema version
SCHEMA_VERSION = 1


# =============================================================================
# Helper Functions
# =============================================================================


def _safe_get(cfg: dict[str, Any] | Any, path: str, default: Any = None) -> Any:
    """Safely get nested value from config using dot notation.

    Parameters
    ----------
    cfg : dict or DictConfig
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
            if hasattr(val, "get"):
                val = val.get(key, {})
            elif hasattr(val, key):
                val = getattr(val, key)
            else:
                return default
        return val if val != {} else default
    except Exception:
        return default


def _infer_goal_from_loop(loop: str) -> str | None:
    """Infer goal from training loop name.

    Parameters
    ----------
    loop : str
        Training loop name (e.g., "transformer_classifier", "ae")

    Returns
    -------
    str | None
        "classification", "anomaly_detection", or None if unknown
    """
    if not loop:
        return None

    loop_lower = loop.lower()

    if "classifier" in loop_lower:
        return "classification"
    if any(x in loop_lower for x in ["ae", "autoencoder", "gan", "diffusion"]):
        return "anomaly_detection"

    return None


def _infer_model_family(loop: str) -> str | None:
    """Infer model family from training loop name.

    Parameters
    ----------
    loop : str
        Training loop name

    Returns
    -------
    str | None
        "transformer", "mlp", "bdt", "ae", or None if unknown
    """
    if not loop:
        return None

    loop_lower = loop.lower()

    if "transformer" in loop_lower:
        return "transformer"
    if "mlp" in loop_lower:
        return "mlp"
    if "bdt" in loop_lower:
        return "bdt"
    if any(x in loop_lower for x in ["ae", "autoencoder", "gan", "diffusion"]):
        return "ae"

    return None


def _extract_dataset_name(cfg: DictConfig | dict) -> str | None:
    """Extract dataset name from config.

    Parameters
    ----------
    cfg : DictConfig or dict
        Full config

    Returns
    -------
    str | None
        Dataset name (e.g., "4topsplitted") or None
    """
    data_path = _safe_get(cfg, "data.path")
    if data_path:
        # Extract basename without extension
        path = Path(str(data_path))
        name = path.stem  # e.g., "4tops_splitted" from "4tops_splitted.h5"
        # Normalize common variations
        return name.replace("_", "").replace("-", "")  # "4topssplitted"

    data_name = _safe_get(cfg, "data.name")
    if data_name:
        return str(data_name)

    return None


def _infer_token_order(cfg: DictConfig | dict) -> str:
    """Infer token ordering from config.

    Current codebase uses input order (no sorting).
    """
    # Check for any shuffle/sort settings
    shuffle = _safe_get(cfg, "data.shuffle_tokens")
    if shuffle is True:
        return "shuffled"

    sort_by = _safe_get(cfg, "data.sort_tokens_by")
    if sort_by == "pt":
        return "pt_sorted"

    # Default: input order (current behavior)
    return "input_order"


def _infer_tokenization(cfg: DictConfig | dict) -> str:
    """Infer high-level tokenization type for WandB grouping.

    Returns "direct", "binned", or "vq".
    """
    use_binned = _safe_get(cfg, "data.use_binned_tokens")
    if use_binned is True or str(use_binned).lower() == "true":
        return "binned"
    tok = _safe_get(cfg, "classifier.model.tokenizer.name")
    if tok == "pretrained":
        return "vq"
    return "direct"


def _infer_pid_encoding(cfg: DictConfig | dict) -> str:
    """Infer PID encoding from config."""
    tokenizer_name = _safe_get(cfg, "classifier.model.tokenizer.name")

    if tokenizer_name == "identity":
        # Identity tokenizer uses embedding
        return "embedded"
    if tokenizer_name == "raw":
        return "none"
    if tokenizer_name == "binned":
        return "embedded"  # Binned uses integer tokens → embedding
    if tokenizer_name == "pretrained":
        return "embedded"

    # Try to infer from id_embed_dim presence
    id_embed_dim = _safe_get(cfg, "classifier.model.tokenizer.id_embed_dim")
    if id_embed_dim and id_embed_dim > 0:
        return "embedded"

    return "unknown"


def _infer_met_rep(cfg: DictConfig | dict) -> str:
    """Infer MET representation from config."""
    globals_present = _safe_get(cfg, "data.globals.present")
    globals_size = _safe_get(cfg, "data.globals.size")

    if globals_present is False:
        return "none"
    if globals_present is True:
        if globals_size == 2:
            return "met_metphi"  # MET and MET_phi
        return "met_metphi"  # Default assumption

    return "unknown"


def _infer_feature_mode(datatreatment: dict[str, Any] | None) -> str | None:
    """Derive a high-level feature_mode string from datatreatment.

    Encodes 4-vector vs 5-vector and presence of MET/globals in a compact form.

    Examples
    --------
    - "4vec"
    - "4vec+MET"
    - "5vec+MET+globals"
    """
    if not datatreatment:
        return None

    pid_encoding = datatreatment.get("pid_encoding")
    met_rep = datatreatment.get("met_rep")
    globals_included = datatreatment.get("globals_included")

    # 4-vector vs 5-vector based on PID encoding
    base = "5vec" if pid_encoding == "embedded" else "4vec"

    suffixes: list[str] = []
    if met_rep and met_rep != "none":
        suffixes.append("MET")
    if globals_included is True:
        suffixes.append("globals")

    if not suffixes:
        return base

    return base + "+" + "+".join(suffixes)


# =============================================================================
# Canonicalization Functions
# =============================================================================


def canonicalize_process_groups(
    label_groups: list[dict[str, Any]],
    preserve_signal_first: bool = False,
) -> list[list[str]] | None:
    """Convert label_groups to canonical list-of-lists with deterministic ordering.

    Canonicalization rules:
    1. Convert ProcessIDs to names using PROCESS_ID_NAMES
    2. Sort processes within each class lexicographically
    3. Sort classes by string signature: "+".join(class)
       Exception: if preserve_signal_first=True (signal_vs_background config),
       keep the original order (background=0, signal=1)

    Parameters
    ----------
    label_groups : list[dict]
        List of {"name": str, "labels": list[int]} dicts from _normalize_label_groups
    preserve_signal_first : bool
        If True, preserve original order (for signal_vs_background configs)

    Returns
    -------
    list[list[str]] | None
        Canonicalized list-of-lists of process names, or None if empty/invalid
    """
    if not label_groups:
        return None

    result = []
    for group in label_groups:
        labels = group.get("labels", [])
        if not labels:
            continue
        # Convert IDs to names and sort lexicographically within class
        names = sorted([PROCESS_ID_NAMES.get(lid, f"unknown_{lid}") for lid in labels])
        result.append(names)

    if not result:
        return None

    # Sort classes by string signature for deterministic ordering
    # Unless preserving signal_vs_background order
    if not preserve_signal_first:
        result = sorted(result, key=lambda cls: "+".join(cls))

    return result


def build_process_groups_key(process_groups: list[list[str]] | None) -> str | None:
    """Build compact key for W&B grouping (no spaces).

    Format: "4t|ttH+ttW+ttWW+ttZ"

    Parameters
    ----------
    process_groups : list[list[str]] | None
        Canonicalized process groups

    Returns
    -------
    str | None
        Compact key or None
    """
    if not process_groups:
        return None
    return "|".join("+".join(cls) for cls in process_groups)


def build_class_def_str(process_groups: list[list[str]] | None) -> str | None:
    """Build human-readable class definition string (with spaces).

    Format: "4t | ttH+ttW+ttWW+ttZ"

    Parameters
    ----------
    process_groups : list[list[str]] | None
        Canonicalized process groups

    Returns
    -------
    str | None
        Human-readable string or None
    """
    if not process_groups:
        return None
    return " | ".join("+".join(cls) for cls in process_groups)


def canonicalize_datatreatment(cfg: DictConfig | dict) -> tuple[dict[str, Any], str]:
    """Extract and canonicalize data treatment fields from config.

    Never bakes in guesses - uses "unknown" if not provable from config.

    Parameters
    ----------
    cfg : DictConfig or dict
        Full config

    Returns
    -------
    tuple[dict, str]
        (treatment_dict, confidence)
    """
    treatment = {
        "token_order": _infer_token_order(cfg),
        "tokenization": _infer_tokenization(cfg),
        "pid_encoding": _infer_pid_encoding(cfg),
        "id_embed_dim": _safe_get(cfg, "classifier.model.tokenizer.id_embed_dim"),
        "met_rep": _infer_met_rep(cfg),
        "globals_included": _safe_get(cfg, "data.globals.present", "unknown"),
        "feature_set_version": FEATURE_SET_VERSION,
        "normalization": _safe_get(cfg, "data.norm", "unknown"),
    }

    # Count unknowns for confidence
    unknown_count = sum(1 for v in treatment.values() if v == "unknown")
    if unknown_count == 0:
        confidence = "high"
    elif unknown_count < 3:
        confidence = "medium"
    else:
        confidence = "low"

    return treatment, confidence


def compute_meta_hash(meta: dict[str, Any]) -> str:
    """Compute stable hash of canonical fields for dedup/grouping.

    Parameters
    ----------
    meta : dict
        Meta dictionary (must have canonical fields populated)

    Returns
    -------
    str
        Hash in format "sha1:abc123def456"
    """
    canonical = {
        "schema_version": meta.get("schema_version"),
        "level": meta.get("level"),
        "goal": meta.get("goal"),
        "dataset_name": meta.get("dataset_name"),
        "model_family": meta.get("model_family"),
        "process_groups": meta.get("process_groups"),
        "datatreatment": meta.get("datatreatment"),
    }
    content = json.dumps(canonical, sort_keys=True, default=str)
    hash_val = hashlib.sha1(content.encode()).hexdigest()[:12]
    return f"sha1:{hash_val}"


# =============================================================================
# Main Builder
# =============================================================================


def build_meta(cfg: DictConfig | dict) -> dict[str, Any]:
    """Build canonical meta.json from resolved Hydra config.

    This is the main entry point for generating run metadata.

    Parameters
    ----------
    cfg : DictConfig or dict
        Full resolved Hydra config

    Returns
    -------
    dict
        Complete meta dictionary ready to write to facts/meta.json
    """
    # Import here to avoid circular imports
    try:
        from thesis_ml.data.h5_loader import _normalize_label_groups

        has_h5_loader = True
    except ImportError:
        has_h5_loader = False
        logger.warning("Could not import _normalize_label_groups, process_groups will be null")

    confidences: dict[str, str] = {}
    review_reasons: list[str] = []

    # --- Goal: explicit > inferred ---
    goal = _safe_get(cfg, "meta.goal")
    if goal:
        confidences["goal"] = "high"
    else:
        loop = _safe_get(cfg, "loop", "")
        goal = _infer_goal_from_loop(loop)
        confidences["goal"] = "high" if goal else "low"
        if not goal:
            review_reasons.append("missing_goal")

    # --- Dataset name ---
    dataset_name = _extract_dataset_name(cfg)
    confidences["dataset_name"] = "high" if dataset_name else "low"
    if not dataset_name:
        review_reasons.append("missing_dataset_name")

    # --- Model family ---
    loop = _safe_get(cfg, "loop", "")
    model_family = _infer_model_family(loop)
    confidences["model_family"] = "high" if model_family else "low"
    if not model_family:
        review_reasons.append("missing_model_family")

    # --- Process groups ---
    process_groups: list[list[str]] | None = None
    if has_h5_loader:
        try:
            # Check if signal_vs_background was used (preserve order)
            is_signal_vs_bg = _safe_get(cfg, "data.classifier.signal_vs_background") is not None

            label_groups, _, _ = _normalize_label_groups(cfg)
            process_groups = canonicalize_process_groups(label_groups, preserve_signal_first=is_signal_vs_bg)
            confidences["process_groups"] = "high" if process_groups else "low"
        except Exception as e:
            logger.warning("Could not normalize label groups: %s", e)
            confidences["process_groups"] = "low"

    if not process_groups:
        review_reasons.append("missing_process_groups")
        confidences["process_groups"] = "low"

    # --- Datatreatment ---
    datatreatment, dt_conf = canonicalize_datatreatment(cfg)
    confidences["datatreatment"] = dt_conf
    if dt_conf == "low":
        review_reasons.append("ambiguous_datatreatment")

    # --- Overall confidence = min ---
    conf_order = {"high": 0, "medium": 1, "low": 2}
    overall_conf = min(confidences.values(), key=lambda c: conf_order.get(c, 2))

    # --- Build meta dict ---
    meta: dict[str, Any] = {
        # Canonical fields
        "schema_version": _safe_get(cfg, "meta.schema_version", SCHEMA_VERSION),
        "level": _safe_get(cfg, "meta.level", "sim_event"),
        "goal": goal,
        "dataset_name": dataset_name,
        "model_family": model_family,
        "process_groups": process_groups,
        "datatreatment": datatreatment,
        "feature_mode": _infer_feature_mode(datatreatment),
        "meta_hash": None,  # computed below
        "meta_source": "live",
        "meta_confidence": overall_conf,
        "meta_confidence_fields": confidences,
        "needs_review": len(review_reasons) > 0,
        "needs_review_reason": review_reasons,
        # Derived fields (cache)
        "n_classes": len(process_groups) if process_groups else None,
        "processes_all": (sorted(set(p for cls in process_groups for p in cls)) if process_groups else None),
        "class_def_str": build_class_def_str(process_groups),
        "process_groups_key": build_process_groups_key(process_groups),
        "row_key": None,  # computed below
    }

    # Compute hash and row_key
    meta["meta_hash"] = compute_meta_hash(meta)
    if meta["process_groups_key"] and dataset_name:
        meta["row_key"] = f"{dataset_name}::{meta['process_groups_key']}"

    return meta


# =============================================================================
# I/O Functions
# =============================================================================


def write_meta(meta: dict[str, Any], path: Path | str) -> None:
    """Write meta dictionary to JSON file.

    Parameters
    ----------
    meta : dict
        Meta dictionary from build_meta()
    path : Path or str
        Output path (typically run_dir / "facts" / "meta.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("[meta] wrote %s", path)


def read_meta(path: Path | str) -> dict[str, Any] | None:
    """Read meta dictionary from JSON file.

    Parameters
    ----------
    path : Path or str
        Path to meta.json file

    Returns
    -------
    dict | None
        Meta dictionary or None if file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_meta_override(run_dir: Path | str) -> dict[str, Any] | None:
    """Load manual override file if it exists.

    Parameters
    ----------
    run_dir : Path or str
        Run directory

    Returns
    -------
    dict | None
        Override dictionary or None if no override file
    """
    override_path = Path(run_dir) / "facts" / "meta_override.json"
    if not override_path.exists():
        return None

    with open(override_path, encoding="utf-8") as f:
        return json.load(f)


def merge_meta_with_override(meta: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge inferred meta with manual override values.

    Override values win. Updates meta_source for overridden fields.

    Parameters
    ----------
    meta : dict
        Inferred meta dictionary
    override : dict
        Override values

    Returns
    -------
    dict
        Merged meta dictionary
    """
    merged = meta.copy()

    for key, value in override.items():
        if key in merged and value is not None:
            merged[key] = value
            # Mark that this field came from override
            if "meta_confidence_fields" in merged and key in merged["meta_confidence_fields"]:
                merged["meta_confidence_fields"][key] = "high"

    # Update source to reflect override
    if override:
        merged["meta_source"] = "override"

    # Recalculate derived fields if process_groups changed
    if "process_groups" in override and override["process_groups"] is not None:
        pg = override["process_groups"]
        merged["n_classes"] = len(pg) if pg else None
        merged["processes_all"] = sorted(set(p for cls in pg for p in cls)) if pg else None
        merged["class_def_str"] = build_class_def_str(pg)
        merged["process_groups_key"] = build_process_groups_key(pg)
        if merged["process_groups_key"] and merged.get("dataset_name"):
            merged["row_key"] = f"{merged['dataset_name']}::{merged['process_groups_key']}"

    # Recalculate hash
    merged["meta_hash"] = compute_meta_hash(merged)

    # Clear review status if key fields are now present
    if merged.get("process_groups") and merged.get("goal") and merged.get("dataset_name"):
        merged["needs_review"] = False
        merged["needs_review_reason"] = []

    return merged
