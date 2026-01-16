from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _parse_hydra_overrides(cfg: dict[str, Any]) -> dict[str, str]:
    """Parse all Hydra override parameters from config.

    Parameters
    ----------
    cfg : dict[str, Any]
        Full config dict with hydra section

    Returns
    -------
    dict[str, str]
        Mapping of override keys to values (as strings)
    """
    overrides: dict[str, str] = {}
    hydra_cfg = cfg.get("hydra", {})
    if not hydra_cfg:
        return overrides

    task_overrides = hydra_cfg.get("overrides", {}).get("task", [])
    if not task_overrides:
        return overrides

    for override in task_overrides:
        override_str = str(override)
        if "=" in override_str:
            # Split on first '=' to handle values that might contain '='
            key, val = override_str.split("=", 1)
            overrides[key] = val

    return overrides


def _normalize_override_key(key: str) -> str:
    """Normalize override key to DataFrame-friendly column name.

    Converts Hydra override keys to simpler column names:
    - `phase1/latent_space` -> `latent_space`
    - `phase1.decoder.globals_beta` -> `globals_beta` (or `decoder.globals_beta`)

    Parameters
    ----------
    key : str
        Raw override key from Hydra

    Returns
    -------
    str
        Normalized column name
    """
    # Remove config group prefix (e.g., "phase1/" -> "")
    if "/" in key:
        key = key.split("/", 1)[1]

    # For dot-separated paths, try to extract meaningful part
    # Prefer last component if it's a single word, otherwise keep structure
    if "." in key:
        parts = key.split(".")
        # If last part is a common parameter name, use just that
        # Otherwise keep the path structure
        if len(parts) > 1 and parts[-1] in ("globals_beta", "epochs", "seed", "lr"):
            return parts[-1]
        # Keep structure but remove leading phase1 if present
        if parts[0] == "phase1":
            return ".".join(parts[1:])
        return key

    return key


def _extract_value_from_composed_cfg(cfg: dict[str, Any], path: str, value_type: type | None = None) -> Any:
    """Extract value from composed config using a dot-separated path.

    Handles multiple value types:
    - Strings: return as-is
    - Dicts with 'name' key: return name
    - Dicts with '_target_' key: infer from class name
    - Scalars: return with optional type conversion

    Parameters
    ----------
    cfg : dict[str, Any]
        Composed config dict
    path : str
        Dot-separated path (e.g., "phase1.latent_space" or "phase1.decoder.globals_beta")
    value_type : type | None
        Optional type to convert result to (e.g., float)

    Returns
    -------
    Any
        Extracted value, or None if path doesn't exist or can't be inferred
    """
    try:
        # Convert path to list of keys, handling both '/' and '.' separators
        keys = path.replace("/", ".").split(".")

        # Navigate through nested dict
        value = cfg
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
            if value is None:
                return None

        # Handle different value types
        if isinstance(value, str):
            result = value
        elif isinstance(value, dict):
            # Try 'name' key first (common in config groups)
            if "name" in value and isinstance(value["name"], str):
                result = value["name"]
            # Try '_target_' key (Hydra instantiation)
            elif "_target_" in value and isinstance(value["_target_"], str):
                # Extract class name from _target_ path
                class_name = value["_target_"].split(".")[-1].lower()
                result = class_name
            else:
                # Can't infer from dict structure
                return None
        else:
            # Scalar value (int, float, bool, etc.)
            result = value

        # Apply type conversion if requested
        if value_type is not None and result is not None:
            try:
                return value_type(result)
            except (ValueError, TypeError):
                return None

        return result
    except Exception:
        return None


@dataclass
class RunFacts:
    run_dir: Path
    cfg: dict[str, Any]
    scalars: pd.DataFrame
    events: list[dict[str, Any]] | None
    config_sha1: str | None
    git_commit: str | None


def _read_cfg(run_dir: Path) -> tuple[dict[str, Any], str | None]:
    # Prefer .hydra/config.yaml as canonical record (no cfg.yaml duplication)
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    if hydra_cfg_path.exists():
        txt = hydra_cfg_path.read_text(encoding="utf-8")
        sha1 = hashlib.sha1(txt.encode("utf-8")).hexdigest()
        cfg = OmegaConf.to_container(OmegaConf.load(hydra_cfg_path), resolve=True)  # type: ignore
        assert isinstance(cfg, dict)
        return cfg, sha1
    # Fallback to cfg.yaml for backward compatibility with old runs
    cfg_path = run_dir / "cfg.yaml"
    if cfg_path.exists():
        txt = cfg_path.read_text(encoding="utf-8")
        sha1 = hashlib.sha1(txt.encode("utf-8")).hexdigest()
        cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)  # type: ignore
        assert isinstance(cfg, dict)
        return cfg, sha1
    raise FileNotFoundError(f"Missing .hydra/config.yaml or cfg.yaml in {run_dir}")


def _read_events(run_dir: Path) -> list[dict[str, Any]] | None:
    fp = run_dir / "facts" / "events.jsonl"
    if not fp.exists():
        return None
    events: list[dict[str, Any]] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def _has_on_train_end(events: list[dict[str, Any]] | None) -> bool:
    if not events:
        return False
    return any(rec.get("moment") == "on_train_end" for rec in events)


def _read_scalars(run_dir: Path) -> pd.DataFrame:
    fp = run_dir / "facts" / "scalars.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing facts/scalars.csv in {run_dir}")
    df = pd.read_csv(fp)
    return df


def _extract_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata from config, including all override parameters.

    Extracts both fixed metadata fields and all Hydra override parameters
    generically. For override parameters, attempts to extract from overrides
    first, then falls back to composed config values.

    Parameters
    ----------
    cfg : dict[str, Any]
        Full config dict

    Returns
    -------
    dict[str, Any]
        Metadata dict with fixed fields and all override parameters
    """
    # Best-effort extraction; tolerate missing keys
    encoder = cfg.get("phase1", {}).get("encoder") or cfg.get("phase1", {}).get("encoder", {})
    if isinstance(encoder, dict):
        encoder = encoder.get("name")
    tokenizer = cfg.get("phase1", {}).get("tokenizer") or cfg.get("phase1", {}).get("tokenizer", {})
    if isinstance(tokenizer, dict):
        tokenizer = tokenizer.get("name")
    seed = cfg.get("trainer", {}).get("seed")
    latent_dim = cfg.get("model", {}).get("latent_dim") or cfg.get("phase1", {}).get("model", {}).get("latent_dim")
    codebook_size = cfg.get("model", {}).get("codebook_size") or cfg.get("phase1", {}).get("model", {}).get("codebook_size")
    dataset_name = cfg.get("data", {}).get("path") or cfg.get("data", {}).get("name")

    # Parse all Hydra overrides generically
    overrides = _parse_hydra_overrides(cfg)

    # Extract all override parameters with fallback to composed config
    override_params: dict[str, Any] = {}
    for override_key, override_value in overrides.items():
        # Normalize key for column name
        normalized_key = _normalize_override_key(override_key)

        # Try to get value from override first
        value = override_value

        # Determine if we should try type conversion based on key patterns
        # Common numeric parameters
        if normalized_key in ("globals_beta", "epochs", "lr", "seed") or "beta" in normalized_key.lower():
            try:
                value = float(override_value)
            except (ValueError, TypeError):
                # Fallback to composed config
                composed_value = _extract_value_from_composed_cfg(cfg, override_key, float)
                value = composed_value if composed_value is not None else override_value
        else:
            # For config groups (like latent_space, encoder), fallback to composed config
            # if override value looks like a simple string identifier
            if "/" in override_key or override_key.count(".") <= 1:
                composed_value = _extract_value_from_composed_cfg(cfg, override_key)
                if composed_value is not None:
                    value = composed_value

        override_params[normalized_key] = value

    # Build result dict with fixed metadata and all override parameters
    result = {
        "encoder": encoder,
        "tokenizer": tokenizer,
        "seed": seed,
        "latent_dim": latent_dim,
        "codebook_size": codebook_size,
        "dataset_name": dataset_name,
        "overrides": overrides,  # Keep raw overrides for backward compatibility
    }

    # Add all override parameters (with normalized keys)
    result.update(override_params)

    return result


def _discover_pointer_targets(runs_dir: Path) -> list[Path]:
    targets: list[Path] = []
    for p in runs_dir.iterdir():
        if p.is_file() and p.suffix == ".json" and p.name.endswith(".pointer.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                raw = data.get("path")
                if not raw:
                    continue
                tgt = Path(raw)
                if not tgt.is_absolute():
                    tgt = (p.parent / tgt).resolve()
                if tgt.exists() and tgt.is_dir():
                    targets.append(tgt)
            except Exception:
                continue
    return targets


def discover_runs(sweep_dir: Path | None, run_dirs: Iterable[Path] | None) -> list[Path]:
    """Discover run directories from a sweep directory or explicit run list.

    Parameters
    ----------
    sweep_dir : Path | None
        Path to a multirun directory (e.g., outputs/multiruns/exp_...)
    run_dirs : Iterable[Path] | None
        Explicit list of run directories

    Returns
    -------
    list[Path]
        Sorted list of discovered run directories
    """
    if (sweep_dir is None) == (not run_dirs):
        raise ValueError("Provide exactly one of sweep_dir or run_dirs")
    candidates: list[Path] = []
    if sweep_dir is not None:
        if not sweep_dir.exists() or not sweep_dir.is_dir():
            raise FileNotFoundError(f"sweep_dir not found: {sweep_dir}")

        # Extract timestamp from multirun directory name (format: exp_TIMESTAMP_name)
        # Match runs in outputs/runs/ with the same timestamp
        multirun_name = sweep_dir.name
        if multirun_name.startswith("exp_"):
            # Extract timestamp: exp_YYYYMMDD-HHMMSS_name -> YYYYMMDD-HHMMSS
            parts = multirun_name.replace("exp_", "").split("_", 1)
            if parts:
                timestamp = parts[0]  # e.g., "20251103-140953"
                # Extract experiment name if present
                exp_name = parts[1] if len(parts) > 1 else None

                # Infer output_root from sweep_dir
                parts = sweep_dir.parts
                try:
                    multiruns_idx = parts.index("multiruns")
                    output_root = Path(*parts[:multiruns_idx])
                    runs_dir = output_root / "runs"

                    if runs_dir.exists():
                        # Scan for runs matching the timestamp pattern
                        # Pattern: run_TIMESTAMP_name_jobN or run_TIMESTAMP_name
                        for run_path in runs_dir.iterdir():
                            if not run_path.is_dir():
                                continue
                            run_name = run_path.name
                            # Check if run matches timestamp and optionally experiment name
                            if timestamp in run_name and (exp_name is None or exp_name in run_name) and ((run_path / ".hydra" / "config.yaml").exists() or (run_path / "cfg.yaml").exists()):
                                candidates.append(run_path)
                except ValueError:
                    pass

        # Fallback: direct child dirs with .hydra/config.yaml or cfg.yaml (legacy nested structure)
        for p in sweep_dir.iterdir():
            if p.is_dir() and ((p / ".hydra" / "config.yaml").exists() or (p / "cfg.yaml").exists()):
                candidates.append(p)
    else:
        assert run_dirs is not None
        for p in run_dirs:
            pp = Path(p)
            if pp.exists() and pp.is_dir():
                candidates.append(pp)

    # Deterministic order: by folder name timestamp if present, else mtime, else lexicographic
    def sort_key(p: Path):
        name = p.name
        # Expect YYYYMMDD-HHMMSS or similar; fallback to mtime
        return (name, p.stat().st_mtime)

    candidates.sort(key=sort_key)
    return candidates


def load_runs(sweep_dir: str | None = None, run_dirs: list[str] | None = None, *, require_complete: bool = True) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    """Load facts from multiple runs into DataFrames.

    Parameters
    ----------
    sweep_dir : str | None
        Path to multirun sweep directory
    run_dirs : list[str] | None
        Explicit list of run directories
    require_complete : bool
        If True, skip runs without on_train_end event

    Returns
    -------
    runs_df : pd.DataFrame
        Summary DataFrame with one row per run
    per_epoch : dict[str, pd.DataFrame]
        Per-epoch metrics for each run (key is run_dir string)
    order : list[str]
        List of run_dir strings in discovery order
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    sd = Path(sweep_dir) if sweep_dir else None
    rds = [Path(p) for p in (run_dirs or [])] or None
    found = discover_runs(sd, rds)
    summaries: list[dict[str, Any]] = []
    per_epoch: dict[str, pd.DataFrame] = {}
    order: list[str] = []

    for rd in found:
        try:
            cfg, cfg_sha1 = _read_cfg(rd)
            events = _read_events(rd)
            if require_complete and not _has_on_train_end(events):
                logger.warning("Skip incomplete run (no on_train_end): %s", rd)
                continue
            scalars = _read_scalars(rd)
            # Validate minimal required columns
            required_cols = {"epoch", "split", "val_loss", "epoch_time_s"}
            if not required_cols.issubset(set(scalars.columns)):
                logger.warning("Skip run missing minimal columns %s: %s", required_cols - set(scalars.columns), rd)
                continue

            meta = _extract_metadata(cfg)
            git_commit = None
            gc_file = rd / "facts" / "git_commit.txt"
            if gc_file.exists():
                git_commit = gc_file.read_text(encoding="utf-8").strip() or None

            # compute per-run aggregates
            val_df = scalars[scalars["split"] == "val"].copy()
            val_df.sort_values("epoch", inplace=True)
            best_idx = int(val_df["val_loss"].astype(float).idxmin())
            best_row = val_df.loc[best_idx]
            final_row = val_df.iloc[-1]
            total_time_s = None
            if events:
                for rec in events:
                    if rec.get("moment") == "on_train_end":
                        total_time_s = rec.get("total_time_s")
                        break
            if total_time_s is None:
                total_time_s = float(val_df["epoch_time_s"].astype(float).sum())

            # stable loss components (best-effort from events histories)
            loss_recon_best = None
            loss_commit_best = None
            loss_codebook_best = None
            loss_rec_globals_best = None
            if events:
                # search last on_epoch_end with histories
                hist = None
                for rec in events:
                    if rec.get("moment") == "on_epoch_end":
                        hist = rec
                if hist:
                    # pick component values at best epoch index if arrays are present
                    be = int(best_row["epoch"]) if not pd.isna(best_row["epoch"]) else -1
                    try:
                        if isinstance(hist.get("history_rec_tokens"), list):
                            loss_recon_best = float(hist["history_rec_tokens"][be])
                    except Exception:
                        pass
                    try:
                        if isinstance(hist.get("history_perplex"), list):
                            pass
                    except Exception:
                        pass
                    try:
                        if isinstance(hist.get("history_codebook"), list):
                            loss_codebook_best = float(hist["history_codebook"][be])
                    except Exception:
                        pass
                    try:
                        # commit not explicitly in histories as loss; may be inside aux totals
                        if isinstance(hist.get("history_perplex"), list):
                            pass
                    except Exception:
                        pass
                    # Extract rec_globals from history
                    if hist and be >= 0:
                        try:
                            if isinstance(hist.get("history_rec_globals"), list):
                                arr = hist["history_rec_globals"]
                                if be < len(arr):
                                    loss_rec_globals_best = float(arr[be])
                        except Exception:
                            pass

            summary = {
                "run_dir": str(rd),
                "encoder": meta.get("encoder"),
                "tokenizer": meta.get("tokenizer"),
                "seed": meta.get("seed"),
                "latent_dim": meta.get("latent_dim"),
                "codebook_size": meta.get("codebook_size"),
                "dataset_name": meta.get("dataset_name"),
                "config_sha1": cfg_sha1,
                "git_commit": git_commit,
                "epochs": int(val_df["epoch"].max()) + 1 if len(val_df) else 0,
                "best_epoch": int(best_row["epoch"]) if len(val_df) else None,
                "loss.total_best": float(best_row["val_loss"]) if len(val_df) else None,
                "loss.total_final": float(final_row["val_loss"]) if len(val_df) else None,
                "loss.recon_best": loss_recon_best,
                "loss.commit_best": loss_commit_best,
                "loss.codebook_best": loss_codebook_best,
                "total_time_s": float(total_time_s) if total_time_s is not None else None,
                "mean_epoch_time_s": float(val_df["epoch_time_s"].astype(float).mean()) if len(val_df) else None,
                "throughput_mean": float(val_df.get("throughput", pd.Series(dtype=float)).astype(float).mean()) if "throughput" in val_df else None,
                "max_memory_mib_max": float(val_df.get("max_memory_mib", pd.Series(dtype=float)).astype(float).max()) if "max_memory_mib" in val_df else None,
                "metric_perplex_final": float(val_df.get("metric_perplex", pd.Series(dtype=float)).astype(float).iloc[-1]) if "metric_perplex" in val_df and len(val_df) else None,
                "loss.rec_globals_best": loss_rec_globals_best,
            }

            # Dynamically add all override parameters from metadata
            # Exclude fixed metadata fields and the raw 'overrides' dict
            fixed_fields = {"encoder", "tokenizer", "seed", "latent_dim", "codebook_size", "dataset_name", "overrides"}
            for key, value in meta.items():
                if key not in fixed_fields:
                    summary[key] = value
            summaries.append(summary)
            per_epoch[str(rd)] = val_df.reset_index(drop=True)
            order.append(str(rd))
        except Exception as e:
            logger.warning("Skip run %s due to error: %s", rd, e)
            continue

    runs_df = pd.DataFrame(summaries)
    return runs_df, per_epoch, order


# =============================================================================
# Metadata Extraction Helpers
# =============================================================================


def extract_classifier_metadata(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract classifier model metadata from runs.

    Extracts common classifier hyperparameters:
    - norm_policy
    - positional
    - pooling
    - dim, depth, heads (for model size)
    - dropout, weight_decay, lr (for regularization)

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information (from load_runs)

    Returns
    -------
    pd.DataFrame
        DataFrame with added metadata columns
    """
    df = runs_df.copy()

    # Define all parameters to extract
    params_config = {
        # Model architecture
        "norm_policy": "classifier.model.norm.policy",
        "positional": "classifier.model.positional",
        "pooling": "classifier.model.pooling",
        "dim": "classifier.model.dim",
        "depth": "classifier.model.depth",
        "heads": "classifier.model.heads",
        # Regularization
        "dropout": "classifier.model.dropout",
        "weight_decay": "classifier.trainer.weight_decay",
        "lr": "classifier.trainer.lr",
        "label_smoothing": "classifier.trainer.label_smoothing",
    }

    for param_name, config_path in params_config.items():
        # Check if column already exists (from override extraction)
        if param_name in df.columns and not df[param_name].isna().all():
            continue

        # Check for matching columns (override keys might be formatted differently)
        matching_cols = [col for col in df.columns if param_name in col.lower()]
        if matching_cols:
            df[param_name] = df[matching_cols[0]]
            continue

        # Read from config files
        param_values = []
        for run_dir in df["run_dir"]:
            try:
                cfg, _ = _read_cfg(Path(run_dir))
                # Determine type based on parameter
                if param_name in ("dim", "depth", "heads"):
                    value = _extract_value_from_composed_cfg(cfg, config_path, int)
                elif param_name in ("dropout", "weight_decay", "lr", "label_smoothing"):
                    value = _extract_value_from_composed_cfg(cfg, config_path, float)
                else:
                    value = _extract_value_from_composed_cfg(cfg, config_path)
                param_values.append(value)
            except Exception as e:
                logger.warning(f"Failed to extract {param_name} from {run_dir}: {e}")
                param_values.append(None)

        df[param_name] = param_values

    return df


def extract_model_size(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Add model size estimate and size label to runs DataFrame.

    Computes:
    - model_size: estimated parameter count
    - size_label: human-readable label (e.g., "256d6L")

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information

    Returns
    -------
    pd.DataFrame
        DataFrame with added model_size and size_label columns
    """
    df = runs_df.copy()

    # Ensure dim and depth are present
    if "dim" not in df.columns or "depth" not in df.columns:
        df = extract_classifier_metadata(df)

    # Compute model size estimate
    model_sizes = []
    size_labels = []

    for _, row in df.iterrows():
        dim = row.get("dim")
        depth = row.get("depth")

        if pd.notna(dim) and pd.notna(depth):
            # Rough parameter estimate
            mlp_ratio = 4
            block_params = 4 * dim * dim + 2 * dim * dim * mlp_ratio
            transformer_params = depth * block_params
            # Add embedding and head (rough estimate)
            total_params = transformer_params + 10000  # Rough estimate for other params
            model_sizes.append(int(total_params))
            size_labels.append(f"{int(dim)}d{int(depth)}L")
        else:
            model_sizes.append(None)
            size_labels.append(None)

    df["model_size"] = model_sizes
    df["size_label"] = size_labels

    return df


def extract_regularization_params(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Add regularization parameter columns to runs DataFrame.

    Extracts:
    - dropout
    - weight_decay
    - lr
    - label_smoothing

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information

    Returns
    -------
    pd.DataFrame
        DataFrame with added regularization columns
    """
    df = runs_df.copy()

    params_to_extract = {
        "dropout": "classifier.model.dropout",
        "weight_decay": "classifier.trainer.weight_decay",
        "lr": "classifier.trainer.lr",
        "label_smoothing": "classifier.trainer.label_smoothing",
    }

    for param_name, config_path in params_to_extract.items():
        if param_name in df.columns and not df[param_name].isna().all():
            continue

        param_values = []
        for run_dir in df["run_dir"]:
            try:
                cfg, _ = _read_cfg(Path(run_dir))
                value = _extract_value_from_composed_cfg(cfg, config_path, float)
                param_values.append(value)
            except Exception:
                param_values.append(None)

        df[param_name] = param_values

        # Convert to numeric
        with contextlib.suppress(Exception):
            df[param_name] = pd.to_numeric(df[param_name], errors="coerce")

    return df


def extract_positional_encoding(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Add positional encoding column to runs DataFrame.

    Parameters
    ----------
    runs_df : pd.DataFrame
        DataFrame with run information

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'positional' column
    """
    df = runs_df.copy()

    if "positional" in df.columns and not df["positional"].isna().all():
        return df

    pos_values = []
    for run_dir in df["run_dir"]:
        try:
            cfg, _ = _read_cfg(Path(run_dir))
            value = _extract_value_from_composed_cfg(cfg, "classifier.model.positional")
            pos_values.append(value)
        except Exception:
            pos_values.append(None)

    df["positional"] = pos_values

    return df
