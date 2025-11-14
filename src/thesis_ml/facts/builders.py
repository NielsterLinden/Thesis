from __future__ import annotations

import os
import socket
import subprocess
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

import torch


def _get_git_commit() -> str | None:
    """Get current git commit hash (short). Returns None if not in a git repo.

    This function works even when called from subdirectories (like outputs/runs/...)
    by first finding the git repository root, then getting the commit hash from there.
    """
    try:
        # First, find the git repository root
        # This works even if we're in a subdirectory (like outputs/runs/...)
        repo_root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if repo_root_result.returncode != 0:
            return None

        repo_root = repo_root_result.stdout.strip()

        # Now get the commit hash from the repo root
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _get_cuda_info() -> dict[str, Any]:
    """Get CUDA availability and device info."""
    info = {"available": torch.cuda.is_available()}
    if info["available"]:
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return info


def _collect_metadata(**extra_fields: Any) -> dict[str, Any]:
    """Collect runtime metadata for reproducibility.

    Parameters
    ----------
    **extra_fields
        May include 'cfg' for extracting seed/hydra info

    Returns
    -------
    dict
        Metadata block with timestamp, run_id, git_commit, hostname, etc.
    """
    meta = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": str(uuid.uuid4()),
        "git_commit": _get_git_commit(),
        "hostname": socket.gethostname(),
        "cuda_info": _get_cuda_info(),
    }

    # Extract seed from config if available
    cfg = extra_fields.get("cfg")
    if cfg is not None:
        # Try various paths where seed might live
        seed = None
        if hasattr(cfg, "phase1") and hasattr(cfg.phase1, "trainer"):
            seed = getattr(cfg.phase1.trainer, "seed", None)
        elif hasattr(cfg, "trainer"):
            seed = getattr(cfg.trainer, "seed", None)
        elif hasattr(cfg, "general") and hasattr(cfg.general, "trainer"):
            seed = getattr(cfg.general.trainer, "seed", None)
        elif hasattr(cfg, "data"):
            seed = getattr(cfg.data, "seed", None)
        if seed is not None:
            meta["seed"] = int(seed)

    # Extract Hydra job ID if available
    hydra_job_id = os.environ.get("HYDRA_JOB_ID")
    if hydra_job_id:
        meta["hydra_job_id"] = hydra_job_id

    return meta


def build_event_payload(
    *,
    moment: Literal["on_start", "on_epoch_end", "on_train_end", "on_test_end"],
    run_dir: str | None = None,
    epoch: int | None = None,
    step: int | None = None,
    split: str | None = None,
    train_loss: float | None = None,
    val_loss: float | None = None,
    metrics: Mapping[str, float] | None = None,
    epoch_time_s: float | None = None,
    total_time_s: float | None = None,
    throughput: float | None = None,
    max_memory_mib: float | None = None,
    histories: Mapping[str, list[float]] | None = None,
    **extra_fields: Any,
) -> dict[str, Any]:
    """Build standardized event payload for logging/plotting.

    Creates a consistent payload structure with:
    - Schema versioning for safe evolution
    - Rich metadata (timestamp, git commit, hostname, etc.)
    - Both structured and flattened history formats for backward compatibility

    Parameters
    ----------
    moment : Literal["on_start", "on_epoch_end", "on_train_end", "on_test_end"]
        Event type identifier
    run_dir : str | None
        Run output directory path
    epoch, step : int | None
        Training progress indicators
    split : str | None
        Data split: "train", "val", "test"
    train_loss, val_loss : float | None
        Current losses
    metrics : Mapping[str, float] | None
        Additional scalar metrics (perplex, acc, etc.)
    epoch_time_s, total_time_s : float | None
        Timing information in seconds
    throughput : float | None
        Training throughput (samples/sec)
    max_memory_mib : float | None
        GPU memory usage in MiB
    histories : Mapping[str, list] | None
        Full training histories (e.g., {"train_loss": [...], "perplex": [...]})
        Will be stored both as structured dict and flattened history_<key> fields
    **extra_fields
        Loop-specific fields that don't fit the standard schema.
        Special handling: if 'cfg' is passed, metadata will extract seed/hydra info.

    Returns
    -------
    dict
        Standardized payload ready for append_jsonl_event() or handle_event()

    Examples
    --------
    >>> payload = build_event_payload(
    ...     moment="on_epoch_end",
    ...     run_dir="/path/to/run",
    ...     epoch=10,
    ...     split="val",
    ...     train_loss=0.5,
    ...     val_loss=0.6,
    ...     metrics={"perplex": 128.5},
    ...     epoch_time_s=12.3,
    ...     throughput=1500.0,
    ...     histories={
    ...         "train_loss": [1.0, 0.8, 0.5],
    ...         "val_loss": [1.1, 0.9, 0.6],
    ...     },
    ... )
    """
    # Core payload structure
    payload: dict[str, Any] = {
        "schema_version": 1,
        "moment": moment,
        "run_dir": str(run_dir) if run_dir else "",
        "epoch": int(epoch) if epoch is not None else None,
        "step": int(step) if step is not None else None,
        "split": split,
        "train_loss": float(train_loss) if train_loss is not None else None,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "metrics": dict(metrics) if metrics else {},
        "epoch_time_s": float(epoch_time_s) if epoch_time_s is not None else None,
        "total_time_s": float(total_time_s) if total_time_s is not None else None,
        "throughput": float(throughput) if throughput is not None else None,
    }

    # Optional standard fields
    if max_memory_mib is not None:
        payload["max_memory_mib"] = float(max_memory_mib)

    # Metadata block
    payload["meta"] = _collect_metadata(**extra_fields)

    # Histories: store both structured and flattened for backward compatibility
    if histories:
        # Structured format (new)
        payload["histories"] = {k: list(v) for k, v in histories.items()}

        # Flattened format (legacy compatibility)
        for key, values in histories.items():
            history_key = key if key.startswith("history_") else f"history_{key}"
            payload[history_key] = list(values)

    # Extra loop-specific fields (excluding 'cfg' which was only for metadata)
    for k, v in extra_fields.items():
        if k != "cfg":  # cfg is special, used only for metadata extraction
            payload[k] = v

    return payload
