"""Utilities for creating report manifests with dataset provenance."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


def compute_file_fingerprint(file_path: Path) -> dict[str, Any]:
    """Compute fingerprint (size + sha256) for a dataset file.

    Parameters
    ----------
    file_path : Path
        Path to dataset file

    Returns
    -------
    dict[str, Any]
        Fingerprint with size_bytes and sha256_hash
    """
    if not file_path.exists():
        return {"size_bytes": None, "sha256_hash": None}

    size = file_path.stat().st_size
    sha256 = hashlib.sha256()
    # Read in chunks to handle large files
    with file_path.open("rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return {"size_bytes": size, "sha256_hash": sha256.hexdigest()}


def create_manifest(
    report_id: str,
    run_ids: list[str],
    output_root: Path | str,
    dataset_cfg: DictConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create manifest.yaml content for a report.

    Parameters
    ----------
    report_id : str
        Report ID
    run_ids : list[str]
        List of run IDs used in this report
    output_root : Path | str
        Root output directory (for resolving run paths)
    dataset_cfg : DictConfig | dict[str, Any] | None
        Dataset config (optional) for extracting dataset path and metadata

    Returns
    -------
    dict[str, Any]
        Manifest data structure
    """
    output_root = Path(output_root)

    # Build models list
    models = []
    for run_id in run_ids:
        run_dir = output_root / "runs" / run_id
        checkpoint_path = run_dir / "best_val.pt"
        config_path = run_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            config_path = run_dir / "cfg.yaml"  # Fallback for old runs

        config_sha1 = None
        if config_path.exists():
            config_text = config_path.read_text(encoding="utf-8")
            config_sha1 = hashlib.sha1(config_text.encode("utf-8")).hexdigest()

        models.append(
            {
                "run_id": run_id,
                "checkpoint": "best_val.pt" if checkpoint_path.exists() else None,
                "config_sha1": config_sha1,
            }
        )

    # Build datasets list
    datasets = []
    if dataset_cfg:
        dataset_dict = OmegaConf.to_container(dataset_cfg, resolve=True) if hasattr(dataset_cfg, "__class__") else dataset_cfg
        dataset_path = dataset_dict.get("path") or dataset_dict.get("name")
        if dataset_path:
            dataset_file = Path(dataset_path)
            fingerprint = compute_file_fingerprint(dataset_file) if dataset_file.exists() else {}
            datasets.append(
                {
                    "name": dataset_dict.get("name", str(dataset_path)),
                    "path": str(dataset_path),
                    "split": dataset_dict.get("split", "val"),
                    "fingerprint": fingerprint,
                }
            )

    manifest = {
        "report_id": report_id,
        "created_at": datetime.now(UTC).isoformat(),
        "models": models,
        "datasets": datasets,
        "inference": {
            "persist_raw_scores": False,  # Default
            "metrics_computed": [],  # Will be populated by inference code
        },
    }

    return manifest


def write_manifest(
    report_dir: Path,
    report_id: str,
    report_name: str,
    run_ids: list[str],
    output_root: Path | str,
    dataset_cfg: DictConfig | dict[str, Any] | None = None,
) -> None:
    """Create and write manifest.yaml to report directory.

    Parameters
    ----------
    report_dir : Path
        Path to report directory
    report_id : str
        Report ID
    report_name : str
        Report name
    run_ids : list[str]
        List of run IDs used in this report
    output_root : Path | str
        Root output directory (for resolving run paths)
    dataset_cfg : DictConfig | dict[str, Any] | None
        Dataset config (optional) for extracting dataset path and metadata
    """
    manifest_data = create_manifest(
        report_id=report_id,
        run_ids=run_ids,
        output_root=output_root,
        dataset_cfg=dataset_cfg,
    )

    manifest_path = report_dir / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)
