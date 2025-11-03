from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from thesis_ml.utils.paths import get_report_id, resolve_report_dir


def resolve_report_output_dir(
    report_id: str | None,
    report_name: str | None,
    output_root: Path | str,
) -> Path:
    """Resolve report directory to outputs/reports/<report_id>/.

    Parameters
    ----------
    report_id : str | None
        Pre-computed report ID. If None, generates one from report_name.
    report_name : str | None
        Report name for ID generation if report_id is None.
    output_root : Path | str
        Root output directory (e.g., from env.output_root)

    Returns
    -------
    Path
        Path to report directory
    """
    if report_id is None:
        report_id = get_report_id(report_name)
    return resolve_report_dir(report_id, output_root)


def ensure_report_dirs(report_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Create report directory structure with training/ and inference/ subdirs.

    Parameters
    ----------
    report_dir : Path
        Path to report directory (e.g., outputs/reports/report_.../)

    Returns
    -------
    tuple[Path, Path, Path, Path]
        (training_dir, inference_dir, training_figs_dir, inference_figs_dir)
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    training_dir = report_dir / "training"
    inference_dir = report_dir / "inference"
    training_figs = training_dir / "figures"
    inference_figs = inference_dir / "figures"
    training_dir.mkdir(exist_ok=True)
    inference_dir.mkdir(exist_ok=True)
    training_figs.mkdir(exist_ok=True)
    inference_figs.mkdir(exist_ok=True)
    return training_dir, inference_dir, training_figs, inference_figs


def save_json(obj: dict[str, Any], path: Path) -> None:
    """Save dict as formatted JSON"""
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def get_fig_config(cfg: DictConfig) -> dict[str, Any]:
    """Extract figure save settings from config"""
    return {"fig_format": str(cfg.outputs.get("fig_format", "png")), "dpi": int(cfg.outputs.get("dpi", 150))}
