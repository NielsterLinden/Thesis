from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from thesis_ml.facts.readers import load_runs
from thesis_ml.utils.paths import get_report_id, get_run_id, resolve_report_dir

logger = logging.getLogger(__name__)


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


def setup_report_environment(
    cfg: DictConfig,
) -> tuple[Path, Path, Path, Path, pd.DataFrame, dict[str, pd.DataFrame], list[str], Path, str, str]:
    """Common setup for all reports: load runs, create directories, validate.

    Parameters
    ----------
    cfg : DictConfig
        Report configuration

    Returns
    -------
    tuple[Path, Path, Path, Path, pd.DataFrame, dict, list, Path, str, str]
        (training_dir, inference_dir, training_figs_dir, inference_figs_dir,
         runs_df, per_epoch, order, report_dir, report_id, report_name)
    """
    # Load runs
    runs_df, per_epoch, order = load_runs(
        sweep_dir=str(cfg.inputs.sweep_dir) if cfg.inputs.sweep_dir else None,
        run_dirs=list(cfg.inputs.run_dirs) if cfg.inputs.run_dirs else None,
        require_complete=True,
    )

    if runs_df.empty:
        raise RuntimeError("No valid runs found for reporting.")

    # Resolve report directory
    report_id = cfg.get("report_id")
    report_name = cfg.get("report_name", "report")
    output_root = Path(cfg.env.output_root)
    report_dir = resolve_report_output_dir(report_id, report_name, output_root)
    training_dir, inference_dir, training_figs_dir, inference_figs_dir = ensure_report_dirs(report_dir)

    return training_dir, inference_dir, training_figs_dir, inference_figs_dir, runs_df, per_epoch, order, report_dir, report_id, report_name


def finalize_report(
    cfg: DictConfig,
    report_dir: Path,
    runs_df: pd.DataFrame,
    output_root: Path,
    report_id: str | None = None,
    report_name: str | None = None,
) -> None:
    """Common finalization: manifest, backlinks, logging.

    Parameters
    ----------
    cfg : DictConfig
        Report configuration
    report_dir : Path
        Path to report directory
    runs_df : pd.DataFrame
        DataFrame with run information (must have 'run_dir' column)
    output_root : Path
        Root output directory
    report_id : str | None
        Report ID (if None, uses cfg.get("report_id"))
    report_name : str | None
        Report name (if None, uses cfg.get("report_name"))
    """
    from ..utils.backlinks import append_report_pointer
    from ..utils.manifest import write_manifest

    report_id = report_id or cfg.get("report_id")
    report_name = report_name or cfg.get("report_name", "report")

    # Write manifest
    write_manifest(
        report_dir=report_dir,
        report_id=report_id or get_report_id(report_name),
        report_name=report_name,
        run_ids=[get_run_id(Path(rd)) for rd in runs_df["run_dir"].dropna().unique()],
        output_root=output_root,
        dataset_cfg=cfg.get("data") if hasattr(cfg, "data") else None,
    )

    # Append report pointers to each run
    for rd in runs_df["run_dir"].dropna().unique():
        run_dir_path = Path(str(rd))
        append_report_pointer(run_dir_path, report_id or get_report_id(report_name))

    logger.info("Report written to %s", report_dir)
