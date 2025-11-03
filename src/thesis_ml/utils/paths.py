"""Path resolution utilities for runs, multiruns, and reports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def get_run_id(run_dir: Path | str) -> str:
    """Extract run_id from directory name.

    Parameters
    ----------
    run_dir : Path | str
        Path to run directory (e.g., `outputs/runs/run_20251024-152016_baseline_test/`)

    Returns
    -------
    str
        Run ID (e.g., `run_20251024-152016_baseline_test`)
    """
    run_path = Path(run_dir)
    return run_path.name


def get_multirun_id(multirun_dir: Path | str) -> str:
    """Extract multirun ID from directory name.

    Parameters
    ----------
    multirun_dir : Path | str
        Path to multirun directory (e.g., `outputs/multiruns/exp_20251024-180000_compare_tokenizers/`)

    Returns
    -------
    str
        Multirun ID (e.g., `exp_20251024-180000_compare_tokenizers`)
    """
    multirun_path = Path(multirun_dir)
    return multirun_path.name


def get_report_id(name: str | None = None) -> str:
    """Generate report ID with timestamp and optional name.

    Parameters
    ----------
    name : str | None
        Optional report name (default: "report")

    Returns
    -------
    str
        Report ID (e.g., `report_20251024-190000_compare_tokenizers`)
    """
    if name is None:
        name = "report"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"report_{timestamp}_{name}"


def resolve_run_dir(run_id: str, output_root: Path | str) -> Path:
    """Get path to run directory from run_id.

    Parameters
    ----------
    run_id : str
        Run ID (e.g., `run_20251024-152016_baseline_test`)
    output_root : Path | str
        Root output directory (e.g., `outputs/`)

    Returns
    -------
    Path
        Path to run directory
    """
    return Path(output_root) / "runs" / run_id


def resolve_report_dir(report_id: str, output_root: Path | str) -> Path:
    """Get path to report directory from report_id.

    Parameters
    ----------
    report_id : str
        Report ID (e.g., `report_20251024-190000_compare_tokenizers`)
    output_root : Path | str
        Root output directory (e.g., `outputs/`)

    Returns
    -------
    Path
        Path to report directory
    """
    return Path(output_root) / "reports" / report_id


def resolve_multirun_dir(exp_id: str, output_root: Path | str) -> Path:
    """Get path to multirun directory from exp_id.

    Parameters
    ----------
    exp_id : str
        Multirun ID (e.g., `exp_20251024-180000_compare_tokenizers`)
    output_root : Path | str
        Root output directory (e.g., `outputs/`)

    Returns
    -------
    Path
        Path to multirun directory
    """
    return Path(output_root) / "multiruns" / exp_id


def validate_run_dir(run_dir: Path | str) -> bool:
    """Validate that run_dir is under outputs/runs/.

    Parameters
    ----------
    run_dir : Path | str
        Path to run directory

    Returns
    -------
    bool
        True if run_dir is under outputs/runs/, False otherwise
    """
    run_path = Path(run_dir).resolve()
    # Check if path contains "runs" component
    parts = run_path.parts
    try:
        runs_idx = parts.index("runs")
        # Should be directly under runs/ (not nested)
        return runs_idx == len(parts) - 2
    except ValueError:
        return False
