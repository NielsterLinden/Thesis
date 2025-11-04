from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def ensure_figures_dir(run_dir: str, cfg_logging: Mapping[str, Any]) -> Path:
    """Ensure the figures directory exists for a run.

    Parameters
    ----------
    run_dir : str
        Path to the run directory
    cfg_logging : Mapping[str, Any]
        Logging configuration

    Returns
    -------
    Path
        Path to the figures directory
    """
    root = Path(run_dir)
    # Use figures/ (or train_figures/) subdirectory
    sub = str(cfg_logging.get("figures_subdir", "figures"))
    out = root / sub
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_filename(
    *,
    cfg_logging: Mapping[str, Any],
    family: str,
    moment: str,
    payload: Mapping[str, Any],
    index: int | None = None,
) -> str:
    """Build a filename for a monitoring figure.

    Parameters
    ----------
    cfg_logging : Mapping[str, Any]
        Logging configuration
    family : str
        Plot family name (e.g., "losses", "metrics")
    moment : str
        Training moment (e.g., "on_epoch_end", "on_train_end")
    payload : Mapping[str, Any]
        Event payload with epoch/step information
    index : int | None
        Optional index for multiple figures in same family/moment

    Returns
    -------
    str
        Filename without extension
    """
    # prefer epoch, else step
    epoch = payload.get("epoch")
    step = payload.get("step")
    if epoch is not None:
        epoch_or_step = f"e{int(epoch):03d}"
    elif step is not None:
        epoch_or_step = f"s{int(step):06d}"
    else:
        epoch_or_step = "eNA"

    pat = str(cfg_logging.get("file_naming", "{family}-{moment}-{epoch_or_step}"))
    name = pat.format(family=family, moment=moment, epoch_or_step=epoch_or_step)
    if index is not None and index > 0:
        name = f"{name}-{index}"
    return name


def save_figure(fig, figures_dir: Path, base_name: str, cfg_logging: Mapping[str, Any]) -> Path:
    """Save a matplotlib figure to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    figures_dir : Path
        Directory to save to
    base_name : str
        Base filename (without extension)
    cfg_logging : Mapping[str, Any]
        Logging configuration (for format and DPI)

    Returns
    -------
    Path
        Path to saved figure
    """
    fmt = str(cfg_logging.get("fig_format", "png"))
    dpi = int(cfg_logging.get("dpi", 150))
    path = figures_dir / f"{base_name}.{fmt}"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    return path
