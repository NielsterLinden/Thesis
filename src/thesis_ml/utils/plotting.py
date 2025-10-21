from __future__ import annotations

import contextlib
import warnings
from collections.abc import Sequence
from pathlib import Path

from thesis_ml.plots import handle_event


def plot_loss_curve(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    *,
    show: bool = False,
    save: bool = False,
    out_path: Path | None = None,
    close: bool = True,
):
    """Deprecated shim: route to plots orchestrator.

    This function remains for one release to avoid breaking callers.
    It forwards to the `losses` family via the orchestrator. Visual
    defaults are governed by the `logging` Hydra config.
    """
    warnings.warn(
        "thesis_ml.utils.plotting.plot_loss_curve is deprecated; use the plots orchestrator via logging policy",
        DeprecationWarning,
        stacklevel=2,
    )
    # Attempt a sensible fallback if caller passes an output path
    run_dir = str(Path(out_path).parent.parent) if out_path else ""
    payload = {
        "run_dir": run_dir,
        "history_train_loss": list(train_losses),
        "history_val_loss": list(val_losses) if val_losses is not None else None,
    }
    # Minimal logging policy fallback
    cfg_logging = {
        "make_plots": bool(save or out_path),
        "show_plots": bool(show),
        "figures_subdir": str(Path(out_path).parent.name) if out_path else "figures",
        "fig_format": (Path(out_path).suffix.lstrip(".") if out_path else "png"),
        "dpi": 150,
        "file_naming": "{family}-{moment}-{epoch_or_step}",
        "families": {"losses": True},
        "moments": {"on_train_end_full": True},
        "destinations": "file",
    }
    with contextlib.suppress(Exception):
        handle_event(cfg_logging, {"losses"}, "on_train_end", payload)
    return None
