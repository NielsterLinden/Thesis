from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from .io_utils import (
    build_filename,
    ensure_figures_dir,
    save_figure,
)
from .registry import get_enabled_families

logger = logging.getLogger(__name__)


def handle_event(
    cfg_logging: Mapping[str, Any],
    supported_families: Iterable[str],
    moment: str,
    payload: Mapping[str, Any],
    *,
    dry_run: bool | None = None,
) -> None:
    """Route a lifecycle event to enabled plot families.

    Parameters
    ----------
    cfg_logging: Mapping
        The composed Hydra `logging` config.
    supported_families: Iterable[str]
        Families the current loop supports (capability intersection).
    moment: str
        One of: "on_start", "on_epoch_end", "on_validation_end",
        "on_train_end", "on_exception", "on_test_end".
    payload: Mapping[str, Any]
        Standard payload with required/optional fields.
    dry_run: bool | None
        If True, print the actions without producing figures.
    """

    # Resolve dry_run from explicit argument or config
    policy_dry_run = bool(cfg_logging.get("dry_run", False))
    do_dry_run = policy_dry_run if dry_run is None else bool(dry_run)

    run_dir = str(payload.get("run_dir", ""))
    if not run_dir:
        # Without run_dir, we cannot save figures; bail early but do not crash
        logger.warning("[plots] skip (no run_dir) moment=%s", moment)
        return

    # Resolve which families should run at this moment according to policy
    families = get_enabled_families(cfg_logging, supported_families, moment)

    if do_dry_run:
        enabled_names = [f.name for f in families]
        logger.info("[plots] dry_run moment=%s families=%s", moment, enabled_names)
        return

    # Prepare filesystem destination
    figures_dir = ensure_figures_dir(run_dir, cfg_logging)

    # Dispatch to each family and persist figures
    for family in families:
        # Validate required payload keys
        missing = [k for k in family.required_keys if k not in payload or payload.get(k) is None]
        if missing:
            logger.warning("[plots] skip family=%s moment=%s missing=%s", family.name, moment, missing)
            continue

        try:
            figs = family.handle(moment, payload, cfg_logging)
        except Exception as e:  # friendly failure: never crash the run
            logger.error("[plots] family=%s moment=%s failed: %s", family.name, moment, e)
            continue

        if not figs:
            continue

        # Save each figure
        for idx, fig in enumerate(figs):
            fname = build_filename(
                cfg_logging=cfg_logging,
                family=family.name,
                moment=moment,
                payload=payload,
                index=idx,
            )
            save_figure(fig, figures_dir, fname, cfg_logging)
