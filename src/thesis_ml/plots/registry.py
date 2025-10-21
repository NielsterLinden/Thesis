from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .families.adversarial import AdversarialHandler
from .families.codebook import CodebookHandler
from .families.diffusion import DiffusionHandler
from .families.latency import LatencyHandler
from .families.losses import LossesHandler
from .families.metrics import MetricsHandler
from .families.recon import ReconHandler


@dataclass
class FamilyRecord:
    name: str
    handler: Any


def build_registry(cfg_logging: Mapping[str, Any]) -> dict[str, FamilyRecord]:
    """Create the registry with handlers configured from policy.

    Keep it explicit and small.
    """
    reg: dict[str, FamilyRecord] = {}

    # losses
    losses = LossesHandler()
    losses.configure(cfg_logging.get("families", {}).get("losses", True))
    reg[losses.name] = FamilyRecord(losses.name, losses)

    # metrics
    metrics_cfg = cfg_logging.get("families", {}).get("metrics", True)
    metrics = MetricsHandler()
    metrics.configure(metrics_cfg)
    reg[metrics.name] = FamilyRecord(metrics.name, metrics)

    # recon
    recon_cfg = cfg_logging.get("families", {}).get("recon", {"enabled": False, "mode": "curves"})
    recon = ReconHandler(mode=str(recon_cfg.get("mode", "curves")))
    recon.configure(recon_cfg)
    reg[recon.name] = FamilyRecord(recon.name, recon)

    # codebook
    codebook_cfg = cfg_logging.get("families", {}).get("codebook", True)
    codebook = CodebookHandler()
    codebook.configure(codebook_cfg)
    reg[codebook.name] = FamilyRecord(codebook.name, codebook)

    # latency
    latency_cfg = cfg_logging.get("families", {}).get("latency", {"enabled": True, "mode": "light"})
    latency = LatencyHandler(mode=str(latency_cfg.get("mode", "light")))
    latency.configure(latency_cfg)
    reg[latency.name] = FamilyRecord(latency.name, latency)

    # adversarial (disabled by default unless configured)
    adv_cfg = cfg_logging.get("families", {}).get("adversarial", False)
    adv = AdversarialHandler()
    adv.configure(adv_cfg)
    reg[adv.name] = FamilyRecord(adv.name, adv)

    # diffusion (disabled by default unless configured)
    dif_cfg = cfg_logging.get("families", {}).get("diffusion", False)
    dif = DiffusionHandler()
    dif.configure(dif_cfg)
    reg[dif.name] = FamilyRecord(dif.name, dif)

    return reg


def _moment_allowed(cfg_logging: Mapping[str, Any], moment: str) -> bool:
    moments = cfg_logging.get("moments", {})
    # Support both explicit names and quick/full toggles
    if moment in moments:
        return bool(moments.get(moment))
    if moment == "on_epoch_end":
        return bool(moments.get("on_epoch_end_quick", True))
    if moment == "on_train_end":
        return bool(moments.get("on_train_end_full", True))
    if moment == "on_validation_end":
        return bool(moments.get("on_validation_end", False))
    if moment == "on_test_end":
        return bool(moments.get("on_test_end", True))
    return True


def get_enabled_families(
    cfg_logging: Mapping[str, Any],
    supported_families: Iterable[str],
    moment: str,
) -> list[Any]:
    """Return configured handlers that are both enabled and supported.

    Applies quick/full policy by allowing heavy families only at train end by default.
    """
    if not _moment_allowed(cfg_logging, moment):
        return []

    reg = build_registry(cfg_logging)

    requested: set[str] = set()
    families_cfg = cfg_logging.get("families", {})
    # Boolean families: True/False
    for name, subcfg in families_cfg.items():
        if isinstance(subcfg, bool) and subcfg or isinstance(subcfg, dict) and bool(subcfg.get("enabled", True)):
            requested.add(name)

    # Tags support (optional)
    tags_cfg = cfg_logging.get("tags", {})
    enable_tags = cfg_logging.get("enable_tags", []) or []
    for tag in enable_tags:
        members = tags_cfg.get(tag, [])
        requested.update(members)

    # Intersect with loop capability
    requested &= set(supported_families)

    # Apply quick/full: do not run heavy families during quick moments
    quick = moment == "on_epoch_end"

    handlers: list[Any] = []
    for name in sorted(requested):
        rec = reg.get(name)
        if rec is None:
            continue
        handler = rec.handler
        if moment not in handler.supported_moments:
            continue
        if quick and getattr(handler, "heavy", False):
            # Skip heavy families during quick moments
            continue
        handlers.append(handler)

    return handlers
