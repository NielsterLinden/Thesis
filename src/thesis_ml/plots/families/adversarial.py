from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class AdversarialHandler:
    name = "adversarial"
    supported_moments = {"on_epoch_end", "on_train_end"}
    required_keys = {"run_dir"}
    heavy = False

    def configure(self, cfg: Any) -> None:  # cfg may be bool or dict
        return

    def handle(self, moment: str, payload: Mapping[str, Any], cfg_logging) -> list:
        # Stub: no figures yet
        return []
