from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt


class CodebookHandler:
    name = "codebook"
    supported_moments = {"on_epoch_end", "on_train_end"}
    required_keys = {"run_dir"}
    heavy = False

    def configure(self, cfg: Any) -> None:
        # no-op for now
        return

    def handle(self, moment: str, payload: Mapping[str, Any], cfg_logging) -> list:
        perplex: Sequence[float] | None = payload.get("history_perplex")
        codebook: Sequence[float] | None = payload.get("history_codebook")
        if not perplex and not codebook:
            return []
        fig, ax = plt.subplots()
        if perplex:
            ax.plot(perplex, label="perplex")
        if codebook:
            ax.plot(codebook, label="codebook")
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
        ax.legend()
        return [fig]
