from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt


class LossesHandler:
    name = "losses"
    supported_moments = {"on_epoch_end", "on_train_end"}
    required_keys = {"run_dir"}
    heavy = False

    def configure(self, cfg: Any) -> None:  # cfg may be bool
        # no-op for now
        return

    def handle(self, moment: str, payload: Mapping[str, Any], cfg_logging) -> list:
        # Expect scalar series in payload history if present, else do nothing
        train_hist: Sequence[float] | None = payload.get("history_train_loss")
        val_hist: Sequence[float] | None = payload.get("history_val_loss")
        if not train_hist and not val_hist:
            return []
        fig, ax = plt.subplots()
        if train_hist:
            ax.plot(train_hist, label="train")
        if val_hist:
            ax.plot(val_hist, label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        return [fig]


def make_loss_figure(train_losses: Sequence[float], val_losses: Sequence[float] | None = None):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="train")
    if val_losses is not None:
        ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    return fig
