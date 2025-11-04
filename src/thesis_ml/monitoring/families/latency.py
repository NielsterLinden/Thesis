from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt


class LatencyHandler:
    name = "latency"
    supported_moments = {"on_epoch_end", "on_train_end"}
    required_keys = {"run_dir"}
    heavy = False

    def __init__(self, *, mode: str = "light") -> None:
        self.mode = mode

    def configure(self, cfg: Any) -> None:
        if isinstance(cfg, dict):
            self.mode = str(cfg.get("mode", self.mode))

    def handle(self, moment: str, payload: Mapping[str, Any], cfg_logging) -> list:
        times: Sequence[float] | None = payload.get("history_epoch_time_s")
        thr: Sequence[float] | None = payload.get("history_throughput")
        if not times and not thr:
            return []
        figs = []
        if times:
            fig, ax = plt.subplots()
            ax.plot(times, label="epoch_time_s")
            ax.set_xlabel("epoch")
            ax.set_ylabel("seconds")
            figs.append(fig)
        if thr and self.mode != "light":
            fig, ax = plt.subplots()
            ax.plot(thr, label="throughput")
            ax.set_xlabel("epoch")
            ax.set_ylabel("samples/s")
            figs.append(fig)
        return figs
