from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt


class MetricsHandler:
    name = "metrics"
    supported_moments = {"on_epoch_end", "on_train_end"}
    required_keys = {"run_dir"}
    heavy = False

    def configure(self, cfg: Any) -> None:  # cfg may be bool
        # no-op for now
        return

    def handle(self, moment: str, payload: Mapping[str, Any], cfg_logging) -> list:
        # Prefer history plot if available
        hist = payload.get("history_metrics")  # dict[str, list[float]]
        if isinstance(hist, dict) and hist:
            fig, ax = plt.subplots()
            for k, series in sorted(hist.items()):
                if not series:
                    continue
                ax.plot(series, label=str(k))
            ax.set_xlabel("epoch")
            ax.set_ylabel("metric")
            ax.legend()
            return [fig]
        # Fallback: single-epoch bar for current metrics
        metrics = payload.get("metrics")
        if isinstance(metrics, dict) and metrics:
            keys = list(sorted(metrics.keys()))
            vals = [float(metrics[k]) for k in keys]
            fig, ax = plt.subplots()
            ax.bar(range(len(keys)), vals)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=45, ha="right")
            ax.set_ylabel("value")
            fig.tight_layout()
            return [fig]
        return []
