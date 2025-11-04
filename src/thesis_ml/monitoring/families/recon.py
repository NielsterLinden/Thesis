from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt


class ReconHandler:
    name = "recon"
    supported_moments = {"on_epoch_end", "on_validation_end", "on_train_end"}
    required_keys = {"run_dir"}
    heavy = True  # visuals can be heavy; curves are light but we keep conservative

    def __init__(self, *, mode: str = "curves") -> None:
        self.mode = mode

    def configure(self, cfg: Any) -> None:
        if isinstance(cfg, dict):
            self.mode = str(cfg.get("mode", self.mode))

    def handle(self, moment: str, payload: Mapping[str, Any], cfg_logging) -> list:
        if self.mode == "curves":
            return self._handle_curves(payload)
        if self.mode == "visuals":
            return self._handle_visuals(payload)
        return []

    def _handle_curves(self, payload: Mapping[str, Any]) -> list:
        # Plot scalar histories related to reconstruction if present
        rec_tok: Sequence[float] | None = payload.get("history_rec_tokens")
        rec_gl: Sequence[float] | None = payload.get("history_rec_globals")
        if not rec_tok and not rec_gl:
            return []
        fig, ax = plt.subplots()
        if rec_tok:
            ax.plot(rec_tok, label="rec_tokens")
        if rec_gl:
            ax.plot(rec_gl, label="rec_globals")
        ax.set_xlabel("epoch")
        ax.set_ylabel("recon loss")
        ax.legend()
        return [fig]

    def _handle_visuals(self, payload: Mapping[str, Any]) -> list:
        # Requires examples and model or an inference callable
        examples = payload.get("examples")
        infer = payload.get("inference")  # callable(model, examples)-> outputs
        model = payload.get("model")
        if examples is None or (infer is None and model is None):
            return []
        # Try to obtain outputs
        try:
            outputs = infer(model, examples) if callable(infer) else model(examples)  # type: ignore[operator]
        except Exception:
            return []

        # Minimal visualization: plot first sample predictions vs targets if available
        preds = getattr(outputs, "preds", None) if hasattr(outputs, "preds") else outputs
        targets = payload.get("targets")
        figs: list = []
        if preds is not None and targets is not None:
            fig, ax = plt.subplots()
            try:
                ax.plot(targets[0], label="target")
                ax.plot(preds[0], label="pred")
                ax.legend()
                ax.set_title("Reconstruction")
                figs.append(fig)
            except Exception:
                plt.close(fig)
        return figs
