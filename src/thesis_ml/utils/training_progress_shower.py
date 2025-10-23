from __future__ import annotations

import sys
from collections.abc import Mapping


class TrainingProgressShower:
    """Simple reusable per-epoch progress bar with ETA and metrics.

    Usage:
        p = TrainingProgressShower(total_epochs=100)
        for ep in range(100):
            # ... run epoch ...
            p.update(ep, epoch_time_s, train_loss=..., val_loss=..., extras={"acc": 0.9})
    """

    def __init__(self, total_epochs: int, bar_width: int = 30, stream=None):
        self.total = int(total_epochs)
        self.width = int(bar_width)
        self.times: list[float] = []
        self.stream = stream or sys.stdout

    def _fmt_time(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def update(
        self,
        epoch_idx: int,
        epoch_time_s: float,
        *,
        train_loss: float | None = None,
        val_loss: float | None = None,
        extras: Mapping[str, float] | None = None,
    ) -> str:
        self.times.append(float(epoch_time_s))
        done = epoch_idx + 1
        frac = (done / self.total) if self.total > 0 else 1.0
        if frac < 0.0:
            frac = 0.0
        if frac > 1.0:
            frac = 1.0
        filled = int(round(self.width * frac))
        bar = "[" + "#" * filled + "-" * (self.width - filled) + "]"

        avg = (sum(self.times) / len(self.times)) if self.times else 0.0
        remaining_epochs = self.total - done
        if remaining_epochs < 0:
            remaining_epochs = 0
        eta_s = avg * remaining_epochs

        parts = [bar, f"{done}/{self.total}", f"ETA {self._fmt_time(eta_s)}", f"time {epoch_time_s:.2f}s"]
        if train_loss is not None:
            parts.append(f"train {train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val {val_loss:.4f}")
        if extras:
            for k, v in extras.items():
                try:
                    parts.append(f"{k} {float(v):.4f}")
                except Exception:
                    parts.append(f"{k} {v}")

        line = " | ".join(parts)
        end = "\n" if done >= self.total else "\r"
        self.stream.write(line + end)
        self.stream.flush()
        return line
