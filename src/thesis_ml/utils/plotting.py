from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curve(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    *,
    show: bool = False,
    save: bool = False,
    out_path: Path | None = None,
    close: bool = True,
):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="train")
    if val_losses is not None:
        ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()

    if save and out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    if show:
        plt.show(block=False)
    if close:
        plt.close(fig)
    return fig
