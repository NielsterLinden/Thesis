from __future__ import annotations

import warnings

from omegaconf import DictConfig

warnings.warn(
    "thesis_ml.train.vq_ae_loop is deprecated; use thesis_ml.phase1.train.ae_loop",
    DeprecationWarning,
    stacklevel=2,
)


def train(cfg: DictConfig):
    from thesis_ml.phase1.train.ae_loop import train as _train

    return _train(cfg)
