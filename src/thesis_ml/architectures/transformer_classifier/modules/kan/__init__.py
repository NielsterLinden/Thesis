"""Kolmogorov-Arnold Network (KAN) modules.

Provides a vendored :class:`KANLinear` B-spline layer and utilities for
collecting spline regularisation across a model.
"""

from thesis_ml.architectures.transformer_classifier.modules.kan.kan_linear import (
    KANLinear,
)
from thesis_ml.architectures.transformer_classifier.modules.kan.utils import (
    build_bias_mlp,
    collect_kan_spline_loss,
    collect_kan_stats,
    update_all_kan_grids,
)

__all__ = [
    "KANLinear",
    "build_bias_mlp",
    "collect_kan_spline_loss",
    "collect_kan_stats",
    "update_all_kan_grids",
]
