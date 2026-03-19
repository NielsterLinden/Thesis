"""KAN utilities: spline regularisation collection and bias-MLP builder.

The regularisation helpers mirror the *plumbing* of
``ffn.moe.collect_moe_aux_loss`` (walk modules → sum) but carry **separate
semantics**: KAN regularisation is a smoothness prior on learned activation
functions, not a routing-behaviour signal.  Naming and logging reflect this.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from thesis_ml.architectures.transformer_classifier.modules.kan.kan_linear import (
    KANLinear,
)

__all__ = [
    "collect_kan_spline_loss",
    "collect_kan_stats",
    "build_bias_mlp",
    "update_all_kan_grids",
]


# ------------------------------------------------------------------
# Spline regularisation collection
# ------------------------------------------------------------------


def collect_kan_spline_loss(model: nn.Module) -> torch.Tensor:
    """Sum ``regularization_loss()`` from every :class:`KANLinear` in *model*.

    Returns ``tensor(0.0)`` when no KAN layers are present.
    """
    total = torch.tensor(0.0)
    for m in model.modules():
        if isinstance(m, KANLinear):
            total = total + m.regularization_loss().to(total.device)
    return total


def collect_kan_stats(model: nn.Module) -> dict:
    """Aggregate lightweight statistics from all KAN layers.

    Returns
    -------
    dict
        ``num_kan_layers``, ``total_spline_params``, ``mean_spline_magnitude``.
    """
    num_layers = 0
    total_params = 0
    total_mag = 0.0
    for m in model.modules():
        if isinstance(m, KANLinear):
            num_layers += 1
            n = m.spline_weight.numel()
            total_params += n
            total_mag += m.spline_weight.abs().sum().item()

    return {
        "num_kan_layers": num_layers,
        "total_spline_params": total_params,
        "mean_spline_magnitude": total_mag / max(total_params, 1),
    }


# ------------------------------------------------------------------
# Grid adaptation
# ------------------------------------------------------------------


def update_all_kan_grids(model: nn.Module, sample: torch.Tensor, margin: float = 0.01) -> None:
    """Call ``update_grid`` on every :class:`KANLinear` in *model*.

    Parameters
    ----------
    sample : Tensor
        A representative batch ``(N, D)`` (or ``(N, T, D)``; reshaped
        internally by each KANLinear).
    """
    flat = sample.reshape(-1, sample.size(-1)) if sample.dim() > 2 else sample
    for m in model.modules():
        if isinstance(m, KANLinear) and flat.size(-1) == m.in_features:
            m.update_grid(flat, margin=margin)


# ------------------------------------------------------------------
# Bias-MLP builder (used by Subproject C: KAN Bias MLPs)
# ------------------------------------------------------------------


def build_bias_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    mlp_type: str = "standard",
    kan_cfg: dict | None = None,
) -> nn.Module:
    """Build a small MLP or KAN equivalent for physics-bias modules.

    Parameters
    ----------
    in_dim, hidden_dim, out_dim : int
        Layer sizes.
    mlp_type : str
        ``"standard"`` — ``Linear → GELU → Linear``.
        ``"kan"`` — ``KANLinear → KANLinear``.
    kan_cfg : dict | None
        Global KAN hyperparameters (``grid_size``, ``spline_order``,
        ``grid_range``).  Only used when ``mlp_type="kan"``.
    """
    if mlp_type == "kan":
        gs = int((kan_cfg or {}).get("grid_size", 5))
        so = int((kan_cfg or {}).get("spline_order", 3))
        gr_raw = (kan_cfg or {}).get("grid_range", [-2.0, 2.0])
        gr = (float(gr_raw[0]), float(gr_raw[1]))
        return nn.Sequential(
            KANLinear(in_dim, hidden_dim, grid_size=gs, spline_order=so, grid_range=gr),
            KANLinear(hidden_dim, out_dim, grid_size=gs, spline_order=so, grid_range=gr),
        )

    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )
