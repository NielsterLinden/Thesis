"""Efficient B-spline Kolmogorov-Arnold Network linear layer.

Vendored from `Blealtan/efficient-kan <https://github.com/Blealtan/efficient-kan>`_
(MIT License) with minor adaptations:

- Explicit ``grid_range`` typing.
- ``regularization_loss`` returns a scalar tensor (not a tuple).
- ``forward`` preserves arbitrary leading batch dimensions.

Original license
----------------
MIT License — Copyright (c) 2024 Blealtan
Full text: https://github.com/Blealtan/efficient-kan/blob/master/LICENSE
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """Single KAN layer: learns a univariate B-spline activation per edge.

    Each edge computes ``base_activation(x) @ base_weight + B(x) @ spline_weight``
    where ``B(x)`` is a B-spline basis expansion evaluated on a fixed grid.

    Parameters
    ----------
    in_features, out_features : int
        Input / output dimensionality.
    grid_size : int
        Number of B-spline grid intervals.
    spline_order : int
        B-spline polynomial degree (``3`` = cubic).
    scale_noise : float
        Noise scale for spline weight initialisation.
    scale_base : float
        Kaiming init scale for ``base_weight``.
    scale_spline : float
        Scale for spline weight initialisation.
    enable_standalone_scale_spline : bool
        If ``True``, add a learnable per-edge ``spline_scaler`` parameter.
    base_activation : type
        Activation applied in the base (residual) path.
    grid_eps : float
        Blend factor between uniform and adaptive grid in ``update_grid``.
    grid_range : tuple[float, float]
        ``(lo, hi)`` — initial range of the B-spline grid.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features),
        )
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order),
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.empty(out_features, in_features),
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self._curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    # ------------------------------------------------------------------
    # B-spline basis
    # ------------------------------------------------------------------

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline bases.

        Parameters
        ----------
        x : Tensor
            Shape ``(N, in_features)``.

        Returns
        -------
        Tensor
            Shape ``(N, in_features, grid_size + spline_order)``.
        """
        grid: torch.Tensor = self.grid  # (in_features, G)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fit spline coefficients via least-squares."""
        A = self._b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply KAN layer, preserving leading batch dimensions.

        Parameters
        ----------
        x : Tensor
            Any shape ``(..., in_features)``.

        Returns
        -------
        Tensor
            Shape ``(..., out_features)``.
        """
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self._b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_features)

    # ------------------------------------------------------------------
    # Grid adaptation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Adapt the B-spline grid to the data distribution of *x*.

        Parameters
        ----------
        x : Tensor
            ``(N, in_features)`` — a representative data sample.
        margin : float
            Padding beyond the data range.
        """
        if x.dim() != 2:
            x = x.reshape(-1, self.in_features)
        batch = x.size(0)

        splines = self._b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self._curve2coeff(x, unreduced_spline_output))

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        """L1 + entropy regularisation on spline weights.

        Returns a scalar tensor suitable for direct addition to the main loss.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        reg_activation = l1_fake.sum()
        p = l1_fake / (reg_activation + 1e-12)
        reg_entropy = -torch.sum(p * p.log().clamp(min=-100))
        return regularize_activation * reg_activation + regularize_entropy * reg_entropy
