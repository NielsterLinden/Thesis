from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class ClassifierHead(nn.Module):
    """Classifier head with pooling and configurable classifier layer.

    Always outputs n_classes logits (unified approach for binary and
    multi-class).  The classifier layer is selected by ``head_type``:

    - ``"linear"`` (default) — single ``nn.Linear``.
    - ``"kan"`` — single :class:`KANLinear` (B-spline activation per edge).
    - ``"moe"`` — N expert linear classifiers with learned router
      (activated when ``moe_cfg["scope"] == "head"``).
    """

    def __init__(
        self,
        dim: int,
        n_classes: int,
        pooling: str = "cls",
        head_type: str = "linear",
        kan_cfg: dict[str, Any] | None = None,
        moe_cfg: dict[str, Any] | None = None,
    ):
        """Initialize classifier head.

        Parameters
        ----------
        dim : int
            Model dimension.
        n_classes : int
            Number of output classes.
        pooling : str
            Pooling strategy: ``"cls"``, ``"mean"``, or ``"max"``.
        head_type : str
            ``"linear"`` | ``"kan"`` | ``"moe"``.
        kan_cfg : dict | None
            Global KAN hyperparameters (``grid_size``, ``spline_order``,
            ``grid_range``).  Only used when ``head_type="kan"``.
        moe_cfg : dict | None
            When provided with ``scope="head"`` and ``enabled=True``,
            activates MoE classification.
        """
        super().__init__()
        self.pooling = pooling
        self.n_classes = n_classes
        self.head_type = head_type

        # Resolve effective head type: explicit head_type takes priority,
        # but legacy moe_cfg with scope="head" still works.
        use_moe = head_type == "moe" or (head_type == "linear" and moe_cfg is not None and moe_cfg.get("enabled", False) and moe_cfg.get("scope") == "head")
        use_kan = head_type == "kan" and not use_moe

        if use_moe:
            _moe = moe_cfg or {}
            self.num_experts = _moe.get("num_experts", 4)
            self.top_k = _moe.get("top_k", 1)
            noisy_gating = _moe.get("noisy_gating", False)

            self.router = nn.Linear(dim, self.num_experts, bias=False)
            self.experts = nn.ModuleList([nn.Linear(dim, n_classes) for _ in range(self.num_experts)])
            self.classifier = None

            if noisy_gating:
                self.noise_linear = nn.Linear(dim, self.num_experts, bias=False)
            else:
                self.noise_linear = None

            self.last_aux_loss: torch.Tensor | None = None
            self.last_routing_stats: dict | None = None
        elif use_kan:
            from thesis_ml.architectures.transformer_classifier.modules.kan import (
                KANLinear,
            )

            _kcfg = kan_cfg or {}
            self.classifier = KANLinear(
                dim,
                n_classes,
                grid_size=int(_kcfg.get("grid_size", 5)),
                spline_order=int(_kcfg.get("spline_order", 3)),
                grid_range=tuple(float(v) for v in _kcfg.get("grid_range", [-2.0, 2.0])),
            )
            self.num_experts = 0
            self.top_k = 0
            self.router = None
            self.experts = None
            self.noise_linear = None
            self.last_aux_loss = None
            self.last_routing_stats = None
        else:
            self.num_experts = 0
            self.top_k = 0
            self.router = None
            self.experts = None
            self.noise_linear = None
            self.last_aux_loss = None
            self.last_routing_stats = None
            self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, T, D] (or [B, T+1, D] if CLS token prepended)
        mask : torch.Tensor, optional
            Attention mask [B, T] or [B, T+1] (True=valid, False=padding)
            Required for mean/max pooling, optional for CLS pooling

        Returns
        -------
        torch.Tensor
            Logits [B, n_classes]
        """
        if self.pooling == "cls":
            # CLS pooling: take first token (CLS token should be at index 0)
            pooled = x[:, 0]  # [B, D]
        elif self.pooling == "mean":
            # Mean pooling: average over sequence with mask
            if mask is None:
                # If no mask, assume all tokens are valid
                pooled = x.mean(dim=1)  # [B, D]
            else:
                # Use mask directly (True=valid, False=padding)
                # x: [B, T, D], mask: [B, T] (True=valid)
                # Expand mask for broadcasting: [B, T, 1]
                mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]

                # Sum over sequence dimension (only valid tokens contribute)
                masked_sum = (x * mask_expanded).sum(dim=1)  # [B, D]

                # Count valid tokens per sample, guard all-pad rows
                denom = mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)  # [B, 1]

                # Average
                pooled = masked_sum / denom  # [B, D]
        elif self.pooling == "max":
            # Max pooling: masked max over sequence
            if mask is None:
                pooled = x.max(dim=1)[0]  # [B, D]
            else:
                mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]
                # Set padding to -inf so it doesn't affect max
                x_masked = x.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
                pooled = x_masked.max(dim=1)[0]  # [B, D]
                # If all tokens are padding, max gives -inf; replace with zeros
                pooled = torch.where(
                    torch.isfinite(pooled),
                    pooled,
                    torch.zeros_like(pooled),
                )
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}. Choose 'cls', 'mean', or 'max'.")

        # Classify
        logits = self._moe_classify(pooled) if self.router is not None else self.classifier(pooled)
        return logits

    # ------------------------------------------------------------------
    # MoE classification (head_only scope)
    # ------------------------------------------------------------------

    def _moe_classify(self, pooled: torch.Tensor) -> torch.Tensor:
        """Route pooled representation through expert classifiers."""
        router_logits = self.router(pooled)  # [B, E]

        if self.training and self.noise_linear is not None:
            noise_std = F.softplus(self.noise_linear(pooled))
            router_logits = router_logits + noise_std * torch.randn_like(router_logits)

        router_probs = F.softmax(router_logits, dim=-1)  # [B, E]

        top_k_values, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        gates = top_k_values / (top_k_values.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute all expert outputs and gather selected
        expert_outputs = torch.stack([expert(pooled) for expert in self.experts], dim=1)  # [B, E, n_classes]

        # Weighted combination of top-k expert outputs
        # top_k_indices: [B, k], gates: [B, k]
        selected = torch.gather(
            expert_outputs,
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, self.n_classes),
        )  # [B, k, n_classes]
        logits = (gates.unsqueeze(-1) * selected).sum(dim=1)  # [B, n_classes]

        # Aux loss (load-balancing over events)
        E = self.num_experts
        one_hot = F.one_hot(top_k_indices[:, 0], num_classes=E).float()  # [B, E]
        f = one_hot.mean(dim=0)  # [E]
        P = router_probs.mean(dim=0)  # [E]
        self.last_aux_loss = E * (f * P).sum()

        expert_counts = one_hot.sum(dim=0)
        self.last_routing_stats = {
            "expert_counts": expert_counts.detach(),
            "expert_mean_prob": P.detach(),
            "expert_utilization": (expert_counts > 0).float().mean().item(),
        }

        return logits


def build_classifier_head(
    cfg: DictConfig,
    dim: int,
    n_classes: int,
    moe_cfg: dict[str, Any] | None = None,
    kan_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    """Build classifier head (pooling + linear/KAN/MoE classifier).

    Parameters
    ----------
    cfg : DictConfig
        Configuration with ``classifier.model.head.*`` keys.
    dim : int
        Model dimension.
    n_classes : int
        Number of output classes (2 for binary, >2 for multi-class).
    moe_cfg : dict | None
        MoE configuration.  Only used when ``scope="head"``.
    kan_cfg : dict | None
        Global KAN hyperparameters.  Only used when ``head.type="kan"``.

    Returns
    -------
    nn.Module
        Classifier head that maps ``[B, T, D] -> [B, n_classes]``.
    """
    model_cfg = cfg.classifier.model
    head_cfg = model_cfg.get("head", {})

    # New config path: head.pooling; legacy fallback: top-level pooling
    pooling = head_cfg.get("pooling", model_cfg.get("pooling", "cls"))
    head_type = str(head_cfg.get("type", "linear"))

    return ClassifierHead(
        dim=dim,
        n_classes=n_classes,
        pooling=pooling,
        head_type=head_type,
        kan_cfg=kan_cfg,
        moe_cfg=moe_cfg,
    )
