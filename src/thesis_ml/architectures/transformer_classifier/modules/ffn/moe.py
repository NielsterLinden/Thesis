"""Mixture-of-Experts feed-forward network for transformer encoder blocks.

Implements sparse MoE as a drop-in replacement for :class:`StandardFFN`.
Each expert is a full :class:`StandardFFN` instance; a learned router
selects the top-k experts per token (or per event) and combines their
outputs with softmax-normalised gates.

Router stability (v1):
    - Load-balancing auxiliary loss: YES (Switch Transformer formulation).
    - Capacity limits: NO (omitted — short HEP sequences make extreme
      imbalance unlikely; the load-balancing loss provides soft balancing).
    - Noisy gating: optional, behind ``noisy_gating`` flag, default off.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis_ml.architectures.transformer_classifier.modules.ffn.standard import StandardFFN


class MoEFFN(nn.Module):
    """Sparse Mixture-of-Experts FFN.

    Parameters
    ----------
    dim : int
        Model dimension (input / output).
    mlp_dim : int
        Hidden dimension for each expert FFN.
    num_experts : int
        Number of expert FFNs.
    top_k : int
        Experts selected per routing decision (1, 2, or 3).
    routing_level : str
        ``"token"`` — each token routed independently.
        ``"event"`` — routing computed from CLS token (or masked mean),
        applied identically to all valid tokens.
    dropout : float
        Dropout rate for each expert.
    norm_policy : str
        ``"pre"`` / ``"post"`` / ``"normformer"``.
    block_norm_type : str
        ``"layernorm"`` / ``"rmsnorm"``.
    use_cls_token : bool
        If True and ``routing_level="event"``, use ``x[:, 0]`` as the
        routing input.  Otherwise fall back to masked mean pooling.
    noisy_gating : bool
        If True, add learnable Gaussian noise to router logits during
        training (Shazeer et al., 2017).
    """

    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        routing_level: str = "token",
        dropout: float = 0.1,
        norm_policy: str = "pre",
        block_norm_type: str = "layernorm",
        use_cls_token: bool = True,
        noisy_gating: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_level = routing_level
        self.use_cls_token = use_cls_token
        self.noisy_gating = noisy_gating

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([StandardFFN(dim, mlp_dim, dropout, norm_policy, block_norm_type) for _ in range(num_experts)])

        if noisy_gating:
            self.noise_linear = nn.Linear(dim, num_experts, bias=False)
        else:
            self.noise_linear = None

        # Populated each forward pass for the training loop to collect.
        self.last_aux_loss: torch.Tensor | None = None
        self.last_routing_stats: dict | None = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply MoE FFN.

        Parameters
        ----------
        x : torch.Tensor
            ``[B, T, D]``.
        mask : torch.Tensor | None
            ``[B, T]`` with ``True = valid``, ``False = padding``.

        Returns
        -------
        torch.Tensor
            ``[B, T, D]``.
        """
        B, T, D = x.shape

        if self.routing_level == "event":
            return self._forward_event(x, mask)
        return self._forward_token(x, mask)

    # ------------------------------------------------------------------
    # Token-level routing
    # ------------------------------------------------------------------

    def _forward_token(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        B, T, D = x.shape

        router_logits = self.router(x)  # [B, T, E]

        if self.training and self.noise_linear is not None:
            noise_std = F.softplus(self.noise_linear(x))
            router_logits = router_logits + noise_std * torch.randn_like(router_logits)

        router_probs = F.softmax(router_logits, dim=-1)  # [B, T, E]

        # Zero out padded positions (softmax of -inf is NaN, so we zero after)
        if mask is not None:
            router_probs = router_probs * mask.unsqueeze(-1).float()

        # Top-k selection
        top_k_values, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)  # [B, T, k]
        # Re-normalise gates over selected experts
        gates = top_k_values / (top_k_values.sum(dim=-1, keepdim=True) + 1e-9)  # [B, T, k]

        # Compute expert outputs and combine
        # Flatten to [B*T, D] for per-token dispatch
        x_flat = x.reshape(B * T, D)
        output_flat = torch.zeros_like(x_flat)

        for k_idx in range(self.top_k):
            expert_idx = top_k_indices[:, :, k_idx]  # [B, T]
            gate_weight = gates[:, :, k_idx]  # [B, T]

            for e in range(self.num_experts):
                token_mask = expert_idx == e  # [B, T]
                if not token_mask.any():
                    continue
                flat_mask = token_mask.reshape(B * T)
                expert_input = x_flat[flat_mask]  # [N_e, D]
                expert_out = self.experts[e](expert_input)  # [N_e, D]
                weighted = expert_out * gate_weight.reshape(B * T)[flat_mask].unsqueeze(-1)
                output_flat[flat_mask] = output_flat[flat_mask] + weighted

        output = output_flat.reshape(B, T, D)

        # Zero out padded positions
        if mask is not None:
            output = output * mask.unsqueeze(-1).float()

        # Aux loss and routing stats
        self._compute_aux_loss_and_stats(router_probs, top_k_indices[:, :, 0], mask)

        return output

    # ------------------------------------------------------------------
    # Event-level routing
    # ------------------------------------------------------------------

    def _forward_event(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        B, T, D = x.shape

        # Routing input: CLS token or masked mean
        if self.use_cls_token:
            router_input = x[:, 0]  # [B, D]
        elif mask is not None:
            mask_f = mask.unsqueeze(-1).float()  # [B, T, 1]
            router_input = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            router_input = x.mean(dim=1)  # [B, D]

        router_logits = self.router(router_input)  # [B, E]

        if self.training and self.noise_linear is not None:
            noise_std = F.softplus(self.noise_linear(router_input))
            router_logits = router_logits + noise_std * torch.randn_like(router_logits)

        router_probs = F.softmax(router_logits, dim=-1)  # [B, E]

        # Top-k selection
        top_k_values, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)  # [B, k]
        gates = top_k_values / (top_k_values.sum(dim=-1, keepdim=True) + 1e-9)  # [B, k]

        # Apply same expert selection to all tokens in the event
        output = torch.zeros_like(x)  # [B, T, D]

        for k_idx in range(self.top_k):
            expert_idx = top_k_indices[:, k_idx]  # [B]
            gate_weight = gates[:, k_idx]  # [B]

            for e in range(self.num_experts):
                event_mask = expert_idx == e  # [B]
                if not event_mask.any():
                    continue
                expert_input = x[event_mask]  # [N_e, T, D]
                expert_out = self.experts[e](expert_input.reshape(-1, D)).reshape(expert_input.shape)
                weighted = expert_out * gate_weight[event_mask].unsqueeze(-1).unsqueeze(-1)
                output[event_mask] = output[event_mask] + weighted

        # Zero out padded positions
        if mask is not None:
            output = output * mask.unsqueeze(-1).float()

        # Aux loss — for event-level routing, expand to token level for
        # consistency with the load-balancing formula
        # router_probs: [B, E] -> [B, T, E] (broadcast)
        router_probs_expanded = router_probs.unsqueeze(1).expand(B, T, -1)
        top1_expanded = top_k_indices[:, 0].unsqueeze(1).expand(B, T)  # [B, T]
        self._compute_aux_loss_and_stats(router_probs_expanded, top1_expanded, mask)

        return output

    # ------------------------------------------------------------------
    # Load-balancing auxiliary loss (Switch Transformer)
    # ------------------------------------------------------------------

    def _compute_aux_loss_and_stats(
        self,
        router_probs: torch.Tensor,
        top1_indices: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        """Compute load-balancing loss and routing statistics.

        Parameters
        ----------
        router_probs : Tensor
            ``[B, T, E]`` softmax probabilities.
        top1_indices : Tensor
            ``[B, T]`` index of the top-1 expert per token.
        mask : Tensor | None
            ``[B, T]`` True=valid.
        """
        B, T, E = router_probs.shape

        valid = mask.float() if mask is not None else torch.ones(B, T, device=router_probs.device)

        num_valid = valid.sum().clamp(min=1.0)

        # f_i: fraction of valid tokens assigned to expert i (hard assignment)
        one_hot = F.one_hot(top1_indices, num_classes=E).float()  # [B, T, E]
        one_hot = one_hot * valid.unsqueeze(-1)
        f = one_hot.sum(dim=(0, 1)) / num_valid  # [E]

        # P_i: mean router probability for expert i over valid tokens
        weighted_probs = router_probs * valid.unsqueeze(-1)
        P = weighted_probs.sum(dim=(0, 1)) / num_valid  # [E]

        # L_aux = E * sum(f_i * P_i)
        self.last_aux_loss = E * (f * P).sum()

        # Routing stats
        expert_counts = one_hot.sum(dim=(0, 1))  # [E]
        utilization = (expert_counts > 0).float().mean()

        self.last_routing_stats = {
            "expert_counts": expert_counts.detach(),
            "expert_mean_prob": P.detach(),
            "expert_utilization": utilization.item(),
        }


# ---------------------------------------------------------------------------
# Utilities for the training loop
# ---------------------------------------------------------------------------


def _has_moe_aux_loss(module: nn.Module) -> bool:
    """Check whether *module* exposes a MoE auxiliary loss."""
    return hasattr(module, "last_aux_loss") and hasattr(module, "last_routing_stats")


def collect_moe_aux_loss(model: nn.Module) -> torch.Tensor:
    """Sum ``last_aux_loss`` from all MoE modules in *model*.

    Works with both :class:`MoEFFN` (encoder blocks) and
    :class:`ClassifierHead` (head MoE).  Returns ``0.0`` when no MoE
    layers exist or none have been forward-passed yet.
    """
    total = torch.tensor(0.0)
    for m in model.modules():
        if _has_moe_aux_loss(m) and m.last_aux_loss is not None:
            total = total + m.last_aux_loss.to(total.device)
    return total


def collect_moe_routing_stats(model: nn.Module) -> dict:
    """Aggregate routing stats from all MoE layers.

    Returns a dict with:
    - ``layer_stats``: list of per-layer stat dicts
    - ``mean_utilization``: average expert utilisation across layers
    - ``total_aux_loss``: sum of aux losses
    """
    layer_stats = []
    for name, m in model.named_modules():
        if _has_moe_aux_loss(m) and m.last_routing_stats is not None:
            layer_stats.append({"name": name, **m.last_routing_stats})

    if not layer_stats:
        return {"layer_stats": [], "mean_utilization": 1.0, "total_aux_loss": 0.0}

    mean_util = sum(s["expert_utilization"] for s in layer_stats) / len(layer_stats)
    total_aux = collect_moe_aux_loss(model).item()

    return {
        "layer_stats": layer_stats,
        "mean_utilization": mean_util,
        "total_aux_loss": total_aux,
    }
