"""Optional interpretability artifacts (attention, KAN, MoE routing, grad norms)."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from thesis_ml.architectures.transformer_classifier.modules.ffn.moe import MoEFFN
from thesis_ml.architectures.transformer_classifier.modules.kan.kan_linear import KANLinear


def compute_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    """Call after ``loss.backward()``, before ``optimizer.step()``.

    Returns L2 norm per module group (embedding, per encoder block / attention / ffn, head, bias).
    """
    groups: dict[str, float] = defaultdict(float)

    def add_sq(prefix: str, tensor: torch.Tensor) -> None:
        groups[prefix] += tensor.detach().float().pow(2).sum().item()

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        parts = name.split(".")
        if parts[0] == "embedding":
            add_sq("grad/embedding", g)
            continue
        if parts[0] == "head":
            add_sq("grad/head", g)
            continue
        if parts[0] == "bias_composer":
            sub = ".".join(parts[:3]) if len(parts) >= 3 else ".".join(parts)
            add_sq(f"grad/bias/{sub}", g)
            continue
        if parts[0] == "encoder" and len(parts) > 2 and parts[1] == "blocks":
            try:
                idx = int(parts[2])
            except ValueError:
                add_sq(f"grad/other/{name}", g)
                continue
            if "attention" in parts:
                add_sq(f"grad/encoder/block_{idx}/attention", g)
            elif "ffn" in parts:
                add_sq(f"grad/encoder/block_{idx}/ffn", g)
            else:
                add_sq(f"grad/encoder/block_{idx}", g)
            continue
        add_sq(f"grad/other/{parts[0]}", g)

    return {k: math.sqrt(v) for k, v in groups.items() if v > 0.0}


def _batch_to_encoder_args(
    batch: tuple,
    device: torch.device,
) -> tuple[tuple[Any, ...], torch.Tensor | None]:
    if len(batch) == 5:
        tokens_cont, tokens_id, globals_, mask, _label = batch
        tokens_cont = tokens_cont.to(device)
        tokens_id = tokens_id.to(device)
        globals_ = globals_.to(device)
        mask = mask.to(device)
        return (tokens_cont, tokens_id, globals_), mask
    integer_tokens, _gi, mask, _label = batch
    integer_tokens = integer_tokens.to(device)
    mask = mask.to(device)
    return (integer_tokens,), mask


def log_attention_maps(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    outdir: str,
    epoch: int,
    n_events: int = 100,
    filename_suffix: str | None = None,
) -> None:
    """Save per-layer attention weights for up to ``n_events`` validation samples."""
    if not hasattr(model, "prepare_encoder_inputs") or not hasattr(model, "encoder"):
        return

    model.eval()
    per_layer_rows: list[list[torch.Tensor]] = []

    with torch.no_grad():
        for batch in loader:
            enc_args, mask = _batch_to_encoder_args(batch, device)
            x, mask, attention_bias, *_rest = model.prepare_encoder_inputs(*enc_args, mask=mask)
            enc_out = model.encoder(x, mask=mask, attention_bias=attention_bias, capture_attention=True)
            if not isinstance(enc_out, tuple):
                continue
            _x_enc, weights = enc_out
            if not per_layer_rows and weights:
                per_layer_rows = [[] for _ in weights]
            for li, w in enumerate(weights):
                if w is None:
                    continue
                per_layer_rows[li].append(w.cpu())
            n_so_far = sum(t.size(0) for t in per_layer_rows[0]) if per_layer_rows and per_layer_rows[0] else 0
            if n_so_far >= n_events:
                break

    stacked: dict[str, torch.Tensor] = {}
    n_saved = 0
    for li, rows in enumerate(per_layer_rows):
        if not rows:
            continue
        cat = torch.cat(rows, dim=0)
        if cat.size(0) > n_events:
            cat = cat[:n_events]
        stacked[f"layer_{li}"] = cat
        n_saved = cat.size(0)

    os.makedirs(os.path.join(outdir, "interpretability"), exist_ok=True)
    tag = filename_suffix if filename_suffix is not None else str(epoch)
    path = os.path.join(outdir, "interpretability", f"attention_epoch_{tag}.pt")
    torch.save({"epoch": epoch, "n_events": n_saved, "layers": stacked}, path)


def log_kan_splines(model: torch.nn.Module, outdir: str, epoch: int, filename_suffix: str | None = None) -> None:
    """Save all ``KANLinear.spline_weight`` tensors."""
    state: dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if isinstance(m, KANLinear):
            state[name] = m.spline_weight.detach().cpu()
    os.makedirs(os.path.join(outdir, "interpretability"), exist_ok=True)
    tag = filename_suffix if filename_suffix is not None else str(epoch)
    path = os.path.join(outdir, "interpretability", f"kan_splines_epoch_{tag}.pt")
    torch.save({"epoch": epoch, "splines": state}, path)


def _moe_primary_expert_per_event(m: MoEFFN, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Top-1 expert index per batch row (mirrors MoEFFN routing)."""
    if m.routing_level == "event":
        if m.use_cls_token:
            router_input = x[:, 0]
        elif mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            router_input = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            router_input = x.mean(dim=1)
        logits = m.router(router_input)
        return logits.argmax(dim=-1)

    # token-level: use CLS position if present, else mean pooled expert vote
    logits = m.router(x)
    probs = F.softmax(logits, dim=-1)
    if mask is not None:
        probs = probs * mask.unsqueeze(-1).float()
    # primary expert by max mass per event
    return probs.sum(dim=1).argmax(dim=-1)


def log_moe_routing(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    outdir: str,
    epoch: int,
    filename_suffix: str | None = None,
) -> None:
    """Aggregate (expert, class) counts over validation batches (per MoEFFN module)."""
    moe_modules: list[MoEFFN] = [m for m in model.modules() if isinstance(m, MoEFFN)]
    if not moe_modules:
        return

    model.eval()
    n_classes: int | None = None
    counts: dict[str, torch.Tensor] = {}

    def _full_forward(enc_args: tuple[Any, ...], mask: torch.Tensor | None) -> None:
        if len(enc_args) == 3:
            model(enc_args[0], enc_args[1], enc_args[2], mask=mask)
        elif len(enc_args) == 2:
            model(enc_args[0], enc_args[1], mask=mask)
        else:
            model(enc_args[0], mask=mask)

    with torch.no_grad():
        for batch in loader:
            enc_args, mask = _batch_to_encoder_args(batch, device)
            labels = batch[-1].to(device)
            if n_classes is None:
                n_classes = int(labels.max().item()) + 1
                counts = {f"moe_{i}": torch.zeros(n_classes, m.num_experts, dtype=torch.float64) for i, m in enumerate(moe_modules)}

            last_expert: dict[int, torch.Tensor] = {}

            def _make_hook(mm: MoEFFN, storage: dict[int, torch.Tensor]) -> Any:
                mid = id(mm)

                def hook(
                    _mod: torch.nn.Module,
                    inp: tuple[Any, ...],
                    _out: Any,
                    *,
                    _mm: MoEFFN = mm,
                    _mid: int = mid,
                    _storage: dict[int, torch.Tensor] = storage,
                ) -> None:
                    x_in = inp[0]
                    mask_in = inp[1] if len(inp) > 1 else None
                    _storage[_mid] = _moe_primary_expert_per_event(_mm, x_in, mask_in).detach().cpu()

                return hook

            hooks = [m.register_forward_hook(_make_hook(m, last_expert)) for m in moe_modules]
            try:
                _full_forward(enc_args, mask)
            finally:
                for h in hooks:
                    h.remove()

            for li, moe in enumerate(moe_modules):
                expert_idx = last_expert.get(id(moe))
                if expert_idx is None:
                    continue
                for b in range(expert_idx.size(0)):
                    c = int(labels[b].item())
                    e = int(expert_idx[b].item())
                    counts[f"moe_{li}"][c, e] += 1.0

    os.makedirs(os.path.join(outdir, "interpretability"), exist_ok=True)
    tag = filename_suffix if filename_suffix is not None else str(epoch)
    path = os.path.join(outdir, "interpretability", f"moe_routing_epoch_{tag}.pt")
    torch.save({"epoch": epoch, "counts": {k: v.cpu() for k, v in counts.items()}}, path)
