"""Batch inference execution with masked loss computation."""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LegacyModelAdapter(nn.Module):
    """Adapter to wrap models with old signature to uniform API.

    Wraps models with signature `forward(tokens_cont, tokens_id, globals_vec)`
    to new signature `forward(tokens_cont, tokens_id, globals_vec, mask)`.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        tokens_cont: torch.Tensor,
        tokens_id: torch.Tensor,
        globals_vec: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass with mask parameter (ignored for legacy models).

        Parameters
        ----------
        tokens_cont : torch.Tensor
            Continuous tokens [B, T, 4]
        tokens_id : torch.Tensor
            Token IDs [B, T]
        globals_vec : torch.Tensor
            Global features [B, 2]
        mask : torch.Tensor | None
            Mask [B, T] (ignored for legacy models)

        Returns
        -------
        dict[str, Any]
            Model output dict with at least 'x_hat' key
        """
        # Call original model (mask is ignored)
        return self.model(tokens_cont, tokens_id, globals_vec)


def _has_mask_parameter(model: nn.Module) -> bool:
    """Check if model forward signature includes mask parameter.

    Parameters
    ----------
    model : nn.Module
        Model to check

    Returns
    -------
    bool
        True if forward signature includes mask parameter
    """
    if not hasattr(model, "forward"):
        return False

    sig = inspect.signature(model.forward)
    return "mask" in sig.parameters


def create_model_adapter(model: nn.Module) -> nn.Module:
    """Create adapter for model if needed.

    Parameters
    ----------
    model : nn.Module
        Model to wrap

    Returns
    -------
    nn.Module
        Model with uniform API (possibly wrapped)
    """
    if _has_mask_parameter(model):
        return model
    return LegacyModelAdapter(model)


def _create_default_mask(tokens_id: torch.Tensor) -> torch.Tensor:
    """Create default mask from token IDs (assuming 0 is pad token).

    Parameters
    ----------
    tokens_id : torch.Tensor
        Token IDs [B, T]

    Returns
    -------
    torch.Tensor
        Mask [B, T] with 1 for valid tokens, 0 for padding
    """
    return (tokens_id != 0).long()


def _extract_batch_components(batch: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Extract components from batch tuple.

    Parameters
    ----------
    batch : tuple
        Batch tuple of length 3 or 4

    Returns
    -------
    tuple
        (tokens_cont, tokens_id, globals_vec, mask, weights)
    """
    if len(batch) == 3:
        tokens_cont, tokens_id, globals_vec = batch
        mask = None
        weights = None
    elif len(batch) == 4:
        tokens_cont, tokens_id, globals_vec, mask_or_weights = batch
        # Check if last element is mask (2D) or weights (1D)
        if mask_or_weights.dim() == 2:
            mask = mask_or_weights
            weights = None
        else:
            mask = None
            weights = mask_or_weights
    elif len(batch) == 5:
        tokens_cont, tokens_id, globals_vec, mask, weights = batch
    else:
        raise ValueError(f"Expected batch tuple of length 3, 4, or 5, got {len(batch)}")

    return tokens_cont, tokens_id, globals_vec, mask, weights


def run_batch_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    autocast: bool = False,
    model_adapter: nn.Module | None = None,
) -> dict[str, Any]:
    """Run batch inference and compute reconstruction errors.

    Parameters
    ----------
    model : nn.Module
        Model to run inference with (should support forward(tokens_cont, tokens_id, globals, mask))
    dataloader : DataLoader
        DataLoader to iterate over
    device : torch.device
        Device to run inference on
    autocast : bool
        Whether to use autocast for mixed precision
    model_adapter : nn.Module | None
        Optional adapter to wrap model (if None, will create adapter if needed)

    Returns
    -------
    dict[str, Any]
        Results dict with keys:
            - per_event: list of dicts with id, weight, mse, mae
            - reconstruction_errors: list of MSE per event (for backward compat)
            - latent_codes: optional list of latent representations
            - reconstructions: optional list of reconstructed outputs
    """
    if model_adapter is None:
        model_adapter = create_model_adapter(model)

    model_adapter.eval()
    model_adapter.to(device)

    per_event_results = []
    reconstruction_errors = []
    latent_codes = []
    reconstructions = []

    event_id = 0

    with torch.inference_mode():
        # Set up autocast context if enabled and on CUDA
        if autocast and device.type == "cuda":
            autocast_context = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            # Use nullcontext or autocast with enabled=False for non-CUDA or disabled autocast
            from contextlib import nullcontext

            autocast_context = nullcontext()

        for batch in dataloader:
            # Extract batch components
            tokens_cont, tokens_id, globals_vec, mask, weights = _extract_batch_components(batch)

            # Move to device (pin device once)
            tokens_cont = tokens_cont.to(device, non_blocking=True)
            tokens_id = tokens_id.to(device, non_blocking=True)
            globals_vec = globals_vec.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True) if mask is not None else _create_default_mask(tokens_id)

            weights = weights.to(device, non_blocking=True) if weights is not None else torch.ones(tokens_cont.shape[0], device=device)

            # Forward pass
            with autocast_context:
                out = model_adapter(tokens_cont, tokens_id, globals_vec, mask)

            # Extract reconstruction
            x_hat = out["x_hat"]  # [B, T, 4]

            # Compute masked reconstruction errors per event
            # mse = ((x_hat - tokens_cont)^2 * mask.unsqueeze(-1)).sum(dim=-1).sum(dim=-1) / mask.sum(dim=-1)
            mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
            squared_error = (x_hat - tokens_cont) ** 2  # [B, T, 4]
            masked_squared_error = squared_error * mask_expanded  # [B, T, 4]
            mse_per_event = masked_squared_error.sum(dim=-1).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)  # [B]

            # MAE
            abs_error = torch.abs(x_hat - tokens_cont)  # [B, T, 4]
            masked_abs_error = abs_error * mask_expanded  # [B, T, 4]
            mae_per_event = masked_abs_error.sum(dim=-1).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)  # [B]

            # Store per-event results
            B = tokens_cont.shape[0]
            for b in range(B):
                per_event_results.append(
                    {
                        "id": event_id,
                        "weight": float(weights[b].item()),
                        "mse": float(mse_per_event[b].item()),
                        "mae": float(mae_per_event[b].item()),
                    }
                )
                reconstruction_errors.append(float(mse_per_event[b].item()))
                event_id += 1

            # Optionally store latent codes and reconstructions
            if "z_e" in out:
                latent_codes.append(out["z_e"].detach().cpu())
            if "x_hat" in out:
                reconstructions.append(x_hat.detach().cpu())

    return {
        "per_event": per_event_results,
        "reconstruction_errors": reconstruction_errors,
        "latent_codes": latent_codes if latent_codes else None,
        "reconstructions": reconstructions if reconstructions else None,
    }
