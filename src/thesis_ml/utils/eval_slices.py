"""Evaluation slice utilities for interpretable validation metrics.

Computes per-event slice membership from batches (tokens_id, tokens_cont, globals)
for reporting AUROC/accuracy per slice (jet mult, b-jet mult, MET bins, etc.).
"""

from __future__ import annotations

import contextlib

import torch


def _digitize(bins: list[float], values: torch.Tensor) -> torch.Tensor:
    """Assign each value to bin index (0-based). Values >= bins[-1] go to len(bins)-2."""
    out = torch.zeros_like(values, dtype=torch.long)
    for i in range(len(bins) - 1):
        mask = (values >= bins[i]) & (values < bins[i + 1])
        out[mask] = i
    out[values >= bins[-1]] = len(bins) - 2
    return out


def compute_slice_indices(
    tokens_id: torch.Tensor,
    tokens_cont: torch.Tensor,
    globals_: torch.Tensor | None,
    mask: torch.Tensor | None,
    cont_dim: int = 4,
) -> dict[str, torch.Tensor]:
    """Compute per-event slice bin indices for standard slices.

    Parameters
    ----------
    tokens_id : torch.Tensor
        [B, T] particle IDs (0=pad, 1=jet, 2=b-jet, 3-6=leptons, 7=photon).
    tokens_cont : torch.Tensor
        [B, T, C] continuous features (E, Pt, eta, phi or Pt, eta, phi).
    globals_ : torch.Tensor | None
        [B, 2] (MET, METphi) or None.
    mask : torch.Tensor | None
        [B, T] True=valid.
    cont_dim : int
        Number of continuous columns (3 or 4).

    Returns
    -------
    dict[str, torch.Tensor]
        Keys: n_jets, n_bjets, n_leptons, ht, met.
        Values: [B] long tensor of bin indices (0-based).
    """
    B = tokens_id.size(0)
    valid = mask if mask is not None else torch.ones(B, tokens_id.size(1), dtype=torch.bool, device=tokens_id.device)
    ids = tokens_id.clamp(0, 7)

    n_jets = ((ids == 1) | (ids == 2)) & valid
    n_bjets = (ids == 2) & valid
    n_leptons = ((ids >= 3) & (ids <= 6)) & valid

    n_jets_per_event = n_jets.sum(dim=1).float()
    n_bjets_per_event = n_bjets.sum(dim=1).float()
    n_leptons_per_event = n_leptons.sum(dim=1).float()

    pt_col = 1 if cont_dim >= 4 else 0
    pt = tokens_cont[..., pt_col]
    ht = (pt * valid.float()).sum(dim=1)

    met = globals_[:, 0] if globals_ is not None else torch.zeros(B, device=tokens_id.device)

    default_bins = {
        "n_jets": [0, 2, 4, 6, 8, 18],
        "n_bjets": [0, 1, 2, 3, 8],
        "n_leptons": [0, 1, 2, 8],
        "ht": [0.0, 200.0, 400.0, 600.0, 1000.0, 5000.0],
        "met": [0.0, 50.0, 100.0, 150.0, 200.0, 500.0],
    }
    values = {
        "n_jets": n_jets_per_event,
        "n_bjets": n_bjets_per_event,
        "n_leptons": n_leptons_per_event,
        "ht": ht,
        "met": met,
    }

    return {k: _digitize(default_bins[k], v) for k, v in values.items()}


def slice_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    slice_indices: dict[str, torch.Tensor],
    slice_name: str,
    n_bins: int,
) -> dict[str, float]:
    """Compute accuracy and AUROC per slice bin.

    Parameters
    ----------
    logits : torch.Tensor
        [B, n_classes] or [B].
    labels : torch.Tensor
        [B] long.
    slice_indices : dict[str, torch.Tensor]
        From compute_slice_indices.
    slice_name : str
        Key in slice_indices (e.g. "n_jets", "met").
    n_bins : int
        Number of bins.

    Returns
    -------
    dict[str, float]
        Per-bin metrics when bin has >= 10 samples; empty for sparse bins.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    idx = slice_indices[slice_name]
    probs = logits.sigmoid().squeeze(-1) if logits.size(-1) == 1 else logits.softmax(dim=-1)[:, 1]
    preds = (probs > 0.5).long().cpu().numpy()
    y = labels.cpu().numpy()

    out: dict[str, float] = {}
    for b in range(n_bins):
        mask = (idx == b).cpu().numpy()
        if mask.sum() < 10:
            continue
        y_b = y[mask]
        pred_b = preds[mask]
        prob_b = probs[mask].cpu().numpy()
        out[f"{slice_name}_bin{b}_acc"] = float(accuracy_score(y_b, pred_b))
        with contextlib.suppress(ValueError):
            out[f"{slice_name}_bin{b}_auroc"] = float(roc_auc_score(y_b, prob_b))
    return out
