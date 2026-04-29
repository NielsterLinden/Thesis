"""Extra Tier-1 metrics for Phase 2 eval (used by stage_b)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

_trapz = getattr(np, "trapezoid", np.trapz)


def _binary_scores_probs(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = labels.astype(int)
    if probs.ndim == 2 and probs.shape[1] >= 2:
        s = probs[:, 1]
    else:
        s = probs.reshape(-1)
    return y, s


def cross_entropy_mean(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels.long(), reduction="mean").item())


def brier_binary(labels: np.ndarray, probs: np.ndarray) -> float:
    y, s = _binary_scores_probs(labels, probs)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(brier_score_loss(y, s))


def log_loss_mc(labels: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs, eps, 1.0 - eps)
    p /= p.sum(axis=1, keepdims=True)
    return float(log_loss(labels, p))


def ece_multiclass(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (equal-mass bins on confidence = max prob)."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == labels).astype(np.float64)
    order = np.argsort(conf)
    n = len(conf)
    ece = 0.0
    bin_n = max(1, n // n_bins)
    for b in range(n_bins):
        sl = order[b * bin_n : (b + 1) * bin_n if b < n_bins - 1 else n]
        if len(sl) == 0:
            continue
        ece += len(sl) / n * abs(acc[sl].mean() - conf[sl].mean())
    return float(ece)


def partial_auroc_fpr(y: np.ndarray, s: np.ndarray, fpr_max: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    mask = fpr <= fpr_max
    if mask.sum() < 2 or len(np.unique(y)) < 2:
        return float("nan")
    f, t = fpr[mask], tpr[mask]
    if len(f) < 2:
        return float("nan")
    return float(_trapz(t, f))


def partial_auroc_tpr(y: np.ndarray, s: np.ndarray, tpr_min: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    mask = tpr >= tpr_min
    if mask.sum() < 2 or len(np.unique(y)) < 2:
        return float("nan")
    f, t = fpr[mask], tpr[mask]
    if len(f) < 2:
        return float("nan")
    return float(_trapz(t, f))


def eps_s_at_background_rejection(y: np.ndarray, s: np.ndarray, inv_b: float) -> float:
    """Signal efficiency when background survival fraction is 1/inv_b (binary, high s = signal)."""
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    bg = s[y == 0]
    sg = s[y == 1]
    if len(bg) == 0 or len(sg) == 0:
        return float("nan")
    target_frac = 1.0 / float(inv_b)
    thr = np.quantile(bg, 1.0 - target_frac)
    return float((sg >= thr).mean())


def per_class_auroc(labels: np.ndarray, probs: np.ndarray, n_classes: int) -> dict[str, float]:
    out: dict[str, float] = {}
    y = labels.astype(int)
    for c in range(n_classes):
        y_bin = (y == c).astype(int)
        if y_bin.sum() == 0 or (1 - y_bin).sum() == 0:
            out[str(c)] = float("nan")
            continue
        try:
            out[str(c)] = float(roc_auc_score(y_bin, probs[:, c]))
        except ValueError:
            out[str(c)] = float("nan")
    return out


def score_histograms_binary(y: np.ndarray, s: np.ndarray, n_bins: int = 50) -> tuple[list[float], list[float]]:
    y = y.astype(int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    h_sig, _ = np.histogram(s[y == 1], bins=edges)
    h_bg, _ = np.histogram(s[y == 0], bins=edges)
    return h_sig.astype(float).tolist(), h_bg.astype(float).tolist()


def confusion_dict(cm: list[list[int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            out[f"{i}_{j}"] = int(v)
    return out


def flops_analytic_transformer(dim: int, depth: int, heads: int, ffn_dim: int, seq: int, n_classes: int) -> float:
    """Rough per-forward FLOPs estimate (order of magnitude, documented in eval_spec)."""
    # Per block: self-attn ~ 4*T*D^2, FFN ~ 2*T*D*F (factor 4 from QKTV projections lumped)
    per_block = 4.0 * seq * (dim**2) * 4 + 2.0 * seq * dim * ffn_dim
    enc = depth * per_block
    head = seq * dim * n_classes
    return float(enc + head)


def tier3_placeholder() -> dict[str, str]:
    """Keys reserved for interpretability; filled incrementally as APIs stabilize."""
    return {
        "eval_v2/diff_attn/lambda_mean_abs": "<not_applicable>",
        "eval_v2/moe/expert_utilization_mean": "<not_applicable>",
        "eval_v2/lorentz/feature_gate_active_count": "<not_applicable>",
        "eval_v2/kan/spline_complexity": "<not_applicable>",
        "eval_v2/typepair/table_norm": "<not_applicable>",
        "eval_v2/typepair/table_drift_from_init": "<not_applicable>",
    }


def flatten_metrics(prefix: str, d: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, (dict, list)):
            out[key] = json.dumps(v)
        elif v is None or (isinstance(v, float) and np.isnan(v)):
            out[key] = ""
        else:
            out[key] = str(v)
    return out
