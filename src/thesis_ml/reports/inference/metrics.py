"""Anomaly detection metrics computation."""

from __future__ import annotations

import numpy as np
import torch

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


def compute_reconstruction_errors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict[str, list[float]]:
    """Compute reconstruction errors per event with masked loss.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted values [B, T, D]
    targets : torch.Tensor
        Target values [B, T, D]
    mask : torch.Tensor
        Mask [B, T] with 1 for valid tokens, 0 for padding
    weights : torch.Tensor | None
        Event weights [B] (optional)

    Returns
    -------
    dict[str, list[float]]
        Dict with "mse" and "mae" lists of per-event errors
    """
    mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]

    # MSE per event: ((x_hat - x)^2 * mask).sum() / mask.sum()
    squared_error = (predictions - targets) ** 2  # [B, T, D]
    masked_squared_error = squared_error * mask_expanded  # [B, T, D]
    mse_per_event = masked_squared_error.sum(dim=-1).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)  # [B]

    # MAE per event: |x_hat - x| * mask).sum() / mask.sum()
    abs_error = torch.abs(predictions - targets)  # [B, T, D]
    masked_abs_error = abs_error * mask_expanded  # [B, T, D]
    mae_per_event = masked_abs_error.sum(dim=-1).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)  # [B]

    return {
        "mse": mse_per_event.tolist(),
        "mae": mae_per_event.tolist(),
    }


def aggregate_metrics(
    errors_dict: dict[str, list[float]],
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Aggregate per-event errors into statistics.

    Parameters
    ----------
    errors_dict : dict[str, list[float]]
        Dict with metric names as keys and lists of per-event values
    weights : list[float] | None
        Optional event weights for weighted statistics

    Returns
    -------
    dict[str, float]
        Aggregated statistics with keys like:
            - {metric}_mean, {metric}_std, {metric}_p50, {metric}_p95, {metric}_p99
            - {metric}_weighted_mean, {metric}_weighted_std (if weights provided)
    """
    result = {}

    for metric_name, values in errors_dict.items():
        values_arr = np.array(values)

        # Unweighted statistics
        result[f"{metric_name}_mean"] = float(np.mean(values_arr))
        result[f"{metric_name}_std"] = float(np.std(values_arr))
        result[f"{metric_name}_p50"] = float(np.percentile(values_arr, 50))
        result[f"{metric_name}_p95"] = float(np.percentile(values_arr, 95))
        result[f"{metric_name}_p99"] = float(np.percentile(values_arr, 99))

        # Weighted statistics if weights provided
        if weights is not None:
            weights_arr = np.array(weights)
            if len(weights_arr) == len(values_arr):
                # Normalize weights
                weights_arr = weights_arr / weights_arr.sum() * len(weights_arr)

                # Weighted mean
                weighted_mean = np.average(values_arr, weights=weights_arr)
                result[f"{metric_name}_weighted_mean"] = float(weighted_mean)

                # Weighted std: sqrt(sum(w * (x - weighted_mean)^2) / sum(w))
                weighted_variance = np.average((values_arr - weighted_mean) ** 2, weights=weights_arr)
                result[f"{metric_name}_weighted_std"] = float(np.sqrt(weighted_variance))

    return result


def compute_auroc(
    baseline_scores: list[float] | np.ndarray,
    corrupted_scores: list[float] | np.ndarray,
    weights: list[float] | np.ndarray | None = None,
) -> float | None:
    """Compute AUROC between baseline and corrupted scores.

    Parameters
    ----------
    baseline_scores : list[float] | np.ndarray
        Reconstruction errors for baseline (normal) data
    corrupted_scores : list[float] | np.ndarray
        Reconstruction errors for corrupted (anomalous) data
    weights : list[float] | np.ndarray | None
        Optional event weights (should be concatenated: baseline_weights + corrupted_weights)

    Returns
    -------
    float | None
        AUROC score, or None if sklearn not available or error occurs
    """
    if roc_auc_score is None:
        return None

    try:
        baseline_arr = np.array(baseline_scores)
        corrupted_arr = np.array(corrupted_scores)

        # Combine scores and labels
        scores = np.concatenate([baseline_arr, corrupted_arr])
        labels = np.concatenate([np.zeros(len(baseline_arr)), np.ones(len(corrupted_arr))])

        # Compute AUROC
        if weights is not None:
            weights_arr = np.array(weights)
            auroc = roc_auc_score(labels, scores, sample_weight=weights_arr) if len(weights_arr) == len(scores) else roc_auc_score(labels, scores)
        else:
            auroc = roc_auc_score(labels, scores)

        return float(auroc)
    except Exception:
        return None
