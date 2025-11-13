"""Classification metrics computation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    all_logits: list[torch.Tensor],
    all_labels: list[torch.Tensor],
    all_probs: list[torch.Tensor],
    n_classes: int,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics.

    Parameters
    ----------
    all_logits : list[torch.Tensor]
        List of logit tensors [B, n_classes] from all batches (CPU tensors)
    all_labels : list[torch.Tensor]
        List of label tensors [B] from all batches (CPU tensors)
    all_probs : list[torch.Tensor]
        List of probability tensors [B, n_classes] from all batches (CPU tensors)
    n_classes : int
        Number of classes

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
            - accuracy: float
            - auroc: float | None
            - precision_macro: float
            - recall_macro: float
            - f1_macro: float
            - precision_per_class: list[float]
            - recall_per_class: list[float]
            - f1_per_class: list[float]
            - support_per_class: list[int]
            - confusion_matrix: list[list[int]]
            - confusion_matrix_normalized: list[list[float]]
            - roc_curves: dict[int, dict[str, list[float]]]
            - pr_curves: dict[int, dict[str, list[float]]]
    """
    # Concatenate all labels and probabilities
    labels_cat = torch.cat(all_labels, dim=0).numpy()  # [N]
    probs_cat = torch.cat(all_probs, dim=0).numpy()  # [N, n_classes]

    # Compute predictions
    preds = probs_cat.argmax(axis=-1)  # [N]

    # Accuracy
    accuracy = accuracy_score(labels_cat, preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(labels_cat, preds, average=None, zero_division=0)

    # Macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels_cat, preds, average="macro", zero_division=0)

    # AUROC
    unique_labels = np.unique(labels_cat)
    if len(unique_labels) < 2:
        auroc = None
    elif n_classes == 2:
        # Binary classification: use positive class probability
        auroc = roc_auc_score(labels_cat, probs_cat[:, 1])
    else:
        # Multi-class: use one-vs-rest macro-averaged
        auroc = roc_auc_score(labels_cat, probs_cat, multi_class="ovr", average="macro")

    # Confusion matrix
    cm = confusion_matrix(labels_cat, preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # Handle division by zero
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)

    # ROC curves (one per class for multi-class, single for binary)
    roc_curves = {}
    if n_classes == 2:
        # Binary: single curve using positive class
        fpr, tpr, _ = roc_curve(labels_cat, probs_cat[:, 1])
        roc_curves[1] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    else:
        # Multi-class: one-vs-rest for each class
        for class_idx in range(n_classes):
            y_binary = (labels_cat == class_idx).astype(int)
            if y_binary.sum() > 0:  # Class exists in data
                fpr, tpr, _ = roc_curve(y_binary, probs_cat[:, class_idx])
                roc_curves[class_idx] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # PR curves
    pr_curves = {}
    if n_classes == 2:
        # Binary: single curve using positive class
        precision_vals, recall_vals, _ = precision_recall_curve(labels_cat, probs_cat[:, 1])
        pr_curves[1] = {"precision": precision_vals.tolist(), "recall": recall_vals.tolist()}
    else:
        # Multi-class: one-vs-rest for each class
        for class_idx in range(n_classes):
            y_binary = (labels_cat == class_idx).astype(int)
            if y_binary.sum() > 0:  # Class exists in data
                precision_vals, recall_vals, _ = precision_recall_curve(y_binary, probs_cat[:, class_idx])
                pr_curves[class_idx] = {"precision": precision_vals.tolist(), "recall": recall_vals.tolist()}

    result = {
        "accuracy": float(accuracy),
        "auroc": float(auroc) if auroc is not None else None,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
    }

    # Store per-event scores and labels for plotting (binary classification only)
    if n_classes == 2:
        result["per_event_scores"] = probs_cat[:, 1].tolist()  # Signal probability
        result["per_event_labels"] = labels_cat.tolist()  # True labels

    return result
