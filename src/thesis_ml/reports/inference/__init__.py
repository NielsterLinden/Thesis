"""Inference infrastructure for anomaly detection and classification."""

from .anomaly_detection import run_anomaly_detection
from .classification import run_classification_inference
from .classification_metrics import compute_classification_metrics
from .data_corruption import create_corrupted_dataloader
from .forward_pass import LegacyModelAdapter, create_model_adapter, run_batch_inference
from .metrics import aggregate_metrics, compute_auroc, compute_reconstruction_errors

__all__ = [
    "run_anomaly_detection",
    "run_classification_inference",
    "compute_classification_metrics",
    "create_corrupted_dataloader",
    "run_batch_inference",
    "LegacyModelAdapter",
    "create_model_adapter",
    "compute_reconstruction_errors",
    "aggregate_metrics",
    "compute_auroc",
]
