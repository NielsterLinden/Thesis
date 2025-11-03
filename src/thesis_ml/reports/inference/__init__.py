"""Inference infrastructure for anomaly detection."""

from .anomaly_detection import run_anomaly_detection
from .data_corruption import create_corrupted_dataloader
from .forward_pass import LegacyModelAdapter, create_model_adapter, run_batch_inference
from .metrics import aggregate_metrics, compute_auroc, compute_reconstruction_errors

__all__ = [
    "run_anomaly_detection",
    "create_corrupted_dataloader",
    "run_batch_inference",
    "LegacyModelAdapter",
    "create_model_adapter",
    "compute_reconstruction_errors",
    "aggregate_metrics",
    "compute_auroc",
]
