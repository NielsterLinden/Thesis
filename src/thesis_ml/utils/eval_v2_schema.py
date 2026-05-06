"""Shared constants for the eval_v2 pipeline.

Imported by both the inline training path (eval_v2.py) and the
batch backfill script (scripts/wandb/eval_pipeline/phase2_upload_to_wandb.py).
Any change here must remain backward-compatible with W&B run summaries already
written at spec_version = "eval_v2.1".
"""

from __future__ import annotations

# Written to eval_v2/spec_version on every run.
# Must match eval_spec.yaml::spec_version used during batch inference.
SPEC_VERSION = "eval_v2.1"

# All scalar keys pushed to run.summary.
# JSON/array columns (roc_fpr, pr_recall, etc.) are NOT listed here — they go
# into the per-run Artifact as wandb.Tables.
SUMMARY_SCALAR_COLS: list[str] = [
    # discrimination
    "eval_v2/test_loss",
    "eval_v2/test_acc",
    "eval_v2/test_f1",
    "eval_v2/test_auroc",
    "eval_v2/log_loss",
    "eval_v2/brier_score",
    "eval_v2/ece",
    # HEP figures of merit (binary only; omitted for multiclass)
    "eval_v2/eps_S_at_invB_10",
    "eval_v2/eps_S_at_invB_50",
    "eval_v2/eps_S_at_invB_100",
    "eval_v2/eps_S_at_invB_1000",
    "eval_v2/auroc_at_low_fpr",
    "eval_v2/auroc_at_high_tpr",
    # cost
    "eval_v2/flops_per_event_analytic",
    "eval_v2/flops_per_event_measured",  # always <not_applicable>
    "eval_v2/num_parameters_total",
    "eval_v2/num_parameters_trainable",
    "eval_v2/num_parameters_by_module_encoder",
    "eval_v2/num_parameters_by_module_head",
    "eval_v2/num_parameters_by_module_tokenizer",
    "eval_v2/num_parameters_by_module_biases",
    "eval_v2/inference_latency_ms_b1_mean",
    "eval_v2/inference_latency_ms_b1_p50",
    "eval_v2/inference_latency_ms_b1_p95",
    "eval_v2/inference_latency_ms_b1_p99",
    "eval_v2/inference_latency_ms_b64_mean",
    "eval_v2/inference_latency_ms_b512_mean",
    "eval_v2/throughput_samples_per_s_b512",
    "eval_v2/peak_memory_mib_inference_b512",
    "eval_v2/checkpoint_size_mb",
    "eval_v2/runtime_seconds",
    # interpretability scalars (architecture-conditional, omitted when not applicable)
    "eval_v2/diff_attn/lambda_mean_abs",
    "eval_v2/moe/expert_utilization_mean",
    "eval_v2/lorentz/feature_gate_active_count",
    "eval_v2/kan/spline_complexity",
    "eval_v2/typepair/table_norm",
    "eval_v2/typepair/table_drift_from_init",
    # provenance
    "eval_v2/spec_version",
    "eval_v2/timestamp",
    "eval_v2/torch_version",
    "eval_v2/cuda_version",
    "eval_v2/gpu_name",
    "eval_v2/checkpoint_status",
    "eval_v2/checkpoint_kind",
    "eval_v2/checkpoint_sha256",
    "eval_v2/test_set_hash",
    "eval_v2/checkpoint_epoch",
]

# Keys that are expected to be absent for certain runs (not a bug if missing).
# The completeness check in run_eval_v2 skips these before warning.
OPTIONAL_SCALAR_COLS: frozenset[str] = frozenset(
    {
        # never implemented inline or in backfill
        "eval_v2/flops_per_event_measured",
        # architecture-conditional interpretability
        "eval_v2/diff_attn/lambda_mean_abs",
        "eval_v2/moe/expert_utilization_mean",
        "eval_v2/lorentz/feature_gate_active_count",
        "eval_v2/kan/spline_complexity",
        "eval_v2/typepair/table_norm",
        "eval_v2/typepair/table_drift_from_init",
        # binary-only HEP FoM (not applicable for multiclass tasks)
        "eval_v2/eps_S_at_invB_10",
        "eval_v2/eps_S_at_invB_50",
        "eval_v2/eps_S_at_invB_100",
        "eval_v2/eps_S_at_invB_1000",
        "eval_v2/auroc_at_low_fpr",
        "eval_v2/auroc_at_high_tpr",
        # latency (skipped on OOM or CPU-only nodes)
        "eval_v2/inference_latency_ms_b1_mean",
        "eval_v2/inference_latency_ms_b1_p50",
        "eval_v2/inference_latency_ms_b1_p95",
        "eval_v2/inference_latency_ms_b1_p99",
        "eval_v2/inference_latency_ms_b64_mean",
        "eval_v2/inference_latency_ms_b512_mean",
        "eval_v2/throughput_samples_per_s_b512",
        "eval_v2/peak_memory_mib_inference_b512",
        # epoch number absent for non-epoch checkpoints
        "eval_v2/checkpoint_epoch",
    }
)
