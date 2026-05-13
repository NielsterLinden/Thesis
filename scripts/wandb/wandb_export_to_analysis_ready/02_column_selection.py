#!/usr/bin/env python3
"""Stage 02 — Column selection → thesis_results/02_eval_combined.csv

Reads 01_raw_export.csv and keeps only the exact columns present in
04_cleaned_backfilled_analysis_ready.csv (the canonical analysis table).
Any column not in KEEP_COLUMNS is silently dropped.

Usage::

    python3 scripts/wandb/wandb_export_to_analysis_ready/02_column_selection.py
    python3 scripts/wandb/wandb_export_to_analysis_ready/02_column_selection.py \\
        --in thesis_results/01_raw_export.csv --out thesis_results/02_eval_combined.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Exact column allowlist — derived from thesis_results/04_cleaned_backfilled_analysis_ready.csv.
# Only these columns are kept; everything else is dropped silently.
KEEP_COLUMNS = [
    "meta_run/id",
    "meta_run/name",
    "meta_run/created_at",
    "meta_run/state",
    "meta_run/tags",
    "meta_run/group",
    "meta_run/project",
    "config/axes/A1_Normalization Policy",
    "config/axes/A2_Normalization Type",
    "config/axes/A3-a_Differential Attention Bias Mode",
    "config/axes/A3_Attention Type",
    "config/axes/A4_Attention Internal Normalization",
    "config/axes/A5_Causal Masking",
    "config/axes/B1-G1_Global-Conditioned Mode",
    "config/axes/B1-G2_Global-Conditioned MLP Type",
    "config/axes/B1-G3_Global-Conditioned Global Dimension",
    "config/axes/B1-L1_Lorentz Feature Set",
    "config/axes/B1-L2_Lorentz MLP Type",
    "config/axes/B1-L3_Lorentz Hidden Dimension",
    "config/axes/B1-L4_Lorentz Per-Head Mode",
    "config/axes/B1-L5_Lorentz Sparse Gating",
    "config/axes/B1-S1_SM Interaction Mode",
    "config/axes/B1-S2_SM Mask Value",
    "config/axes/B1-T1_Type-Pair Initialization",
    "config/axes/B1-T2_Type-Pair Freeze Table",
    "config/axes/B1-T3_Type-Pair Kinematic Gate",
    "config/axes/B1-T4_Type-Pair Kinematic Feature",
    "config/axes/B1-T5_Type-Pair Mask Value",
    "config/axes/B1_Bias Activation Set",
    "config/axes/C1_Head Realization",
    "config/axes/C2_Pooling Strategy",
    "config/axes/D1_Feature Set",
    "config/axes/D2_MET Treatment",
    "config/axes/D3_Token Ordering",
    "config/axes/E1-a1_PE Dimension Mask",
    "config/axes/E1-a_PE Space",
    "config/axes/E1-b_Rotary Base Frequency",
    "config/axes/E1_PE Type",
    "config/axes/F1-a1_KAN FFN Bottleneck Dimension",
    "config/axes/F1-a_KAN FFN Variant",
    "config/axes/F1-b_MoE Encoder Scope",
    "config/axes/F1-eff_FFN Realization",
    "config/axes/F1-moe_MoE Enabled",
    "config/axes/F1_FFN Type",
    "config/axes/G1_Task Type",
    "config/axes/G2_Model Family",
    "config/axes/G3_Classification Task",
    "config/axes/H10_Model Size Label",
    "config/axes/H1_Model Dimension",
    "config/axes/H2_Encoder Depth",
    "config/axes/H3_Attention Heads",
    "config/axes/H4_FFN Hidden Dimension",
    "config/axes/H5_Dropout",
    "config/axes/K1_KAN Grid Size",
    "config/axes/K2_KAN Spline Order",
    "config/axes/K3_KAN Grid Range",
    "config/axes/K4_KAN Spline Regularization Weight",
    "config/axes/K5_KAN Grid Update Frequency",
    "config/axes/L1_Log PID Embeddings",
    "config/axes/L2_Interpretability Enabled",
    "config/axes/L3_Save Attention Maps",
    "config/axes/L4_Save KAN Splines",
    "config/axes/L5_Save MoE Routing",
    "config/axes/L6_Save Gradient Norms",
    "config/axes/L7_Checkpoint Epochs",
    "config/axes/M1_MoE Number of Experts",
    "config/axes/M2_MoE Top K",
    "config/axes/M3_MoE Routing Level",
    "config/axes/M4_MoE Load Balance Weight",
    "config/axes/M5_MoE Noisy Gating",
    "config/axes/P1-a_Nodewise Mass Neighbourhood Sizes",
    "config/axes/P1-b_Nodewise Mass Hidden Dimension",
    "config/axes/P1_Nodewise Mass Enabled",
    "config/axes/P2-a_MIA Placement",
    "config/axes/P2-b_MIA Number of Blocks",
    "config/axes/P2-c_MIA Interaction Dimension",
    "config/axes/P2-d_MIA Reduction Dimension",
    "config/axes/P2-e_MIA Dropout",
    "config/axes/P2_MIA Pre-Encoder Enabled",
    "config/axes/R10_Early Stop Enabled",
    "config/axes/R11_Early Stop Patience",
    "config/axes/R12_Early Stop Min Delta",
    "config/axes/R13_Restore Best Weights",
    "config/axes/R14_PID Schedule Mode",
    "config/axes/R15_PID Transition Epoch",
    "config/axes/R16_PID Reinit Mode",
    "config/axes/R17_PID Separate LR",
    "config/axes/R1_Epochs",
    "config/axes/R2_Learning Rate",
    "config/axes/R3_Weight Decay",
    "config/axes/R4_Batch Size",
    "config/axes/R5_Seed",
    "config/axes/R6_Warmup Steps",
    "config/axes/R7_LR Schedule",
    "config/axes/R8_Label Smoothing",
    "config/axes/R9_Gradient Clipping",
    "config/axes/S1_Shared Backbone Enabled",
    "config/axes/S2_Shared Backbone Features",
    "config/axes/T1-a_PID Embedding Mode",
    "config/axes/T1-b_PID Embedding Dimension",
    "config/axes/T1-c_Pretrained Model Type",
    "config/axes/T1_Tokenizer Family",
    "eval_v2/test_loss",
    "eval_v2/test_acc",
    "eval_v2/test_f1",
    "eval_v2/test_auroc",
    "eval_v2/log_loss",
    "eval_v2/brier_score",
    "eval_v2/eps_S_at_invB_10",
    "eval_v2/eps_S_at_invB_50",
    "eval_v2/eps_S_at_invB_100",
    "eval_v2/eps_S_at_invB_1000",
    "eval_v2/auroc_at_low_fpr",
    "eval_v2/auroc_at_high_tpr",
    "eval_v2/ece",
    "eval_v2/flops_per_event_analytic",
    "eval_v2/flops_per_event_measured",
    "eval_v2/num_parameters_total",
    "eval_v2/num_parameters_trainable",
    "eval_v2/num_parameters_by_module_encoder",
    "eval_v2/num_parameters_by_module_head",
    "eval_v2/num_parameters_by_module_tokenizer",
    "eval_v2/num_parameters_by_module_biases",
    "eval_v2/checkpoint_size_mb",
    "eval_v2/checkpoint_epoch",
    "eval_v2/checkpoint_status",
    "eval_v2/checkpoint_kind",
    "eval_v2/checkpoint_sha256",
    "eval_v2/test_set_hash",
    "eval_v2/spec_version",
    "eval_v2/timestamp",
    "eval_v2/torch_version",
    "eval_v2/cuda_version",
    "eval_v2/gpu_name",
    "eval_v2/runtime_seconds",
    "eval_v2/inference_latency_ms_b1_mean",
    "eval_v2/inference_latency_ms_b1_p50",
    "eval_v2/inference_latency_ms_b1_p95",
    "eval_v2/inference_latency_ms_b1_p99",
    "eval_v2/inference_latency_ms_b64_mean",
    "eval_v2/inference_latency_ms_b512_mean",
    "eval_v2/throughput_samples_per_s_b512",
    "eval_v2/peak_memory_mib_inference_b512",
    "eval_v2/diff_attn/lambda_mean_abs",
    "eval_v2/moe/expert_utilization_mean",
    "eval_v2/lorentz/feature_gate_active_count",
    "eval_v2/kan/spline_complexity",
    "eval_v2/typepair/table_norm",
    "eval_v2/typepair/table_drift_from_init",
    "eval_v2/score_hist_signal",
    "eval_v2/score_hist_background",
    "eval_v2/per_class_auroc_json",
    "eval_v2/per_class_precision_json",
    "eval_v2/per_class_recall_json",
    "eval_v2/cm_json",
    "eval_v2/roc_fpr",
    "eval_v2/roc_tpr",
    "eval_v2/pr_precision",
    "eval_v2/pr_recall",
    "config/meta.needs_review",
]

_KEEP_SET = set(KEEP_COLUMNS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=REPO_ROOT / "thesis_results" / "01_raw_export.csv")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "thesis_results" / "02_eval_combined.csv")
    args = ap.parse_args()

    inp = args.inp.resolve()
    if not inp.is_file():
        print(f"error: input not found: {inp}", file=sys.stderr)
        return 1

    with inp.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("error: empty CSV", file=sys.stderr)
            return 1
        in_header = list(reader.fieldnames)
        rows = list(reader)

    kept_cols = [c for c in KEEP_COLUMNS if c in set(in_header)]
    dropped = len(in_header) - len(kept_cols)
    missing = [c for c in KEEP_COLUMNS if c not in set(in_header)]

    print(f"[info] input cols={len(in_header)}  kept={len(kept_cols)}  dropped={dropped}")
    if missing:
        print(f"[warn] {len(missing)} expected columns not found in input (will be absent in output):")
        for c in missing:
            print(f"  {c}")

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kept_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out)

    print(f"[info] wrote {out} ({len(rows)} rows, {len(kept_cols)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
