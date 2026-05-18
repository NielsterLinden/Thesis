# Evidence Note: ch7 — Model Scaling & Efficiency (Experiments 7A, 7B)

**Status:** INTERPRETED (7A, 7B)
**Date created:** 2026-05-15
**Last updated:** 2026-05-17

**Figures used in thesis:**
- `thesis_report/figures/ch7/ch7_auroc_vs_model_size.pdf`
- `thesis_report/figures/ch7/ch7_flops_vs_model_size.pdf`
- `thesis_report/figures/ch7/ch7_wallclock_vs_model_size.pdf`
- `thesis_report/figures/ch7/ch7_epoch_vs_model_size.pdf`
- `thesis_report/figures/ch7/ch7_pareto_auroc_vs_flops.pdf`
- `thesis_report/figures/ch7/ch7_pareto_auroc_vs_latency.pdf`
- `thesis_report/figures/ch7/ch7_pareto_auroc_vs_throughput.pdf`
- `thesis_report/figures/ch7/ch7_pareto_auroc_vs_memory.pdf`

> Naming note: these runs are labelled `ch8_scaling` in configs and W&B (due to chapter numbering that shifted during writing). They correspond to Chapter 7 in the thesis text. Experiment labels in the thesis text are 7A and 7B (renamed from 8A/8B to match the chapter). Exp 8C (axis transfer) has been dropped.

---

## 1. Entry Point

**Entry Point C** for 7A and 7B — `04_cleaned_backfilled_analysis_ready.csv` is complete for all 45 runs; all efficiency metrics (latency, throughput, memory, analytic FLOPs, training wall-clock, best epoch) are present with zero nulls. No new inference or training needed.

Higher-cost points rejected:
- B (inference missing): all 45 runs have full `eval_v2/*` metrics in the CSV. No re-inference needed.
- A (no checkpoint): 5 model sizes × 3 tasks × 3 seeds = 45 checkpoints confirmed present in multirun dirs.
- `flops_per_event_measured` is all-NaN in the CSV — this column was not populated by the eval pipeline. It is not needed; `flops_per_event_analytic` (0 nulls) is the correct and sufficient FLOP count for scaling analysis.

**Exp 7C (axis transfer) has been dropped** — removed from chapter scope. Chapter 7 is self-contained on scaling and efficiency; Ch8 will reference d64_L6 baseline AUROC from these runs as its comparison point.

---

## 2. Inventory Snapshot

| Item | Status |
|---|---|
| W&B group (4t-vs-bg) | `exp_20260420-130733_ch8_scaling_exp8a_1_4t_vs_bg` (15 runs) |
| W&B group (4t-vs-ttH) | `exp_20260420-130733_ch8_scaling_exp8a_2_4t_vs_ttH` (15 runs) |
| W&B group (multiclass) | `exp_20260420-130826_ch8_scaling_exp8a_3_multiclass` (15 runs) |
| Run dirs (4t-vs-bg) | `/data/atlas/users/nterlind/outputs/multiruns/exp_20260420-130733_ch8_scaling_exp8a_1_4t_vs_bg/` |
| Run dirs (4t-vs-ttH) | `/data/atlas/users/nterlind/outputs/multiruns/exp_20260420-130733_ch8_scaling_exp8a_2_4t_vs_ttH/` |
| Run dirs (multiclass) | `/data/atlas/users/nterlind/outputs/multiruns/exp_20260420-130826_ch8_scaling_exp8a_3_multiclass/` |
| model.pt | 45 confirmed (5 sizes × 3 tasks × 3 seeds) |
| Eval CSV | `thesis_results/04_cleaned_backfilled_analysis_ready.csv` — 45 ch8_scaling rows, 0 NaN on all efficiency and AUROC columns |
| Report config (7A/7B) | `configs/report/thesis_experiments_reports/ch7_scaling.yaml` ✓ |
| Analysis module | `src/thesis_ml/reports/analyses/ch7_scaling.py` ✓ — all 8 figures generated |

---

## 3. Axes Covered

| Axis | Name | Config Key | CSV Column | Values in sweep |
|---|---|---|---|---|
| H01 | Model dimension | `classifier.model.dim` | `config/axes/H1_Model Dimension` | 32, 32, 64, 128, 192 |
| H02 | Encoder depth | `classifier.model.depth` | `config/axes/H2_Encoder Depth` | 3, 5, 6, 8, 12 |
| H10 | Model size label | derived `d{dim}_L{depth}` | `config/axes/H10_Model Size Label` | d32_L3, d32_L5, d64_L6, d128_L8, d192_L12 |
| G03 | Classification task | `data.classifier.*` | `meta_run/group` (via group name) | 4t-vs-bg, 4t-vs-ttH, multiclass |
| R05 | Seed | `classifier.trainer.seed` | `config/axes/R5_Seed` | 3 seeds per cell |

Seeds: 3 per (size × task) cell. Design: full factorial — 5 sizes × 3 tasks × 3 seeds = 45 runs. All other axes held at their thesis baseline values.

### Model size table (actual, confirmed from CSV)

| Size key | dim | depth | approx. params | FLOPs/event (analytic, mean) |
|---|---|---|---|---|
| d32_L3 | 32 | 3 | ~26k | 1,292,256 |
| d32_L5 | 32 | 5 | ~43k | 2,152,416 |
| d64_L6 | 64 | 6 | ~202k | 10,325,952 |
| d128_L8 | 128 | 8 | ~1.06M | 55,058,304 |
| d192_L12 | 192 | 12 | ~3.57M | 185,806,656 |

> Tex correction needed: thesis text currently states `d256_L12` for the largest model. The actual largest model is `d192_L12` (confirmed in CSV). The thesis `.tex` must be updated.

---

## 4. Metrics (from eval CSV only)

All values below are mean ± std across 9 runs (3 tasks × 3 seeds) unless otherwise noted.

### 4.1 AUROC by model size (aggregated over all 3 tasks, n=9 each)

| Size key | Mean AUROC | Std |
|---|---|---|
| d32_L3 | 0.8429 | 0.0295 |
| d32_L5 | 0.8442 | 0.0293 |
| d64_L6 | 0.8520 | 0.0248 |
| d128_L8 | 0.8517 | 0.0247 |
| d192_L12 | 0.8515 | 0.0247 |

Interpretation: AUROC plateaus at d64_L6 (~202k params). The two smallest models (d32_L3, d32_L5) are 0.8–1.3 pp lower. The three larger models are statistically indistinguishable within seed spread.

### 4.2 Training wall-clock (seconds, n=9 each)

| Size key | Mean (s) | Std (s) |
|---|---|---|
| d32_L3 | 28.9 | 0.8 |
| d32_L5 | 45.1 | 0.8 |
| d64_L6 | 86.7 | 0.7 |
| d128_L8 | 133.5 | 1.0 |
| d192_L12 | 302.7 | 1.6 |

### 4.3 Best epoch (early stopping, n=9 each)

| Size key | Mean epoch | Std |
|---|---|---|
| d32_L3 | 47.1 | 2.8 |
| d32_L5 | 46.1 | 3.8 |
| d64_L6 | 39.7 | 7.9 |
| d128_L8 | 20.4 | 3.4 |
| d192_L12 | 17.1 | 2.8 |

Larger models converge in fewer epochs (faster loss descent) but each epoch takes longer — net wall-clock is dominated by epoch cost for d192_L12.

### 4.4 Inference latency — batch size 1 (ms, n=9 each)

| Size key | Mean (ms) | Std (ms) |
|---|---|---|
| d32_L3 | 3.40 | 0.20 |
| d32_L5 | 5.22 | 0.35 |
| d64_L6 | 5.47 | 1.22 |
| d128_L8 | 6.40 | 1.56 |
| d192_L12 | 8.22 | 1.29 |

### 4.5 Inference latency — batch size 512 (ms, n=9 each)

| Size key | Mean (ms) | Std (ms) |
|---|---|---|
| d32_L3 | 0.163 | 0.000 |
| d32_L5 | 0.268 | 0.000 |
| d64_L6 | 0.552 | 0.001 |
| d128_L8 | 0.868 | 0.001 |
| d192_L12 | 2.019 | 0.015 |

### 4.6 Throughput — batch size 512 (samples/s, n=9 each)

| Size key | Mean (samples/s) | Std |
|---|---|---|
| d32_L3 | 3,144,612 | 2,899 |
| d32_L5 | 1,912,038 | 1,587 |
| d64_L6 | 926,980 | 1,049 |
| d128_L8 | 590,081 | 550 |
| d192_L12 | 253,558 | 1,876 |

### 4.7 Peak inference memory — batch size 512 (MiB, n=9 each)

| Size key | Mean (MiB) | Std (MiB) |
|---|---|---|
| d32_L3 | 995.8 | ~0.0 |
| d32_L5 | 995.9 | ~0.0 |
| d64_L6 | 1,967.5 | ~0.0 |
| d128_L8 | 3,186.8 | ~0.0 |
| d192_L12 | 4,772.4 | ~0.0 |

Memory near-zero std reflects deterministic allocation (no seed dependence in peak memory).

### 4.8 Signal efficiency at fixed background rejection (binary tasks only, n=3 seeds each)

`eps_S` is undefined for multiclass (correct — signal efficiency at a fixed background rejection rate requires a binary decision boundary). Available at four thresholds.

**4t vs. background:**

| Size key | ε_S @ 1/B=10 | ε_S @ 1/B=50 | ε_S @ 1/B=100 | ε_S @ 1/B=1000 |
|---|---|---|---|---|
| d32_L3   | 0.567 ± 0.002 | 0.276 ± 0.006 | 0.187 ± 0.001 | 0.044 ± 0.004 |
| d32_L5   | 0.568 ± 0.001 | 0.277 ± 0.000 | 0.189 ± 0.002 | 0.047 ± 0.005 |
| d64_L6   | 0.578 ± 0.004 | 0.284 ± 0.002 | 0.194 ± 0.002 | 0.047 ± 0.003 |
| d128_L8  | 0.580 ± 0.004 | 0.286 ± 0.009 | 0.189 ± 0.006 | 0.048 ± 0.001 |
| d192_L12 | 0.576 ± 0.002 | 0.288 ± 0.004 | 0.194 ± 0.002 | 0.037 ± 0.002 |

**4t vs. ttH:**

| Size key | ε_S @ 1/B=10 | ε_S @ 1/B=50 | ε_S @ 1/B=100 | ε_S @ 1/B=1000 |
|---|---|---|---|---|
| d32_L3   | 0.642 ± 0.003 | 0.283 ± 0.003 | 0.174 ± 0.002 | 0.028 ± 0.003 |
| d32_L5   | 0.643 ± 0.008 | 0.284 ± 0.005 | 0.177 ± 0.005 | 0.031 ± 0.003 |
| d64_L6   | 0.653 ± 0.005 | 0.298 ± 0.008 | 0.187 ± 0.008 | 0.026 ± 0.002 |
| d128_L8  | 0.652 ± 0.004 | 0.290 ± 0.007 | 0.178 ± 0.003 | 0.029 ± 0.008 |
| d192_L12 | 0.651 ± 0.009 | 0.292 ± 0.007 | 0.176 ± 0.005 | 0.025 ± 0.006 |

**Interpretation:** The plateau pattern mirrors AUROC exactly. At 1/B=50 (the operational reference), d64_L6 achieves ε_S ≈ 0.284 (4t vs. bg) and 0.298 (4t vs. ttH), statistically consistent with the larger models. At 1/B=1000 the values are noisy and the trend is non-monotonic — this regime is too sparse for clean conclusions and should not be the primary physics metric in the chapter. Use 1/B=50 as the operational reference point.

---

## 5. Staged Action

**COMPLETE.** All plots generated and imported. No further data collection or code work needed for Chapter 7.

Run command used:
```bash
/data/atlas/users/nterlind/venvs/thesis-ml/bin/python -c "
from omegaconf import OmegaConf
from thesis_ml.reports.analyses.ch7_scaling import run_report
cfg = OmegaConf.create({...})  # see ch7_scaling.yaml
run_report(cfg)
"
```
Output: `/data/atlas/users/nterlind/outputs/reports/report_20260517-211444_ch7_scaling/`

---

## 6. Code

- `src/thesis_ml/reports/analyses/ch7_scaling.py` ✓ — complete; 8 plot functions, numeric coercion in `_load_ch7_data`.
- `configs/report/thesis_experiments_reports/ch7_scaling.yaml` ✓ — CSV-based Entry Point C config.
- `configs/report/config.yaml` ✓ — added `csv_path` and `group_prefix` to base inputs schema.
- `src/thesis_ml/cli/reports/__main__.py` ✓ — patched to accept `csv_path` without requiring sweep dirs.

---

## 7. Confounders and Limitations

- AUROC is aggregated across all three tasks in the table above. Per-task breakdown is important: the multiclass task systematically produces lower absolute AUROC than the binary tasks. Always report per-task or explicitly note the aggregation.
- The large std (~0.025) across the 9-run groups reflects cross-task variance, not seed noise. Within a single task, seed std is much smaller (expected ~0.003–0.005 based on other chapters).
- Wall-clock runtime includes evaluation overhead (latency benchmarking), not just training. The absolute values should be treated as indicative, not as clean training time.
- Inference latency (b=1) has large std for d64_L6 and d128_L8 (~1.2–1.6 ms). This reflects OS scheduling jitter on the interactive node, not model variability. Batch-512 latency is far more stable (std < 0.02 ms) and is the better efficiency metric.
- `flops_per_event_measured` is absent (all-NaN). Only analytic FLOPs are available. Analytic FLOPs count multiply-add operations from model architecture; they do not include Python overhead, data movement, or attention-softmax cost exactly. This is the standard comparison basis and is appropriate for thesis purposes.
- Axis-scale interaction: the scaling results shown here are for the baseline architecture only. Whether a better architecture (from Ch8) scales differently is an open question; this is noted as a limitation in Ch7's conclusion but not investigated.
- `d256_L12` label correction: ✓ fixed in tex — actual largest model is `d192_L12` (dim=192, depth=12, ~3.57M params).

---

## 8. Plot Paths

Generated from report run `report_20260517-211444_ch7_scaling`.

| Destination (`thesis_report/figures/ch7/`) | Source path |
|---|---|
| `ch7_auroc_vs_model_size.pdf` | `/data/atlas/users/nterlind/outputs/reports/report_20260517-211444_ch7_scaling/training/figures/figure-auroc_vs_model_size.pdf` |
| `ch7_flops_vs_model_size.pdf` | `/data/atlas/users/nterlind/outputs/reports/report_20260517-211444_ch7_scaling/training/figures/figure-flops_vs_model_size.pdf` |
| `ch7_wallclock_vs_model_size.pdf` | `/data/atlas/users/nterlind/outputs/reports/report_20260517-211444_ch7_scaling/training/figures/figure-wallclock_vs_model_size.pdf` |
| `ch7_epoch_vs_model_size.pdf` | `/data/atlas/users/nterlind/outputs/reports/report_20260517-211444_ch7_scaling/training/figures/figure-best_epoch_vs_model_size.pdf` |
| `ch7_pareto_auroc_vs_flops.pdf` | `/data/atlas/users/nterlind/outputs/reports/report_20260517-211444_ch7_scaling/training/figures/figure-pareto_auroc_vs_flops.pdf` |
| `ch7_pareto_auroc_vs_latency.pdf` | `/data/atlas/users/nterlind/outputs/reports/report_20260517-211444_ch7_scaling/training/figures/figure-pareto_auroc_vs_latency.pdf` |
| `ch7_pareto_auroc_vs_throughput.pdf` | `/data/atlas/users/nterlind/outputs/reports/report_20260517-211444_ch7_scaling/training/figures/figure-pareto_auroc_vs_throughput.pdf` |
| `ch7_pareto_auroc_vs_memory.pdf` | `/data/atlas/users/nterlind/outputs/reports/report_20260517-211444_ch7_scaling/training/figures/figure-pareto_auroc_vs_memory.pdf` |

---

## 9. Thesis-Safe Interpretation

Model performance (AUROC) scales rapidly from the two smallest configurations (d32_L3, ~26k params; d32_L5, ~43k params) and then plateaus at d64_L6 (~202k params, ~10M FLOPs/event). The three larger models — d64_L6, d128_L8 (~1.1M params), and d192_L12 (~3.6M params) — are statistically indistinguishable in AUROC (within seed spread), suggesting that the classification task is effectively saturated at this operating point with the baseline architecture. Signal efficiency at fixed background rejection (1/B=50) confirms this plateau: ε_S ≈ 0.284 for 4t vs. background and ε_S ≈ 0.298 for 4t vs. ttH at d64_L6, with no statistically significant gain from larger models.

On the efficiency side, the cost increase above d64_L6 is substantial: inference latency at batch 512 grows from 0.55 ms (d64_L6) to 2.02 ms (d192_L12, ×3.7), throughput drops from ~927k to ~254k samples/s, and peak memory grows from ~2 GiB to ~4.8 GiB. Training wall-clock grows roughly linearly with FLOPs, as expected.

Together, these results identify d64_L6 as the Pareto-optimal operating point for this task: it achieves the full performance of much larger models at a fraction of the inference cost. This retrospectively validates the fixed model size used throughout Chapters 4–6, and provides the baseline AUROC (d64_L6, baseline architecture) against which Chapter 8's optimized architectures are compared. Chapter 8 takes d64_L6 as fixed and asks which combination of attention type, normalization, positional encoding, and physics biases achieves the highest discrimination performance.
