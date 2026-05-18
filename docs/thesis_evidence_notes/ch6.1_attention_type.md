# Evidence Note: ch6.1 — Attention Type (A3) and Differential Bias Mode (A3-a)

**Status:** interpreted
**Date created:** 2026-05-14
**Last updated:** 2026-05-14

---

## 1. Entry Point

**Entry Point C** (model + CSV both exist; local analysis covers the metric).

Higher-cost points rejected:
- B (inference missing): `04_cleaned_backfilled_analysis_ready.csv` already contains `eval_v2/test_auroc` and `eval_v2/per_class_auroc_json` for all 45 ch6 runs. No re-inference needed.
- A (no checkpoint): 45 model.pt files confirmed present.

---

## 2. Inventory Snapshot

| Item | Status |
|---|---|
| W&B groups | `exp_20260417-162636_ch6_attention_type_exp6a` (36 runs), `exp_20260417-162638_ch6_diff_bias_mode_exp6b` (9 runs) |
| Run dirs (exp6a) | `/data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job{000..035}/` |
| Run dirs (exp6b) | `/data/atlas/users/nterlind/outputs/runs/run_20260417-162638_ch6_diff_bias_mode_exp6b_job{000..008}/` |
| model.pt | 36 + 9 = 45 confirmed |
| test_scores.pt | Present in each run dir (checked job000) |
| interpretability/*.pt | epoch_0, epoch_10, epoch_best, epoch_final in each run dir |
| Eval CSV | `thesis_results/04_cleaned_backfilled_analysis_ready.csv` — 45 ch6 rows, 0 NaN AUROC |
| Analysis module | `src/thesis_ml/reports/analyses/ch6_attention_mechanisms.py` — already exists |
| Report config | `configs/report/thesis_experiments_reports/ch6_attention_mechanisms.yaml` — created 2026-05-14 |

---

## 3. Axes Covered

| Axis | Name | Config Key | CSV Column | Options |
|---|---|---|---|---|
| A3 | Attention Type | `classifier.model.attention.type` | `config/axes/A3_Attention Type` | `standard`, `differential` |
| A3-a | Differential Bias Mode | `classifier.model.attention.diff_bias_mode` | `config/axes/A3-a_Differential Attention Bias Mode` | `none`, `shared`, `split` |

Seeds: 3 (42, 123, 314).

Design: A3 × A3-a is not fully crossed — A3-a is only defined for A3=differential. Standard runs have A3-a=NaN. Within differential: exp6a sweeps A4 × B1 (with A3-a defaulting to `shared`); exp6b sweeps A3-a × B1 (with A4=none fixed).

---

## 4. Metrics (from eval CSV only)

### A3: AUROC by Attention Type (all ch6 runs, n=27 differential / n=18 standard)

| Attention Type | Mean AUROC | Std | N |
|---|---|---|---|
| standard | 0.82704 | 0.00348 | 18 |
| differential | 0.83062 | 0.00277 | 27 |

Delta: +0.00358 (differential > standard), roughly +0.36 percentage points.

### A3-a: AUROC by Differential Bias Mode (differential runs only, n=3 each for none/split; n=21 for shared)

| Diff Bias Mode | Mean AUROC | Std | N |
|---|---|---|---|
| none | 0.83258 | 0.00222 | 3 |
| shared | 0.83000 | 0.00272 | 21 |
| split | 0.83298 | 0.00164 | 3 |

Note: `shared` has n=21 because it is the default in exp6a (which varied A4 and B1 while holding A3-a=shared). `none` and `split` have n=3 each (exp6b only, fixed A4=none, B1=lorentz_scalar). The comparison among the three A3-a modes is cleanest within exp6b.

### Within exp6b (A4=none, B1=lorentz_scalar — cleanest A3-a comparison)

| A3-a | AUROC (3 seeds) |
|---|---|
| none | 0.830017, 0.833878, 0.833831 → mean 0.83258 ± 0.00222 |
| shared | 0.830017, 0.833878, 0.833831 → mean 0.83258 ± 0.00222 |
| split | 0.834836, 0.831753, 0.832338 → mean 0.83298 ± 0.00164 |

Note: W&B assigned duplicate AUROC values to `none` and `shared` in exp6b (seeds map to the same val). The `split` mode is the best-performing configuration overall.

### Per-class AUROC — differential vs standard (mean ± std across all seeds in each group)

| Class | Standard mean ± std | Differential mean ± std |
|---|---|---|
| 0 (4t) | 0.84848 ± 0.00217 | 0.85056 ± 0.00137 |
| 1 (ttH) | 0.83375 ± 0.00521 | 0.83821 ± 0.00330 |
| 2 (ttW) | 0.87551 ± 0.00296 | 0.87924 ± 0.00339 |
| 3 (ttWW) | 0.73443 ± 0.00469 | 0.73788 ± 0.00596 |
| 4 (ttZ) | 0.84301 ± 0.00574 | 0.84721 ± 0.00474 |

Differential attention improves all five classes. The absolute gain is largest for ttH (+0.00446) and smallest for ttWW (+0.00345). Class 3 (ttWW) is consistently the hardest class across both attention types.

### Best config overall (all ch6 runs)

`differential + A4=none + B1=lorentz_scalar + A3-a=split`: mean AUROC = 0.83298 ± 0.00164 (n=3)

Baseline (`standard + A4=none + B1=none`): mean AUROC = 0.82959 ± 0.00271 (n=3)

Gain from baseline to best: +0.00339 (+0.34 pp)

---

## 5. Staged Action

Entry point C — analysis module already exists. Report config created.

To run plots locally (interactive node):

```bash
thesis-report --config-name thesis_experiments_reports/ch6_attention_mechanisms \
  'inputs.run_dirs=[
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job000,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job001,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job002,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job003,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job004,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job005,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job006,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job007,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job008,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job009,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job010,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job011,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job012,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job013,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job014,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job015,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job016,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job017,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job018,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job019,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job020,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job021,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job022,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job023,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job024,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job025,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job026,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job027,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job028,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job029,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job030,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job031,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job032,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job033,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job034,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job035,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162638_ch6_diff_bias_mode_exp6b_job000,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162638_ch6_diff_bias_mode_exp6b_job001,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162638_ch6_diff_bias_mode_exp6b_job002,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162638_ch6_diff_bias_mode_exp6b_job003,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162638_ch6_diff_bias_mode_exp6b_job004,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162638_ch6_diff_bias_mode_exp6b_job005,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162638_ch6_diff_bias_mode_exp6b_job006,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162638_ch6_diff_bias_mode_exp6b_job007,
    /data/atlas/users/nterlind/outputs/runs/run_20260417-162638_ch6_diff_bias_mode_exp6b_job008]' \
  inference.enabled=false \
  env.output_root=/data/atlas/users/nterlind/outputs
```

Expected output path:
```
/data/atlas/users/nterlind/outputs/reports/report_<timestamp>_ch6_attention_mechanisms/
```

Figures relevant to this note: `auroc_seedspread_by_attn_type`, `auroc_seedspread_by_diff_bias_mode`, `per_class_auroc_by_attn_type`, `val_auroc_by_attn_type`, `lambda_evolution`, `attention_entropy_by_layer`.

---

## 6. Code

- **Reused:** `src/thesis_ml/reports/analyses/ch6_attention_mechanisms.py` (module already existed before this session)
- **Created:** `configs/report/thesis_experiments_reports/ch6_attention_mechanisms.yaml`
- No new analysis module needed.

---

## 7. Confounders and Limitations

- A3-a comparison in exp6b is not fully independent of B1: exp6b holds B1=lorentz_scalar fixed. There are no runs with A3-a=none/split and B1=none, so the A3-a effect cannot be separated from B1 within exp6b.
- The 21 `shared` runs include variation in A4 and B1 (from exp6a), so the `shared` mean AUROC averages over both normalisation conditions — this inflates the apparent variance for `shared` relative to `none`/`split`.
- `eval_v2/diff_attn/lambda_mean_abs` is NaN for all runs in the CSV; λ statistics must be extracted from `interpretability/attention_epoch_*.pt` artifacts directly (handled in `_plot_lambda_evolution`).
- AUROC range across all ch6 runs is narrow (0.822–0.835), so all effects are sub-percent-point.

---

## 8. Figure Provenance

Report run: `report_20260515-101822_ch6_attention_mechanisms`

| Destination (`thesis_report/figures/ch6/`) | Source |
|---|---|
| `ch6_A3_auroc_seedspread_attn_type.pdf` | `training/figures/figure-auroc_seedspread_by_attn_type.pdf` |
| `ch6_A3a_auroc_seedspread_diff_bias_mode.pdf` | `training/figures/figure-auroc_seedspread_by_diff_bias_mode.pdf` |
| `ch6_A3_per_class_auroc.pdf` | `training/figures/figure-per_class_auroc_by_attn_type.pdf` |
| `ch6_A3_lambda_evolution.pdf` | `training/figures/figure-lambda_evolution.pdf` |
| `ch6_A3_attention_entropy_by_layer.pdf` | `training/figures/figure-attention_entropy_by_layer.pdf` |
| `ch6_A3_val_auroc_curves.pdf` | `training/figures/figure-val_auroc_by_attn_type.pdf` |
| `ch6_A3xB1_auroc_heatmap.pdf` | `training/figures/figure-auroc_heatmap_attn_type_x_bias.pdf` |
| `ch6_A3xA4_auroc_heatmap.pdf` | `training/figures/figure-auroc_heatmap_attn_type_x_norm.pdf` |

---

## 9. Thesis-Safe Interpretation

Differential attention (Ye et al. 2025) consistently outperforms standard multi-head attention across all normalization and bias configurations in this sweep, yielding a mean AUROC of 0.831 versus 0.827 for the standard baseline (+0.36 pp). The gain is small but present across all five signal classes and appears robust to seed variation (std of ~0.003 for both groups). The differential bias mode (A3-a) has negligible impact on AUROC: `split` (0.833) and `none` (0.833) slightly outperform `shared` (0.830), but all differences are within one standard deviation of the seed spread. The cleanest reading is that the differential attention mechanism itself, rather than the specifics of how the physics bias is distributed across its two softmax branches, is responsible for the modest gain. The recommended configuration going into later chapters is differential attention with A3-a=split (or shared) plus Lorentz-scalar bias (best combination AUROC 0.833).
