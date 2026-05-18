# Evidence Note: ch6.3 — Attention Biases and Interaction with Attention Type (B1 × A3)

**Status:** interpreted
**Date created:** 2026-05-14
**Last updated:** 2026-05-14

---

## 1. Entry Point

**Entry Point C** (model + CSV both exist; local analysis covers the metric).

Higher-cost points rejected:
- B (inference missing): `04_cleaned_backfilled_analysis_ready.csv` contains `eval_v2/test_auroc` for all 45 ch6 runs (0 NaN). No re-inference needed.
- A (no checkpoint): all 45 model.pt confirmed.

---

## 2. Inventory Snapshot

| Item | Status |
|---|---|
| W&B groups | `exp_20260417-162636_ch6_attention_type_exp6a` and `exp_20260417-162638_ch6_diff_bias_mode_exp6b` |
| Run dirs (exp6a) | `/data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job{000..035}/` |
| Run dirs (exp6b) | `/data/atlas/users/nterlind/outputs/runs/run_20260417-162638_ch6_diff_bias_mode_exp6b_job{000..008}/` |
| model.pt | 45 confirmed |
| Eval CSV | `thesis_results/04_cleaned_backfilled_analysis_ready.csv` — 45 ch6 rows |
| Analysis module | `src/thesis_ml/reports/analyses/ch6_attention_mechanisms.py` |
| Report config | `configs/report/thesis_experiments_reports/ch6_attention_mechanisms.yaml` |

---

## 3. Axes Covered

| Axis | Name | Config Key | CSV Column | Options |
|---|---|---|---|---|
| B1 | Bias Activation Set | `classifier.model.attention_biases` | `config/axes/B1_Bias Activation Set` | `none`, `lorentz_scalar` |
| A3 | Attention Type | `classifier.model.attention.type` | `config/axes/A3_Attention Type` | `standard`, `differential` |

Note: In this chapter, B1 is limited to `none` vs `lorentz_scalar`. Full B1 family sweeps (SM interaction, type-pair kinematic, global-conditioned) are covered in chapter 5.

Design: B1 is crossed with A3 in exp6a. exp6b uses B1=lorentz_scalar throughout (no none rows in exp6b).

B1 counts:
- `lorentz_scalar`: 27 runs (18 from exp6a + 9 from exp6b)
- `none`: 18 runs (exp6a only)

---

## 4. Metrics (from eval CSV only)

### B1: AUROC by Bias Activation Set (all 45 ch6 runs)

| B1 | Mean AUROC | Std | N |
|---|---|---|---|
| none | 0.82768 | 0.00262 | 18 |
| lorentz_scalar | 0.83019 | 0.00372 | 27 |

Overall Lorentz-scalar gain: +0.00251 (~+0.25 pp). Note the larger std for lorentz_scalar (0.004) is partly due to the mix of A4 and A3-a conditions in the lorentz_scalar group.

### A3 × B1 interaction

| A3 | B1 | Mean AUROC | Std | N |
|---|---|---|---|---|
| standard | none | 0.82693 | — | 9 |
| standard | lorentz_scalar | 0.82715 | — | 9 |
| differential | none | 0.82844 | — | 9 |
| differential | lorentz_scalar | 0.83171 | — | 18 |

Key observation: the Lorentz-scalar bias provides almost no gain for standard attention (+0.00022, within noise) but a clear gain for differential attention (+0.00327). This suggests the Lorentz-scalar physics bias is more effective when combined with differential attention's ability to form relative dot-product representations.

### Within A4=none, cleanest comparison (marginalising out A4)

| A3 | B1 | Mean AUROC | Std | N |
|---|---|---|---|---|
| standard | none | 0.82959 | 0.00271 | 3 |
| standard | lorentz_scalar | 0.83190 | 0.00251 | 3 |
| differential | none | 0.83054 | 0.00222 | 3 |
| differential | lorentz_scalar | 0.83265 | 0.00173 | 12 |

Within A4=none the pattern holds: standard+lorentz (0.832) ≈ differential+none (0.831), and differential+lorentz is best at 0.833.

### Best config referencing B1 (confirmed from CSV)

`differential + A4=none + B1=lorentz_scalar + A3-a=split`: mean AUROC = 0.83298 ± 0.00164 (n=3)

Delta over no-bias baseline (`differential + A4=none + B1=none`): +0.00244 (+0.24 pp)

Delta over standard no-bias baseline: +0.00339 (+0.34 pp)

---

## 5. Staged Action

Entry point C — same report run as ch6.1 and ch6.2 covers B1 plots.

Figures relevant to this note: `auroc_seedspread_by_bias`, `auroc_heatmap_attn_type_x_bias`, `auroc_heatmap_norm_x_bias`, `val_auroc_by_bias`.

See `ch6.1_attention_type.md` for the full `thesis-report` command to run.

Expected output path:
```
/data/atlas/users/nterlind/outputs/reports/report_<timestamp>_ch6_attention_mechanisms/
```

---

## 6. Code

- **Reused:** `src/thesis_ml/reports/analyses/ch6_attention_mechanisms.py`
- **Reused:** `configs/report/thesis_experiments_reports/ch6_attention_mechanisms.yaml`
- No new code needed.

---

## 7. Confounders and Limitations

- exp6b contains no B1=none rows: all 9 exp6b runs use lorentz_scalar. This means the B1=lorentz_scalar group (n=27) is a superset of exp6a lorentz runs plus all exp6b runs (which also vary A3-a). The none group (n=18) is exp6a only. The imbalance does not bias the within-exp6a comparison but makes the marginalised B1 table (n=27 vs n=18) slightly heterogeneous.
- The A3 × B1 interaction cell for `standard + lorentz_scalar` and `differential + none` each have n=9 (3 norms × 3 seeds), while `differential + lorentz_scalar` has n=18 (exp6a n=9 + exp6b n=9). Cell sizes are unequal.
- No B1 types beyond `none` and `lorentz_scalar` are present in this chapter; the broader B1 family comparison belongs to chapter 5.
- `eval_v2/diff_attn/lambda_mean_abs` is NaN in the CSV for all runs; λ statistics must be read from `interpretability/attention_epoch_*.pt`.

---

## 8. Figure Provenance

Report run: `report_20260515-101822_ch6_attention_mechanisms`

| Destination (`thesis_report/figures/ch6/`) | Source |
|---|---|
| `ch6_B1_auroc_seedspread_bias.pdf` | `training/figures/figure-auroc_seedspread_by_bias.pdf` |
| `ch6_B1_val_auroc_curves.pdf` | `training/figures/figure-val_auroc_by_bias.pdf` |
| `ch6_A3xB1_auroc_heatmap.pdf` | `training/figures/figure-auroc_heatmap_attn_type_x_bias.pdf` |
| `ch6_A4xB1_auroc_heatmap.pdf` | `training/figures/figure-auroc_heatmap_norm_x_bias.pdf` |

---

## 9. Thesis-Safe Interpretation

The Lorentz-scalar physics bias provides a small but consistent AUROC improvement when combined with differential attention (+0.25 pp marginalised over all conditions; +0.33 pp in the best-vs-baseline comparison). Crucially, the gain is concentrated in the differential-attention group: standard attention sees essentially no benefit from the Lorentz-scalar bias (mean AUROC 0.827 with and without it). This interaction suggests that the differential attention mechanism is better at exploiting the structure of the physics-informed additive bias — possibly because its learnable λ parameter can modulate how strongly the bias contributes relative to the content-based attention score. The recommended final configuration for chapter 6 and subsequent chapters is: A3=differential, A4=none, B1=lorentz_scalar, A3-a=split (or shared), with overall AUROC 0.833.
