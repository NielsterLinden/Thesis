# Evidence Note: ch6.2 — Attention Internal Normalization (A4)

**Status:** run-complete
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
| W&B group | `exp_20260417-162636_ch6_attention_type_exp6a` (36 runs; A4 swept here) |
| Run dirs | `/data/atlas/users/nterlind/outputs/runs/run_20260417-162636_ch6_attention_type_exp6a_job{000..035}/` |
| model.pt | 36 confirmed for exp6a |
| Eval CSV | `thesis_results/04_cleaned_backfilled_analysis_ready.csv` — 45 ch6 rows |
| Analysis module | `src/thesis_ml/reports/analyses/ch6_attention_mechanisms.py` |
| Report config | `configs/report/thesis_experiments_reports/ch6_attention_mechanisms.yaml` |

Note: A4 is only varied in exp6a (36 runs). exp6b (9 runs) fixes A4=none.

---

## 3. Axes Covered

| Axis | Name | Config Key | CSV Column | Options |
|---|---|---|---|---|
| A4 | Attention-Internal Normalization | `classifier.model.attention.norm` | `config/axes/A4_Attention Internal Normalization` | `none`, `layernorm`, `rmsnorm` |

Design: A4 is crossed with A3 (both attention types) and B1 (both bias options) in exp6a.
- `none`: 21 runs (A3=differential×12 + A3=standard×6, plus 3 from exp6b)
- `layernorm`: 12 runs (6 differential + 6 standard)
- `rmsnorm`: 12 runs (6 differential + 6 standard)

Seeds: 3 per cell.

---

## 4. Metrics (from eval CSV only)

### A4: AUROC by Attention Internal Normalization (all 45 ch6 runs)

| A4 | Mean AUROC | Std | N |
|---|---|---|---|
| none | 0.83181 | 0.00221 | 21 |
| layernorm | 0.82736 | 0.00232 | 12 |
| rmsnorm | 0.82643 | 0.00319 | 12 |

`none` outperforms both normalisation modes. LayerNorm and RMSNorm are nearly identical, with RMSNorm slightly lower.

### A3 × A4 interaction (exp6a only)

| A3 | A4 | Mean AUROC | Std | N |
|---|---|---|---|---|
| standard | none | 0.83075 | — | 6 |
| standard | layernorm | 0.82603 | — | 6 |
| standard | rmsnorm | 0.82434 | — | 6 |
| differential | none | 0.83223 | — | 15 |
| differential | layernorm | 0.82870 | — | 6 |
| differential | rmsnorm | 0.82851 | — | 6 |

The pattern is consistent: A4=none is best for both attention types. The gap from none to normalised is larger for standard (~0.005–0.006 pp) than for differential (~0.004 pp).

### A4 × B1 interaction

| A4 | B1 | Mean AUROC | Std | N |
|---|---|---|---|---|
| none | lorentz_scalar | 0.83250 | — | 15 |
| none | none | 0.83006 | — | 6 |
| layernorm | lorentz_scalar | 0.82851 | — | 6 |
| layernorm | none | 0.82622 | — | 6 |
| rmsnorm | lorentz_scalar | 0.82608 | — | 6 |
| rmsnorm | none | 0.82677 | — | 6 |

Lorentz-scalar bias helps when A4=none and when A4=layernorm, but not when A4=rmsnorm (where it slightly hurts). This is a weak and non-systematic interaction; differences are within one seed-spread std.

---

## 5. Staged Action

Entry point C — same report run as ch6.1 covers A4 plots.

Figures relevant to this note: `auroc_seedspread_by_attn_norm`, `val_auroc_by_attn_norm`, `auroc_heatmap_attn_type_x_norm`, `auroc_heatmap_norm_x_bias`.

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

- A4 is not crossed with A3-a: exp6b (which varies A3-a) holds A4=none fixed. The A4 effect on differential+split or differential+none is therefore untested.
- The A4=none group has more runs (21) than layernorm or rmsnorm (12 each) because exp6b also uses A4=none. This makes the none group mean more stable but also means it includes the exp6b A3-a=split/none runs which tend to be slightly higher (AUROC ~0.833). This could marginally inflate the `none` mean relative to a fully balanced comparison.
- All AUROC differences (~0.005 pp) are within the seed-spread standard deviation for any given group. The result is directional but should not be overstated.

---

## 8. Thesis-Safe Interpretation

Applying LayerNorm or RMSNorm inside the attention module (A4) consistently reduces AUROC compared to using no normalisation (A4=none), across both attention types and both bias conditions. The mean AUROC for A4=none is 0.832, versus 0.827 for LayerNorm and 0.826 for RMSNorm — a ~0.005 absolute gap (~0.5 pp). The effect is small relative to the spread across seeds but is consistent in direction across all four (A3 × B1) sub-groups. A plausible explanation is that attention-internal normalisation disrupts the raw dot-product scale that the Lorentz-scalar bias is calibrated against, or that it introduces unnecessary smoothing in a regime where the model is not particularly deep (6 layers). The recommended setting is A4=none going forward; LayerNorm and RMSNorm appear to offer no benefit and a small cost in this architecture.
