# Evidence Note: ch4.2 — Data Treatment (D02, D03)

**Status: interpreted**
**Chapter:** 4
**Section:** 4.2 — Data Treatment
**Created:** 2026-05-07
**Last updated:** 2026-05-10 (plots finalised)

---

## 1. Inventory Snapshot (data-inventory)

| Item | Value |
|---|---|
| W&B group — Exp 4B | `exp_20260410-163637_ch4_data_treatment_exp4b` |
| Run count | 24 runs (8 conditions × 3 seeds) |
| model.pt present | Yes (confirmed via report manifest — loaded in the same 57-model report) |
| `04_cleaned_backfilled_analysis_ready.csv` rows | 24 rows in group `exp_20260410-163637_ch4_data_treatment_exp4b` |
| Report directory | `/data/atlas/users/nterlind/outputs/reports/report_20260420-104926_ch4_best_input_repr/` |
| Plots present | Yes — see Section 6 |

**Entry point chosen: D** (Uniform report bundle). The existing report directory contains all relevant D02 and D03 figures. No new inference or training is required.

Higher-cost entry points rejected:
- Entry point C not needed: bar charts, seed-spread plots, and ROC curves for D02/D03 are already produced.
- Entry points B/A not applicable: all checkpoints and eval CSV rows are present.

---

## 2. Axes Covered

| Axis ID | Name | Values Swept | Config Key | axes key | W&B key |
|---|---|---|---|---|---|
| D02 | MET Treatment | `False`, `True` | `classifier.globals.include_met` | `include_met` | `axes/include_met` |
| D03 | Token Ordering | `input_order`, `shuffled` | `data.sort_tokens_by`, `data.shuffle_tokens` | `token_order` | `axes/token_order` |
| D01 | Feature Set | `[0,1,2,3]`, `[1,2,3]` | `data.cont_features` | `cont_features` | `axes/cont_features` |

Note: D01 is also covered in ch4.1; its Exp 4B values are included here for completeness since the Exp 4B sweep crosses D01 × D02 × D03 simultaneously.

**Primary CSV metric:** `eval_v2/test_auroc`

---

## 3. Baseline Tokenizer Configuration — Confirmed

**The Exp 4B baseline tokenizer configuration is confirmed from the CSV:**

| Key | Value |
|---|---|
| T1 (Tokenizer Family) | `identity` |
| T1-a (PID Mode) | `learned` |
| T1-b (PID Embed Dim) | `8` |

This is confirmed by reading `config/axes/T1_Tokenizer Family`, `config/axes/T1-a_PID Embedding Mode`, and `config/axes/T1-b_PID Embedding Dimension` directly from the 24 Exp 4B CSV rows. All 24 rows use `identity` + `learned` + `8`. The previously unconfirmed question of whether `one_hot` or `learned` was used as the baseline is resolved: **Exp 4B used `identity` + `learned` + `dim=8`** throughout.

---

## 4. What Was Held Fixed (Confounders)

All Exp 4B runs held fixed:
- **Tokenizer (T1):** `identity`, T1-a = `learned`, T1-b = `8` (see Section 3)
- **Pooling (C2):** `cls`
- **Positional encoding (E1):** `sinusoidal` (default) — this is critical context for D03, see Section 7
- **Attention biases (B1):** `none`
- **FFN (F1):** `standard`
- **Seeds (R05):** 42, 123, 314 for every cell
- **Model size:** `d64_L4` baseline

The 2 × 2 × 2 design sweeps D01 (`[0,1,2,3]` vs `[1,2,3]`) × D02 (`False` vs `True`) × D03 (`input_order` vs `shuffled`), giving 8 conditions × 3 seeds = 24 runs.

---

## 5. Result Summary (from `thesis_results/04_cleaned_backfilled_analysis_ready.csv`)

### 5.1 Full 2×2×2 factorial table

| D01 | D02 (MET) | D03 (Order) | mean AUROC | std | n |
|---|---|---|---|---|---|
| `[0,1,2,3]` | False | input_order | 0.8431 | 0.0007 | 3 |
| `[0,1,2,3]` | False | shuffled | 0.8398 | 0.0012 | 3 |
| `[0,1,2,3]` | True | input_order | 0.8434 | 0.0015 | 3 |
| `[0,1,2,3]` | True | shuffled | 0.8420 | 0.0008 | 3 |
| `[1,2,3]` | False | input_order | 0.8424 | 0.0010 | 3 |
| `[1,2,3]` | False | shuffled | 0.8406 | 0.0013 | 3 |
| `[1,2,3]` | True | input_order | 0.8436 | 0.0004 | 3 |
| `[1,2,3]` | True | shuffled | 0.8401 | 0.0021 | 3 |

### 5.2 D02 — MET Treatment marginal effect

Marginal comparison collapsing over D01 and D03:

| MET (D02) | mean AUROC | std | n |
|---|---|---|---|
| `False` (no MET) | 0.8415 | 0.0017 | 12 |
| `True` (include MET) | 0.8423 | 0.0019 | 12 |

Delta D02: +0.0008 (MET=True vs MET=False). This is smaller than one standard deviation across the 12-run pool; the effect is not statistically distinguishable from noise at this sample size.

Comparing only on input_order runs (where D03 does not introduce a confound):

| MET (D02) | D03 | mean AUROC | std | n |
|---|---|---|---|---|
| False | input_order | 0.8428 | 0.0008 | 6 |
| True | input_order | 0.8435 | 0.0010 | 6 |

Delta: +0.0007. Again within one standard deviation.

**Important note:** The key issue description mentions a −0.026 MET effect. The CSV data does NOT reproduce this. The observed direction is a small positive effect of MET inclusion (+0.0007 to +0.0008) or no effect, not a negative −0.026. The −0.026 figure may refer to a different experiment group, a different metric, or a previous sweep version. The thesis writer should use the values from this CSV, not the −0.026 figure, for any quantitative statement about Exp 4B.

### 5.3 D03 — Token Ordering marginal effect

Marginal comparison (D02=False only, to avoid MET interaction):

| Order (D03) | mean AUROC | std | n |
|---|---|---|---|
| `input_order` | 0.8428 | 0.0008 | 6 |
| `shuffled` | 0.8402 | 0.0012 | 6 |

Delta D03: −0.0026 (shuffled vs. input_order), with MET=False. This is approximately 2–3× the within-group standard deviation and is consistent across both D01 conditions.

With MET=True:

| Order (D03) | D01 | mean AUROC | std | n |
|---|---|---|---|---|
| input_order | `[0,1,2,3]` | 0.8434 | 0.0015 | 3 |
| shuffled | `[0,1,2,3]` | 0.8420 | 0.0008 | 3 |
| input_order | `[1,2,3]` | 0.8436 | 0.0004 | 3 |
| shuffled | `[1,2,3]` | 0.8401 | 0.0021 | 3 |

The ordering effect persists under MET=True.

---

## 6. Existing Plot Paths

All plots under `/data/atlas/users/nterlind/outputs/reports/report_20260420-104926_ch4_best_input_repr/inference/figures/`:

| Plot | Filename |
|---|---|
| AUROC bar chart by MET treatment | `figure-auroc_bar_by_met.png` |
| AUROC seed spread by MET | `figure-auroc_seedspread_by_met.png` |
| ROC curves by MET | `figure-roc_curves_by_met.png` |
| AUROC bar chart by token shuffle | `figure-auroc_bar_by_shuffle.png` |
| AUROC seed spread by shuffle | `figure-auroc_seedspread_by_shuffle.png` |
| ROC curves by shuffle | `figure-roc_curves_by_shuffle.png` |
| AUROC bar chart by feature set | `figure-auroc_bar_by_features.png` |

Training-phase figures under `training/figures/`:
- `figure-val_auroc_by_met.png`
- `figure-val_auroc_by_shuffle.png`
- `figure-val_auroc_by_features.png`

Per-run confusion matrices and score distributions for all 24 Exp 4B runs are in `inference/figures/` with prefix `figure-confusion_matrices_run_20260410-163637_ch4_data_treatment_exp4b_job*` and `figure-score_distributions_run_20260410-163637_ch4_data_treatment_exp4b_job*`.

---

## 7. Confounders, Hypotheses, and Known Limitations

### D02 — MET Treatment: causal mechanism is a hypothesis

The PE-disruption hypothesis — that including MET as a token disrupts the sinusoidal positional encoding because MET does not correspond to a well-defined particle position in the pt-sorted ordering — is a plausible post-hoc explanation for any observed degradation under MET=True. However, this hypothesis has NOT been tested. The CSV data from Exp 4B does not show MET hurting performance (the observed delta is +0.0007 to +0.0008, not negative). Any statement attributing a MET performance effect to PE disruption in the thesis must be clearly labelled as a hypothesis, not a measured result. The mechanism would require a dedicated experiment comparing MET=True with E1=none (no PE) to isolate the PE contribution, which has not been run.

### D03 — Token Ordering: sinusoidal PE attribution requires E1=none control

The observed ordering effect (input_order outperforms shuffled by ~0.003) is consistent with the sinusoidal PE providing useful positional information when tokens are presented in a consistent order. However, the attribution of this effect to sinusoidal PE specifically requires a control experiment with E1=none (no positional encoding): if D03 still matters under E1=none, the effect must be attributed to something other than PE (e.g. the transformer's residual stream attending to token-position-correlated features). This E1=none control has NOT been run. The thesis writer should state the observed ordering effect as a measured result but hedge the PE attribution explicitly.

### Baseline tokenizer for Exp 4B

The Exp 4B baseline is confirmed as T1=identity, T1-a=learned, T1-b=8. This means the D02/D03 results are obtained with a specific tokenizer choice. It cannot be assumed that the same D02/D03 effects hold for the raw or binned tokenizer; those combinations were not swept in this experiment.

### Three seeds is a small replication

With n=3 seeds per cell, the standard errors are large relative to the observed deltas (~0.001–0.003). The D03 ordering effect at 0.0026 is suggestive but would require n≥6 per cell to approach 95% confidence with typical seed-spread variability.

---

## 8. Thesis-Safe Interpretation

Including MET in the event representation (D02=True) produces no measurable change in test AUROC in Exp 4B: across the 2×2×2 factorial design the difference between MET-included and MET-excluded conditions is at most +0.001, well within the seed-spread of individual cells. This result is inconsistent with any strong claim that MET is harmful or beneficial to classification performance at the baseline tokenizer configuration used here; the data simply do not resolve the question given the small seed count. Token ordering (D03) shows a more consistent pattern: input-order presentation outperforms shuffled tokens by approximately 0.003 AUROC (marginal over D01 and D02), suggesting the model makes use of consistent token ordering, possibly through the sinusoidal positional encoding. However, the PE-attribution is a hypothesis — an E1=none control experiment would be needed to confirm whether the ordering effect is mediated by PE or by some other mechanism. The feature-set comparison (D01: with vs. without energy) yields a difference of ~0.001, below the seed-spread level, indicating that energy is not a critical feature for this task at the tested model scale. Taken together, Exp 4B suggests that the identity tokenizer baseline is robust to moderate changes in data treatment, with token ordering being the largest of the three effects examined.

---

## 9. Staged Action and Final Plot Paths

Entry point D for original report. All reviewer-agreed fixes plus new style system applied 2026-05-10 via `scripts/one_off/ch4_final_plots.py` (entry point C, interactive node). Final figures land in:

**Final plots path:** `/data/atlas/users/nterlind/outputs/reports/report_ch4_final/figures/`

| Final figure file | LaTeX label | Thesis path | Key finding |
|---|---|---|---|
| `figure-auroc_bar_by_met.pdf` | `fig:auroc-bar-met` | `thesis_report/figures/ch4/figure-auroc_bar_by_met.pdf` | delta=+0.0008 (MET=True vs MET=False), source: CSV test_auroc |
| `figure-auroc_bar_by_shuffle.pdf` | `fig:auroc-bar-shuffle` | `thesis_report/figures/ch4/figure-auroc_bar_by_shuffle.pdf` | delta=−0.0026 (shuffled vs input_order), source: CSV test_auroc |
| `figure-auroc_bar_by_features.pdf` | `fig:auroc-bar-features` | `thesis_report/figures/ch4/figure-auroc_bar_by_features.pdf` | identity tokenizer only; null result (delta=−0.0007) |
| `figure-val_auroc_by_met.pdf` | (not in main body) | `thesis_report/figures/ch4/figure-val_auroc_by_met.pdf` | Style updated; MET curves show near-identical convergence |
| `figure-val_auroc_by_shuffle.pdf` | (not in main body) | `thesis_report/figures/ch4/figure-val_auroc_by_shuffle.pdf` | Style updated |

IMPORTANT: The original report's `inference/summary.json` showed wrong MET/shuffle values (~0.81 for
MET=True and shuffle=True conditions). Root cause: the re-run inference at report time had a data
loading bug where the val-split dataset did not correctly include the MET token for MET=True models,
causing near-random predictions. The `eval_v2/test_auroc` values in the CSV (stored at training time)
are correct. The final plots use CSV values only.

Note: `inference/summary.json` does store correct per-event scores and ROC curves per run (under
flat top-level run-name keys, with keys `per_event_scores`, `per_event_labels`, `roc_curves`). The
bug only affected the re-inference step in the report pipeline, not the stored summary data.

Script: `scripts/one_off/ch4_final_plots.py`. Re-run with:
```
/data/atlas/users/nterlind/venvs/thesis-ml/bin/python3 scripts/one_off/ch4_final_plots.py
```

If the PE-disruption hypothesis (D02 × E1) needs to be tested for the thesis argument, that would require a new training sweep (entry point A). Specifically, a 2×2 design crossing D02 ∈ {False, True} with E1 ∈ {sinusoidal, none} under otherwise identical conditions. This has not been proposed here because the current data does not establish a negative MET effect that would motivate the test; the thesis argument can be made without it by noting the null result and stating the PE hypothesis as untested.

---

## 10. Open Questions

- The −0.026 MET figure mentioned in the task description does not appear in the Exp 4B CSV data. Origin is unknown — it may be from an earlier sweep version or a different group. Do not use it without tracing its source.
- The D03 ordering effect attribution to sinusoidal PE requires an E1=none control (not run).
- Exp 4B does not include a `pt_sorted` token ordering condition; only `input_order` and `shuffled` are present. If pt-sorted is needed for the thesis, a new run would be required.
