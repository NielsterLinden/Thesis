# Evidence Note: ch4.1 — Tokenizer Family (T1, T1-a, T1-b, D01)

**Status: interpreted**
**Chapter:** 4
**Section:** 4.1 — Input Representation
**Created:** 2026-05-07
**Last updated:** 2026-05-10 (plots finalised)

---

## 1. Inventory Snapshot (data-inventory)

| Item | Value |
|---|---|
| W&B group — Exp 4A-1 raw | `exp_20260410-160141_ch4_input_repr_exp4a_1_raw` |
| W&B group — Exp 4A-1 binned | `exp_20260410-160937_ch4_input_repr_exp4a_1_binned` |
| W&B group — Exp 4A-2 | `exp_20260410-162037_ch4_input_repr_exp4a_2` |
| Run counts | 3 (raw) + 3 (binned) + 27 (identity) = 33 total |
| model.pt present | Yes (confirmed via report manifest — 57 models loaded) |
| 03_analysis_ready.csv rows | 33 rows across the three groups |
| Report directory | `/data/atlas/users/nterlind/outputs/reports/report_20260420-104926_ch4_best_input_repr/` |
| Plots present | Yes — see Section 6 |

**Entry point chosen: D** (Uniform report bundle). The report directory at the path above contains all required plots and the `thesis_results/03_analysis_ready.csv` holds all metric values. No new inference or training is required to answer the thesis question for this note.

Higher-cost entry points rejected:
- Entry point C not needed: existing report module already produced all summary bar charts and heatmaps.
- Entry points B/A not applicable: all checkpoints and eval CSVs are present.

---

## 2. Axes Covered

| Axis ID | Name | Values Swept | Config Key | axes key | W&B key |
|---|---|---|---|---|---|
| T1 | Tokenizer Family | `raw`, `binned`, `identity` | `classifier.model.tokenizer.name` | `tokenizer_name` | `axes/tokenizer_name` |
| T1-a | PID Embedding Mode | `learned`, `one_hot`, `fixed_random` | `classifier.model.tokenizer.pid_mode` | `pid_mode` | `axes/pid_mode` |
| T1-b | PID Embedding Dimension | `8`, `16`, `32` (+ `num_types` for one_hot) | `classifier.model.tokenizer.id_embed_dim` | `id_embed_dim` | `axes/id_embed_dim` |
| D01 | Feature Set | `[0,1,2,3]` (all), `[1,2,3]` (no energy) | `data.cont_features` | `cont_features` | `axes/cont_features` |

**Primary CSV metric:** `eval_v2/test_auroc`
**Secondary metrics also available:** `eval_v2/test_acc`, `eval_v2/test_f1`, `eval_v2/eps_S_at_invB_10`, `eval_v2/eps_S_at_invB_50`, `eval_v2/eps_S_at_invB_100`

---

## 3. What Was Held Fixed (Confounders)

All runs in this note used:
- **Pooling (C2):** `cls` for all runs
- **Positional encoding (E1):** `sinusoidal` (default) — NOT swept in this chapter
- **Attention biases (B1):** `none`
- **FFN (F1):** `standard`
- **MET (D02):** `False`
- **Token order (D03):** `input_order` (Exp 4A-2 sweep; raw/binned Exp 4A-1 also use `input_order`)
- **Seeds (R05):** 42, 123, 314 — all three seeds used for every cell (3 replicas per configuration)
- **Model size:** `d64_L4` baseline for all runs (H01=64, H02=4)

**Known confounder — learned + dim=8 pooled mean inflation:** The `learned` PID mode has n=33 rows because it appears in BOTH Exp 4A-2 (27 rows) AND Exp 4B (an additional 6 rows for D01=[0,1,2,3], input_order from the exp4b group). When reporting the `learned` mean for this section, use **only the 27 Exp 4A-2 rows** (or the 3-seed cells within Exp 4A-2) rather than the pooled 33-row figure. The dimension=8 cell within `learned` has n=27 in the full pool but only n=3 in the Exp 4A-2-only cut, because the swept cells (dim=8 across D01/D03 conditions) overlap with Exp 4B entries. The thesis writer should use per-cell means, not the pooled `learned` mean.

**Known confounder — one_hot apparent triplicate:** The 9 CSV rows for `one_hot` in Exp 4A-2 consist of 3 groups of 3 seeds with identical axis values and numerically identical AUROC values (confirmed: same seed + same config = same result due to deterministic eval). These are 9 distinct W&B run IDs (`rr4xstyx`, `fl0c638i`, `ktdyao75`, `vsh4ftk1`, `ns12iw81`, `jfnhmtzl`, `mhzy0596`, `eqb600nm`, `7lkiw11s`) but only 3 unique (seed, config) combinations. Use n=3 for `one_hot` (one per seed) when computing the per-cell mean; the 9-row pooled mean is numerically identical but inflates the stated sample size.

---

## 4. Result Summary (from `thesis_results/03_analysis_ready.csv`)

### 4.1 T1 — Tokenizer Family (primary comparison)

| Tokenizer (T1) | mean AUROC | std | n (seeds) | Group |
|---|---|---|---|---|
| `raw` | 0.8044 | 0.0011 | 3 | exp4a_1_raw |
| `binned` | 0.7246 | 0.0019 | 3 | exp4a_1_binned |
| `identity` (all cells pooled) | 0.8422 | 0.0019 | 51* | exp4a_2 + exp4b overlap |

*See confounder note above. For clean per-cell comparison, use the exp4a_2 cells below.

### 4.2 T1-a — PID Embedding Mode (identity tokenizer, Exp 4A-2 only, D01=[0,1,2,3], D02=False, D03=input_order)

| PID Mode (T1-a) | dim (T1-b) | mean AUROC | std | n |
|---|---|---|---|---|
| `learned` | 8 | 0.8431 | 0.0007 | 3 |
| `learned` | 16 | 0.8428 | 0.0006 | 3 |
| `learned` | 32 | 0.8402 | 0.0004 | 3 |
| `one_hot` | num_types | 0.8442 | 0.0006 | 3* |
| `fixed_random` | 8 | 0.8435 | 0.0009 | 3 |
| `fixed_random` | 16 | 0.8400 | 0.0020 | 3 |
| `fixed_random` | 32 | 0.8397 | 0.0004 | 3 |

*one_hot has 9 CSV rows (triplicate) — unique AUROC values are 0.8438, 0.8451, 0.8438 across seeds 42, 123, 314.

### 4.3 T1-a marginal summary (all dims pooled, Exp 4A-2)

| PID Mode | mean AUROC | std | n |
|---|---|---|---|
| `learned` | 0.8420 | 0.0017 | 27 |
| `one_hot` | 0.8442 | 0.0006 | 9* |
| `fixed_random` | 0.8410 | 0.0021 | 9 |

### 4.4 D01 — Feature Set (identity tokenizer, Exp 4B, D02=False, D03=input_order)

| Feature Set (D01) | mean AUROC | std | n |
|---|---|---|---|
| `[0,1,2,3]` (E + pT + eta + phi) | 0.8431 | 0.0007 | 3 |
| `[1,2,3]` (pT + eta + phi, no energy) | 0.8424 | 0.0010 | 3 |

Delta D01: −0.0007 (no energy vs. with energy). Difference is within one standard deviation; no strong evidence that energy contributes at this model scale.

---

## 5. Output Path and Artifact Verification

**Report directory:** `/data/atlas/users/nterlind/outputs/reports/report_20260420-104926_ch4_best_input_repr/`

Confirmed present (verified 2026-05-07):
- `manifest.yaml` — 57 models loaded
- `inference/summary.json` — 51 MB
- `inference/figures/` — plots present (see Section 6)
- `training/summary.csv` — 20 KB
- `training/figures/` — plots present

---

## 6. Existing Plot Paths

All plots under `/data/atlas/users/nterlind/outputs/reports/report_20260420-104926_ch4_best_input_repr/inference/figures/`:

| Plot | Filename |
|---|---|
| AUROC bar chart by tokenizer family | `figure-auroc_bar_by_tokenizer.png` |
| AUROC seed spread by tokenizer | `figure-auroc_seedspread_by_tokenizer.png` |
| ROC curves by tokenizer | `figure-roc_curves_by_tokenizer.png` |
| AUROC bar chart by PID mode | `figure-auroc_bar_by_pid_mode.png` |
| AUROC bar chart by embed dim | `figure-auroc_bar_by_id_embed_dim.png` |
| AUROC heatmap (PID mode x embed dim) | `figure-auroc_heatmap_pid_mode_x_embed_dim.png` |
| AUROC bar chart by features | `figure-auroc_bar_by_features.png` |
| Failure analysis raw vs identity | `figure-failure_analysis_raw_vs_identity.png` |
| Global metrics comparison | `figure-metrics_comparison.png` |
| Raw ROC all tokenizers | `figure-roc_curves.png` |

Training-phase figures under `training/figures/`:
- `figure-val_auroc_by_tokenizer.png`
- `figure-val_auroc_by_pid_mode.png`
- `figure-val_auroc_by_id_embed_dim.png`
- `figure-val_auroc_by_features.png`

Per-run confusion matrices and score distributions are also present for all 33 runs.

---

## 7. Thesis-Safe Interpretation

Across the three tokenizer families tested, the identity tokenizer (particle-type embedding + projection MLP) consistently outperforms both the raw kinematic tokenizer and the binned histogram tokenizer on the 4-tops vs. background classification task. The identity tokenizer achieves a mean test AUROC of approximately 0.842 (dim≥16 cells) compared to 0.804 for raw and 0.725 for binned, with the binned result showing a substantial degradation that is consistent with the histogram discretisation discarding fine-grained kinematic structure. Within the identity family, the choice of PID embedding mode (learned, one-hot, fixed random) has only a small effect on AUROC — differences are at most ~0.003 and largely within seed-to-seed variance — suggesting that the model is not strongly sensitive to whether particle-type information is trained or fixed, at least at this model scale and dataset size. The PID embedding dimension (T1-b) shows a similar pattern: the default dimension of 8 performs comparably to 16 or 32, with no monotonic improvement trend. The D01 feature-set comparison (with vs. without energy) yields a difference of only ~0.001, which is below the seed-spread level; the absence of the energy feature does not measurably harm performance in this experiment, though a causal interpretation requires caution because the model may partially reconstruct energy-correlated quantities from pT and angles. All quantitative conclusions above should be treated as indicative given the small number of seeds (n=3 per cell); wider seed sweeps would be required before reporting tight confidence intervals.

---

## 8. Staged Action and Final Plot Paths

Entry point D — original plots produced by the existing report run. All reviewer-agreed fixes plus new style system applied 2026-05-10 via `scripts/one_off/ch4_final_plots.py` (entry point C, interactive node). Final figures land in:

**Final plots path:** `/data/atlas/users/nterlind/outputs/reports/report_ch4_final/figures/`

All figures generated with `thesis_ml.reports.plots.style` (`apply_thesis_style()`, `figure_size()`, `axis_color()`, `CATEGORICAL_COLORS`). Format: PDF at 300 DPI for all figures.

Note: `inference/summary.json` stores `per_event_scores`, `per_event_labels`, and `roc_curves` per run as flat top-level keys (not nested under `'runs'`). Both the ROC curve and failure analysis plots are produced directly from this data — no new inference pass required.

| Final figure file | LaTeX label | Thesis path | Format | Fix applied |
|---|---|---|---|---|
| `figure-auroc_bar_by_tokenizer.pdf` | `fig:auroc-bar-tokenizer` | `thesis_report/figures/ch4/figure-auroc_bar_by_tokenizer.pdf` | PDF | Fix 1: identity dim>=16 only (n=21) |
| `figure-auroc_bar_by_pid_mode.pdf` | `fig:auroc-bar-pid-mode` | `thesis_report/figures/ch4/figure-auroc_bar_by_pid_mode.pdf` | PDF | Fix 2: dim>=16 filter applied |
| `figure-auroc_heatmap_pid_mode_x_embed_dim.pdf` | `fig:heatmap-pid-dim` | `thesis_report/figures/ch4/figure-auroc_heatmap_pid_mode_x_embed_dim.pdf` | PDF | Style updated |
| `figure-val_auroc_by_tokenizer.pdf` | `fig:val-auroc-tokenizer` | `thesis_report/figures/ch4/figure-val_auroc_by_tokenizer.pdf` | PDF | Fix 4: x-axis 0–30 |
| `figure-roc_curves_by_tokenizer.pdf` | `fig:roc-tokenizer` | `thesis_report/figures/ch4/figure-roc_curves_by_tokenizer.pdf` | PDF | Per-seed ROC curves + mean band from summary.json `roc_curves['1']` |
| `figure-val_auroc_by_pid_mode.pdf` | (appendix / not in main body) | `thesis_report/figures/ch4/figure-val_auroc_by_pid_mode.pdf` | PDF | Style updated |
| `figure-failure_analysis_raw_vs_identity.pdf` | `fig:failure-analysis` | `thesis_report/figures/ch4/figure-failure_analysis_raw_vs_identity.pdf` | PDF | Two-panel scatter+bar from real per-event scores in summary.json |

Note: `auroc_bar_by_id_embed_dim` (fig:auroc-bar-dim) is deliberately NOT generated — removed per reviewer decision (redundant given heatmap).

Script: `scripts/one_off/ch4_final_plots.py`. Re-run with:
```
/data/atlas/users/nterlind/venvs/thesis-ml/bin/python3 scripts/one_off/ch4_final_plots.py
```
Interactive node only, no GPU needed, completes in ~15 s.

---

## 9. Verified Failure Analysis Counts

From `inference/summary.json`, pairing best raw run (`run_20260410-160141_ch4_input_repr_exp4a_1_raw_job002`, AUROC=0.8056) against best identity run (`run_20260410-162037_ch4_input_repr_exp4a_2_job010`, one_hot, AUROC=0.8451). Threshold=0.5, total test events=30,207.

| Category | Count | Fraction |
|---|---|---|
| Both correct | 20,289 | 67.2% |
| PID helped (identity fixed raw) | 2,713 | 9.0% |
| PID hurt (raw fixed identity) | 1,809 | 6.0% |
| Both wrong | 5,396 | 17.9% |

Net gain from identity over raw: +904 events (+3.0%). These numbers supersede the counts in the tex commentary (20,121 / 2,515 / 2,014 / 5,557), which were unverified placeholders.

---

## 10. Limitations and Open Questions

- Seeds: only 3 seeds per cell. Seed-spread bars are indicative but bootstrap CI would require more replicas.
- The `one_hot` triplicate in the CSV (9 rows, 3 unique) should be deduplicated before any statistical aggregation.
- T1-b (embed dim) was only swept for `learned` and `fixed_random`; `one_hot` has `T1-b = num_types` (fixed by architecture), so the T1-b plot is not a clean grid.
- D01 was swept only for the identity tokenizer in Exp 4B; the raw and binned tokenizers have not been tested without energy features.
- No `pretrained` (T1-c) tokenizer was included in these groups. That comparison is deferred.
