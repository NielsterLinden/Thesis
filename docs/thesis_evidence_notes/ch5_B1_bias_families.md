# Evidence Note: ch5 — B1 Bias Activation Set (Physics-Informed Attention Biases)

**Status: triaged**
**Chapter:** 5
**Section:** 5 — Physics-Informed Attention Biases
**Created:** 2026-05-13
**Last updated:** 2026-05-13

> The thesis chapter for this material is Chapter 5. The W&B group prefix
> `ch5_*` matches that numbering — there is no naming mismatch.

---

## 1. Inventory Snapshot (data-inventory)

All 102 Ch5 runs are present and evaluated in
`thesis_results/04_cleaned_backfilled_analysis_ready.csv`. All 102 `model.pt`
checkpoints exist on disk under `/data/atlas/users/nterlind/outputs/runs/`.

| Sub-experiment | W&B group | Rows | model.pt | mean AUROC | Purpose |
|---|---|---|---|---|---|
| 5A bias families | `exp_20260511-144128_ch5_bias_families` | 18 | 18 ✓ | ≈ 0.8410 | Compare 4 bias families (+ none + all-four-combined) |
| 5B Lorentz part 1 | `exp_20260511-144127_ch5_lorentz_p1` | 30 | 30 ✓ | ≈ 0.842 | Lorentz B1-L sub-axes (features, MLP, hidden dim) |
| 5B Lorentz part 2 | `exp_20260511-150429_ch5_lorentz_p2` | 30 | 30 ✓ | ≈ 0.842 | Lorentz B1-L sub-axes (per-head, sparse gating) |
| 5C Type-pair part 1 | `exp_20260511-193832_ch5_typepair_p1` | 3 | 3 ✓ | ≈ 0.842 | Type-pair init / freeze (B1-T1, B1-T2) |
| 5C Type-pair part 2 | `exp_20260511-200334_ch5_typepair_p2` | 12 | 12 ✓ | ≈ 0.842 | Type-pair gate / feature / mask (B1-T3..T5) |
| 5D SM mode | `exp_20260511-214641_ch5_sm_mode` | 9 | 9 ✓ | ≈ 0.841 | SM-interaction mode (B1-S1) |
| **Total** | — | **102** | **102 ✓** | **0.838–0.846 range** | — |

Run-dir verification commands used:

```bash
find /data/atlas/users/nterlind/outputs/runs/run_*_ch5_bias_families_job*/ -name model.pt | wc -l  # 18
find /data/atlas/users/nterlind/outputs/runs/run_*_ch5_lorentz_p1_job*/    -name model.pt | wc -l  # 30
find /data/atlas/users/nterlind/outputs/runs/run_*_ch5_lorentz_p2_job*/    -name model.pt | wc -l  # 30
find /data/atlas/users/nterlind/outputs/runs/run_*_ch5_typepair_p1_job*/   -name model.pt | wc -l  # 3
find /data/atlas/users/nterlind/outputs/runs/run_*_ch5_typepair_p2_job*/   -name model.pt | wc -l  # 12
find /data/atlas/users/nterlind/outputs/runs/run_*_ch5_sm_mode_job*/       -name model.pt | wc -l  # 9
```

The Hydra-managed sweep directories under
`/data/atlas/users/nterlind/outputs/multiruns/exp_20260511-*_ch5_*/` contain
only `multirun.yaml`; per-run artifacts (including `model.pt`) live under the
sibling `runs/run_*_job*/` directories — this is the project default
(`hydra.sweep.subdir` points back to `runs/`).

**Entry point chosen: C** for the AUROC bar charts (5A/5B/5C/5D). The eval CSV
already holds the per-run AUROC values; no inference re-run is needed for the
top-level metric comparison.

**Entry point E** is queued for the interpretability follow-ups (attention
maps for Exp 5A, KAN-spline visualisations for Exp 5B, type-pair heatmaps for
Exp 5C) — those require loading `model.pt` and bespoke analysis code, but
they go in separate evidence notes (see Section 9).

Higher-cost entry points rejected:
- **D** rejected: there is no existing report directory under
  `/data/atlas/users/nterlind/outputs/reports/` that aggregates the cleaned
  `04_cleaned_backfilled_analysis_ready.csv` Ch5 cohort with the V2 axis
  schema. The pre-2026-05 report directories used the legacy axes.
- **B** rejected: 102/102 `model.pt` present and 102/102 eval rows present.
- **A** rejected: 5A spans all required B1 families with 3 seeds each.

---

## 2. Axes Covered

Primary axis under study (5A): **B1 — Bias Activation Set**. Sub-experiments
sweep one B1 sub-family axis at a time:

| Axis ID | Name | Values Swept (this cohort) | Config Key | CSV Column |
|---|---|---|---|---|
| B1 | Bias Activation Set | `none`, `lorentz_scalar`, `typepair_kinematic`, `sm_interaction`, `global_conditioned`, `lorentz_scalar+typepair_kinematic+sm_interaction+global_conditioned` | `classifier.model.attention_biases` | `config/axes/B1_Bias Activation Set` |
| B1-L1 | Lorentz Feature Set | swept in 5B | `classifier.model.bias_config.lorentz_scalar.features` | `config/axes/B1-L1_Lorentz Feature Set` |
| B1-L2 | Lorentz MLP Type | swept in 5B | `classifier.model.bias_config.lorentz_scalar.mlp_type` | `config/axes/B1-L2_Lorentz MLP Type` |
| B1-L3 | Lorentz Hidden Dimension | swept in 5B | `classifier.model.bias_config.lorentz_scalar.hidden_dim` | `config/axes/B1-L3_Lorentz Hidden Dimension` |
| B1-L4 | Lorentz Per-Head Mode | swept in 5B | `classifier.model.bias_config.lorentz_scalar.per_head` | `config/axes/B1-L4_Lorentz Per-Head Mode` |
| B1-L5 | Lorentz Sparse Gating | swept in 5B | `classifier.model.bias_config.lorentz_scalar.sparse_gating` | `config/axes/B1-L5_Lorentz Sparse Gating` |
| B1-T1..T5 | Type-pair sub-axes | swept in 5C | `classifier.model.bias_config.typepair_kinematic.*` | `config/axes/B1-T*_*` |
| B1-S1 | SM Interaction Mode | swept in 5D | `classifier.model.bias_config.sm_interaction.mode` | `config/axes/B1-S1_SM Interaction Mode` |
| R5 | Seed | 42, 123, 314 | `classifier.trainer.seed` | `config/axes/R5_Seed` |

Source: `docs/AXES_REFERENCE_V2.md` rows 444–559, 773–787.

Metric column: **`eval_v2/test_auroc`** (single confirmed AUROC column in the
cleaned CSV). Secondary metrics also available:
`eval_v2/auroc_at_low_fpr`, `eval_v2/auroc_at_high_tpr`,
`eval_v2/per_class_auroc_json`.

---

## 3. What Was Held Fixed (Confounders + Fixed-Baseline NaN Issue)

From `configs/classifier/experiment/thesis_experiments/db_completeness/ch5_physics_bias_rerun/exp5a_bias_families.yaml`:

| Axis (V2) | Value | Source |
|---|---|---|
| D02 — MET treatment | `True` (`classifier.globals.include_met: true`) | YAML `classifier.globals.include_met` |
| D03 — Token ordering | `input_order` (project default; not overridden) | default |
| H01 — Embedding dim | 128 (`classifier.model.dim`) | YAML |
| H02 — Depth | 3 (`classifier.model.depth`) | YAML |
| H03 — Heads | 4 (`classifier.model.heads`) | YAML |
| H04 — MLP dim | 256 (`classifier.model.mlp_dim`) | YAML |
| H05 — Dropout | 0.1 (`classifier.model.dropout`) | YAML |
| H10 — Total params | ≈ derived from above (see `eval_v2/num_parameters_*`) | derived |
| T1 — Tokenizer family | `identity` (default) | default + `tokenizer.name: identity` |
| T1-a — PID embedding mode | `learned` | YAML `tokenizer.pid_mode: learned` |
| T1-b — PID embedding dim | 8 | YAML `tokenizer.id_embed_dim: 8` |
| E1 — Positional encoding | `sinusoidal` | YAML `positional: sinusoidal` |
| B1-G1 — Global-cond mode (only when `global_conditioned ∈ B1`) | `met_direction` | YAML `bias_config.global_conditioned.mode` |
| R1 — Trainer epochs / batch | 50 / 1024 | YAML |
| Data — signal vs background | 1 vs {2,3,4,5} | YAML |
| Data — selected_labels | `null` (all events kept) | YAML |
| Cohort tag | `db_completeness_2026_05`, sweep `ch5_physics_bias_rerun` | YAML |

**Fixed-baseline NaN issue.** For all 102 Ch5 rows in
`04_cleaned_backfilled_analysis_ready.csv`, the columns capturing the *fixed*
baseline axis values are `NaN`:

- `config/axes/D2_MET Treatment` → recoverable from YAML: `True`
- `config/axes/D3_Token Ordering` → recoverable from default: `input_order`
- `config/axes/E1_Positional Encoding` → recoverable from YAML: `sinusoidal`
- `config/axes/H1_Embedding Dimension` → 128
- `config/axes/H2_Depth` → 3
- `config/axes/H3_Number of Heads` → 4
- `config/axes/H4_MLP Dimension` → 256
- `config/axes/H5_Dropout` → 0.1
- `config/axes/H10_Total Parameters` → derivable from `eval_v2/num_parameters_total`
- `config/axes/R1_Trainer Type` → recoverable from default
- `config/axes/T1-a_PID Embedding Mode` → `learned`
- `config/axes/T1-b_PID Embedding Dimension` → 8

These are not bugs in the runs themselves — they are an export gap in the
W&B → CSV pipeline (the `axes/*` keys for *fixed* axes were not always
written when the axis was not swept). All values are recoverable from the
Hydra config. Any analysis script that filters Ch5 rows by these fixed
columns must inject the values from the YAML rather than relying on the CSV.

The B1-family swept-axis columns (`config/axes/B1_Bias Activation Set` and
its sub-axes) are populated for the corresponding sub-experiment.

---

## 4. Run Group(s) and Checkpoint Paths

Checkpoint root pattern:
`/data/atlas/users/nterlind/outputs/runs/run_<TS>_<group-name>_job<NNN>/model.pt`

Concrete groups (and `<TS>` prefix to glob):
- 5A: `run_20260511-144128_ch5_bias_families_job{000..017}/model.pt` — 18
- 5B p1: `run_20260511-144127_ch5_lorentz_p1_job{000..029}/model.pt` — 30
- 5B p2: `run_20260511-150429_ch5_lorentz_p2_job{000..029}/model.pt` — 30
- 5C p1: `run_20260511-193832_ch5_typepair_p1_job{000..002}/model.pt` — 3
- 5C p2: `run_20260511-200334_ch5_typepair_p2_job{000..011}/model.pt` — 12
- 5D: `run_20260511-214641_ch5_sm_mode_job{000..008}/model.pt` — 9

Total: **102 / 102 model.pt confirmed on disk.**

---

## 5. Action Staged

### 5.1 Exp 5A AUROC bar chart (this turn — DONE)

**Entry point: C.** Local interactive-node run; no Condor.

Script: `scripts/one_off/ch5_bias_families_plots.py`
Run command (re-runnable):

```bash
/data/atlas/users/nterlind/venvs/thesis-ml/bin/python3 \
    scripts/one_off/ch5_bias_families_plots.py
```

Inputs:
- `thesis_results/04_cleaned_backfilled_analysis_ready.csv`
- Filter: `meta_run/group == "exp_20260511-144128_ch5_bias_families"`
- Metric: `eval_v2/test_auroc`
- Grouping: `config/axes/B1_Bias Activation Set` (6 levels × 3 seeds = 18 rows)

Outputs (under
`/data/atlas/users/nterlind/outputs/reports/report_ch5_bias_families/`):

| File | Description |
|---|---|
| `figure-auroc_bar_by_bias_family.pdf` | Seed-mean AUROC bar per B1 family (6 bars). Error bars: per-cell seed standard deviation (n=3). Individual seed AUROC values overplotted as black scatter (alpha=0.5). Dashed reference line at the `none` baseline mean. |
| `exp5a_auroc_summary.csv` | Per-B1 aggregate table: B1 value, mean, std, count. |

Error-bar choice: **per-cell standard deviation across 3 seeds**. This is
indicative, not a confidence interval — see Section 8.

### 5.2 Reused vs new code

- **Reused:** thesis style system (`thesis_ml.reports.plots.style` —
  `apply_thesis_style`, `figure_size`, `axis_color`).
- **New:** `scripts/one_off/ch5_bias_families_plots.py` (small, ≈ 140 lines).
  No existing analysis module under `src/thesis_ml/reports/analyses/` fits
  this exact axis-group comparison cleanly; the ch4 precedent
  (`scripts/one_off/ch4_final_plots.py`) is the same pattern.

---

## 6. Result Summary (from `04_cleaned_backfilled_analysis_ready.csv`)

### 6.1 Exp 5A — B1 Bias Activation Set (n=3 seeds per cell)

| B1 value | mean AUROC | std (seed, n=3) | Δ vs `none` |
|---|---|---|---|
| `none` (baseline) | 0.84125 | 0.00068 | 0 |
| `lorentz_scalar` | 0.84156 | 0.00029 | +0.00031 |
| `typepair_kinematic` | 0.84155 | 0.00080 | +0.00030 |
| `sm_interaction` | 0.84080 | 0.00136 | −0.00045 |
| `global_conditioned` | 0.84170 | 0.00091 | +0.00045 |
| all-four combined | 0.83937 | 0.00090 | −0.00188 |

All deltas from baseline are within ~2× the per-cell seed standard
deviation. The all-four combined cell is the only one whose mean sits
visibly below the `none` baseline.

### 6.2 Exp 5B / 5C / 5D — placeholders

Sub-axis-level aggregates are deferred to the dedicated evidence notes
(Section 9). Group-level means are in the inventory table above:
all in the 0.838–0.846 range.

---

## 7. Plot Paths

- `/data/atlas/users/nterlind/outputs/reports/report_ch5_bias_families/figure-auroc_bar_by_bias_family.pdf`

No figure imported to `thesis_report/figures/ch5/` yet — pending user review.
Once approved, use the `figure-import` skill to copy into the LaTeX tree.

---

## 8. Confounders / Limitations

- **n=3 seeds per cell.** Per-cell stds are ~0.0003–0.0014 AUROC. The
  inter-cell range of mean AUROC is also ~0.002. The bias-family effect is
  therefore on the same order as seed variance and should not be over-stated
  in the thesis prose without a larger seed budget.
- **All-four-combined cell:** ~0.0019 below baseline. This is suggestive of
  optimisation interference (too many extra parameters / multiple gating
  signals competing) but cannot be cleanly attributed without per-bias
  ablations at matched parameter counts. The 5B/5C/5D sub-experiments help
  but do not fully isolate the interaction effect.
- **Fixed-baseline NaN issue (see Section 3).** Any filtering or grouping
  that touches the fixed columns will silently drop or misclassify Ch5 rows.
- **Single seed list reused across all sub-experiments** (42, 123, 314).
  Seed effects across sub-experiments are correlated, so cross-experiment
  std pooling is not statistically clean.
- **No matched-parameter control.** 5A bias families add bias-specific
  parameters (Lorentz / type-pair / SM / global-cond), so an apparently null
  effect at fixed encoder dims is consistent with either "the bias is
  uninformative" or "the bias is informative but the encoder already learned
  the same information." Distinguishing these requires the Ch5 interpretability
  work (attention maps, KAN splines) — entry point E, separate notes.

---

## 9. Next Steps — Follow-up Evidence Notes (to be authored next)

Sketches only; **not yet created.** Will live alongside this note under
`docs/thesis_evidence_notes/ch5_*.md`.

| Planned file | Scope | Entry point | Primary outputs |
|---|---|---|---|
| `ch5_B1L_lorentz.md` | Exp 5B (60 runs). B1-L sub-axes: features, MLP type, hidden dim, per-head, sparse gating. AUROC bar(s) + KAN spline visualisations + gate-value histograms. | C (bars) + E (splines / gates) | per-B1-L bar charts, KAN spline panels, gate histograms |
| `ch5_B1T_typepair.md` | Exp 5C (15 runs). B1-T sub-axes: init, freeze, gate, feature, mask. AUROC bar + 21×21 type-pair learned-bias heatmap. | C (bars) + E (heatmap) | per-B1-T bar, type-pair heatmap, init-vs-freeze comparison |
| `ch5_B1S_sm_mode.md` | Exp 5D (9 runs). B1-S1 sub-axis sweep. AUROC bar. | C | per-B1-S1 bar |
| `ch5_attention_maps.md` | Exp 5A interpretability. Per-bias attention-pattern visualisations from the 18 5A checkpoints; head-level inspection of where each bias family redistributes attention vs `none`. | E | per-bias attention heatmaps (test event subset), comparison panels |

---

## 10. Thesis-Safe Interpretation

> Across the six bias-activation regimes tested in Exp 5A (`none`, four
> single-family biases, and all-four combined), the seed-mean test AUROC
> spans only ~0.002 (0.8394–0.8417) with per-cell seed-standard deviations
> of 0.0003–0.0014. None of the four single-family biases moves the
> seed-mean AUROC by more than approximately one inter-seed standard
> deviation away from the `none` baseline (0.84125 ± 0.00068, n=3). The
> all-four-combined configuration is the only cell whose mean (0.83937)
> sits clearly below the baseline, a deficit of about 0.002 AUROC, which
> is consistent with mild optimisation interference once multiple bias
> sources compete for the same attention signal. On the AUROC axis alone,
> the physics-informed biases therefore neither help nor hurt the
> 4-tops-vs-background classifier appreciably at this model scale (128-dim,
> depth-3, 4-head transformer) and seed budget (n=3). Whether this null
> AUROC result reflects redundancy with information the unbiased encoder
> already extracts, or insensitivity of the AUROC metric to where in the
> attention pattern that information is encoded, is best addressed by the
> 5A interpretability follow-ups (attention maps) and by the sub-family
> sub-axis sweeps in 5B/5C/5D, which probe the bias-family internals at
> finer granularity.
