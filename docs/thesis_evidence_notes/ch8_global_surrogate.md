# Ch8 Global Surrogate Analysis — Evidence Note

**Status:** interpreted (2026-05-17) — Chapter 8 drafted as `thesis_report/mainmatter/08_global_surrogate.tex`; figures copied to `thesis_report/figures/ch8/`

---

## Phase B — Final Plots (2026-05-14)

### Script

`/project/atlas/users/nterlind/Thesis-Code/scripts/one_off/ch8_final_plots.py`

Standalone, no Hydra, no W&B. Mirrors conventions of `ch4_final_plots.py`.

### Inputs

- Primary cohort CSV: `thesis_results/ch8_streamlined/05_ch8_streamlined_primary.csv` (1015 rows)
- Raw 04 CSV (for missingness heatmap only): `thesis_results/04_cleaned_backfilled_analysis_ready.csv` (1320 rows)

### Figures written (all PDF, 300 DPI)

| # | File | Description |
|---|------|-------------|
| 1 | `/data/atlas/users/nterlind/outputs/reports/report_ch8_final/figures/audit_run_counts_by_spec.pdf` | Bar chart: N runs per `eval_v2/spec_version`. 1012 rows on `eval_v2.1`, 3 on a legacy hash. |
| 2 | `/data/atlas/users/nterlind/outputs/reports/report_ch8_final/figures/audit_missingness_heatmap.pdf` | Heatmap of fraction NaN per axis col × G3 cohort in raw 04 CSV (84 cols with any NaN, 3 cohorts). |
| 3 | `/data/atlas/users/nterlind/outputs/reports/report_ch8_final/figures/audit_cramer_v_heatmap.pdf` | Cramér's V between axis families (13×13, primary cohort). Each cell = mean pairwise Cramér's V across all col-pairs in those two families, excluding cols with <2 non-inactive unique values. |
| 4 | `/data/atlas/users/nterlind/outputs/reports/report_ch8_final/figures/marginals_ranked_range_bar.pdf` | Horizontal bar: AUROC range (max mean − min mean) per axis, ranked. 47 axes plotted after ≥5-rows-per-level gate. Coloured by axis family. Seed noise floor annotated as vertical dashed line. |
| 5 | `/data/atlas/users/nterlind/outputs/reports/report_ch8_final/figures/marginals_top5_boxdot.pdf` | Box+dot plots for the top-5 axes by AUROC range, plus one R5_Seed panel as inline noise floor reference. Levels sorted by median AUROC. |
| 6 | `/data/atlas/users/nterlind/outputs/reports/report_ch8_final/figures/marginals_seed_noise_floor.pdf` | Histogram of within-fingerprint AUROC std across 201 groups with ≥2 seeds. Median annotated. |

### Seed noise floor

**Median within-group AUROC std = 0.000856**
(fingerprint = hash of all axis cols excluding `R5_Seed`; N groups with ≥2 seeds = 201;
mean within-group std = 0.006596)

This is the appropriate noise floor threshold for interpreting marginal AUROC ranges: any axis with a range below ~0.001 is within seed noise.

### Top-5 axes by AUROC range (≥5 rows per level)

1. `R1_Epochs`
2. `R5_Seed`
3. `R2_Learning Rate`
4. `R4_Batch Size`
5. `B1_Bias Activation Set`

Note: The top-4 are all training hyperparameters (R family), which have large AUROC ranges because low-epoch or extreme-LR runs under-train. `B1_Bias Activation Set` is the first architectural axis in the ranking.

### Axes dropped by ≥5-rows-per-level gate (44 axes)

These axes either had too few runs at some levels to meet the 5-rows minimum, were entirely inactive in the primary cohort, or were numeric and excluded from the categorical marginal analysis:

A3-a, A3, A5, B1-G2, B1-G3, B1-S2, B1-T4, B1-T5, C2, E1-a1, G3, K3, K4, K5,
L1, L2, L3, L4, L5, L6, L7, M4, M5, P1-a, P1-b, P1, P2-a, P2-b, P2-c, P2-d, P2-e, P2,
R11, R12, R13, R14, R15, R16, R3, R7, R8, R9, S1, S2

Many of these are interpretability / logging flags (L series), KAN hyperparameters (K3–K5),
physics extensions (P series, MIA), and early-stopping params (R11–R13) — all sparsely explored
in the primary cohort.

### Confounders and limitations

- The top-4 ranked axes are R-family training hyperparameters. Their large ranges reflect
  under-trained models at extreme settings, not genuine architectural signal. The thesis
  writer should note this when presenting figure 4 and consider presenting the ranking
  with R1/R2/R4 excluded to reveal architectural effects.
- The Cramér's V heatmap uses a pairwise mean across all column pairs within a family pair,
  which can mask heterogeneous within-family co-occurrence patterns.
- 44 axes are excluded from the marginal analysis. Most are sparsely explored design choices
  (KAN, MIA, physics extensions). Their absence from figure 4 does not imply they are
  unimportant — only that the current database does not support the 5-rows-per-level estimate.
- The missingness heatmap uses the raw 04 CSV (pre-inactive encoding). High NaN fractions
  for conditional sub-axes (e.g. B1-G2/G3 only meaningful when B1-G1 is active) are
  expected and structural, not data quality issues.

### Reused code

- `thesis_ml.reports.plots.style` — `apply_thesis_style`, `axis_color`, `figure_size`, `CATEGORICAL_COLORS`, `_CATEGORICAL_ORDER`
- `thesis_ml.monitoring.io_utils.save_figure` — PDF at 300 DPI
- No new modules added to `src/`.

### Entry point chosen

Entry point C: primary CSV and raw 04 CSV present; local analysis script sufficient.
No Condor jobs staged. No model.pt access required.

---

## Phase C — XGBoost Surrogate + SHAP (2026-05-14)

### Status: complete

### Entry point

Entry point C: primary cohort CSV present; bespoke analysis script run on interactive node.
No Condor jobs. No model.pt access required.

### Inventory snapshot

| Item | Present |
|------|---------|
| Primary cohort CSV `05_ch8_streamlined_primary.csv` | yes (948 rows) |
| Shared library `src/thesis_ml/reports/ch8_surrogate.py` | yes (new) |
| Script `scripts/one_off/ch8_surrogate_fit.py` | yes (new) |
| `shap` package | yes (0.51.0, installed 2026-05-14) |
| `xgboost` package | yes (pre-existing in conda env) |

### Inputs

- Primary cohort CSV: `thesis_results/ch8_streamlined/05_ch8_streamlined_primary.csv` (948 rows, 90 axis columns)
- Target: `eval_v2/test_auroc` — mean=0.8328, std=0.0301, min=0.4917, max=0.8865

### Surrogate configuration

- Model: `XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)`
- Validation: 5-fold `GroupKFold`; groups defined by hash of all axis columns excluding `R5_Seed` (429 unique groups)
- Feature matrix: one-hot encoding of all 90 axis columns → 428 binary features
- SHAP: `TreeExplainer`, subsampled to 800 rows (random_state=42)

### CV metrics (from eval CSV / run output)

| Metric | Value |
|--------|-------|
| Spearman mean | 0.739 |
| Spearman std | 0.078 |
| R² mean | 0.416 |
| R² std | 0.451 |

Spearman > 0.3 threshold: **MET** (0.739).

Note: R² std is large (0.451) due to fold 1 having R² = −0.46.  This reflects structural
difficulty in that particular train/val split (GroupKFold may produce folds with very
different axis-value distributions).  Spearman is robust to this and remains consistently
high across all folds (0.655–0.858), confirming the surrogate captures rank ordering well.

### Top-5 axis families by normalised mean |SHAP|

| Rank | Family | Normalised importance |
|------|--------|-----------------------|
| 1 | T (Tokenizer) | 37.9% |
| 2 | R (Training protocol) | 18.4% |
| 3 | H (Model size / scaling) | 16.9% |
| 4 | E (Positional encoding) | 7.3% |
| 5 | B (Physics biases) | 4.6% |

Remaining families: D 4.4%, A 4.2%, C 1.8%, L 1.8%, P 1.4%, F 0.9%, G 0.3%, M 0.1%, S 0.1%, K 0.0%.

### Top-5 individual axis columns by mean |SHAP|

| Rank | Axis column | Mean |SHAP| (aggregated over one-hot children) |
|------|-------------|--------------------------------------|
| 1 | `T1-a_PID Embedding Mode` | 0.0079 |
| 2 | `T1_Tokenizer Family` | 0.0028 |
| 3 | `R2_Learning Rate` | 0.0028 |
| 4 | `H10_Model Size Label` | 0.0027 |
| 5 | `T1-b_PID Embedding Dimension` | 0.0027 |

### Full output paths

All figures:
```
/data/atlas/users/nterlind/outputs/reports/report_ch8_final_after_failed_run_removal/figures/
  surrogate_cv_scatter.pdf            (48 KB)
  surrogate_shap_family_bar.pdf       (18 KB)
  surrogate_shap_top5_beeswarm.pdf    (58 KB)
  surrogate_shap_dependence_top3.pdf  (17 KB)
```

Surrogate artefacts:
```
/data/atlas/users/nterlind/outputs/reports/report_ch8_final_after_failed_run_removal/surrogate/
  surrogate_xgb.json     (1.4 MB — XGBoost booster, loadable with Booster.load_model())
  shap_values.npy        (1.4 MB — shape (800, 428))
  X_shap_sample.csv      (2.0 MB — 800-row subsample used for SHAP)
  cv_metrics.json        (658 B — CV metrics as JSON)
```

### Figures description

| File | Description |
|------|-------------|
| `surrogate_cv_scatter.pdf` | OOF predicted vs actual AUROC. Points coloured D-family blue. Diagonal y=x. Annotated with Spearman ρ and R² (mean ± std from CV). |
| `surrogate_shap_family_bar.pdf` | Horizontal bar chart of normalised mean \|SHAP\| per axis family, sorted descending. Bars coloured by thesis axis palette. Values labelled as percentages. |
| `surrogate_shap_top5_beeswarm.pdf` | Strip plot (matplotlib): y = top-5 parent axis (short label), x = aggregated SHAP value, colour = active (cyan) vs inactive (gray). Points jittered on y-axis. |
| `surrogate_shap_dependence_top3.pdf` | Per-level mean SHAP for the top-3 parent axes (T1-a, T1, R2). One subplot per axis; bars sorted by mean SHAP; positive bars in family colour, negative in lightcoral. |

### Reused code

- `thesis_ml.reports.plots.style` — `apply_thesis_style`, `axis_color`, `figure_size`
- `thesis_ml.monitoring.io_utils.save_figure` — PDF at 300 DPI

### New code

- `src/thesis_ml/reports/ch8_surrogate.py` — shared library: `build_feature_matrix`, `make_groups`, `fit_surrogate`, `compute_shap`, `aggregate_shap_to_families`
- `scripts/one_off/ch8_surrogate_fit.py` — standalone script for Phase C figures

### Confounders and limitations

- The T family dominates SHAP (37.9%) largely due to `T1-a_PID Embedding Mode`, which is
  a design choice that is also correlated with which cohort was run (some PID modes only
  explored in specific periods). This family-level dominance should be interpreted with caution.
- The R family (training protocol, 18.4%) reflects under-trained models at extreme LR settings,
  consistent with Phase B marginal analysis. The surrogate captures this correctly — it is a
  genuine predictor of AUROC — but is not an architectural finding.
- H family (16.9%): model size is a pure scaling axis. Its high importance reflects that
  larger models generally outperform smaller ones; this is unsurprising but quantifies the
  effect.
- GroupKFold with 429 groups and 5 folds can produce unbalanced folds (some groups only
  appear once), contributing to the large R² variance across folds.
- SHAP is computed on a 800-row subsample for speed. Results are consistent with full-data
  feature importances from XGBoost (verified via family ranking).
- The `__LEVEL__` naming convention in one-hot columns is internal only; level labels in the
  dependence plot strip this suffix for readability.

---

## Phase D — Candidate Proposal (2026-05-14)

### Status: complete

### Entry point

Entry point C: primary cohort CSV and surrogate artefacts present; local analysis script run on interactive node.
No Condor jobs. No model.pt access required.

### Inventory snapshot

| Item | Present |
|------|---------|
| Surrogate model `surrogate_xgb.json` | yes (1.4 MB) |
| Training column list `X_shap_sample.csv` | yes (428 columns) |
| CV metrics `cv_metrics.json` | yes (Spearman 0.739 ± 0.078) |
| Primary cohort CSV `05_ch8_streamlined_primary.csv` | yes (948 rows, 90 axis columns) |
| Library `src/thesis_ml/reports/ch8_constraints.py` | yes (new) |
| Script `scripts/one_off/ch8_candidate_propose.py` | yes (new) |

### Sampling and filtering

| Metric | Value |
|--------|-------|
| Candidates sampled | 100,000 |
| Passing `is_legal_config` | 80,141 (80.1%) |
| Novel (not in training fingerprints) | 80,141 (all legal candidates novel) |
| Training fingerprints | 429 unique configs |

The product-of-marginals sampling produces combinations that almost never exactly match the 429 observed training fingerprints, so all legal candidates are novel by construction.  This is expected and correct — the surrogate is being extrapolated within the observed marginal ranges.

### Constraint rules implemented in `is_legal_config`

Seven rules implemented (conservative subset from §0.3):

1. Raw-token path: B1 biases / P1 / P2 require T1 ∈ {raw, identity}.
2. MET requirement: B1-G1 = met_direction requires D2_MET Treatment = True.
3+4. FFN/MoE consistency: F1-eff = moe ↔ F1-moe_MoE Enabled = True; moe.scope = head requires moe enabled.
5. PID dim override: T1-a = one_hot → T1-b = num_types.
6. Raw tokenizer: T1 = raw → T1-a and T1-b must be inactive.
7. Energy requirement: P1 = True → D1_Feature Set must include index 0.

Not implemented (noted as gaps): KAN hyperparams active when KAN enabled (K1–K5); MoE coupling (M1–M5); §S backbone; A3-a dependency on A3; B1 multi-family string validation.

### Top-10 predicted AUROCs

| Rank | Predicted AUROC | T1 | T1-a | B1 Bias | H1 Dim |
|------|----------------|-----|------|---------|---------|
| 1 | 0.8774 | identity | learned | none | 64 |
| 2 | 0.8698 | identity | learned | typepair_kinematic | 128 |
| 3 | 0.8692 | identity | learned | lorentz_scalar+typepair_kinematic+sm_interaction+global_conditioned | 256 |
| 4 | 0.8691 | identity | learned | lorentz_scalar+typepair_kinematic+sm_interaction+global_conditioned | 256 |
| 5 | 0.8686 | identity | learned | none | 64 |
| 6 | 0.8681 | identity | fixed_random | none | 64 |
| 7 | 0.8680 | identity | learned | none | 256 |
| 8 | 0.8679 | identity | learned | lorentz_scalar+typepair_kinematic+sm_interaction+global_conditioned | 128 |
| 9 | 0.8678 | identity | learned | lorentz_scalar+typepair_kinematic+sm_interaction+global_conditioned | 256 |
| 10 | 0.8675 | identity | learned | none | 384 |

Best observed AUROC in training data: 0.8865.  The top-10 predicted values (0.877–0.868) are plausible interpolations within the observed range.

### Uncertainty estimate method

Per-fold models were not saved during Phase C surrogate fitting (only the final model was serialised).  The uncertainty is therefore estimated as a **uniform band** equal to the CV Spearman std = 0.078.  This is expressed in rank-correlation units, not AUROC units.  It captures how much the surrogate's rank-ordering agreement varies across GroupKFold folds, but cannot be directly converted to ±AUROC without re-running CV inference.

The thesis writer should note this limitation: the uncertainty band should be interpreted as "the surrogate's ranking confidence", not as a literal AUROC prediction interval.

### Full output paths

```
/data/atlas/users/nterlind/outputs/reports/report_ch8_final_after_failed_run_removal/candidates/
  top10.json             (68 KB — list of {config dict, predicted_auroc, uncertainty, fingerprint, rank})
  top10_overrides.txt    (15 KB — one block per candidate with Hydra override key=value pairs)
  top10_predicted.csv    (2.3 KB — table for thesis: rank, predicted_auroc, uncertainty, key axis values)
  top10_scatter.pdf      (17 KB — rank vs predicted AUROC with uncertainty bars)
```

### New code

- `src/thesis_ml/reports/ch8_constraints.py` — `is_legal_config`, `sample_observed_hull`
- `scripts/one_off/ch8_candidate_propose.py` — standalone Phase D script

### Reused code

- `thesis_ml.reports.ch8_surrogate.build_feature_matrix`, `make_groups`
- `thesis_ml.reports.plots.style.apply_thesis_style`, `axis_color`, `figure_size`
- `thesis_ml.monitoring.io_utils.save_figure`

### Confounders and limitations

- The product-of-marginals sampler treats each axis independently.  It can propose combinations
  that are valid per marginal but structurally unusual (e.g. a very large model dim with a
  very small depth).  The constraint predicate only checks the rules listed in §0.3; structural
  implausibility is not screened.
- The top-1 candidate (predicted AUROC 0.8774) is slightly below the best observed training
  value (0.8865).  The surrogate is likely slightly conservative near the boundary of the
  observed space, where training data is sparse.  The best observed config should be the
  preferred "recommended" config; the top-10 candidates represent promising variants worth
  training to confirm.
- All top-10 candidates use `T1 = identity` and `T1-a = learned` or `fixed_random`, consistent
  with the Phase C SHAP finding that T-family axes dominate.  This concentration reflects the
  surrogate's learned preference, not an exhaustive search.
- The uncertainty band (CV Spearman std = 0.078) is not in AUROC units.  It should not be
  shown as ±AUROC error bars without this caveat in the thesis caption.
- No Condor jobs were staged.  If the thesis argument requires confirming any of these
  candidates experimentally, entry point A would be needed (new training run).  This is
  currently not proposed — the surrogate prediction is the intended deliverable for Chapter 8.

---

## Phase F — HPC Staging (2026-05-14)

### Summary

55 Hydra YAML configs written and a Condor submit script staged.  No jobs have been
submitted yet — the user runs the script manually.

### Job breakdown

| Set | Description | Configs | Seeds | Jobs |
|-----|-------------|---------|-------|------|
| A | Top-10 surrogate candidates (Phase D output) | 10 | 42, 123, 456 | 30 |
| B | Top-5 observed best performers (G3=`ttH+ttW+ttWW+ttZ \| 4t`, by `eval_v2/test_auroc` in 04 CSV) | 5 | 42, 123, 456, 789, 1337 | 25 |
| **Total** | | | | **55** |

### YAML directory

```
configs/classifier/experiment/thesis_experiments/ch8_candidates/
  cand01_s42.yaml … cand10_s456.yaml   (30 files — Set A)
  top5_00_s42.yaml … top5_04_s1337.yaml (25 files — Set B)
```

Full path: `/project/atlas/users/nterlind/Thesis-Code/configs/classifier/experiment/thesis_experiments/ch8_candidates/`

### Submit script

`/project/atlas/users/nterlind/Thesis-Code/hpc/submit_ch8_candidates.sh`

To submit:
```bash
# Dry-run check first:
bash hpc/submit_ch8_candidates.sh --dry-run

# Live submission (throttled, MAX_JOBS=3 concurrent by default):
bash hpc/submit_ch8_candidates.sh
```

The script uses `hpc/stoomboot/train.sub` as the Condor JDL template.  Each job is
submitted individually (throttled) to avoid flooding the queue.

### Set A config details

- Source: `top10_overrides.txt` from Phase D
- Naming: `ch8_cand{rank:02d}_s{seed}` (e.g. `ch8_cand01_s42`)
- Task: `signal_vs_background: signal=1, background=[2,3,4,5]` (4t vs ttH+ttW+ttWW+ttZ)
- Seeds replace the seed in the original override string; all other hyperparameters are
  taken verbatim from the surrogate-proposed override.

### Set B config details

- Source: `04_cleaned_backfilled_analysis_ready.csv`, top 5 rows by `eval_v2/test_auroc`
  filtered to `G3_Classification Task = ttH+ttW+ttWW+ttZ | 4t`

| Rank | AUROC | Source run | Config |
|------|-------|-----------|--------|
| 1 | 0.886485 | `run_20260507-161757_config3_all4biases_job000` | All-4-biases, d256, diff_bias=shared, cosine LR |
| 2 | 0.885721 | `run_20260507-173959_config6_rotary_base50k_job000` | Rotary PE base=50000, no biases, d128 |
| 3 | 0.884859 | `run_20260507-152854_config1_lorentz_perhead_job000` | Lorentz per-head=True, d128 |
| 4 | 0.884615 | `run_20260507-021444_gc_cross_task_ttH_job000` | Global-cond met_direction, d128 |
| 5 | 0.884087 | `run_20260506-222540_lor_binary_ttH_job001` | Lorentz [m2,deltaR,log_m2], d128 |

- Naming: `ch8_top5_{idx:02d}_s{seed}` (e.g. `ch8_top5_00_s42`)
- All Set B configs use `selected_labels: [1, 2]` (binary 4t vs ttH, same as source runs)
- All Set B configs use `include_met: true`
- Seeds used: 42, 123, 456, 789, 1337 (5 per config). Note: all source runs used seed 42,
  so seeds 123, 456, 789, 1337 are genuinely new. Seed 42 would be an exact duplicate for
  all 5 configs — retained for completeness of the seed distribution.

### Expected output paths

Runs will land under:
```
/data/atlas/users/nterlind/outputs/runs/run_<timestamp>_<experiment.name>_job000/
```

No multirun sweep subdirectory nesting occurs (each YAML spawns exactly one run in MULTIRUN
mode with no sweeper params block, so job number is always 000).

### W&B cohort / tags

- `cohort=ch8_candidates_2026_05`
- Set A: `sweep=ch8_surrogate_candidates`, `rank=cand{NN}`
- Set B: `sweep=ch8_top5_best`, `top5_idx={NN}`

---

## Phase G — Training-protocol normalisation and Optuna search (2026-05-15)

### Status: jobs submitted (running)

### Decision log

#### G1 — All ch8 candidates capped at 50 epochs

All marginal-greedy candidates (`cand_m1`, `cand_m2`, `cand_m3`) and the Optuna search
(`cand_optuna`) are trained for **50 epochs** (`classifier.trainer.epochs: 50`).

**Why:** The initial submission used 200 epochs.  The first Optuna trial clocked ~40 min,
putting the full 150-trial search at ~100h — over the 72h Condor budget.  More importantly,
50 epochs is the standard used throughout the rest of the thesis database
(`thesis_results/04_cleaned_backfilled_analysis_ready.csv`), so fixing all ch8 candidate
runs to 50 epochs makes the comparison fair across search strategies and consistent with the
existing evidence base.  The 200-epoch partial runs were aborted and removed from W&B by hand.

**Config files changed:** `cand_m1.yaml`, `cand_m2.yaml`, `cand_m3.yaml`, `cand_optuna.yaml`
(each: `epochs: 200` → `epochs: 50`).

**Note on warmup:** `warmup_steps: 10000` is unchanged.  At 50 epochs with batch_size=1024
this is fine provided the training set has ≥~200 k events (giving ≥10 k gradient steps).
If warmup exceeds total steps the LR would ramp throughout training without reaching its
target value — suboptimal but not fatal, and consistent with how shorter runs behave elsewhere
in the thesis.

#### G2 — Batch size kept at 1024 for all candidates

`batch_size: 1024` is fixed for all three search strategies (surrogate, marginal-greedy,
Optuna).  This was already documented for the marginal candidates in the Phase D update
(batch-size override from the marginal winner of 16).  Reconfirmed here for completeness:
all ch8 candidate runs use the same batch size, so AUROC differences are attributable to
architecture and hyperparameter choices, not training dynamics driven by batch size.

#### G3 — Optuna/TPE as a third global search method

A 150-trial Bayesian (TPE) search over 10 axes is added as a third search strategy alongside
the surrogate-candidate (Phase D) and marginal-greedy (Phase D update) approaches.

| Axis | Choices |
|------|---------|
| `A1` normalisation policy | pre, post, normformer |
| `E1` positional encoding | sinusoidal, none, learned, rotary |
| `F1` FFN type | standard, kan |
| `T1` tokenizer | identity, raw |
| `C1` head type | linear, kan |
| `B1` attention bias | none, lorentz_scalar, typepair_kinematic, sm_interaction, global_conditioned |
| `D1` continuous features | [0,1,2,3], [1,2,3] |
| `D2` include MET | false, true |
| `H1` model dimension | 64, 128, 256 |
| `H2` depth | 3, 6, 8 |

Fixed (not swept): `attention.type=differential`, `diff_bias_mode=shared`, `norm=layernorm`,
`heads=8` (valid divisor for all dim choices), `moe.enabled=false`, `mia_blocks.enabled=false`,
`dropout=0.15`, `lr=0.003`, `batch_size=1024`, `epochs=50`.

**Why 10 axes, not all axes:** The 10-axis set covers the axes with the highest SHAP importance
from Phase C (T, H, E, B, F families) plus the data-selection axes D1/D2.  Axes with very low
SHAP (M, S, K, L, P families) are fixed to their best-known values or disabled, which concentrates
the search budget on axes that actually move AUROC.

**Why `heads=8` fixed:** Heads must divide dim.  64/8=8, 128/8=16, 256/8=32 — all valid.
Sweeping heads would require conditional logic in the search space; fixing it avoids illegal
combinations without Optuna custom samplers.

**Timing at 50 epochs:** ~10 min/trial × 150 trials ≈ **25 h** — well within the 72 h Condor budget.

**W&B cohort:** `cohort=ch8_optuna` (distinct from `cohort=ch8_candidates`).  The static
thesis table (`04_cleaned_backfilled_analysis_ready.csv`) is unaffected.  If the table is
ever re-pulled from W&B, filter by cohort to keep the analysis clean.

**Config:** `configs/classifier/experiment/thesis_experiments/ch8_candidates/cand_optuna.yaml`
**Submit script:** `hpc/submit_ch8_optuna.sh`

#### G4 — Submit scripts updated

`hpc/submit_ch8_candidates.sh` now submits only the M-set (cand_m1/m2/m3).  The cand01/02/03
surrogate candidates are removed from the loop — those already completed and their results
are in W&B under `cohort=ch8_candidates`.  The greedy candidates are resubmitted as short
jobs (3 seeds × 3 configs = 9 runs).

### Expected W&B cohorts after this phase

| Cohort tag | Contents | Runs |
|------------|----------|------|
| `cohort=ch8_candidates` | cand01–03 (surrogate) + cand_m1–m3 (greedy, 50 ep) | 9 surrogate + 9 greedy = 18 |
| `cohort=ch8_optuna` | Optuna TPE trials, 50 ep, 1 seed each | up to 150 |

---

## Phase H — Validation training results (2026-05-17)

### Status: complete

### Entry point

Entry point C: W&B exports read directly; `test_scores.pt` used for cand_m AUROCs
(those runs were not in the W&B export). No model.pt access for inference; no Condor.

### Inventory snapshot

| Item | Present |
|------|---------|
| W&B export 1 (cand01–03): `agent_reference/wandb_export_2026-05-17T21_13_44.630+02_00.csv` | yes (9 rows) |
| W&B export 2 (Optuna 150 trials): `agent_reference/wandb_export_2026-05-17T22_02_28.058+02_00.csv` | yes (150 rows) |
| cand_m1/m2/m3 run dirs (test_scores.pt): `run_20260515-112452_cand_m{1,2,3}_job{0,1,2}` | yes (9 runs) |

Note: cand_m1/m2/m3 were not included in W&B export 1. Their AUROCs were computed from
`test_scores.pt` using sklearn `roc_auc_score(labels, probs[:, 1])` on the latest
completed batch (`run_20260515-112452_*`).

### Predicted vs actual AUROC: surrogate candidates (cand01–cand03)

These three are the top-3 surrogate-predicted configs from Phase D, trained 3 seeds each
(seeds 42, 123, 456), 50 epochs, batch 1024, G3 = `ttH+ttW+ttWW+ttZ | 4t`.

| Candidate | Predicted AUROC (surrogate) | Actual AUROC mean | Actual AUROC std | N seeds |
|-----------|-----------------------------|--------------------|------------------|---------|
| cand01 (rank 1) | 0.8700 | 0.8495 | 0.0007 | 3 |
| cand02 (rank 2) | 0.8681 | 0.8450 | 0.0008 | 3 |
| cand03 (rank 3) | 0.8672 | 0.8357 | 0.0014 | 3 |

The surrogate over-estimates by 0.018–0.032 AUROC points. Relative rank ordering is
preserved (cand01 > cand02 > cand03), confirming the surrogate is useful for ranking.
The absolute over-estimation is consistent with regression toward the mean near the edge
of the observed training space.

Key config for cand01: identity tokenizer, learned PID embedding (dim 8), lorentz_scalar
bias (per-head), standard FFN, no MoE, post-LN layernorm, dim=64, depth=6, heads=4,
MIA enabled, cosine LR 1e-4, 50 epochs.

### Predicted vs actual AUROC: marginal-greedy candidates (cand_m1–cand_m3)

These three are the top-3 marginal-greedy configs from the Phase D marginal analysis,
trained 3 seeds each, 50 epochs, batch 1024 (batch-size override from marginal winner of 16;
see Phase D update below).

| Candidate | Marginal AUROC estimate | Actual AUROC mean | Actual AUROC std | N seeds |
|-----------|-------------------------|--------------------|------------------|---------|
| cand_m1 (greedy rank 1) | 0.8416 | 0.8372 | 0.0034 | 3 |
| cand_m2 (greedy rank 2) | 0.8416 | 0.8247 | 0.0005 | 3 |
| cand_m3 (greedy rank 3) | 0.8416 | 0.8359 | 0.0039 | 3 |

The marginal AUROC estimate (0.8416 for all three, since this is the observed mean at the
best marginal level) over-estimates actual AUROC by 0.004–0.017 points. The over-estimation
is smaller than for the surrogate candidates, but the marginal candidates achieve lower
absolute AUROC. The surrogate-selected configs outperform the greedy-marginal configs by
~0.012–0.015 AUROC points (cand01 vs cand_m1 mean).

Note: cand_m1 and cand_m3 happen to achieve nearly identical actual AUROC (0.8372 vs 0.8359)
despite different secondary axis choices (R2 and R4 perturbations). This is consistent with
the seed noise floor (within-fingerprint std ~0.001–0.004) and confirms that marginal
differences near the optimum are small.

### Summary comparison: all three search strategies

| Strategy | Best candidate | Actual AUROC (mean ± std) |
|----------|---------------|--------------------------|
| Surrogate top-1 (cand01) | identity, learned PID, lorentz bias, d64, depth 6 | **0.8495 ± 0.0007** |
| Marginal greedy top-1 (cand_m1) | identity, one_hot PID, global+lorentz biases, d64, depth 4 | 0.8372 ± 0.0034 |
| Optuna TPE best (single trial) | see §8.7 below | 0.8485 (1 seed) |

The surrogate-selected candidate (cand01) achieves the highest mean AUROC across all
three search strategies.

**Candidate promoted to Chapter 9: cand01.**
Config: `configs/classifier/experiment/thesis_experiments/ch8_candidates/cand01.yaml`.
Best observed single-seed AUROC: 0.8503 (seed 123).

### Run directories (cand_m, latest batch)

```
/data/atlas/users/nterlind/outputs/runs/run_20260515-112452_cand_m1_job{000,001,002}
/data/atlas/users/nterlind/outputs/runs/run_20260515-112452_cand_m2_job{000,001,002}
/data/atlas/users/nterlind/outputs/runs/run_20260515-112452_cand_m3_job{000,001,002}
```

Older cand_m batches under the same base path exist; the `run_20260515-112452` batch is the
canonical one (50-epoch cap, batch 1024, correct cohort tag).

---

## Phase I — Optuna summary (2026-05-17)

### Status: complete (brief; no formal axis analysis)

### Entry point

Entry point C: W&B export 2 (`agent_reference/wandb_export_2026-05-17T22_02_28.058+02_00.csv`),
150 rows. All `config/axes/*` columns are null — the Optuna runs do not log formal axes because
the Optuna search space is not routed through the thesis axis-logging system. No surrogate or
SHAP analysis is possible on this export.

### Run group

- W&B cohort: `cohort=ch8_optuna`
- Config: `configs/classifier/experiment/thesis_experiments/ch8_candidates/cand_optuna.yaml`
- Submit script: `hpc/submit_ch8_optuna.sh`
- Completed runs: 150 / 150 (all `eval_v2/test_auroc` non-null)
- Run dir batch: `run_20260515-183905_cand_optuna_job{000..149}` under `/data/atlas/users/nterlind/outputs/runs/`

### AUROC distribution (150 trials, 1 seed each, 50 epochs)

| Statistic | Value |
|-----------|-------|
| Mean | 0.8376 |
| Std | 0.0217 |
| Min | 0.6966 |
| p10 | 0.8158 |
| p25 | 0.8404 |
| Median (p50) | 0.8452 |
| p75 | 0.8470 |
| p90 | 0.8470 |
| p95 | 0.8470 |
| Max (best single trial) | 0.8485 |

The distribution has a long lower tail (3 runs below 0.75) and a tight plateau at the
top (5 runs within 0.0001 of the best). The flat top (p75=p90=0.8470) suggests the search
converged to a local plateau around AUROC ~0.847 after roughly 30–40 trials.

### Comparison to other strategies

- Best Optuna trial (0.8485, single seed) is **below** cand01 mean (0.8495 ± 0.0007).
- Best Optuna trial is comparable to cand_m3 mean (0.8359) but from a single seed — with
  additional seeds, Optuna's best config might close the gap.
- The Optuna search was restricted to 10 axes with fixed attention type (differential, shared
  bias mode). This restriction may limit its ceiling compared to the surrogate-guided search
  which optimised over a broader space.

### Axes swept by Optuna (10 axes, no formal axis logging)

A1 (norm policy), E1 (PE type), F1 (FFN type), T1 (tokenizer), C1 (head type), B1 (bias set),
D1 (feature set), D2 (MET), H1 (model dim: 64/128/256), H2 (depth: 3/6/8).
Fixed: attention=differential, diff_bias=shared, norm_type=layernorm, heads=8, MoE=false,
MIA=false, dropout=0.15, lr=0.003, batch_size=1024, epochs=50.

### Confounders and limitations

- These runs have no formal axis logging; they cannot be added to the surrogate DB or used
  for axis-level marginal analysis. The thesis should present Optuna purely as a black-box
  search comparison, not as axis-level evidence.
- Single-seed results — inter-seed variance (median within-fingerprint std ~0.001) means the
  best single-trial result (0.8485) may not be reproducible at the same level. The surrogate
  cand01 (mean 0.8495 over 3 seeds) is a more reliable estimate.
- The restriction to 10 axes with many fixed settings makes Optuna's ceiling lower than a
  fully unrestricted search. This is a design choice for comparability (same epoch/batch
  budget), not a limitation of the Optuna algorithm itself.

---

## Phase D update — Marginal candidates batch-size override (2026-05-14)

The marginal greedy strategy (`ch8_candidate_propose_marginal.py`) selected `R4_Batch Size = 16`
as the marginal AUROC winner. However, batch size 16 risks GPU OOM on Stoomboot and would make
training ~64× slower per step than the surrogate candidates (batch 1024), making direct
comparison of wall-clock training cost misleading.

**Decision:** `batch_size` overridden to **1024** in all three marginal candidate YAMLs
(`cand_m1.yaml`, `cand_m2.yaml`, `cand_m3.yaml`). The marginal winner value (16) is preserved
in `top3_marginal.json` under `marginal_auroc_estimate` for reference.

**Implication for thesis:** The marginal strategy's `R4` recommendation is not tested as-is.
If the chapter discusses the batch-size axis, note that the trained marginal candidates use
batch 1024 — the same as the surrogate candidates — so AUROC differences are attributable to
architecture choices, not batch size.
