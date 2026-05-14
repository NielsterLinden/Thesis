# Evidence Note: ch5 — B1-L Lorentz Scalar Bias (Exp 5B)

**Status: triaged**
**Chapter:** 5
**Section:** 5 — Physics-Informed Attention Biases (Lorentz sub-axes)
**Created:** 2026-05-13
**Last updated:** 2026-05-13

> Follow-up to `ch5_B1_bias_families.md`. Scope: the Lorentz-scalar bias
> family (B1-L sub-axes), Exp 5B.

---

## 1. Inventory Snapshot (data-inventory)

| W&B group | Rows in CSV | model.pt on disk | Purpose |
|---|---|---|---|
| `exp_20260511-144127_ch5_lorentz_p1` | 30 | 30 ✓ | Sparse gating **off** sub-grid |
| `exp_20260511-150429_ch5_lorentz_p2` | 30 | 30 ✓ | Sparse gating **on** sub-grid |
| **Total** | **60** | **60 ✓** | 5 × 2 × 2 × 3 = 60 |

Run-dir verification:

```bash
find /data/atlas/users/nterlind/outputs/runs/run_*_ch5_lorentz_p1_job*/ -name model.pt | wc -l  # 30
find /data/atlas/users/nterlind/outputs/runs/run_*_ch5_lorentz_p2_job*/ -name model.pt | wc -l  # 30
```

**Entry point chosen: C.** AUROC values for all 60 runs are present in
`thesis_results/04_cleaned_backfilled_analysis_ready.csv`; no inference re-run
needed.

Higher-cost rejected:
- **D** rejected: no existing report aggregates Exp 5B with the cleaned CSV.
- **B** rejected: 60/60 model.pt present, 60/60 eval rows present.
- **A** rejected: the 5×2×2×3 grid spans the planned B1-L1/L2/L5 axes with
  3 seeds per cell.

KAN-spline visualisations and gate-value histograms (entry **E**) are queued
as a separate follow-up note (`ch5_B1L_lorentz_interpretability.md`) — they
require loading `model.pt` checkpoints and were explicitly out of scope for
this turn.

---

## 2. Axes Covered

| Axis ID | Name | Values Swept | Config Key | CSV Column |
|---|---|---|---|---|
| B1-L1 | Lorentz Feature Set | 5 levels: `['deltaR']`, `['m2']`, `['m2','deltaR']`, `['log_kt','z','deltaR','log_m2']`, `['m2','deltaR','log_m2','log_kt','z','deltaR_ptw']` | `classifier.model.bias_config.lorentz_scalar.features` | `config/axes/B1-L1_Lorentz Feature Set` |
| B1-L2 | Lorentz MLP Type | `standard`, `kan` | `classifier.model.bias_config.lorentz_scalar.mlp_type` | `config/axes/B1-L2_Lorentz MLP Type` |
| B1-L5 | Lorentz Sparse Gating | `False`, `True` | `classifier.model.bias_config.lorentz_scalar.sparse_gating` | `config/axes/B1-L5_Lorentz Sparse Gating` |
| R5 | Seed | 42, 123, 314 | `classifier.trainer.seed` | `config/axes/R5_Seed` |

B1-L3 (hidden dim) and B1-L4 (per-head mode) are held fixed in this cohort
(`B1-L3` empty in CSV; `B1-L4` constant False). The all-`False` B1-L4 means
the bias uses a shared-across-heads MLP — per-head mode is not exercised by
Exp 5B and is queued for a future sweep if needed.

All fixed-baseline confounders (D02, D03, H01–H05, T1-a/b, E1) — see
`ch5_B1_bias_families.md` Section 3. The NaN-on-fixed-axes issue applies
identically to this cohort.

---

## 3. Run Group(s) and Checkpoint Paths

```
/data/atlas/users/nterlind/outputs/runs/run_20260511-144127_ch5_lorentz_p1_job{000..029}/model.pt   # 30
/data/atlas/users/nterlind/outputs/runs/run_20260511-150429_ch5_lorentz_p2_job{000..029}/model.pt   # 30
```

---

## 4. Action Staged

**Entry point: C.** Local interactive-node run; no Condor.

Script: `scripts/one_off/ch5_lorentz_plots.py` (new; mirrors `ch5_bias_families_plots.py`).
Decision to split into a dedicated file (rather than extending the 5A
module): 5B has its own multi-axis layout and is large enough that mixing it
with 5A muddies the entry-point structure.

```bash
/data/atlas/users/nterlind/venvs/thesis-ml/bin/python3 \
    scripts/one_off/ch5_lorentz_plots.py
```

Inputs:
- `thesis_results/04_cleaned_backfilled_analysis_ready.csv`
- Filter: `meta_run/group ∈ {exp_20260511-144127_ch5_lorentz_p1, exp_20260511-150429_ch5_lorentz_p2}`
- Metric: `eval_v2/test_auroc`
- Grouping: (`B1-L1`, `B1-L2`, `B1-L5`) — 20 cells × 3 seeds = 60 rows.

**Layout choice:** two-panel layout (gate=off left, gate=on right), feature
set on the x-axis, MLP type (`standard` = baseline-grey, `kan` = teal) as hue.
Twenty cells in one grouped bar would have four hue groups per feature-set
and be hard to read; two panels separate the gating dimension cleanly while
keeping feature-set × MLP-type readable per panel.

Outputs (under
`/data/atlas/users/nterlind/outputs/reports/report_ch5_lorentz/`):

| File | Description |
|---|---|
| `figure-auroc_bar_by_lorentz.pdf` | Two-panel grouped bar. Hue = MLP type. Error bars = per-cell seed std (n=3). Black scatter = individual seed AUROCs. |
| `exp5b_auroc_summary.csv` | Per-cell aggregate: `(B1-L1, B1-L2, B1-L5)` × `mean`/`std`/`count`. |

### 4.1 Reused vs new code

- **Reused:** thesis style system (`apply_thesis_style`, `figure_size`,
  `axis_color`); plotting pattern from 5A.
- **New:** `scripts/one_off/ch5_lorentz_plots.py` (≈ 170 lines).

---

## 5. Result Summary (from `04_cleaned_backfilled_analysis_ready.csv`)

Per-cell seed-mean AUROC (n=3 seeds per cell):

### Sparse gating OFF

| Feature set | MLP type | mean | std |
|---|---|---|---|
| ΔR | standard | 0.84126 | 0.00123 |
| ΔR | kan | 0.84245 | 0.00089 |
| m² | standard | 0.84131 | 0.00142 |
| m² | kan | 0.84154 | 0.00159 |
| m², ΔR | standard | 0.84114 | 0.00042 |
| m², ΔR | kan | 0.84256 | 0.00063 |
| log kT, z, ΔR, log m² | standard | 0.84200 | 0.00085 |
| log kT, z, ΔR, log m² | kan | 0.84399 | 0.00101 |
| m², ΔR, log m², log kT, z, ΔR/pT | standard | 0.84257 | 0.00106 |
| m², ΔR, log m², log kT, z, ΔR/pT | kan | 0.84436 | 0.00214 |

### Sparse gating ON

| Feature set | MLP type | mean | std |
|---|---|---|---|
| ΔR | standard | 0.84155 | 0.00126 |
| ΔR | kan | 0.84206 | 0.00044 |
| m² | standard | 0.84114 | 0.00118 |
| m² | kan | 0.84117 | 0.00118 |
| m², ΔR | standard | 0.84100 | 0.00104 |
| m², ΔR | kan | 0.84190 | 0.00022 |
| log kT, z, ΔR, log m² | standard | 0.84191 | 0.00034 |
| log kT, z, ΔR, log m² | kan | 0.84311 | 0.00076 |
| m², ΔR, log m², log kT, z, ΔR/pT | standard | 0.84170 | 0.00065 |
| m², ΔR, log m², log kT, z, ΔR/pT | kan | 0.84378 | 0.00107 |

Patterns:
- **MLP type:** within every (feature set, gating) cell, `kan` ≥ `standard`
  on the mean. Largest gap is in the 6-feature-set with gate=off
  (+0.0018 AUROC). Gaps are at or just above the per-cell seed std.
- **Feature set:** richest 4- and 6-feature sets consistently beat the 1- and
  2-feature sets, more clearly with `kan`. The 6-feature, kan, gate=off
  cell is the best (0.84436), about +0.0031 above the 5A `none` baseline
  (0.84125, from `ch5_B1_bias_families.md`).
- **Sparse gating:** gating-on cells tend to sit slightly below gating-off
  cells of the same (feature, MLP). The effect is small (~0.0005–0.0009) and
  inside the seed std for most cells. Sparse gating does not clearly help on
  AUROC at this seed budget.

---

## 6. Plot Paths

- `/data/atlas/users/nterlind/outputs/reports/report_ch5_lorentz/figure-auroc_bar_by_lorentz.pdf`
- `/data/atlas/users/nterlind/outputs/reports/report_ch5_lorentz/exp5b_auroc_summary.csv`

No figure imported to `thesis_report/figures/ch5/` yet — pending user review.

---

## 7. Confounders / Limitations

- **n=3 seeds per cell.** Inter-cell mean range is ~0.003 AUROC, comparable
  to per-cell seed stds (~0.0004–0.0021). Effects are suggestive, not
  significant in a strict statistical sense.
- **B1-L4 (per-head) is fixed False.** The shared-MLP regime is the only one
  exercised; whether per-head MLPs would extract more signal is open.
- **Feature-set monotonicity is not strictly tested** — the 5 sets are not
  nested in a strict subset order (`ΔR` ⊂ `m², ΔR`, but `[log_kt, z, ΔR,
  log_m2]` does not contain `m²`). Interpret as "richer Lorentz feature
  panels tend to help" rather than as an information-monotonicity proof.
- Fixed-baseline NaN issue applies identically (see 5A note Section 3).

---

## 8. Thesis-Safe Interpretation

> Across the 20 Lorentz-bias configurations swept in Exp 5B, the test
> AUROC sits in the 0.840–0.844 range (seed-mean, n=3 per cell). The two
> consistent trends are (i) `kan` MLPs sit at or above `standard` MLPs in
> every (feature-set, gating) cell, with the largest gap (~0.0018) at the
> 6-feature gate-off configuration, and (ii) richer Lorentz feature panels
> (4- and 6-feature) systematically outperform the single-feature panels,
> especially when paired with `kan`. Sparse gating on does not improve
> AUROC at this scale and seed budget; gate-on cells tend to sit ≲0.001
> below their gate-off counterparts. The best Lorentz cell (6-feature, kan,
> gate-off) lies approximately 0.003 AUROC above the Exp 5A `none` baseline,
> a margin comparable to several per-cell seed standard deviations. The
> AUROC view alone therefore supports a tentative reading that the Lorentz
> bias provides a small, consistent, but modest gain when given a rich
> kinematic feature set and a `kan` MLP; the AUROC-insensitive interpretive
> questions (which spline shapes the `kan` learns, what fraction of head/key
> pairs gate selects) are deferred to the planned interpretability
> follow-up.

---

## Imported figures

| Destination (thesis_report/figures/ch5/) | Source (/data/atlas/users/nterlind/outputs/reports/) | LaTeX label |
|---|---|---|
| `figure-auroc_bar_by_lorentz.pdf` | `report_ch5_lorentz/figure-auroc_bar_by_lorentz.pdf` | `fig:5b_auroc_bar` |
