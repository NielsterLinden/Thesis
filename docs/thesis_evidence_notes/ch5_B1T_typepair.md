# Evidence Note: ch5 — B1-T Type-Pair Kinematic Bias (Exp 5C)

**Status: triaged**
**Chapter:** 5
**Section:** 5 — Physics-Informed Attention Biases (Type-pair sub-axes)
**Created:** 2026-05-13
**Last updated:** 2026-05-13

> Follow-up to `ch5_B1_bias_families.md`. Scope: the type-pair kinematic
> bias family (B1-T sub-axes), Exp 5C.

---

## 1. Inventory Snapshot

| W&B group | Rows in CSV | model.pt on disk | Purpose |
|---|---|---|---|
| `exp_20260511-193832_ch5_typepair_p1` | 3 | 3 ✓ | `init=none`, freeze axis n/a |
| `exp_20260511-200334_ch5_typepair_p2` | 12 | 12 ✓ | `init ∈ {binary, fixed_coupling}` × `freeze ∈ {False, True}` × 3 seeds |
| **Total** | **15** | **15 ✓** | 3 + 2×2×3 = 15 |

Run-dir verification:

```bash
find /data/atlas/users/nterlind/outputs/runs/run_*_ch5_typepair_p1_job*/ -name model.pt | wc -l  # 3
find /data/atlas/users/nterlind/outputs/runs/run_*_ch5_typepair_p2_job*/ -name model.pt | wc -l  # 12
```

**Entry point chosen: C.** AUROC values present for all 15 runs.

Higher-cost rejected:
- **D**: no existing aggregate report for this cohort.
- **B**: 15/15 model.pt present, 15/15 eval rows present.
- **A**: the planned 5-cell sweep is fully covered.

21×21 learned-bias-table heatmap (entry **E**) is queued separately —
requires loading `model.pt` and extracting the learned type-pair table.

**Naming note.** The user's task description called the non-`none` init
modes `binary` / `physics`; the implemented enum (and the value present in
the CSV) is `binary` / `fixed_coupling`. The "physics-motivated" mode is
`fixed_coupling`. This note uses the CSV value name throughout.

---

## 2. Axes Covered

| Axis ID | Name | Values Swept | Config Key | CSV Column |
|---|---|---|---|---|
| B1-T1 | Type-Pair Initialization | `none`, `binary`, `fixed_coupling` | `classifier.model.bias_config.typepair_kinematic.init` | `config/axes/B1-T1_Type-Pair Initialization` |
| B1-T2 | Type-Pair Freeze Table | `False`, `True` (only when `init != none`) | `classifier.model.bias_config.typepair_kinematic.freeze_table` | `config/axes/B1-T2_Type-Pair Freeze Table` |
| R5 | Seed | 42, 123, 314 | `classifier.trainer.seed` | `config/axes/R5_Seed` |

B1-T3 (kinematic gate), B1-T4 (kinematic feature), B1-T5 (mask value) are
held fixed in this cohort (`T3=True`; `T4`/`T5` empty in CSV — default
values). Sweeping them was descoped to a future cohort.

Fixed-baseline confounders identical to 5A (see Section 3 of
`ch5_B1_bias_families.md`).

---

## 3. Run Group(s) and Checkpoint Paths

```
/data/atlas/users/nterlind/outputs/runs/run_20260511-193832_ch5_typepair_p1_job{000..002}/model.pt   # 3
/data/atlas/users/nterlind/outputs/runs/run_20260511-200334_ch5_typepair_p2_job{000..011}/model.pt   # 12
```

---

## 4. Action Staged

**Entry point: C.** Local interactive-node run.

Script: `scripts/one_off/ch5_typepair_plots.py` (new; mirrors 5A).

```bash
/data/atlas/users/nterlind/venvs/thesis-ml/bin/python3 \
    scripts/one_off/ch5_typepair_plots.py
```

Inputs:
- `thesis_results/04_cleaned_backfilled_analysis_ready.csv`
- Filter: `meta_run/group ∈ {exp_20260511-193832_ch5_typepair_p1, exp_20260511-200334_ch5_typepair_p2}`
- Metric: `eval_v2/test_auroc`
- Grouping: 5 distinct cells: `none`, `binary-free`, `binary-frozen`,
  `fixed_coupling-free`, `fixed_coupling-frozen`.

**Layout choice:** single grouped bar over the 5 cells. `none` is rendered
in baseline-grey (no hatch) — the freeze axis is undefined here. The four
non-`none` cells use the physics-bias teal; the two frozen cells use a `//`
hatch so init and freeze read as separable visual channels. A side-by-side
two-panel layout (one panel per init) would split the small dataset awkwardly;
a single 5-bar row is cleaner.

Outputs (under `/data/atlas/users/nterlind/outputs/reports/report_ch5_typepair/`):

| File | Description |
|---|---|
| `figure-auroc_bar_by_typepair.pdf` | 5-bar grouped chart. Hatched bars = frozen table. Error bars = per-cell seed std (n=3). Black scatter = individual seed AUROCs. Dashed reference at the `none` baseline. |
| `exp5c_auroc_summary.csv` | Per-cell aggregate: `init`, `freeze`, `mean`/`std`/`count`. |

### 4.1 Reused vs new code

- **Reused:** style system; bar-plot pattern from 5A.
- **New:** `scripts/one_off/ch5_typepair_plots.py` (≈ 175 lines). Hatching
  pattern is new but uses standard `matplotlib.patches` API.

---

## 5. Result Summary (from `04_cleaned_backfilled_analysis_ready.csv`)

| Cell | init | freeze | mean AUROC | std (seed, n=3) | Δ vs `none` |
|---|---|---|---|---|---|
| none (baseline) | `none` | (n/a) | 0.84205 | 0.00081 | 0 |
| binary, free | `binary` | False | 0.84075 | 0.00046 | −0.00130 |
| binary, frozen | `binary` | True | 0.84131 | 0.00065 | −0.00074 |
| fixed-coupling, free | `fixed_coupling` | False | 0.84175 | 0.00102 | −0.00030 |
| fixed-coupling, frozen | `fixed_coupling` | True | 0.84138 | 0.00046 | −0.00067 |

Patterns:
- **All four non-`none` cells sit below the `none` baseline** in seed mean.
- Deltas (−0.00030 to −0.00130 AUROC) are within ~1–2× per-cell seed std.
- `fixed_coupling` (physics-motivated initialization) consistently outperforms
  `binary` (interaction allowed / forbidden mask), regardless of freeze
  state — but only by ~0.0006–0.0010 AUROC.
- Freezing has small, non-monotonic effects: it slightly *helps* `binary`
  (+0.00056) and slightly *hurts* `fixed_coupling` (−0.00037). Both are at
  the seed-noise floor.

---

## 6. Plot Paths

- `/data/atlas/users/nterlind/outputs/reports/report_ch5_typepair/figure-auroc_bar_by_typepair.pdf`
- `/data/atlas/users/nterlind/outputs/reports/report_ch5_typepair/exp5c_auroc_summary.csv`

No figure imported to `thesis_report/figures/ch5/` yet.

---

## 7. Confounders / Limitations

- **`init=none` lacks the freeze axis by construction** — there is no
  table to freeze when nothing is initialised. The 5-cell layout is
  appropriate but the `none` cell is not directly comparable to a "frozen
  baseline."
- **B1-T3/T4/T5 fixed** to defaults in this cohort. Whether different
  kinematic features or mask values would shift the picture is open.
- **n=3 seeds per cell.** Deltas of order 0.001 AUROC are at the noise
  floor.
- Fixed-baseline NaN issue applies identically.

---

## 8. Thesis-Safe Interpretation

> The type-pair kinematic bias as configured in Exp 5C does not improve
> AUROC over the unbiased `none` baseline (0.84205 ± 0.00081, n=3) on the
> 4-tops-vs-background task. All four (init, freeze) combinations lie
> 0.0003–0.0013 AUROC below baseline — within one to two per-cell seed
> standard deviations. Among the non-baseline cells, the physics-motivated
> `fixed_coupling` initialization consistently outperforms the `binary`
> mask initialization by ~0.0006–0.0010 AUROC, and freezing the table has
> small, opposite-signed effects for the two inits (helping `binary`,
> hurting `fixed_coupling`), all at the noise floor. Taken together, AUROC
> does not distinguish the type-pair bias from the unbiased baseline at
> this seed budget; whether the learned 21×21 type-pair table is structured
> or essentially noise — and whether the `fixed_coupling` table is being
> exploited by attention — is best addressed by the planned interpretability
> follow-up (learned-bias heatmap, entry point E).
