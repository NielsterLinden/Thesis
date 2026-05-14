# Evidence Note: ch5 — B1-T Type-Pair Kinematic Bias (Exp 5C)

**Status: complete**
**Chapter:** 5
**Section:** 5 — Physics-Informed Attention Biases (Type-pair sub-axes)
**Created:** 2026-05-13
**Last updated:** 2026-05-14

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

**Entry points used: C + E.**

- **C**: AUROC bar chart — all 15 eval rows present in CSV, run locally.
- **E**: Learned-bias heatmap — `model.pt` loaded directly, no inference required.

Higher-cost rejected:
- **D**: no existing aggregate report for this cohort.
- **B**: 15/15 model.pt present, 15/15 eval rows present.
- **A**: the planned 5-cell sweep is fully covered.

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

**Entry point E: heatmap.** Local interactive-node run.

Script: `scripts/one_off/ch5_typepair_heatmap.py` (new).

```bash
/data/atlas/users/nterlind/venvs/thesis-ml/bin/python3 \
    scripts/one_off/ch5_typepair_heatmap.py
```

Inputs: 15 × `model.pt` (state dict key
`bias_composer.bias_modules.typepair_kinematic.table_raw`, shape `[8, 8]`).
Symmetric table reconstructed as `0.5*(raw + raw.T) * pad_mask`.
Gate extracted as `tanh(gate)` but not applied to the displayed table
(table shown in its raw unit scale for interpretability).

Outputs (under `/data/atlas/users/nterlind/outputs/reports/report_ch5_typepair/`):

| File | Description |
|---|---|
| `figure-typepair_reference_tables.pdf` | Two-panel: `binary` and `fixed_coupling` init reference tables (7×7, skip padding row/col). |
| `figure-typepair_learned_tables.pdf` | Five-panel row: seed-mean learned table for each (init, freeze) cell. Shared RdBu_r colormap; annotated cell values; gate mean ± std in x-label. |
| `figure-typepair_diff_from_init.pdf` | Four-panel: seed-mean learned table minus init reference. Diverging colormap. Only `binary` and `fixed_coupling` inits (no reference for `none`). |
| `exp5c_typepair_table_stats.csv` | Per-seed: `frob_norm`, `drift_from_init`, `gate`. |

### 4.1 Reused vs new code

- **Reused:** style system; bar-plot pattern from 5A.
- **New:**
  - `scripts/one_off/ch5_typepair_plots.py` (AUROC bar, entry C).
  - `scripts/one_off/ch5_typepair_heatmap.py` (heatmap, entry E).

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

### 5.2 Learned-table findings (entry E, from `exp5c_typepair_table_stats.csv` and checkpoint extraction)

Parameter shape: `[8, 8]` symmetric (7×7 active after padding row/col removal).
All values below are from the seed-mean symmetric table for each group.

#### Per-group scalar summary

| Group | init | freeze | gate mean ± std | Frobenius norm | Drift from init |
|---|---|---|---|---|---|
| none | none | Free | 0.234 ± 0.015 | 0.434 (7×7) | 0.944 (= norm; zero init) |
| binary, free | binary | Free | 0.097 ± 0.012 | 27.044 | 0.868 |
| binary, frozen | binary | Frozen | 0.094 ± 0.013 | 27.295 | 0.000 (frozen) |
| fc, free | fixed_coupling | Free | 0.104 ± 0.015 | 26.848 | 0.930 |
| fc, frozen | fixed_coupling | Frozen | 0.101 ± 0.015 | 27.094 | ~0 (6.6e-8, frozen) |

#### Key structural findings

**`none` init (zero start, free):**
The network learns a small-magnitude table (Frobenius 0.43). Values range
-0.16 to +0.12. The table is essentially unstructured noise at this seed
budget; no clean SM-interaction pattern emerges. Gate ~0.23 (higher than
the physics-init groups).

**`binary` and `fixed_coupling` init (physics init):**
All physics-init groups have Frobenius ~27, driven by the many mask-value
(-5.0) entries. Gate ~0.094–0.104 (lower than `none`; the physics table
acts as a strong prior that suppresses gate learning).

**`fc_free` seed-mean 7×7 table (active entries only):**
- jet–jet: +1.22, jet–bjet: +1.17, bjet–bjet: +1.23 (QCD pairs — positive)
- e+–e-: +0.54, mu+–mu-: +0.89 (EW pairs — positive)
- All non-interacting pairs: near -5.0 (mask value; suppressed)
- photon row: small positive values for all hadronic and leptonic types (0.10–0.41)

This pattern is structurally faithful to the SM `fixed_coupling` prior.
The drift (~0.93) comes from small learned perturbations (+/-0.1–0.3) on
the active entries, not from collapse of the mask-value entries.

**`fc_frozen`:** Exactly matches the reference table (drift ~6.6e-8). Confirms
the freeze mechanism works. Gate 0.101 — similar to fc_free.

**`binary_free`:** Structurally identical pattern to fc_free (same mask positions),
but with higher learned drift (0.868 vs 0.930). Active entries are positive but
not exactly the coupling-constant values; the network adjusts them freely.

**Gate comparison:** The `none` init group has a meaningfully higher gate
(0.234 ± 0.015) vs all physics-init groups (0.094–0.104). This is consistent
with the gate finding it harder to learn useful structure from a zero start
and compensating by opening further; or, with the network activating the bias
more heavily when the table is initially flat and all pairs look equal.

---

## 6. Plot Paths

### Entry C (AUROC bar)
- `/data/atlas/users/nterlind/outputs/reports/report_ch5_typepair/figure-auroc_bar_by_typepair.pdf`
- `/data/atlas/users/nterlind/outputs/reports/report_ch5_typepair/exp5c_auroc_summary.csv`

### Entry E (heatmaps)
- `/data/atlas/users/nterlind/outputs/reports/report_ch5_typepair/figure-typepair_reference_tables.pdf`
- `/data/atlas/users/nterlind/outputs/reports/report_ch5_typepair/figure-typepair_learned_tables.pdf`
- `/data/atlas/users/nterlind/outputs/reports/report_ch5_typepair/figure-typepair_diff_from_init.pdf`
- `/data/atlas/users/nterlind/outputs/reports/report_ch5_typepair/exp5c_typepair_table_stats.csv`

No figures imported to `thesis_report/figures/ch5/` yet.

---

## 7. Confounders / Limitations

- **`init=none` lacks the freeze axis by construction** — there is no
  table to freeze when nothing is initialised. The 5-cell layout is
  appropriate but the `none` cell is not directly comparable to a "frozen
  baseline."
- **B1-T3/T4/T5 fixed** to defaults in this cohort. Whether different
  kinematic features or mask values would shift the picture is open.
- **n=3 seeds per cell.** Deltas of order 0.001 AUROC are at the noise
  floor. Gate and table-norm differences similarly have high per-seed
  variance at n=3.
- The heatmap shows the raw `table_raw` (symmetric) without the scalar
  `tanh(gate)` multiplier. Effective attention contribution is
  `tanh(gate) * head_proj(table * kinematic_gate_output)`. The shown table
  alone does not reflect the full attention impact.
- Fixed-baseline NaN issue applies identically.

---

## 8. Thesis-Safe Interpretation

> The type-pair kinematic bias as configured in Exp 5C does not improve
> AUROC over the unbiased `none` baseline (0.84205 ± 0.00081, n=3) on the
> 4-tops-vs-background task. All four (init, freeze) combinations lie
> 0.0003–0.0013 AUROC below baseline — within one to two per-cell seed
> standard deviations.
>
> Inspection of the learned 8×8 type-pair tables reveals a qualitatively
> different picture from the AUROC results. When initialised from the
> SM `fixed_coupling` prior and left free to train, the network preserves
> the physics structure: QCD quark–quark pairs (jet–jet, jet–bjet, bjet–bjet)
> retain large positive values (~+1.2), EW lepton pairs (e+e-, mu+mu-)
> retain positive values (~+0.5–0.9), and all non-interacting pairs remain
> near the mask value (−5.0). The drift from the initialization is small
> (~0.93 Frobenius on the 7×7 active sub-matrix), reflecting fine-tuning of
> the coupling-constant magnitudes rather than structural reorganization.
> The `binary` init group shows the same structural pattern (same mask
> positions) but with freely learned entry magnitudes that do not match
> coupling constants.
>
> By contrast, the `none` init (zero start) learns a small-magnitude,
> largely unstructured table (Frobenius 0.43 vs ~27 for physics inits),
> with values scattered in [-0.16, +0.12] and no discernible SM interaction
> pattern. Its gate is noticeably higher (0.23 vs ~0.10) — consistent with
> the table acting as a weaker prior so the gate opens further to compensate.
>
> Taken together, the results suggest that the SM `fixed_coupling`
> initialization provides a strong structural prior that the network broadly
> respects during training. However, this structural fidelity does not
> translate to measurable AUROC improvement at n=3 seeds, indicating either
> that the type-pair table's effective attention contribution (modulated by
> `tanh(gate) ≈ 0.10`) is too small to shift discrimination at this task
> scale, or that the kinematic gate already captures sufficient inter-type
> structure without an explicit table prior.

---

## Imported figures

| Destination (thesis_report/figures/ch5/) | Source (/data/atlas/users/nterlind/outputs/reports/) | LaTeX label |
|---|---|---|
| `figure-typepair_learned_tables.pdf` | `report_ch5_typepair/figure-typepair_learned_tables.pdf` | `fig:5c_tables` |
| `figure-typepair_diff_from_init.pdf` | `report_ch5_typepair/figure-typepair_diff_from_init.pdf` | `fig:5c_diff` |
| `figure-auroc_bar_by_typepair.pdf` | `report_ch5_typepair/figure-auroc_bar_by_typepair.pdf` | (not yet imported — available) |
| `figure-typepair_reference_tables.pdf` | `report_ch5_typepair/figure-typepair_reference_tables.pdf` | (not yet imported — available) |
