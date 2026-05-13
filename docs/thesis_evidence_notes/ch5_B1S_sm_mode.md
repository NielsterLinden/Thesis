# Evidence Note: ch5 — B1-S1 SM Interaction Mode (Exp 5D)

**Status: triaged**
**Chapter:** 5
**Section:** 5 — Physics-Informed Attention Biases (SM-interaction sub-axis)
**Created:** 2026-05-13
**Last updated:** 2026-05-13

> Follow-up to `ch5_B1_bias_families.md`. Scope: SM-interaction mode
> (B1-S1), Exp 5D.

---

## 1. Inventory Snapshot

| W&B group | Rows in CSV | model.pt on disk | Purpose |
|---|---|---|---|
| `exp_20260511-214641_ch5_sm_mode` | 9 | 9 ✓ | 3 SM-interaction modes × 3 seeds |

Run-dir verification:

```bash
find /data/atlas/users/nterlind/outputs/runs/run_*_ch5_sm_mode_job*/ -name model.pt | wc -l  # 9
```

**Entry point chosen: C.** AUROC values for all 9 runs present in the CSV.

Higher-cost rejected:
- **D**: no existing aggregate report for Exp 5D.
- **B**: 9/9 model.pt present, 9/9 eval rows present.
- **A**: B1-S1 has 3 implemented modes; all 3 swept with 3 seeds.

---

## 2. Axes Covered

| Axis ID | Name | Values Swept | Config Key | CSV Column |
|---|---|---|---|---|
| B1-S1 | SM Interaction Mode | `binary`, `fixed_coupling`, `running_coupling` | `classifier.model.bias_config.sm_interaction.mode` | `config/axes/B1-S1_SM Interaction Mode` |
| R5 | Seed | 42, 123, 314 | `classifier.trainer.seed` | `config/axes/R5_Seed` |

B1-S2 (SM mask value) is held fixed (CSV column empty in this cohort).

The `none`-bias reference for the SM family is the `none` cell from Exp 5A
(0.84125 ± 0.00068, see `ch5_B1_bias_families.md` Section 6).

Fixed-baseline confounders identical to 5A.

---

## 3. Run Group(s) and Checkpoint Paths

```
/data/atlas/users/nterlind/outputs/runs/run_20260511-214641_ch5_sm_mode_job{000..008}/model.pt   # 9
```

---

## 4. Action Staged

**Entry point: C.** Local interactive-node run.

Script: `scripts/one_off/ch5_sm_mode_plots.py` (new; mirrors 5A).

```bash
/data/atlas/users/nterlind/venvs/thesis-ml/bin/python3 \
    scripts/one_off/ch5_sm_mode_plots.py
```

Inputs:
- `thesis_results/04_cleaned_backfilled_analysis_ready.csv`
- Filter: `meta_run/group == "exp_20260511-214641_ch5_sm_mode"`
- Metric: `eval_v2/test_auroc`
- Grouping: `B1-S1` (3 levels × 3 seeds = 9 rows).

**Layout choice:** single-row 3-bar chart in the physics-bias teal. No
baseline bar inside this figure — the `none` reference is in Exp 5A and is
cross-referenced in the prose rather than re-plotted (avoids visual
double-counting of the same `none` runs).

Outputs (under `/data/atlas/users/nterlind/outputs/reports/report_ch5_sm_mode/`):

| File | Description |
|---|---|
| `figure-auroc_bar_by_sm_mode.pdf` | 3-bar chart. Error bars = per-cell seed std (n=3). Black scatter = individual seed AUROCs. |
| `exp5d_auroc_summary.csv` | Per-cell aggregate: `B1-S1`, `mean`/`std`/`count`. |

### 4.1 Reused vs new code

- **Reused:** style system, plotting pattern from 5A.
- **New:** `scripts/one_off/ch5_sm_mode_plots.py` (≈ 120 lines).

---

## 5. Result Summary (from `04_cleaned_backfilled_analysis_ready.csv`)

| B1-S1 mode | mean AUROC | std (seed, n=3) | Δ vs 5A `none` (0.84125) |
|---|---|---|---|
| `binary` | 0.84140 | 0.00050 | +0.00015 |
| `fixed_coupling` | 0.84146 | 0.00091 | +0.00021 |
| `running_coupling` | 0.84108 | 0.00081 | −0.00018 |

Patterns:
- All three modes sit within ±0.0002 AUROC of the 5A `none` baseline —
  well inside per-cell seed std.
- The three modes themselves span ~0.0004 AUROC, also at the noise floor.
- `fixed_coupling` (αs at a fixed energy scale) is nominally the best;
  `running_coupling` (αs(Q²) running with momentum transfer) is nominally
  the worst. The differences are not separable from seed noise at n=3.

---

## 6. Plot Paths

- `/data/atlas/users/nterlind/outputs/reports/report_ch5_sm_mode/figure-auroc_bar_by_sm_mode.pdf`
- `/data/atlas/users/nterlind/outputs/reports/report_ch5_sm_mode/exp5d_auroc_summary.csv`

No figure imported to `thesis_report/figures/ch5/` yet.

---

## 7. Confounders / Limitations

- **n=3 seeds per cell.** Inter-cell range (~0.0004) is below per-cell seed
  std (~0.0005–0.0009). No mode is meaningfully separable from any other.
- **No `none` bar in this figure.** Comparison to the unbiased baseline
  relies on the Exp 5A `none` cell (different sweep config but same fixed
  baseline values). Both cohorts share encoder hyperparameters and seed
  list, so the cross-cohort comparison is fair, but it is not strictly
  apples-to-apples and is documented as such.
- **B1-S2 (mask value) fixed** to default; the SM bias has only one mask
  configuration exercised here.
- Fixed-baseline NaN issue applies identically.

---

## 8. Thesis-Safe Interpretation

> The SM-interaction bias as configured in Exp 5D does not produce a
> measurable AUROC effect at the n=3 seed budget. All three modes (`binary`,
> `fixed_coupling`, `running_coupling`) sit within ±0.0002 AUROC of the
> Exp 5A `none` baseline (0.84125 ± 0.00068), and the three modes span
> only ~0.0004 AUROC among themselves — at or below the per-cell seed
> standard deviation. The ranking (`fixed_coupling` > `binary` >
> `running_coupling`) is nominally consistent with the
> running-coupling mode being the most parameter-rich (αs depends on Q²)
> and therefore the most exposed to optimisation noise at fixed
> hyperparameters, but the differences are not separable from seed
> variability and should not be reported as a directional finding. AUROC
> alone is insufficient to discriminate among SM-interaction modes at this
> budget; whether the bias is being used at all by attention is best
> answered by the planned attention-map follow-up (entry point E).
