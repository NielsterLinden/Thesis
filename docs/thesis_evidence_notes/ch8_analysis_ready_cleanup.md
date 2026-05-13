# Ch8 — Analysis-ready CSV cleanup (`03` → `04`)

**Status:** complete (CSV in repo). W&B patches are operator-run via script (dry-run by default).

---

## 1. What this is

- **`thesis_results/03_analysis_ready.csv`** — frozen export from W&B (wide schema: `meta_run/*`, `config/axes/*`, `eval_v2/*`, `config/meta.needs_review`). Used as the **input** to cleanup.
- **`thesis_results/04_cleaned_backfilled_analysis_ready.csv`** — **canonical thesis table** after row cleanup and G-axis backfill. Same columns as `03`, fewer rows, corrected framing fields for the May 2026 DB-completeness cohort.

Builder: [`scripts/thesis_results/build_04_cleaned_analysis_ready.py`](../../scripts/thesis_results/build_04_cleaned_analysis_ready.py). A dated copy is also written under `thesis_results/archive/<YYYY-MM-DD>_04_cleaned_backfilled_analysis_ready.csv`.

---

## 2. Gaps in `03` (why `04` exists)

| Issue | Count | Effect |
|-------|------:|--------|
| Tier-0 eval missing (`eval_v2/test_auroc` empty) | **21** | Not usable for discrimination / Ch8; dropped. |
| Empty `G3` + wrong `G1` on **db_completeness** cohort | **170** | All tagged `cohort=db_completeness_2026_05` (May 2026). Hydra configs use standard `signal_vs_background` (process 1 vs 2–5); see e.g. [`mia_moe_experts.yaml`](../../configs/classifier/experiment/thesis_experiments/db_completeness/mia_orthogonal_density/mia_moe_experts.yaml). W&B axis mirror was incomplete (`G1` showed `transformer` instead of `transformer_classifier`, `G3` blank). |

Other known gaps (ROC artifact holes, measured FLOPs sparsity) were **not** addressed in this pass; they remain as in `03` for kept rows.

---

## 3. What we filled (CSV)

For every **kept** row with `cohort=db_completeness_2026_05` and previously empty `G3`:

| Column | Value |
|--------|--------|
| `config/axes/G1_Task Type` | `transformer_classifier` |
| `config/axes/G2_Model Family` | `transformer` |
| `config/axes/G3_Classification Task` | `ttH+ttW+ttWW+ttZ \| 4t` |

This matches [`build_class_def_str`](../../src/thesis_ml/facts/meta.py) for the canonical binary class definition used elsewhere in the frozen table for the same data setup.

---

## 4. Rows discarded

**21** rows removed: any row with empty or non-numeric `eval_v2/test_auroc` (Tier-0 incomplete — no checkpoint eval block in export). They are **not** copied to `04`.

---

## 5. Status of `04`

| Metric | Value |
|--------|------:|
| Rows | **1320** (= 1341 − 21) |
| Columns | Same as `03` (wide analysis-ready schema) |
| Rows with empty `eval_v2/test_auroc` | **0** |
| Cohort rows with empty `G3` after backfill | **0** |
| Cohort rows with wrong `G1` after backfill | **0** |

**W&B sync (optional):** [`scripts/wandb/patch_wandb_g_axes_from_csv.py`](../../scripts/wandb/patch_wandb_g_axes_from_csv.py) applies the same G1/G2/G3 (and `meta.class_def_str` / `meta.process_groups_key`) to runs listed in `04` that still carry the cohort tag. Default is dry-run; use `--execute` after `WANDB_API_KEY` is set.

---

## 6. Pointers

- Primary table for agents and thesis analysis: **`thesis_results/04_cleaned_backfilled_analysis_ready.csv`** (see root [`CLAUDE.md`](../../CLAUDE.md)).
- Legacy unchanged snapshot: `thesis_results/03_analysis_ready.csv` (retained for diff / audit).
