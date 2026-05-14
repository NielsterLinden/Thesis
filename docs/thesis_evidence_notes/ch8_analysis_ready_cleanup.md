# Ch8 — Analysis-ready CSV

**Status:** `thesis_results/03_analysis_ready.csv` is the frozen W&B export (do not modify). **`thesis_results/04_cleaned_backfilled_analysis_ready.csv`** is derived from `03` by stage 04 only (see below).

---

## Pipeline scripts

| Stage | Script | Notes |
|-------|--------|-------|
| 01 | `scripts/wandb/wandb_export_to_analysis_ready/01_raw_export.py` | W&B → raw CSV |
| 02 | `scripts/wandb/wandb_export_to_analysis_ready/02_column_selection.py` | Column pruning |
| 03 | `scripts/wandb/wandb_export_to_analysis_ready/03_transformer_filter.py` | Transformer-only filter; drops rows missing `eval_v2/test_auroc`; adds `axes_complete` flag. **Last frozen export stage.** |
| 04 | `scripts/wandb/wandb_export_to_analysis_ready/04_final_thesis_table.py` | **Run** after `03`: copies `03` → `04` with two **explicit** row filters only. **No** global AUROC threshold (e.g. not `eval_v2/test_auroc < 0.65`); removals must match the documented cohort rules. |

### Stage 04 exclusion rules

1. **April 2026 ch5:** `meta_run/created_at` starts with `2026-04` **and** `meta_run/group` contains `_ch5_`.
2. **Named failed W&B cohorts** (`meta_run/group` substring): `builtjes_baseline`, `OrthogonalSweep_B_g16`, `order_pe_attention_4t_vs_bg` (see script `FAILED_COHORT_RULES` for exact needles and log slugs).

### Regenerate `04` and streamlined Ch8 `05` files

```bash
cd /project/atlas/users/nterlind/Thesis-Code && source ~/.bashrc && thesis && \
python scripts/wandb/wandb_export_to_analysis_ready/04_final_thesis_table.py \
  --in thesis_results/03_analysis_ready.csv \
  --out thesis_results/04_cleaned_backfilled_analysis_ready.csv
```

```bash
cd /project/atlas/users/nterlind/Thesis-Code && source ~/.bashrc && thesis && \
python scripts/one_off/build_05_ch8_streamlined.py
```

---

## Canonical table

- **`thesis_results/03_analysis_ready.csv`** — **1 426** data rows, 161 cols. Frozen export.
- **`thesis_results/04_cleaned_backfilled_analysis_ready.csv`** — **1 253** data rows, 161 cols. Same schema as `03` after removing **106** April‑2026 ch5 rows and **67** rows from the three named failed cohorts (**173** total dropped vs `03`).
- Prior `04` snapshots live under `thesis_results/archive/` (e.g. `*_pre_04_rewrite_04_cleaned_backfilled_analysis_ready.csv`).
