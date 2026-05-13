# Ch8 — Analysis-ready CSV

**Status:** frozen as of 2026-05-13. `thesis_results/03_analysis_ready.csv` is the canonical thesis table. Do not modify.

---

## Pipeline scripts

| Stage | Script | Notes |
|-------|--------|-------|
| 01 | `scripts/wandb/wandb_export_to_analysis_ready/01_raw_export.py` | W&B → raw CSV |
| 02 | `scripts/wandb/wandb_export_to_analysis_ready/02_column_selection.py` | Column pruning |
| 03 | `scripts/wandb/wandb_export_to_analysis_ready/03_transformer_filter.py` | Transformer-only filter; drops rows missing `eval_v2/test_auroc`; adds `axes_complete` flag. **Last stage run.** |
| 04 | `scripts/wandb/wandb_export_to_analysis_ready/04_final_thesis_table.py` | Not run — Stage 03 output is the final table. |

---

## Canonical table

- **`thesis_results/03_analysis_ready.csv`** — 1 426 rows, 161 cols. Frozen.
- **`thesis_results/04_cleaned_backfilled_analysis_ready.csv`** — identical copy of `03` (for backwards compatibility with scripts that reference the `04` name).
- Previous `04` (1 320 rows, with manual backfills and row drops) archived at `thesis_results/archive/04_cleaned_backfilled_analysis_ready.csv`.
