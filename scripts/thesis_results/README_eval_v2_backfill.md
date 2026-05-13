# Backfilling Tier-3 `eval_v2` artifact columns

Some runs have scalar `eval_v2/test_auroc` in the W&B summary but empty `eval_v2/roc_fpr` (and related JSON columns) in the analysis CSV because the batch summary omitted tables or the export skipped artifact download.

**Primary table:** `thesis_results/04_cleaned_backfilled_analysis_ready.csv` (see `docs/thesis_evidence_notes/ch8_analysis_ready_cleanup.md`).

## Re-export selected runs (W&B API)

From repo root with `WANDB_API_KEY` set:

```bash
python3 scripts/wandb/export/export_analysis_csv.py \
  --out /tmp/ch8_backfill_smoke.csv \
  --only-run-id RUN_ID_1 \
  --only-run-id RUN_ID_2
```

Omit `--no-singular-refetch` unless smoke-testing; singular `api.run()` refetch improves `eval_v2/*` summary completeness.

## Merge into the tracked CSV

Merge artifact-filled cells into `04` (or regenerate `03` from W&B then re-run `build_04_cleaned_analysis_ready.py --write`).

## W&B G-axis cohort patch

For the May 2026 `cohort=db_completeness_2026_05` runs, apply config fixes on W&B:

```bash
python3 scripts/wandb/patch_wandb_g_axes_from_csv.py \
  --csv thesis_results/04_cleaned_backfilled_analysis_ready.csv --limit 3
python3 scripts/wandb/patch_wandb_g_axes_from_csv.py --csv thesis_results/04_cleaned_backfilled_analysis_ready.csv --execute
```
