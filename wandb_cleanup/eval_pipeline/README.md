# Phase 2: W&B eval / re-inference pipeline

CSV-driven stages **A → B → C → D** under this directory. Policies live in `config/eval_spec.yaml` (checkpoint order, G3→task map, dtype) and `config/test_splits.yaml` (per-task HDF5 + label config).

## Prerequisites

- Python env with `torch`, `thesis_ml` deps (same as training), `pandas`, `omegaconf`, `sklearn`.
- HPC paths: code under `/project/atlas/users/nterlind/Thesis-Code`, data under `/data/atlas/users/nterlind/datasets/`, runs under `/data/atlas/users/nterlind/outputs/runs/`.
- Stage A only needs `omegaconf` (no PyTorch).

## Stage A — manifest

```bash
python wandb_cleanup/eval_pipeline/stage_a_manifest.py \
  --raw-csv wandb_cleanup/backfill_pipeline/snapshots/2026-04-29_raw/00_raw_export.csv \
  --out-dir wandb_cleanup/eval_pipeline/snapshots/MY_PHASE2_RUN \
  --runs-root /data/atlas/users/nterlind/outputs/runs \
  --limit 0
```

Writes `00_eval_manifest.csv`. Check `task_status` value counts.

## Stage B — inference (GPU)

Each Condor job writes **only** under one **shard directory** (parallel-safe).

- **Row range:** evaluable rows (sorted by `source_created_at` descending) use **half-open** indices `[row_start, row_start + row_count)`. Example: `--row-start 0 --row-count 100` → global indices **0–99**; `--row-start 100 --row-count 100` → **100–199**.
- **Default shard path:** `<phase-dir>/shards/rows_<start:05d>_<end:05d>/` where `<end>` is **exclusive** (e.g. `rows_00000_00100` holds indices 0..99).
- Contents per shard: `01_eval_results.csv`, `failures/<run_id>_traceback.txt`, `run_log.txt`.
- **`--out-dir`:** phase snapshot directory (same as Stage A `--out-dir`).
- **`--shard-dir`:** optional; overrides the default shard directory.

Inference is **per run** (each row gets its own merged Hydra config and test dataloader).

```bash
export PYTHONPATH=/project/atlas/users/nterlind/Thesis-Code/src
python wandb_cleanup/eval_pipeline/stage_b_inference.py \
  --manifest wandb_cleanup/eval_pipeline/snapshots/MY_PHASE2_RUN/00_eval_manifest.csv \
  --out-dir wandb_cleanup/eval_pipeline/snapshots/MY_PHASE2_RUN \
  --row-start 0 \
  --row-count 100 \
  --limit 5
```

- **`--task`:** optional filter to one `task_canonical`.
- **`--no-resume`:** ignore existing rows in **this shard’s** CSV (delete the shard dir for a clean rewrite).
- **Failures:** full traceback in `failures/<run_id>_traceback.txt`; Condor **stdout** (`.out`) prints progress plus `exception_only` / `traceback_tail` on failure.

### HTCondor — ten separate submits

Use **`-append 'arguments = ...'`** (same pattern as training in [docs/COMMANDS.md](../docs/COMMANDS.md) §2.3.3). Each command is one GPU job; `--row-start` is a literal `0`, `100`, …, `900`.

```bash
SUB=hpc/stoomboot/eval_stage_b.sub

condor_submit $SUB -append 'arguments = --manifest $(EVAL_MANIFEST_CSV) --out-dir $(EVAL_PHASE_DIR) --row-start 0 --row-count $(ROW_COUNT)'
condor_submit $SUB -append 'arguments = --manifest $(EVAL_MANIFEST_CSV) --out-dir $(EVAL_PHASE_DIR) --row-start 100 --row-count $(ROW_COUNT)'
condor_submit $SUB -append 'arguments = --manifest $(EVAL_MANIFEST_CSV) --out-dir $(EVAL_PHASE_DIR) --row-start 200 --row-count $(ROW_COUNT)'
condor_submit $SUB -append 'arguments = --manifest $(EVAL_MANIFEST_CSV) --out-dir $(EVAL_PHASE_DIR) --row-start 300 --row-count $(ROW_COUNT)'
condor_submit $SUB -append 'arguments = --manifest $(EVAL_MANIFEST_CSV) --out-dir $(EVAL_PHASE_DIR) --row-start 400 --row-count $(ROW_COUNT)'
condor_submit $SUB -append 'arguments = --manifest $(EVAL_MANIFEST_CSV) --out-dir $(EVAL_PHASE_DIR) --row-start 500 --row-count $(ROW_COUNT)'
condor_submit $SUB -append 'arguments = --manifest $(EVAL_MANIFEST_CSV) --out-dir $(EVAL_PHASE_DIR) --row-start 600 --row-count $(ROW_COUNT)'
condor_submit $SUB -append 'arguments = --manifest $(EVAL_MANIFEST_CSV) --out-dir $(EVAL_PHASE_DIR) --row-start 700 --row-count $(ROW_COUNT)'
condor_submit $SUB -append 'arguments = --manifest $(EVAL_MANIFEST_CSV) --out-dir $(EVAL_PHASE_DIR) --row-start 800 --row-count $(ROW_COUNT)'
condor_submit $SUB -append 'arguments = --manifest $(EVAL_MANIFEST_CSV) --out-dir $(EVAL_PHASE_DIR) --row-start 900 --row-count $(ROW_COUNT)'
```

Defaults live in `hpc/stoomboot/eval_stage_b.sub` (`EVAL_MANIFEST_CSV`, `EVAL_PHASE_DIR`, `ROW_COUNT`). Edit the `.sub` file if paths or slice list change.

### Merge shards (before Stage C)

```bash
python wandb_cleanup/eval_pipeline/merge_stage_b_shards.py \
  --phase-dir wandb_cleanup/eval_pipeline/snapshots/MY_PHASE2_RUN
```

Writes `01_eval_results_merged.csv` in the phase dir (override with `--out`). Point Stage C at this file.

## Stage C — aggregate

```bash
python wandb_cleanup/eval_pipeline/stage_c_aggregate.py \
  --raw-csv wandb_cleanup/backfill_pipeline/snapshots/2026-04-29_raw/00_raw_export.csv \
  --results wandb_cleanup/eval_pipeline/snapshots/MY_PHASE2_RUN/01_eval_results_merged.csv \
  --out-dir wandb_cleanup/eval_pipeline/snapshots/MY_PHASE2_RUN
```

Produces `02_eval_combined.csv`, `schema_validation_report.txt`, `anomalies_report.csv`.

## Stage D — W&B push

Dry-run by default (writes `push_log.csv` only):

```bash
python wandb_cleanup/eval_pipeline/stage_d_push.py \
  --combined-csv wandb_cleanup/eval_pipeline/snapshots/MY_PHASE2_RUN/02_eval_combined.csv \
  --out-dir wandb_cleanup/eval_pipeline/snapshots/MY_PHASE2_RUN \
  --entity YOUR_ENTITY \
  --project thesis-ml
```

Add `--confirm` to call the API: updates **only** `eval_v2/*` scalar keys in run summary; ROC/PR/histogram JSON columns go to a per-run **Artifact** `eval_v2_curves_<run_id>`.

## Library hook

Classifier checkpoint priority (`best_val` → `last` → `model` → `epoch_*`) and `load_classifier_from_run_dir()` live in `src/thesis_ml/reports/utils/inference.py` for reuse outside this pipeline.
