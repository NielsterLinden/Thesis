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

### HTCondor — parallel jobs

Always pass **`ROW_START`** and **`ROW_COUNT`** on each submit (defaults in the `.sub` file are only placeholders). Same `ROW_COUNT` every time; advance `ROW_START` by `ROW_COUNT` per job: `0`, `100`, `200`, …

- **`EVAL_PHASE_DIR`:** phase dir (`--out-dir` from Stage A).
- **`ROW_COUNT`:** width of each slice (e.g. `100`).
- **`ROW_START`:** first global index for that job (e.g. `0`, `100`, `200`, `300`).
- **Logs:** Condor **`log`** = scheduler events; app output in **`output`** (`.out`). `PYTHONUNBUFFERED=1` is set in the submit file.

```bash
EVAL_MANIFEST_CSV=/project/atlas/users/nterlind/Thesis-Code/wandb_cleanup/eval_pipeline/snapshots/2026-04-29_phase2/00_eval_manifest.csv
EVAL_PHASE_DIR=/project/atlas/users/nterlind/Thesis-Code/wandb_cleanup/eval_pipeline/snapshots/2026-04-29_phase2
SUB=/project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/eval_stage_b.sub

condor_submit "$SUB" EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" ROW_START=0   ROW_COUNT=100
condor_submit "$SUB" EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" ROW_START=100 ROW_COUNT=100
condor_submit "$SUB" EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" ROW_START=200 ROW_COUNT=100
condor_submit "$SUB" EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" ROW_START=300 ROW_COUNT=100
condor_submit "$SUB" EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" ROW_START=400 ROW_COUNT=100
condor_submit "$SUB" EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" ROW_START=500 ROW_COUNT=100
condor_submit "$SUB" EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" ROW_START=600 ROW_COUNT=100
condor_submit "$SUB" EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" ROW_START=700 ROW_COUNT=100
condor_submit "$SUB" EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" ROW_START=800 ROW_COUNT=100
condor_submit "$SUB" EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" ROW_START=900 ROW_COUNT=100
```

Use only the lines you need (`ROW_START` = 0, 100, … up to the last slice). Macros: `EVAL_MANIFEST_CSV`, `EVAL_PHASE_DIR`, `ROW_START`, `ROW_COUNT`, `QUEUE_COUNT`. Do not use a macro named `MANIFEST` (Condor reserved).

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
