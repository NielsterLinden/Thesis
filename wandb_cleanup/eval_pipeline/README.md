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

Each Condor job (or local run) writes **only** under a **shard directory** so many jobs can run in parallel:

- Default shard path: `<phase-dir>/shards/batch_<batch_index:03d>/` (e.g. `.../shards/batch_000/`).
- Contents per shard: `01_eval_results.csv`, `failures/<run_id>_traceback.txt`, `run_log.txt`.
- **`--out-dir`** is the phase snapshot directory (same as Stage A `--out-dir`).
- **`--shard-dir`**: optional override for the shard root (must be empty or resumable via that shard’s CSV only).

Inference is **per run**: each row gets its own merged Hydra config and test dataloader (robustness over speed).

From repo root with `PYTHONPATH=src` (or conda env that has `thesis_ml` installed):

```bash
export PYTHONPATH=/project/atlas/users/nterlind/Thesis-Code/src
python wandb_cleanup/eval_pipeline/stage_b_inference.py \
  --manifest wandb_cleanup/eval_pipeline/snapshots/MY_PHASE2_RUN/00_eval_manifest.csv \
  --out-dir wandb_cleanup/eval_pipeline/snapshots/MY_PHASE2_RUN \
  --batch-index 0 \
  --batch-size 100 \
  --limit 5
```

- **Ordering:** evaluable runs sorted by `source_created_at` descending; batch slice is `[batch_index * batch_size, ...)`.
- **`--task`:** optional filter to one `task_canonical`.
- **`--no-resume`:** ignore existing rows in **this shard’s** `01_eval_results.csv` (delete the shard file first if you want a clean rewrite).
- **Failures:** full traceback in `failures/<run_id>_traceback.txt`; CSV row with `eval_v2/checkpoint_status=failed_*`. Condor **stdout** (`.out`) prints one line per run (`status`, `elapsed`) plus `exception_only` and a short `traceback_tail` on failure.

### HTCondor — parallel jobs (100 runs per slice)

- **`EVAL_PHASE_DIR`:** same directory you passed as `--out-dir` to Stage A (contains `shards/` after jobs run).
- **`BATCH_SLICE`:** integer batch index `0, 1, 2, …` (maps to `shards/batch_000`, `batch_001`, …).
- **`BATCH_SIZE`:** runs per job (default `100`).
- Submit **one job per slice** you need: `N = ceil(num_evaluable / BATCH_SIZE)` (e.g. 400 evaluable → `BATCH_SLICE` 0–3).
- **Logs:** `log` in the submit file is the **Condor event log** (scheduler). **Application** progress and errors are in **`output`** (stdout `.out`) and **`error`** (stderr `.err`). `PYTHONUNBUFFERED=1` is set in the submit file so `.out` updates live.

Example: set variables then submit slices `0`–`9` (use only `0`–`N-1` for your manifest):

```bash
EVAL_MANIFEST_CSV=/project/atlas/users/nterlind/Thesis-Code/wandb_cleanup/eval_pipeline/snapshots/2026-04-29_phase2/00_eval_manifest.csv
EVAL_PHASE_DIR=/project/atlas/users/nterlind/Thesis-Code/wandb_cleanup/eval_pipeline/snapshots/2026-04-29_phase2

condor_submit /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/eval_stage_b.sub \
  EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" BATCH_SLICE=0 BATCH_SIZE=100
condor_submit /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/eval_stage_b.sub \
  EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" BATCH_SLICE=1 BATCH_SIZE=100
condor_submit /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/eval_stage_b.sub \
  EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" BATCH_SLICE=2 BATCH_SIZE=100
condor_submit /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/eval_stage_b.sub \
  EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" BATCH_SLICE=3 BATCH_SIZE=100
condor_submit /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/eval_stage_b.sub \
  EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" BATCH_SLICE=4 BATCH_SIZE=100
condor_submit /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/eval_stage_b.sub \
  EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" BATCH_SLICE=5 BATCH_SIZE=100
condor_submit /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/eval_stage_b.sub \
  EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" BATCH_SLICE=6 BATCH_SIZE=100
condor_submit /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/eval_stage_b.sub \
  EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" BATCH_SLICE=7 BATCH_SIZE=100
condor_submit /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/eval_stage_b.sub \
  EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" BATCH_SLICE=8 BATCH_SIZE=100
condor_submit /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/eval_stage_b.sub \
  EVAL_MANIFEST_CSV="$EVAL_MANIFEST_CSV" EVAL_PHASE_DIR="$EVAL_PHASE_DIR" BATCH_SLICE=9 BATCH_SIZE=100
```

Macros: `EVAL_MANIFEST_CSV`, `EVAL_PHASE_DIR`, `BATCH_SLICE`, `BATCH_SIZE`, `QUEUE_COUNT`. Do not use a macro named `MANIFEST` (Condor reserved).

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
