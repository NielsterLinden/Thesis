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
- **`--task`:** optional filter to one `task_canonical` (loads one test tensor family per job).
- **`--no-resume`:** rewrite from scratch (default resumes using `run_id` in `01_eval_results.csv`).
- **Failures:** tracebacks in `failures/<run_id>_traceback.txt`; CSV row with `eval_v2/checkpoint_status=failed_*`.

HTCondor: edit paths in `hpc/stoomboot/eval_stage_b.sub`, set `queue` to number of batches, then `condor_submit hpc/stoomboot/eval_stage_b.sub`. Jobs run `hpc/stoomboot/thesis_inference.sh` (GPU + thesis-ml conda, same style as `train.sh`); stdout/stderr/stdlog paths are in the submit file.

## Stage C — aggregate

```bash
python wandb_cleanup/eval_pipeline/stage_c_aggregate.py \
  --raw-csv wandb_cleanup/backfill_pipeline/snapshots/2026-04-29_raw/00_raw_export.csv \
  --results wandb_cleanup/eval_pipeline/snapshots/MY_PHASE2_RUN/01_eval_results.csv \
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
