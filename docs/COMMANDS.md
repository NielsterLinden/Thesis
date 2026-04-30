# Commands Reference

Training, reports, and HPC usage. For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. General Overview

### Options at a Glance

| | Local | HPC (Interactive) | HPC (Batch) |
|---|------|-------------------|-------------|
| **Training** | `thesis-train env=local` | SSH → `init_session.sh` → `thesis-train env=stoomboot` | `condor_submit train.sub -append 'arguments = env=stoomboot ...'` |
| **Reporting** | `thesis-report --config-name X inputs.sweep_dir=...` | Same, with HPC paths | `condor_submit report.sub -append 'arguments = --config-name X inputs.sweep_dir=...'` |

### Environment (`env`)

The `env` setting selects which paths to use for data and outputs. It loads from `configs/env/`:

- **`env=local`** — Uses `configs/env/local.yaml`. Data from `data_root`, outputs to `output_root` (e.g. `C:/Users/.../Data` and `.../outputs`).
- **`env=stoomboot`** — Uses `configs/env/stoomboot.yaml`. Data and outputs on HPC filesystem (e.g. `/data/atlas/users/nterlind/datasets`, `/data/atlas/users/nterlind/outputs`).

Always set `env=stoomboot` when running on HPC so paths resolve correctly. Default in `config.yaml` is `env: stoomboot`; override with `env=local` for local runs.

### Hydra Configuration

Hydra composes configs from `configs/`. Override any value from the CLI with `key=value` or `group/option=value`.

**Config groups:** `configs/phase1/` (encoder, decoder, latent_space, trainer, experiment), `configs/classifier/` (model, trainer, experiment), `configs/report/`, `configs/logging/`, `configs/data/`.

**Multirun:** Add `--multirun` and comma-separated values to sweep. Example: `phase1/latent_space=none,linear,vq` runs three jobs. Use `hydra=experiment` for sweep directory layout.

**Experiment configs:** Pre-defined sweeps in `configs/phase1/experiment/` and `configs/classifier/experiment/`. Select with `phase1/experiment=NAME` or `classifier/experiment=NAME`.

---

## 2. Training

**Entry points:** `thesis-train` or `python -m thesis_ml.cli.train`

**Loops:** `ae`, `gan_ae`, `diffusion_ae`, `test_mlp`, `transformer_classifier`, `mlp_classifier`, `bdt_classifier`. Default: `transformer_classifier`. Phase1 experiments require `loop=ae`.

### 2.1 Types of Training

#### 2.1.1 Single Run

One job with one config. Override parameters on the CLI; Hydra writes outputs to `outputs/runs/run_TIMESTAMP_NAME/`.

```bash
thesis-train env=local loop=ae phase1.trainer.epochs=20 phase1/latent_space=vq
```

#### 2.1.2 Multirun from CLI

Sweep over comma-separated values. Use `--multirun` and `hydra=experiment`. Hydra creates `outputs/multiruns/exp_TIMESTAMP_NAME/` and individual runs under `outputs/runs/run_*_job*/`.

```bash
thesis-train --multirun hydra=experiment phase1/latent_space=none,linear,vq phase1.trainer.epochs=20
```

#### 2.1.3 Multirun from Experiment

Use a pre-defined experiment config that specifies the sweep. Phase1: `phase1/experiment=compare_globals_heads` or `compare_latent_spaces`. Classifier: `classifier/experiment=exp_NAME` with `--multirun`.

```bash
thesis-train loop=ae phase1/experiment=compare_globals_heads
thesis-train loop=transformer_classifier classifier/experiment=exp_NAME --multirun
```

### 2.2 Training Locally

Run on your machine with `env=local`. Data and outputs use paths from `configs/env/local.yaml`.

```bash
thesis-train env=local loop=ae phase1.trainer.epochs=20
thesis-train env=local loop=ae phase1/experiment=compare_latent_spaces
```

### WandB on/off (logging)

- **Default:** `logging=default` has W&B **enabled** (`logging.use_wandb=true`), so runs are logged to W&B by default.
- **Disable W&B** (e.g. local smoke tests):

  ```bash
  thesis-train env=local logging=default logging.use_wandb=false ...
  ```

- **Force W&B online** (typical HPC run):

  ```bash
  thesis-train env=stoomboot logging=wandb_online ...
  # or equivalently
  thesis-train env=stoomboot logging=default logging.use_wandb=true logging.wandb.mode=online ...
  ```

- **Force W&B offline** (log locally, sync later):

  ```bash
  thesis-train env=stoomboot logging=wandb_offline ...
  # or equivalently
  thesis-train env=stoomboot logging=default logging.use_wandb=true logging.wandb.mode=offline ...
  ```

### 2.3 Training on HPC

#### 2.3.1 Interactive

SSH to Stoomboot, source the init script (activates conda, sets WandB, cd to project), then run training commands with `env=stoomboot`.

```bash
ssh stbc-i1.nikhef.nl
source /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/init_session.sh
thesis-train env=stoomboot loop=ae phase1/experiment=compare_globals_heads
```

#### 2.3.2 Batch

Submit a job via Condor. Pass arguments with `-append 'arguments = ...'`. The job runs `thesis_ml.cli.train` with those arguments.

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=ae phase1/experiment=compare_globals_heads'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=exp_NAME logging=wandb_online --multirun'
```

**Monitor:** `condor_q`, `condor_tail -f <job_id>`, `condor_rm <job_id>`. Logs: `/data/atlas/users/nterlind/logs/`.

### 2.3.3 Phase2 eval — Stage B (GPU, ten jobs)

Same **`-append 'arguments = ...'`** pattern as training. Each line is one cluster job; `--row-start` is a **literal** (evaluable rows sorted by `source_created_at`, half-open `[start, start+count)`). Shards: `EVAL_PHASE_DIR/shards/rows_<start>_<end>/`.

Defaults in `eval_stage_b.sub`: `EVAL_MANIFEST_CSV`, `EVAL_PHASE_DIR`, `ROW_COUNT=100`. Override paths on the command line if needed, e.g. `EVAL_PHASE_DIR=/other/phase`.

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

Merge shard CSVs for Stage C:

```bash
python wandb_cleanup/eval_pipeline/merge_stage_b_shards.py \
  --phase-dir /project/atlas/users/nterlind/Thesis-Code/wandb_cleanup/eval_pipeline/snapshots/2026-04-29_phase2
```

---

## 3. Reporting

**Entry points:** `thesis-report` or `python -m thesis_ml.cli.reports`

Reports read facts from runs and generate comparative analysis. Use `inputs.sweep_dir` (not `+sweep_dir`) to point at a multirun directory. Output: `outputs/reports/report_TIMESTAMP_NAME/`.

### 3.1 Reporting Locally

Point `inputs.sweep_dir` at a local multirun path (e.g. `outputs/multiruns/exp_*_NAME`).

```bash
thesis-report --config-name compare_tokenizers inputs.sweep_dir=outputs/multiruns/exp_*_NAME
thesis-report --config-name compare_tokenizers inputs.sweep_dir=outputs/multiruns/exp_*_NAME inference.enabled=true
```

### 3.2 Reporting on HPC

#### 3.2.1 Interactive

SSH, source init script, run `thesis-report` with HPC paths for `inputs.sweep_dir`.

```bash
ssh stbc-i1.nikhef.nl
source /project/atlas/users/nterlind/Thesis-Code/hpc/stoomboot/init_session.sh
thesis-report --config-name compare_tokenizers inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_NAME inference.enabled=true
```

#### 3.2.2 Batch

Submit a report job via Condor. Pass `--config-name`, `inputs.sweep_dir`, and optional overrides in the arguments.

```bash
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name compare_tokenizers inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_NAME inference.enabled=true'
```

---

## HPC Paths

| Purpose | Path |
|---------|------|
| Project | `/project/atlas/users/nterlind/Thesis-Code` |
| Outputs | `/data/atlas/users/nterlind/outputs` |
| Data | `/data/atlas/users/nterlind/datasets` |
| Conda | `/data/atlas/users/nterlind/venvs/thesis-ml` |
| Logs | `/data/atlas/users/nterlind/logs/` |
