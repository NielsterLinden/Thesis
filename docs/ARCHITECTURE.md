# System Architecture

Overview of thesis-ml architecture, data flow, and component responsibilities. For practical commands, see [COMMANDS.md](COMMANDS.md).

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────── ─┐
│  thesis-train             thesis-report          notebooks       │
└──────┼─────────────────────────┼─────────────────────┼───────────┘
       ▼                         ▼                     ▼
┌──────────────────┐    ┌─────────────────┐   ┌─────────────────┐
│   cli/train/     │    │  cli/reports/   │   │  Direct Import  │
│   DISPATCH       │    │                 │   │                 │
└────────┬─────────┘    └────────┬────────┘   └────────┬────────┘
         ▼                       ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Training Loops  │     Reports (analyses/)  │   Architectures   │
│  ae, gan_ae,     │     facts.readers        │   autoencoder/    │
│  diffusion_ae,   │     load_runs()          │   simple/         │
│  transformer_    │                          │   transformer_    │
│  classifier,     │                          │   classifier/     │
│  mlp_classifier, │                          │                   │
│  bdt_classifier  │                          │                   │
└────────┬─────────┴────────────┬───────────  ┴───────────────────┘
         ▼                      ▼
   ┌─────────────┐       ┌─────────────┐
   │ Facts       │       │ Data        │
   │ builders,   │       │ h5_loader,  │
   │ writers     │       │ synthetic   │
   └──────┬──────┘       └─────────────┘
          ▼
   outputs/runs/  outputs/multiruns/  outputs/reports/
```

## Data Flow

**Training:** `thesis-train` → Hydra loads `configs/config.yaml` → DISPATCH selects loop → training loop loads data, builds model, runs epochs → emits facts to `{run_dir}/facts/` → optionally creates plots via `monitoring.orchestrator`.

**Reporting:** `thesis-report --config-name X inputs.sweep_dir=...` → loads `configs/report/X.yaml` → `facts.readers.discover_runs()` and `load_runs()` → report generates plots and optionally runs inference → saves to `outputs/reports/report_*_X/`.

## Component Responsibilities

| Component | Purpose |
|-----------|---------|
| `cli/train/` | Hydra entry point, DISPATCH to training loops |
| `cli/reports/` | Hydra entry point, load report config, invoke `run_report()` |
| `training_loops/` | `train(cfg) -> dict`; load data, build model, emit facts |
| `architectures/` | Model definitions (autoencoder, simple, transformer_classifier) |
| `facts/` | builders (event payloads), writers (events.jsonl, scalars.csv), readers (load_runs, discover) |
| `monitoring/` | Optional real-time plots during training |
| `reports/` | analyses/, inference/, plots/, utils/ |
| `data/` | h5_loader, synthetic |

## Config Layout

```
configs/
├── config.yaml          # Root (env, data, phase1, classifier, logging)
├── env/                 # local, stoomboot (data_root, output_root)
├── phase1/              # encoder, decoder, latent_space, trainer, experiment
├── classifier/          # model, trainer, experiment
├── report/              # Report configs (compare_tokenizers, etc.)
├── logging/             # default, wandb_online, wandb_offline, plots_*
└── hydra/               # experiment (multirun sweep dir)
```

Default loop: `transformer_classifier`. Phase1 experiments use `loop=ae`.

## Facts System

**Events** (`facts/events.jsonl`): Lifecycle events (on_start, on_epoch_end, on_train_end) with full training histories. JSON Lines format.

**Scalars** (`facts/scalars.csv`): Per-epoch metrics for DataFrame analysis. Columns: epoch, split, train_loss, val_loss, epoch_time_s, throughput, metric_*.

**Meta** (`facts/meta.json`): Run classification for W&B and reports. Key fields: schema_version, level, goal, dataset_name, model_family, process_groups, datatreatment. See `docs/archive/METADATA_SCHEMA.md` for full schema.

Training loops emit via `facts.builders.build_event_payload()` and `facts.writers.append_jsonl_event()` / `append_scalars_csv()`. Reports consume via `facts.readers.load_runs()`.

## Paths

| Context | Runs | Multiruns | Reports |
|---------|------|-----------|---------|
| Local (`env=local`) | `outputs/runs/` | `outputs/multiruns/exp_*` | `outputs/reports/` |
| HPC (`env=stoomboot`) | `${output_root}/runs/` | `${output_root}/multiruns/exp_*` | `${output_root}/reports/` |

HPC paths (stoomboot): `data_root=/data/atlas/users/nterlind/datasets`, `output_root=/data/atlas/users/nterlind/outputs`, conda at `/data/atlas/users/nterlind/venvs/thesis-ml`, project at `/project/atlas/users/nterlind/Thesis-Code`.

## Extension Points

**New training loop:** Create `training_loops/my_loop.py` with `def train(cfg) -> dict`, emit facts, register in `cli/train/__init__.py` DISPATCH.

**New report:** Create `reports/analyses/my_report.py` with `def run_report(cfg) -> None`, add `configs/report/my_report.yaml`, use `facts.readers.load_runs()`.

**New architecture:** Add to `architectures/autoencoder/` or `architectures/transformer_classifier/`, create config in `configs/phase1/` or `configs/classifier/`.

## WandB

Core: `utils/wandb_utils.py`. Configs: `logging=wandb_online` or `wandb_offline`. Scripts: `scripts/wandb/` (migrate, sync, backfill_labels, cleanup). Auth: `hpc/stoomboot/.wandb_env` or `~/.wandb_api_key`.

## File System Conventions

**Run dir:** `outputs/runs/run_TIMESTAMP_NAME/` contains `.hydra/config.yaml`, `facts/events.jsonl`, `facts/scalars.csv`, `figures/`, `best_val.pt`.

**Report dir:** `outputs/reports/report_TIMESTAMP_NAME/` contains `manifest.yaml`, `training/summary.csv`, `training/figures/`, optionally `inference/`.
