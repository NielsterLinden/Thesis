# Environment and Data Paths

Data input locations and output storage for local vs HPC (Stoomboot at Nikhef).

## HPC (Stoomboot) Layout

**Project directory:** `/project/atlas/users/nterlind/Thesis-Code`

**Data input:** `/data/atlas/users/nterlind/datasets/`
- `4tops_splitted.h5` — main 4-tops dataset (train/val/test splits)
- `4tops_5bins_ours.h5` — binned variant

**Output storage:** `/data/atlas/users/nterlind/outputs/`
- `runs/` — single-run outputs (`run_TIMESTAMP_NAME/`)
- `multiruns/` — multirun sweep outputs (`exp_TIMESTAMP_NAME/`)
- `reports/` — report outputs (`report_TIMESTAMP_NAME/`)
- `wandb/` — WandB offline run files (when `WANDB_DIR` points here)

**Other HPC paths:**
- Conda env: `/data/atlas/users/nterlind/venvs/thesis-ml`
- Logs: `/data/atlas/users/nterlind/logs/` (stderr, stdlog, stdout from Condor)

**Login node / long multiruns:** if you see `RuntimeError: Too many open files` from PyTorch DataLoader, raise the shell limit before training (e.g. `ulimit -n 8192`) or pass `data.num_workers=0`. Condor jobs run `hpc/stoomboot/train.sh`, which bumps `ulimit -n` when the scheduler allows it.

## Local Layout

**Data input:** `C:/Users/niels/Projects/Thesis-Code/Data`
- Place `4tops_splitted.h5` (or symlink) here for local training

**Output storage:** `C:/Users/niels/Projects/Thesis-Code/Code/Niels_repo/outputs/`
- Same structure: `runs/`, `multiruns/`, `reports/`

## Config Selection

Set `env=local` or `env=stoomboot` to switch paths. Configs live in `configs/env/`:
- `configs/env/local.yaml` — Windows paths
- `configs/env/stoomboot.yaml` — HPC paths

Data path is resolved via `data.path: "${env.data_root}/4tops_splitted.h5"` in `configs/data/h5_tokens.yaml`.

## HPC Project Layout (Stoomboot)

```
/project/atlas/users/nterlind/Thesis-Code/   # Code, configs, scripts
/data/atlas/users/nterlind/datasets/          # 4tops_splitted.h5, 4tops_5bins_ours.h5
/data/atlas/users/nterlind/outputs/          # runs/, multiruns/, reports/, wandb/
/data/atlas/users/nterlind/venvs/thesis-ml/  # Conda env
/data/atlas/users/nterlind/logs/             # Condor stderr, stdlog, stdout
```
