# Codebase Overview

**Complete visual walkthrough of the thesis-ml codebase structure and workflows.**

## 📊 System Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          THESIS-ML SYSTEM                             │
│                                                                       │
│  Entry Points:  thesis-train    thesis-report    python notebook    │
│                      │               │                 │             │
└──────────────────────┼───────────────┼─────────────────┼─────────────┘
                       │               │                 │
          ┌────────────┴────────┐      │                 │
          │                     │      │                 │
          ▼                     ▼      ▼                 ▼
    ┌─────────┐          ┌──────────────────────────────────┐
    │ CLI/    │          │         CORE MODULES             │
    │ train   │          │                                  │
    │         │          │  ┌────────────┐  ┌────────────┐ │
    │DISPATCH │──────────┼─►│ Training   │  │Architec-   │ │
    └─────────┘          │  │ Loops      │  │tures       │ │
                         │  └──────┬─────┘  └────────────┘ │
    ┌─────────┐          │         │                        │
    │ CLI/    │          │         │                        │
    │ reports │          │         ▼                        │
    │         │          │  ┌────────────┐                 │
    │ analyze │──────────┼─►│   Facts    │                 │
    └─────────┘          │  │ Writers    │                 │
                         │  └──────┬─────┘                 │
                         │         │                        │
                         │         ▼                        │
                         │  ┌────────────┐  ┌────────────┐ │
                         │  │Monitoring  │  │   Data     │ │
                         │  │ (plots)    │  │  Loaders   │ │
                         │  └────────────┘  └────────────┘ │
                         └──────────────────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────┐
                         │   File System    │
                         │                  │
                         │  outputs/        │
                         │    ├─ runs/      │
                         │    │   ├─facts/  │
                         │    │   └─figures/│
                         │    └─ reports/   │
                         └──────────────────┘
                                    ▲
                                    │
                         ┌──────────┴───────┐
                         │   Facts Readers  │
                         │                  │
                         │   load_runs()    │
                         │   discover()     │
                         └──────────────────┘
```

## 📁 Directory Tree

```
thesis-ml/
│
├── configs/                    # Hydra configuration files
│   ├── config.yaml             # Root composition
│   ├── data/                   # Dataset configs
│   │   ├── h5_tokens.yaml
│   │   └── synthetic.yaml
│   ├── phase1/                 # Autoencoder configs
│   │   ├── encoder/            # Encoder architectures
│   │   ├── decoder/            # Decoder architectures
│   │   ├── latent_space/       # Bottleneck types
│   │   ├── trainer/            # Training hyperparameters
│   │   └── experiment/         # Pre-defined experiments
│   ├── logging/                # Monitoring/plot configs
│   ├── env/                    # Environment (local/stoomboot)
│   └── report/                 # Report configs
│
├── src/thesis_ml/              # Main package
│   │
│   ├── cli/                    # Command-line interface
│   │   ├── train/              # thesis-train entry point
│   │   │   ├── __main__.py     # Hydra CLI
│   │   │   └── __init__.py     # DISPATCH registry
│   │   └── reports/            # thesis-report entry point
│   │       └── __main__.py
│   │
│   ├── training_loops/         # Training implementations
│   │   ├── autoencoder.py      # Standard AE
│   │   ├── gan_autoencoder.py  # GAN AE
│   │   ├── diffusion_autoencoder.py
│   │   └── simple_mlp.py       # Test loop
│   │
│   ├── architectures/          # Model definitions
│   │   ├── autoencoder/
│   │   │   ├── assembly.py     # build_from_config()
│   │   │   ├── encoders/       # MLP, GNN, etc.
│   │   │   ├── decoders/       # MLP, GNN, etc.
│   │   │   ├── bottlenecks/    # VQ, linear, identity
│   │   │   └── losses/         # Reconstruction, adversarial
│   │   └── simple/
│   │       └── mlp.py
│   │
│   ├── facts/                  # Facts system (NEW!)
│   │   ├── builders.py         # build_event_payload()
│   │   ├── writers.py          # append_jsonl_event(), append_scalars_csv()
│   │   └── readers.py          # load_runs(), discover_runs()
│   │
│   ├── monitoring/             # Training-time visualization (was plots/)
│   │   ├── orchestrator.py     # handle_event()
│   │   ├── io_utils.py         # save_figure()
│   │   └── families/           # losses, metrics, recon, etc.
│   │
│   ├── reports/                # Post-training analysis
│   │   ├── analyses/           # Report implementations (was experiments/)
│   │   │   ├── compare_tokenizers.py
│   │   │   └── compare_globals_heads.py
│   │   ├── inference/          # Inference utilities
│   │   ├── plots/              # Report plotting
│   │   └── utils/              # IO, manifest, backlinks
│   │
│   ├── data/                   # Dataset loaders
│   │   ├── h5_loader.py
│   │   └── synthetic.py
│   │
│   └── utils/                  # General utilities
│       ├── seed.py
│       ├── paths.py
│       └── training_progress_shower.py
│
├── outputs/                    # All training/report outputs
│   ├── runs/                   # Single runs
│   │   └── run_YYYYMMDD-HHMMSS_name/
│   │       ├── .hydra/
│   │       ├── facts/
│   │       ├── figures/
│   │       └── *.pt
│   ├── multiruns/              # Sweeps
│   │   └── exp_YYYYMMDD-HHMMSS_name/
│   └── reports/                # Generated reports
│       └── report_YYYYMMDD-HHMMSS_name/
│
├── hpc/stoomboot/              # HPC submission scripts
│   ├── train.sh
│   ├── train.sub
│   ├── report.sh
│   └── report.sub
│
├── tests/                      # Test suite
├── scripts/                    # Utility scripts
└── notebooks/                  # Jupyter notebooks
```

## 🔄 Key Workflows

### Workflow 1: Train Locally

```
1. User: thesis-train phase1.trainer.epochs=20 phase1/latent_space=vq

2. CLI: Parse config via Hydra
   └─> configs/config.yaml + overrides

3. Dispatch: Select training loop
   └─> DISPATCH["ae"] → training_loops/autoencoder.py

4. Training Loop:
   ├─ Load data (data/h5_loader.py)
   ├─ Build model (architectures/autoencoder/assembly.py)
   │   ├─ Encoder (architectures/autoencoder/encoders/mlp.py)
   │   ├─ Bottleneck (architectures/autoencoder/bottlenecks/vq.py)
   │   └─ Decoder (architectures/autoencoder/decoders/mlp.py)
   ├─ Training epochs
   │   └─ Each epoch:
   │       ├─ Emit facts (facts/writers.py)
   │       │   ├─> facts/events.jsonl
   │       │   └─> facts/scalars.csv
   │       └─ Create plots (monitoring/orchestrator.py)
   │           └─> figures/*.png
   └─ Save checkpoints
       └─> best_val.pt, last.pt

5. Output: outputs/runs/run_YYYYMMDD-HHMMSS_/
```

### Workflow 2: Multi-Run Sweep

```
1. User: thesis-train --multirun phase1/latent_space=none,vq,linear

2. Hydra: Create 3 jobs (job0, job1, job2)

3. For each job:
   └─> Run Workflow 1

4. Output:
   ├─> outputs/multiruns/exp_YYYYMMDD-HHMMSS_experiment/
   └─> outputs/runs/
       ├─ run_YYYYMMDD-HHMMSS_experiment_job0/  # latent_space=none
       ├─ run_YYYYMMDD-HHMMSS_experiment_job1/  # latent_space=vq
       └─ run_YYYYMMDD-HHMMSS_experiment_job2/  # latent_space=linear
```

### Workflow 3: Generate Report

```
1. User: thesis-report --config-name compare_tokenizers \
          inputs.sweep_dir=outputs/multiruns/exp_*/

2. CLI: Load report config
   └─> configs/report/compare_tokenizers.yaml

3. Report Execution (reports/analyses/compare_tokenizers.py):
   ├─ Discover runs
   │   └─> facts/readers.py::discover_runs()
   │       └─> Find all runs matching sweep_dir
   │
   ├─ Load facts from all runs
   │   └─> facts/readers.py::load_runs()
   │       ├─ Read .hydra/config.yaml
   │       ├─ Read facts/events.jsonl
   │       ├─ Read facts/scalars.csv
   │       └─ Return (runs_df, per_epoch, order)
   │
   ├─ Aggregate & analyze
   │   ├─ Group by tokenizer
   │   ├─ Compute statistics
   │   └─ Filter/sort runs
   │
   ├─ Generate training plots
   │   └─> reports/plots/*.py
   │       └─> training/figures/*.png
   │
   ├─ (Optional) Run inference
   │   └─> reports/inference/*.py
   │       └─> inference/figures/*.png
   │
   └─ Save summaries
       ├─> training/summary.csv
       ├─> training/summary.json
       └─> manifest.yaml

4. Output: outputs/reports/report_YYYYMMDD-HHMMSS_compare_tokenizers/
```

### Workflow 4: HPC Submission

```
1. User (local): Edit hpc/stoomboot/train.sub
   └─> Set arguments: phase1.trainer.epochs=50 ...

2. User (local): git push

3. SSH to Stoomboot: ssh stoomboot

4. User (HPC):
   ├─ cd /project/atlas/users/nterlind/Thesis-Code
   ├─ git pull
   └─ condor_submit hpc/stoomboot/train.sub

5. Condor: Submit job to cluster

6. Job execution:
   ├─ Activate conda env
   ├─ Run: thesis-train env=stoomboot ...
   └─> (Workflow 1 with HPC paths)

7. Output: /data/atlas/users/nterlind/outputs/runs/...

8. Monitor: condor_q
```

## 🧩 Component Interaction Map

### Training Phase

```
User Command
    │
    ▼
cli/train/__main__.py
    │
    ├─ Load config (Hydra)
    ├─ Validate (legacy key check)
    └─ Dispatch to loop
         │
         ▼
training_loops/autoencoder.py
         │
         ├─ data/h5_loader → DataLoaders
         ├─ architectures/autoencoder/assembly → Model
         ├─ torch.optim → Optimizer
         │
         ├─ FOR each epoch:
         │   ├─ Forward/backward pass
         │   ├─ facts/writers → events.jsonl, scalars.csv
         │   └─ monitoring/orchestrator → figures/*.png
         │
         └─ Save checkpoints → *.pt
```

### Reporting Phase

```
User Command
    │
    ▼
cli/reports/__main__.py
    │
    ├─ Load config (Hydra)
    └─ Dispatch to report
         │
         ▼
reports/analyses/compare_tokenizers.py
         │
         ├─ facts/readers → (runs_df, per_epoch)
         ├─ Aggregate/filter runs
         ├─ reports/plots → training/figures/*.png
         ├─ reports/inference (optional) → inference/figures/*.png
         └─ Save summaries → *.csv, *.json
```

## 📋 Key Files Reference

### Entry Points

| File | Purpose | Command |
|------|---------|---------|
| `cli/train/__main__.py` | Training CLI | `thesis-train` |
| `cli/reports/__main__.py` | Reports CLI | `thesis-report` |

### Core Logic

| File | Purpose |
|------|---------|
| `training_loops/autoencoder.py` | Standard AE training |
| `architectures/autoencoder/assembly.py` | Build model from config |
| `facts/builders.py` | Create event payloads |
| `facts/writers.py` | Write facts to disk |
| `facts/readers.py` | Read facts from runs |
| `monitoring/orchestrator.py` | Route events to plot families |
| `reports/analyses/compare_tokenizers.py` | Compare VQ vs non-VQ |

### Configuration

| File | Purpose |
|------|---------|
| `configs/config.yaml` | Root composition |
| `configs/phase1/encoder/*.yaml` | Encoder configs |
| `configs/phase1/latent_space/*.yaml` | Bottleneck configs |
| `configs/phase1/trainer/ae.yaml` | Training hyperparameters |
| `configs/report/compare_tokenizers.yaml` | Report config |

## 🎯 Design Highlights

### 1. Facts-First Architecture
- **Training emits** → Facts written to disk
- **Reports consume** → Facts read from disk
- **Benefit**: Post-hoc analysis without re-training

### 2. Hydra-Driven Configuration
- **All parameters** via config files
- **CLI overrides** for flexibility
- **Reproducibility** via `.hydra/config.yaml` snapshot

### 3. Modular Components
- **Training loops**: Independent implementations
- **Architectures**: Composable encoders/decoders/bottlenecks
- **Reports**: Reusable analysis patterns

### 4. Environment Agnostic
- **Same code**, different configs
- **Local**: `env=local` (Windows paths)
- **HPC**: `env=stoomboot` (Linux paths)

## 🚀 Quick Reference Commands

```bash
# Training
thesis-train                                      # Default config
thesis-train phase1.trainer.epochs=20             # Override epochs
thesis-train phase1/latent_space=vq               # Use VQ bottleneck
thesis-train --multirun phase1/latent_space=none,vq  # Sweep

# Reports
thesis-report --config-name compare_tokenizers inputs.sweep_dir=outputs/multiruns/exp_*

# HPC
condor_submit hpc/stoomboot/train.sub
condor_q

# Development
pip install -e .
pytest
```

## 📚 Documentation Index

1. **[README.md](README.md)**: Project overview and quick start
2. **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed system design
3. **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**: Using/creating training code
4. **[REPORTS_GUIDE.md](REPORTS_GUIDE.md)**: Using/creating reports
5. **[FACTS_SYSTEM.md](FACTS_SYSTEM.md)**: Facts architecture
6. **[CONFIGS_GUIDE.md](CONFIGS_GUIDE.md)**: Hydra patterns (to be created)
7. **[HPC_GUIDE.md](HPC_GUIDE.md)**: Stoomboot usage (to be created)
8. **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)**: Development setup (to be created)

This document provides the 10,000-foot view. Dive into specific guides for details.
