# Thesis ML: Modular Framework for Particle Physics Machine Learning

A production-ready, reproducible machine learning framework for particle physics experiments, built with Hydra configuration management and designed for both local development and HPC deployment.

## 🎯 Overview

This codebase provides a structured environment for training autoencoder variants, running experiments, and generating comparative analysis reports. It emphasizes:

- **Modularity**: Easy to add new architectures, training loops, and analysis types
- **Reproducibility**: Hydra-based configuration tracking and comprehensive facts logging
- **Scalability**: Seamless deployment from laptop to HPC cluster (Stoomboot at Nikhef)
- **Maintainability**: Clear separation between training, monitoring, and reporting phases

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Niels_repo

# Create conda environment
mamba env create -f environment.yml
mamba activate thesis-ml

# Install in editable mode
pip install -e .
```

### Run a Training

```bash
# Simple autoencoder training (local)
thesis-train

# Or using Python module
python -m thesis_ml.cli.train

```

### Generate a Report

```bash
# Compare multiple runs
thesis-report --config-name compare_tokenizers \
    inputs.sweep_dir=outputs/multiruns/exp_20251103_experiment

# Or using Python module
python -m thesis_ml.cli.reports --config-name compare_tokenizers \
    inputs.sweep_dir=outputs/multiruns/exp_20251103_experiment
```

## 📁 Project Structure

```
src/thesis_ml/
├── cli/                    # Command-line entry points
│   ├── train/              # Training CLI (thesis-train)
│   └── reports/            # Reports CLI (thesis-report)
│
├── training_loops/         # Training loop implementations
│   ├── autoencoder.py      # Standard autoencoder
│   ├── gan_autoencoder.py  # GAN-based autoencoder
│   ├── diffusion_autoencoder.py
│   └── simple_mlp.py       # Simple MLP for testing
│
├── architectures/          # Model architectures
│   ├── autoencoder/        # Autoencoder components
│   │   ├── encoders/       # Encoder architectures (MLP, GNN, etc.)
│   │   ├── decoders/       # Decoder architectures
│   │   ├── bottlenecks/    # Latent space types (VQ, linear, identity)
│   │   └── losses/         # Loss functions
│   └── simple/             # Simple architectures (MLP)
│
├── facts/                  # Training metrics & events system
│   ├── builders.py         # Build standardized event payloads
│   ├── writers.py          # Write facts to disk (JSONL, CSV)
│   └── readers.py          # Read facts for reports
│
├── monitoring/             # Training-time visualization
│   ├── orchestrator.py     # Route events to plot families
│   ├── io_utils.py         # Figure saving utilities
│   └── families/           # Plot families (losses, metrics, etc.)
│
├── reports/                # Post-training analysis
│   ├── analyses/           # Analysis implementations
│   │   ├── compare_tokenizers.py
│   │   └── compare_globals_heads.py
│   ├── inference/          # Inference utilities
│   ├── plots/              # Report plotting functions
│   └── utils/              # Report utilities
│
├── data/                   # Dataset loaders
│   ├── h5_loader.py        # HDF5 dataset loader
│   └── synthetic.py        # Synthetic data generation
│
└── utils/                  # General utilities
    ├── seed.py             # Reproducibility utilities
    ├── paths.py            # Path management
    └── training_progress_shower.py
```

## 🔑 Key Concepts

### Facts System

The **facts system** is the backbone of reproducibility and analysis:

- **Events** (`events.jsonl`): Lifecycle events (on_start, on_epoch_end, on_train_end) with full training histories
- **Scalars** (`scalars.csv`): Per-epoch metrics for easy DataFrame analysis
- **Purpose**: Enables post-hoc analysis without re-running expensive training

All training loops emit facts to `{run_dir}/facts/`. Reports read these facts to generate analyses.

### Training → Monitoring → Reports Pipeline

1. **Training**: Run a training loop (e.g., `autoencoder.py`)
   - Emits facts via `facts.writers`
   - Optionally creates real-time plots via `monitoring.orchestrator`

2. **Monitoring**: Real-time visualization during training
   - Plot families (losses, metrics, reconstruction, etc.)
   - Configured via `logging` config group

3. **Reports**: Post-training analysis
   - Read facts via `facts.readers`
   - Generate comparative plots
   - Run inference on test data
   - Output to `outputs/reports/`

### Environment Switching (Local ↔ HPC)

Switch between local and Stoomboot (Nikhef HPC) via Hydra:

```bash
# Local (default paths)
thesis-train env=local

# Stoomboot
thesis-train env=stoomboot
```

Paths are automatically configured:
- **Local**: Data in `C:\...\Data`, outputs in `outputs/`
- **Stoomboot**: Data in `/data/atlas/users/nterlind/datasets`, outputs in `/data/atlas/users/nterlind/outputs`

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System design and data flow
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**: Using and creating training code
- **[REPORTS_GUIDE.md](REPORTS_GUIDE.md)**: Using and creating reports
- **[FACTS_SYSTEM.md](FACTS_SYSTEM.md)**: Facts architecture in detail
- **[CONFIGS_GUIDE.md](CONFIGS_GUIDE.md)**: Hydra configuration patterns
- **[HPC_GUIDE.md](HPC_GUIDE.md)**: Running on Stoomboot cluster
- **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)**: Contributing and development setup

## 🎓 Typical Workflows

### Local Development

```bash
# 1. Quick smoke test (3 epochs, no artifacts)
thesis-train phase1.trainer.epochs=3 logging.save_artifacts=false

# 2. Full training run with plots
thesis-train phase1.trainer.epochs=20 logging=plots_standard

# 3. Experiment sweep (try different latent spaces)
thesis-train --multirun hydra=experiment \
    phase1/latent_space=none,linear,vq \
    phase1.trainer.epochs=20
```

### HPC Deployment

```bash
# Submit to Stoomboot cluster
condor_submit hpc/stoomboot/train.sub

# Monitor job
condor_q
```

### Analysis & Reporting

```bash
# Generate comparison report from sweep
thesis-report --config-name compare_tokenizers \
    inputs.sweep_dir=outputs/multiruns/exp_20251103-140953_experiment \
    inference.enabled=true

# Output: outputs/reports/report_TIMESTAMP_compare_tokenizers/
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_phase1_assembly.py

# Smoke test (minimal imports)
pytest tests/test_smoke.py
```

## 🛠️ Adding New Components

### New Training Loop

1. Create `src/thesis_ml/training_loops/my_loop.py`
2. Implement `def train(cfg: DictConfig) -> dict`
3. Register in `src/thesis_ml/cli/train/__init__.py`:
   ```python
   def _my_loop(cfg):
       from thesis_ml.training_loops.my_loop import train as _t
       return _t(cfg)

   DISPATCH = {
       ...,
       "my_loop": _my_loop,
   }
   ```
4. Run: `thesis-train loop=my_loop`

### New Architecture

1. Add encoder/decoder/bottleneck to `src/thesis_ml/architectures/autoencoder/`
2. Create config in `configs/phase1/encoder/` (or decoder/latent_space)
3. Run: `thesis-train phase1/encoder=my_encoder`

### New Report

1. Create `src/thesis_ml/reports/analyses/my_report.py`
2. Implement `def run_report(cfg: DictConfig) -> None`
3. Create config in `configs/report/my_report.yaml`
4. Run: `thesis-report --config-name my_report`

## 🤝 Contributing

See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for:
- Code style guidelines
- Testing requirements
- Git workflow
- Documentation standards

## 📊 Output Directory Structure

```
outputs/
├── runs/                   # Single training runs
│   └── run_YYYYMMDD-HHMMSS_[name]/
│       ├── .hydra/         # Hydra config snapshot
│       ├── facts/          # Training facts (events.jsonl, scalars.csv)
│       ├── figures/        # Training-time plots
│       ├── model.pt        # Saved model checkpoint
│       └── *.log           # Logs
│
├── multiruns/              # Multi-run experiments
│   └── exp_YYYYMMDD-HHMMSS_[name]/
│       └── (structure mirrors runs/)
│
└── reports/                # Generated reports
    └── report_YYYYMMDD-HHMMSS_[name]/
        ├── manifest.yaml   # Report metadata
        ├── training/       # Training analysis
        │   ├── summary.csv
        │   └── figures/
        └── inference/      # Inference results (optional)
            ├── summary.json
            └── figures/
```

## 🏆 Design Philosophy

1. **Configuration over Code**: Use Hydra to change behavior without editing Python
2. **Facts-First**: Training emits facts; reports consume facts. Clean separation.
3. **Fail-Fast Validation**: Catch config errors early with guardrails
4. **HPC-Ready**: One codebase, multiple environments, no code changes
5. **Extensibility**: Adding new components should be straightforward

## 📄 License

[Specify your license here]

## 👤 Author

Niels ter Linde - Master's Thesis, Particle Physics
