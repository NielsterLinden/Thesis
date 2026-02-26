# System Architecture

This document provides a detailed overview of the thesis-ml system architecture, component responsibilities, and data flow patterns.

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                        │
│                                                                  │
│  thesis-train             thesis-report          notebooks       │
│      │                         │                     │           │
└──────┼─────────────────────────┼─────────────────────┼───────────┘
       │                         │                     │
       ▼                         ▼                     ▼
┌──────────────────┐    ┌─────────────────┐   ┌─────────────────┐
│   CLI Layer      │    │  Reports CLI    │   │  Direct Import  │
│                  │    │                 │   │                 │
│  cli/train/      │    │  cli/reports/   │   │  (notebooks)    │
│  - __main__.py   │    │  - __main__.py  │   │                 │
│  - DISPATCH      │    │                 │   │                 │
└────────┬─────────┘    └────────┬────────┘   └────────┬────────┘
         │                       │                      │
         │                       │                      │
         ▼                       ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CORE EXECUTION LAYER                        │
│                                                                  │
│  ┌─────────────────┐     ┌──────────────┐    ┌───────────────┐ │
│  │ Training Loops  │     │   Reports    │    │ Architectures │ │
│  │                 │     │              │    │               │ │
│  │ - autoencoder   │     │ - analyses/  │    │ - autoencoder/│ │
│  │ - gan_ae        │     │ - inference/ │    │ - simple/     │ │
│  │ - diffusion_ae  │     │ - plots/     │    │               │ │
│  │ - simple_mlp    │     │              │    │               │ │
│  └────────┬────────┘     └──────┬───────┘    └───────┬───────┘ │
│           │                     │                    │          │
└───────────┼─────────────────────┼────────────────────┼──────────┘
            │                     │                    │
            │                     │                    │
            ▼                     ▼                    │
   ┌─────────────────┐   ┌─────────────────┐          │
   │  Facts System   │   │  Facts Readers  │          │
   │                 │   │                 │          │
   │  - builders     │   │  - load_runs()  │          │
   │  - writers      │◄──┤  - discover     │          │
   │                 │   │                 │          │
   └────────┬────────┘   └─────────────────┘          │
            │                                          │
            │                     ┌────────────────────┘
            ▼                     ▼
   ┌─────────────────┐   ┌─────────────────┐
   │  Monitoring     │   │      Data       │
   │                 │   │                 │
   │  - orchestrator │   │  - h5_loader    │
   │  - families/    │   │  - synthetic    │
   │                 │   │                 │
   └─────────────────┘   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │   File System   │
   │                 │
   │  outputs/runs/  │
   │    ├─ facts/    │
   │    └─ figures/  │
   │                 │
   └─────────────────┘
```

## 🔄 Data Flow: Training Phase

### 1. Initialization

```
User Command
    │
    ▼
thesis-train [args]
    │
    ▼
cli/train/__main__.py
    │
    ├─ Load Hydra config from configs/
    ├─ Validate legacy keys
    └─ Dispatch to training loop via DISPATCH dict
         │
         ▼
training_loops/autoencoder.py::train(cfg)
```

### 2. Training Loop Execution

```
train(cfg)
    │
    ├─ 1. Set random seeds (reproducibility)
    │      └─ utils/seed.py::set_all_seeds()
    │
    ├─ 2. Load data
    │      └─ data/h5_loader.py::make_dataloaders()
    │
    ├─ 3. Build model
    │      └─ architectures/autoencoder/base.py::build_from_config()
    │          ├─ Build encoder (from config)
    │          ├─ Build bottleneck (VQ/linear/identity)
    │          └─ Build decoder (from config)
    │
    ├─ 4. Create optimizer
    │      └─ torch.optim.Adam(...)
    │
    ├─ 5. Setup run directory
    │      └─ Hydra handles this (outputs/runs/run_TIMESTAMP_NAME/)
    │
    ├─ 6. Emit on_start event
    │      ├─ facts/builders.py::build_event_payload()
    │      ├─ facts/writers.py::append_jsonl_event()
    │      └─ monitoring/orchestrator.py::handle_event()
    │
    ├─ 7. Training epochs
    │      │
    │      FOR each epoch:
    │          ├─ Run train batches (forward, backward, optimizer step)
    │          ├─ Run validation batches
    │          ├─ Compute metrics
    │          │
    │          ├─ Emit on_epoch_end event
    │          │   ├─ build_event_payload(histories=...)
    │          │   ├─ append_jsonl_event()
    │          │   ├─ append_scalars_csv()  # CSV for easy DataFrame loading
    │          │   └─ handle_event()  # Creates figures if enabled
    │          │
    │          ├─ Save best checkpoint if val loss improved
    │          └─ Update progress bar
    │
    ├─ 8. Test evaluation
    │      └─ Run test batches
    │
    ├─ 9. Emit on_train_end event
    │      ├─ build_event_payload(total_time_s=...)
    │      ├─ append_jsonl_event()
    │      └─ handle_event()  # Final plots
    │
    └─ 10. Return results dict
            └─ {"best_val_loss": ..., "test_loss": ...}
```

### 3. Facts Emission

Each training loop emits standardized events:

```python
# Build payload
payload = build_event_payload(
    moment="on_epoch_end",
    run_dir=outdir,
    epoch=ep,
    train_loss=tr["loss"],
    val_loss=va["loss"],
    metrics={"perplex": ...},
    histories={
        "train_loss": [0.5, 0.4, 0.3, ...],
        "val_loss": [0.6, 0.5, 0.4, ...],
        ...
    },
    cfg=cfg,  # For metadata extraction
)

# Write to disk
append_jsonl_event(run_dir, payload)      # events.jsonl
append_scalars_csv(run_dir, epoch=ep, ...) # scalars.csv

# Optionally create plots
handle_event(cfg.logging, families, "on_epoch_end", payload)
```

## 📊 Data Flow: Reporting Phase

### 1. Report Invocation

```
thesis-report --config-name compare_tokenizers \
    inputs.sweep_dir=outputs/multiruns/exp_TIMESTAMP_NAME

    │
    ▼
cli/reports/__main__.py
    │
    ├─ Load report config from configs/report/
    ├─ Infer report module from config name
    │   └─ reports/analyses/compare_tokenizers.py
    │
    ├─ Extract data config from first run (for inference)
    └─ Call run_report(cfg)
```

### 2. Report Execution

```
reports/analyses/compare_tokenizers.py::run_report(cfg)
    │
    ├─ 1. Setup environment
    │      ├─ Create report directories
    │      │   ├─ outputs/reports/report_TIMESTAMP_NAME/
    │      │   ├─   ├─ training/
    │      │   └─   └─ inference/
    │      │
    │      └─ Discover runs
    │          └─ facts/readers.py::discover_runs(sweep_dir)
    │
    ├─ 2. Load facts from all runs
    │      └─ facts/readers.py::load_runs()
    │          ├─ Read .hydra/config.yaml
    │          ├─ Read facts/events.jsonl
    │          ├─ Read facts/scalars.csv
    │          ├─ Extract metadata
    │          ├─ Compute aggregates
    │          └─ Return (runs_df, per_epoch, order)
    │
    ├─ 3. Filter runs (optional)
    │      └─ Based on cfg.inputs.select
    │
    ├─ 4. Save training summary
    │      ├─ training/summary.csv
    │      └─ training/summary.json
    │
    ├─ 5. Generate training plots
    │      ├─ reports/plots/curves.py::plot_loss_vs_time()
    │      ├─ Custom analysis plots
    │      └─ Save to training/figures/
    │
    ├─ 6. Run inference (if enabled)
    │      ├─ Load models from runs
    │      ├─ reports/inference/forward_pass.py
    │      ├─ reports/inference/metrics.py
    │      ├─ reports/inference/anomaly_detection.py
    │      ├─ Compute AUROC, reconstruction errors
    │      └─ Save to inference/figures/
    │
    └─ 7. Finalize
           ├─ Create manifest.yaml
           ├─ Create backlinks to runs
           └─ Log completion
```

## 🗂️ Component Responsibilities

### `cli/`

**Purpose**: Command-line interface entry points

- `cli/train/`: Training CLI
  - `__main__.py`: Hydra entry point, validation, dispatch
  - `__init__.py`: DISPATCH dictionary mapping loop names to functions

- `cli/reports/`: Reports CLI
  - `__main__.py`: Hydra entry point, report discovery, invocation

**Responsibilities**:
- Parse command-line arguments via Hydra
- Validate configuration
- Dispatch to appropriate training loop or report
- Handle errors gracefully

### `training_loops/`

**Purpose**: Implement training procedures

Each file implements a `train(cfg: DictConfig) -> dict` function:

- `autoencoder.py`: Standard autoencoder training
- `gan_autoencoder.py`: GAN-based autoencoder
- `diffusion_autoencoder.py`: Diffusion-based autoencoder
- `simple_mlp.py`: Simple MLP for testing/debugging

**Responsibilities**:
- Load data via `data/`
- Build model via `architectures/`
- Run training loop
- Emit facts via `facts.writers`
- Optionally create real-time plots via `monitoring.orchestrator`
- Save checkpoints
- Return final metrics

**Key Pattern**:
```python
def train(cfg: DictConfig) -> dict:
    # Setup
    set_all_seeds(cfg.phase1.trainer.seed)
    device = ...

    # Data & model
    train_dl, val_dl, test_dl, meta = make_dataloaders(cfg)
    model = build_from_config(cfg).to(device)
    opt = torch.optim.Adam(...)

    # Training loop
    for epoch in range(cfg.phase1.trainer.epochs):
        # Train, validate, emit facts
        ...

    # Return
    return {"best_val_loss": ..., "test_loss": ...}
```

### `architectures/`

**Purpose**: Model architecture definitions

- `autoencoder/`: Autoencoder components
  - `base.py`: Assembly logic (`build_from_config`)
  - `encoders/`: Encoder modules (MLP, GNN, diffusion, GAN)
  - `decoders/`: Decoder modules (MLP, GNN, diffusion, GAN)
  - `bottlenecks/`: Latent space transformations (VQ, linear, identity)
  - `losses/`: Loss functions (reconstruction, adversarial, diffusion)

- `simple/`: Simple architectures
  - `mlp.py`: Basic MLP builder

**Responsibilities**:
- Define PyTorch nn.Module classes
- Provide builder functions that accept config dicts
- Remain agnostic to training procedure
- Focus on forward pass logic

**Key Pattern**:
```python
def build_encoder(cfg, input_dim, latent_dim):
    """Build encoder from config."""
    return EncoderMLP(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=cfg.phase1.encoder.hidden_dims,
        ...
    )
```

### `facts/`

**Purpose**: Standardized event and metric logging

- `builders.py`: Create event payloads
  - `build_event_payload()`: Constructs standardized dict with metadata

- `writers.py`: Write facts to disk
  - `append_jsonl_event()`: Append to `facts/events.jsonl`
  - `append_scalars_csv()`: Append to `facts/scalars.csv`
  - `ensure_facts_dir()`: Create facts directory

- `readers.py`: Read facts from runs
  - `discover_runs()`: Find run directories
  - `load_runs()`: Load and aggregate facts into DataFrames
  - Helper functions for metadata extraction

**Responsibilities**:
- Define standardized fact schema (schema_version=1)
- Provide consistent API for emitting facts
- Enable efficient reading for reports
- Extract metadata from Hydra configs

**Key Invariant**: Training loops emit, reports consume. No direct coupling.

### `monitoring/`

**Purpose**: Real-time visualization during training

- `orchestrator.py`: Route events to plot families
  - `handle_event()`: Main dispatch function

- `families/`: Plot family implementations
  - `losses.py`: Loss curves
  - `metrics.py`: Metric plots
  - `recon.py`: Reconstruction visualizations
  - `codebook.py`: VQ codebook analysis
  - `latency.py`: Training speed plots
  - `adversarial.py`: GAN-specific plots
  - `diffusion.py`: Diffusion-specific plots

- `io_utils.py`: Figure utilities
  - `ensure_figures_dir()`, `build_filename()`, `save_figure()`

**Responsibilities**:
- Create figures during training (optional, configured via `logging` group)
- Respect `cfg.logging.families` and `cfg.logging.moments` settings
- Save figures to `{run_dir}/figures/`
- Never crash training (fail-safe, log warnings)

**Key Pattern**:
```python
def handle_event(cfg_logging, supported_families, moment, payload):
    families = get_enabled_families(cfg_logging, supported_families, moment)

    for family in families:
        figs = family.handle(moment, payload, cfg_logging)
        for fig in figs:
            save_figure(fig, figures_dir, ...)
```

### `reports/`

**Purpose**: Post-training analysis and inference

- `analyses/`: Report implementations
  - Each file implements `run_report(cfg) -> None`
  - `compare_tokenizers.py`: Compare VQ vs non-VQ
  - `compare_globals_heads.py`: Analyze globals reconstruction

- `inference/`: Inference utilities
  - `forward_pass.py`: Run models on test data
  - `metrics.py`: Compute reconstruction metrics
  - `anomaly_detection.py`: AUROC for anomaly detection
  - `data_corruption.py`: Data corruption strategies

- `plots/`: Report plotting functions
  - `curves.py`: Loss vs time plots
  - `grids.py`: Grid visualizations
  - `scatter.py`: Scatter plots
  - `anomaly.py`: Anomaly detection plots

- `utils/`: Report utilities
  - `io.py`: File I/O helpers
  - `manifest.py`: Manifest generation
  - `backlinks.py`: Create backlinks to runs
  - `inference.py`: Inference orchestration

**Responsibilities**:
- Read facts via `facts.readers`
- Aggregate and compare runs
- Generate comparative plots
- Optionally run inference
- Save outputs to `outputs/reports/report_TIMESTAMP_NAME/`

### `data/`

**Purpose**: Dataset loading and preprocessing

- `h5_loader.py`: Load HDF5 datasets
- `synthetic.py`: Generate synthetic data for testing

**Responsibilities**:
- Return PyTorch DataLoaders
- Return metadata dict (e.g., `{"n_tokens": 100, "cont_dim": 4}`)
- Handle train/val/test splits
- Remain agnostic to model architecture

### `utils/`

**Purpose**: General-purpose utilities

- `seed.py`: `set_all_seeds()` for reproducibility
- `paths.py`: Path management (run IDs, report IDs)
- `training_progress_shower.py`: ASCII progress bars

**Responsibilities**:
- Provide reusable utilities
- No domain-specific logic
- No dependencies on other thesis_ml modules (except minimal imports)

## 🔐 Key Design Decisions

### 1. Facts-First Architecture

**Decision**: Training loops emit facts; reports consume facts. No direct coupling.

**Rationale**:
- **Reproducibility**: Re-run analysis without re-training
- **Efficiency**: Analysis is cheap; training is expensive
- **Flexibility**: Add new analyses without modifying training code
- **Debugging**: Inspect facts to diagnose issues

**Tradeoff**: Requires consistent fact schema (currently `schema_version=1`)

### 2. Hydra-Driven Configuration

**Decision**: Use Hydra for all configuration, no hardcoded parameters.

**Rationale**:
- **Reproducibility**: `.hydra/config.yaml` captures exact config
- **Flexibility**: Override any parameter from CLI
- **Modularity**: Compose configs from groups
- **Scalability**: Same code, different environments (local/HPC)

**Tradeoff**: Learning curve for Hydra composition

### 3. Separate CLI and Logic

**Decision**: `cli/` contains entry points; logic lives in `training_loops/` and `reports/`.

**Rationale**:
- **Testability**: Import and call `train(cfg)` directly in tests/notebooks
- **Reusability**: Use training loops without CLI
- **Clarity**: Separation of concerns

**Tradeoff**: Slightly more files

### 4. Monitoring is Optional

**Decision**: Training loops emit events; monitoring creates plots if enabled.

**Rationale**:
- **Performance**: Disable plotting on HPC for speed
- **Flexibility**: Different plot policies for different runs
- **Robustness**: Plotting failures don't crash training

**Tradeoff**: Monitoring code must handle missing data gracefully

### 5. Namespace Preservation

**Decision**: Keep `thesis_ml/` as root package, not top-level `train/`, `data/`, etc.

**Rationale**:
- **Avoid Collisions**: Generic names like `train`, `utils` clash with other packages
- **Clean Imports**: `from thesis_ml.facts import ...` is clear
- **pip install**: Works seamlessly with editable installs

**Tradeoff**: Slightly longer import paths

## 🔄 Extension Points

### Adding a New Training Loop

1. Create `training_loops/my_loop.py`
2. Implement `def train(cfg: DictConfig) -> dict`
3. Emit facts via `facts.writers`
4. Register in `cli/train/__init__.py`

### Adding a New Architecture Component

1. Create `architectures/autoencoder/{encoders,decoders,bottlenecks}/my_component.py`
2. Create config in `configs/phase1/{encoder,decoder,latent_space}/my_component.yaml`
3. Update `architectures/autoencoder/base.py` if new type

### Adding a New Report

1. Create `reports/analyses/my_report.py`
2. Implement `def run_report(cfg: DictConfig) -> None`
3. Create config in `configs/report/my_report.yaml`
4. Use `facts.readers.load_runs()` to get data

### Adding a New Plot Family

1. Create `monitoring/families/my_family.py`
2. Implement handler with `handle(moment, payload, cfg) -> list[Figure]`
3. Register in `monitoring/registry.py`

### WandB Integration

**Core code:** `src/thesis_ml/utils/wandb_utils.py` — `init_wandb()`, `extract_wandb_config()`, `log_metrics()`, `finish_wandb()`, `log_artifact()`

**Scripts:** `scripts/wandb/` — `cleanup_wandb.py`, `migrate_runs_to_wandb.py`, `sync_wandb.sh`, `test_wandb_hpc.py`, `backfill_labels.py`

**Configs:** `configs/logging/` — `wandb_online`, `wandb_offline`, `default`

**Auth:** `hpc/stoomboot/.wandb_env` (local and HPC; gitignored). Create with `export WANDB_API_KEY="your_key_here"`.

**SOP for new config keys:**

1. Add key to Hydra config YAML
2. Optionally add curated extraction in `extract_wandb_config()` for clean dashboard UX
3. `raw/*` auto-flatten ensures new keys are never lost
4. Run `python scripts/wandb/backfill_labels.py --dry-run --labels '{"new/key": "default"}'` to stamp old runs with the default value

## 📁 File System Conventions

### Run Directory Structure

```
outputs/runs/run_20251103-140953_experiment_job0/
├── .hydra/
│   ├── config.yaml          # Canonical config snapshot
│   ├── overrides.yaml       # CLI overrides
│   └── hydra.yaml           # Hydra runtime config
├── facts/
│   ├── events.jsonl         # Lifecycle events (one per line)
│   └── scalars.csv          # Per-epoch metrics (DataFrame-friendly)
├── figures/                 # Training-time plots (optional)
│   ├── losses-on_epoch_end-e019.png
│   └── ...
├── model.pt                 # Symlink/copy of best_val.pt
├── best_val.pt              # Best validation checkpoint
└── last.pt                  # Final epoch checkpoint
```

### Report Directory Structure

```
outputs/reports/report_20251103-142813_compare_tokenizers/
├── manifest.yaml            # Report metadata
├── training/
│   ├── summary.csv          # Aggregated metrics across runs
│   ├── summary.json         # Metadata, sweep params
│   └── figures/
│       ├── figure-val_mse_vs_time.png
│       └── ...
└── inference/               # Optional
    ├── summary.json         # Inference results
    └── figures/
        ├── figure-reconstruction_error_distributions.png
        └── ...
```

## 🚀 Performance Considerations

### Training

- **Disable plots on HPC**: `logging.save_artifacts=false` or `logging.make_plots=false`
- **Use GPU**: Automatically detected via `torch.cuda.is_available()`
- **Batch size**: Tune via config (`data.batch_size`)

### Reports

- **Inference is expensive**: Only enable when needed (`inference.enabled=true`)
- **DataFrame operations**: `facts.readers` returns pandas DataFrames for speed
- **Parallel loading**: Can load runs in parallel (future enhancement)

## 🔍 Debugging Tips

### Training Issues

1. Check `.hydra/config.yaml` for actual config used
2. Check `facts/events.jsonl` for emitted events
3. Check `facts/scalars.csv` for per-epoch metrics
4. Enable `logging.make_plots=true` to visualize training

### Report Issues

1. Check `manifest.yaml` for report metadata
2. Check `training/summary.csv` for aggregated metrics
3. Verify runs have `on_train_end` event in `facts/events.jsonl`
4. Check logs for warnings about skipped runs

### Import Issues

1. Ensure editable install: `pip install -e .`
2. Check Python path: `echo $PYTHONPATH`
3. Verify package structure: `python -c "import thesis_ml; print(thesis_ml.__file__)"`

## 🎯 Future Enhancements

- **Distributed training**: Add PyTorch DDP support
- **Experiment tracking**: Integrate W&B/MLflow
- **Auto-scaling**: Adjust batch size based on GPU memory
- **Caching**: Cache data loaders for faster iteration
- **Profiling**: Add performance profiling hooks
- **Schema evolution**: Support multiple fact schema versions
