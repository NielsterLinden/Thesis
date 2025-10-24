# CLI Usage Guide

## Output Directory Structure

All outputs are organized under `outputs/`:

```
outputs/
├── runs/              # Single training runs
│   └── run_YYYYMMDD-HHMMSS_[name]/
└── experiments/       # Multi-run experiments/sweeps
    └── exp_YYYYMMDD-HHMMSS_[name]/
        ├── 0/         # First run
        ├── 1/         # Second run
        └── ...
```

## Single Runs

### Basic run (default name "run")
```powershell
python -m thesis_ml.train
# Output: outputs/runs/run_20251023-173000_run/
```

### Run with custom name
```powershell
python -m thesis_ml.train hydra.job.name=baseline_test
# Output: outputs/runs/run_20251023-173000_baseline_test/
```

### Override epochs
```powershell
python -m thesis_ml.train trainer.epochs=5
```

## Experiments (Multirun/Sweeps)

### Basic multirun sweep (default name "experiment")
```powershell
python -m thesis_ml.train --multirun hydra=experiment phase1/latent_space=none,vq
# Output: outputs/experiments/exp_20251023-180000_experiment/0/, /1/
```

### Multirun with custom name
```powershell
python -m thesis_ml.train --multirun hydra=experiment hydra.job.name=compare_tokenizers phase1/latent_space=none,vq
# Output: outputs/experiments/exp_20251023-180000_compare_tokenizers/0/, /1/
```

### Multirun with multiple parameters
```powershell
python -m thesis_ml.train --multirun hydra=experiment hydra.job.name=grid_search phase1/latent_space=none,vq trainer.epochs=3,5,10
# Creates 6 runs (2 latent_space × 3 epoch settings)
```

## Experiment Configs

Experiment configs in `configs/phase1/experiment/` can define:
- Default sweep parameters
- Specific logging configurations
- Experiment metadata

### Using an experiment config
```powershell
python -m thesis_ml.train --config-name=phase1/experiment/compare_vq_vs_none_mlp_ae
# Uses experiment.name from config: exp_20251023-180000_compare_vq_vs_none/
```

## Common Overrides

### Data
```powershell
python -m thesis_ml.train data=synthetic
```

### Logging
```powershell
python -m thesis_ml.train logging=plots_minimal
python -m thesis_ml.train logging=plots_full
```

### Model architecture
```powershell
python -m thesis_ml.train phase1/encoder=gnn phase1/decoder=gnn
```

### Latent space configuration
```powershell
python -m thesis_ml.train phase1/latent_space=vq
python -m thesis_ml.train phase1/latent_space=none
```

## Report Generation

After running experiments, generate comparison reports:

```powershell
python -m thesis_ml.phase1.reports.compose_experiment --config-name=report/compare_tokenizers experiment.slug=my_experiment run_dirs='["outputs/experiments/exp_*/0","outputs/experiments/exp_*/1"]'
```

The report system automatically extracts sweep parameters and creates:
- `summary.csv` - Aggregated metrics
- `summary.json` - Metadata including sweep_params
- `figures/` - Comparison plots
