# Thesis ML

Modular machine learning framework for particle physics experiments. Hydra-based configuration, facts logging, and deployment from local to HPC (Stoomboot at Nikhef).

## Quick Start

```bash
mamba env create -f environment.yml
mamba activate thesis-ml
pip install -e .
```

```bash
# Train (local)
thesis-train env=local

# Report from sweep
thesis-report --config-name compare_tokenizers inputs.sweep_dir=outputs/multiruns/exp_*_NAME
```

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — System design, configs, data flow, Facts
- **[docs/COMMANDS.md](docs/COMMANDS.md)** — Training, reports, HPC (interactive + Condor)
- **[docs/SCRIPTS.md](docs/SCRIPTS.md)** — Script reference
