# Thesis ML

Local project for event-level inference in multi-top physics.

## Quickstart
```powershell
mamba env create -f environment.yml
mamba activate thesis-ml
pre-commit install
```

## Layout
- src/thesis_ml/ â€” library code
- notebooks/ â€” experiments
- scripts/ â€” cli utilities
- configs/ â€” Hydra/YAML configs
- tests/ â€” unit tests
- outputs/ â€” run artifacts (logs, checkpoints)

## Data
Set DATA_ROOT in your .env to C:\Users\niels\Projects\Thesis-Code\Data. Never commit data.
