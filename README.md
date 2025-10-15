# Thesis ML — Minimal Hydra Project + Notebook Driver

This repository contains a minimal, self-contained PyTorch project with Hydra configs and a hybrid workflow:

- Training logic in `src/` (importable)
- CLI entrypoint via `python -m thesis_ml.train` (Hydra)
- Notebook driver that calls the same `train(cfg)`
- Synthetic data (no external files)
- Simple loss plot and artifact toggle

## Setup
```powershell
mamba env create -f environment.yml
mamba activate thesis-ml
pre-commit install
```

## Run
CLI (Hydra):
```bash
python -m thesis_ml.train trainer.epochs=2 logging.save_artifacts=true
```

Notebook:
```bash
jupyter lab
# open notebooks/00_driver.ipynb and run
```

## Artifacts policy
- When `logging.save_artifacts=false` (default), outputs are written to a temporary directory and removed at the end of the run.
- When `logging.save_artifacts=true`, a run directory is created under `outputs/YYYYMMDD-HHMMSS/` containing:
  - `cfg.yaml`
  - `model.pt`
  - `loss.png` (if `logging.make_plots=true`)

## Configs
Top-level composition is defined in `configs/config.yaml`, with sub-configs for `data`, `model`, `trainer`, and `logging`.

## Tests
```bash
pytest -q
```
