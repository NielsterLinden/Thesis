# 🧠 Thesis

This repository provides a clean, configurable setup for **training and testing ML models** using [Hydra](https://hydra.cc), [PyTorch](https://pytorch.org/), and reproducible experiment management.

It is designed to be lightweight enough for quick iteration, but structured enough to scale with multiple training loops and models.

---

## 📁 Project Structure

```
src/thesis_ml/
│
├─ __init__.py                     # package initializer
│
├─ data/
│  ├─ __init__.py
│  └─ synthetic.py                 # synthetic dataset & dataloaders
│
├─ models/
│  ├─ __init__.py
│  └─ mlp.py                       # tiny configurable MLP (example model)
│
├─ utils/
│  ├─ __init__.py
│  ├─ plotting.py                  # deprecated shim → plots.orchestrator
│  └─ seed.py                      # set_all_seeds(seed) for reproducibility
│
├─ train/
│  ├─ __init__.py
│  ├─ __main__.py                  # CLI dispatcher (Hydra entrypoint)
│  └─ train_test.py                # example training loop (called from __main__)
│
├─ train.py                        # (if single-file version used)
│
├─ configs/
│  ├─ config.yaml                  # composition root for Hydra
│  ├─ data/
│  │  └─ synthetic.yaml            # dataset parameters (size, task, seed, split)
│  ├─ model/
│  │  └─ mlp.yaml                  # model hyperparameters
│  ├─ trainer/
│  │  └─ default.yaml              # training loop parameters
│  └─ logging/
│     ├─ default.yaml              # logging+plotting policy (families, moments)
│     ├─ plots_minimal.yaml        # preset overrides
│     ├─ plots_standard.yaml       # preset overrides
│     └─ plots_full.yaml           # preset overrides
│
└─ tests/
   └─ test_smoke.py                # simple import/forward test
```

---

## ⚙️ Configuration System (Hydra + OmegaConf)

All experiments are configured via YAML files under `configs/`.

Each top-level run composes these automatically:

| Config group | Purpose                             | Example file                   |
| ------------ | ----------------------------------- | ------------------------------ |
| `data`       | Dataset shape and type              | `configs/data/synthetic.yaml`  |
| `model`      | Architecture hyperparameters        | `configs/model/mlp.yaml`       |
| `trainer`    | Runtime training parameters         | `configs/trainer/default.yaml` |
| `logging`    | Artifact saving & plotting behavior | `configs/logging/default.yaml` |

> 🧩 **Why Hydra?**
> Hydra allows overrides from the command line, e.g.:
>
> ```bash
> python -m thesis_ml.train trainer.epochs=10 model.activation=gelu data.task=binary
> ```
>
> This ensures consistent, reproducible experiments without editing code.

---

## 🧱 Core Components

### **1. Data generation — `data/synthetic.py`**

* Creates reproducible synthetic datasets for regression or binary classification.
* Reads parameters from `cfg.data` (`n_samples`, `n_features`, `train_frac`, `seed`, `task`).
* Returns train/val `DataLoader`s and metadata (`input_dim`, `task`).

### **2. Model builder — `models/mlp.py`**

* Constructs an `nn.Sequential` MLP from config (`hidden_sizes`, `dropout`, `activation`).
* Returns a ready-to-train `torch.nn.Module`.
* The output dimension is inferred from the task (regression/binary).

### **3. Training loop — `train/train_test.py`**

* Core PyTorch loop: forward → loss → backward → optimizer step.
* Logs per-epoch train/validation losses (and accuracy if binary).
* Uses Adam optimizer, task-appropriate loss, and device auto-selection.
* Saves model and plots to `outputs/<timestamp>/` if `logging.save_artifacts=true`,
  otherwise runs ephemerally in a temporary directory.

### **4. Entry point — `train/__main__.py`**

* Hydra CLI launcher.
* Imports the desired training loop (`train_test.main`) and executes it.
* Keeps Hydra logic separate from the `train()` function so notebooks can import and run cleanly.

---

## 🧪 Usage

### **1. Environment setup**

```bash
mamba env create -f environment.yml
mamba activate thesis-ml
pre-commit install  # optional
```

### **2. CLI Training**

```bash
# Default config (ephemeral, no artifacts saved)
python -m thesis_ml.train

# Persistent run with artifacts saved
python -m thesis_ml.train logging.save_artifacts=true trainer.epochs=5
```

Artifacts (model, config, plots) are saved to:

```
outputs/YYYYMMDD-HHMMSS/
```

### **3. Notebook Usage**

```python
from omegaconf import OmegaConf
from thesis_ml.train.train_test import train

cfg = OmegaConf.load("configs/config.yaml")
cfg.trainer.epochs = 2
cfg.logging.save_artifacts = False

result = train(cfg)
print(result)
```

This runs the same logic **without** Hydra’s directory changes — perfect for Jupyter.

---

## 📊 Cross-run reports (VQ vs AE)

Generate sweep-level comparison reports across multiple `run_dir`s.

Requirements per run:

- `cfg.yaml`
- `facts/scalars.csv` with minimal columns: `epoch,split,val_loss,epoch_time_s`
- `facts/events.jsonl` containing an `on_train_end` record

CLI (PowerShell examples):

```powershell
python -m thesis_ml.reports --config-name compare_tokenizers inputs.run_dirs='["outputs/20251021-125542","outputs/20251021-131421"]' outputs.report_subdir=report outputs.which_figures='[val_mse_vs_time,throughput_vs_best_val]'
```

or a sweep directory:

```powershell
python -m thesis_ml.reports --config-name compare_tokenizers inputs.sweep_dir='outputs/20251021' outputs.report_subdir=report outputs.which_figures='[val_mse_vs_time,pareto_error_vs_compression,vq_perplexity_boxplot]'
```

Notes:

- Quote list arguments on Windows.
- Figures format/DPI are controlled by the report config, independent of logging policy.
- Outputs are written to `sweep_dir/report/` (or `<common_parent>/report/`) and include `summary.csv`, `summary.json`, and figures.


## 🧩 Logging and Artifact Control

Controlled by `configs/logging/default.yaml`:

Key plotting policy lives under `configs/logging/`. Important keys:

```yaml
save_artifacts: true
make_plots: true
show_plots: false
output_root: "outputs"
figures_subdir: "figures"
fig_format: "png"
dpi: 150
file_naming: "{family}-{moment}-{epoch_or_step}"
destinations: "file"
families:
  losses: true
  metrics: true
  recon:
    enabled: false
    mode: curves  # visuals optional
  codebook: true
  latency:
    enabled: true
    mode: light
moments:
  on_epoch_end_quick: true
  on_train_end_full: true
```

| Mode                   | Behavior                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| `save_artifacts=false` | Writes outputs to a temporary folder, deleted after run.                                        |
| `save_artifacts=true`  | Creates a timestamped folder under `outputs/` and saves `cfg.yaml`, `model.pt`, and `loss.png`. |

---

## 💡 Plot Families and Orchestrator

Training loops emit lifecycle events and JSONL facts. The orchestrator routes events to enabled families based on `logging` config. Figures go to `{run_dir}/figures`, facts to `{run_dir}/facts/`.

| Family   | Required inputs                              | Moments                                        | Cost      |
|----------|-----------------------------------------------|-----------------------------------------------|-----------|
| losses   | `run_dir`, `history_*_loss`                   | `on_epoch_end`, `on_train_end`                 | low       |
| metrics  | `run_dir`, `history_metrics` or `metrics`     | `on_epoch_end`, `on_train_end`                 | low       |
| recon    | curves: histories; visuals: `examples`+hook   | `on_epoch_end`, `on_validation_end`, `on_train_end` | med/heavy |
| codebook | `run_dir`, `history_perplex`/`history_codebook`| `on_epoch_end`, `on_train_end`                 | low       |
| latency  | `run_dir`, `history_epoch_time_s` (and thr)   | `on_epoch_end`, `on_train_end`                 | low/med   |

## 💡 Extending the project

1. **Add new models** under `src/thesis_ml/models/`.
2. **Add new training loops** under `src/thesis_ml/train/` (e.g., `vqae_loop.py`, `transformer_loop.py`).
3. Update `train/__main__.py` with a **dispatcher** if multiple training loops are supported:

   ```python
   DISPATCH = {"mlp": mlp_main, "vqae": vqae_main}
   ```
4. Add new config groups (`configs/model/<new_model>.yaml`, etc.).
5. Run:

   ```bash
   python -m thesis_ml.train trainer.loop=vqae
   ```

---

## 🧠 Design Philosophy

* **Hydra-first** configuration for reproducible and swappable experiments.
* **Separation of concerns**:
  * `data/` handles data generation/loading.
  * `models/` handles architecture creation.
  * `train/` handles training logic.
* **Notebook-friendly** imports (`train(cfg)`) without Hydra side-effects.
* **Minimal dependencies**, pure PyTorch, easily extended.

---

## 📦 Outputs and Experiment Tracking

Each run can produce:

```
outputs/
└── 20251015-134512/
    ├── cfg.yaml           # exact config used
    ├── model.pt           # trained model weights
    ├── loss.png           # loss curve
    └── logs/ (optional)
```

Use these folders to compare models, re-run with the same seeds, or analyze metrics.

---

## 🧰 Dependencies

Defined in `environment.yml`:

* `python=3.11`
* `pytorch`, `torchvision`, `torchaudio`
* `hydra-core`, `omegaconf`
* `numpy`, `pandas`, `matplotlib`, `scikit-learn`
* `black`, `ruff`, `pytest`, `jupyterlab`

---

## ✅ Quick sanity test

```bash
pytest -q
```

Ensures package imports, data generation, and a minimal training step all work.

---

## Phase 1 Standalone Autoencoders

Phase 1 introduces a unified autoencoder assembly with Hydra composition. Choose encoder, tokenizer, decoder, trainer, and logging via config overrides.

Run examples (do not execute here):

```bash
python -m thesis_ml.train logging=plots_minimal phase1/encoder=mlp phase1/decoder=mlp phase1/tokenizer=none phase1/trainer=ae
```

```bash
python -m thesis_ml.train logging=plots_standard phase1/encoder=mlp phase1/decoder=mlp phase1/tokenizer=vq phase1/trainer=ae
```

Artifacts are saved under `outputs/<stamp>/facts` and `outputs/<stamp>/figures`.
