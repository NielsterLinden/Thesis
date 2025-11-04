# Training Guide

Complete guide for running training and creating new training code.

## 📖 Table of Contents

- [Using Existing Training Code](#using-existing-training-code)
- [Creating New Training Code](#creating-new-training-code)

---

## Using Existing Training Code

### Quick Start

```bash
# Basic training (local, default config)
thesis-train

# With custom parameters
thesis-train phase1.trainer.epochs=20 phase1/latent_space=vq

# Multiple runs (sweep)
thesis-train --multirun phase1/latent_space=none,linear,vq
```

### Available Training Loops

| Loop Name | File | Description |
|-----------|------|-------------|
| `ae` | `autoencoder.py` | Standard autoencoder (default) |
| `gan_ae` | `gan_autoencoder.py` | GAN-based autoencoder |
| `diffusion_ae` | `diffusion_autoencoder.py` | Diffusion-based autoencoder |
| `test_mlp` | `simple_mlp.py` | Simple MLP for testing |

**Usage**: `thesis-train loop=ae` (or `gan_ae`, `diffusion_ae`, `test_mlp`)

### Common Configuration Overrides

#### Training Parameters

```bash
# Number of epochs
thesis-train phase1.trainer.epochs=50

# Learning rate
thesis-train phase1.trainer.lr=0.001

# Batch size
thesis-train data.batch_size=256

# Random seed
thesis-train phase1.trainer.seed=42
```

#### Architecture Selection

```bash
# Encoder type
thesis-train phase1/encoder=mlp  # or gnn, gan, diffusion

# Decoder type
thesis-train phase1/decoder=mlp  # or gnn, gan, diffusion

# Latent space type
thesis-train phase1/latent_space=none   # identity (no quantization)
thesis-train phase1/latent_space=linear # learned linear projection
thesis-train phase1/latent_space=vq     # vector quantization
```

#### Logging & Monitoring

```bash
# Logging presets
thesis-train logging=plots_minimal   # minimal plots
thesis-train logging=plots_standard  # standard plots
thesis-train logging=plots_full      # all plots

# Disable saving artifacts (ephemeral run)
thesis-train logging.save_artifacts=false

# Custom plot families
thesis-train logging.families.losses.enabled=true \
             logging.families.recon.enabled=false
```

### Running on HPC (Stoomboot)

#### Interactive

```bash
# SSH to Stoomboot
ssh stoomboot

# Navigate to code
cd /project/atlas/users/nterlind/Thesis-Code

# Activate environment
conda activate /data/atlas/users/nterlind/venvs/thesis-ml

# Run with HPC paths
thesis-train env=stoomboot phase1.trainer.epochs=20
```

#### Batch Submission

```bash
# Submit job
condor_submit hpc/stoomboot/train.sub

# Monitor
condor_q

# Check logs
tail -f /data/atlas/users/nterlind/logs/stdout/train_*.out
```

**Edit `train.sub` for custom arguments:**

```condor
arguments = env=stoomboot phase1/latent_space=vq phase1.trainer.epochs=50
```

### Running Sweeps (Multiruns)

Sweep over parameter combinations:

```bash
# Simple sweep
thesis-train --multirun \
    hydra=experiment \
    phase1/latent_space=none,vq \
    phase1.trainer.epochs=10,20

# Named experiment
thesis-train --multirun \
    hydra=experiment \
    hydra.job.name=latent_comparison \
    phase1/latent_space=none,linear,vq
```

**Output structure:**

```
outputs/multiruns/exp_20251103-140953_latent_comparison/
outputs/runs/
├── run_20251103-140953_latent_comparison_job0/  # none
├── run_20251103-140953_latent_comparison_job1/  # linear
└── run_20251103-140953_latent_comparison_job2/  # vq
```

### Using Experiment Configs

Pre-defined experiment configs in `configs/phase1/experiment/`:

```bash
# Run predefined experiment
thesis-train --config-name=phase1/experiment/compare_vq_vs_none

# Override parameters
thesis-train --config-name=phase1/experiment/compare_vq_vs_none \
    phase1.trainer.epochs=50
```

### Monitoring Training

#### During Training

- **Progress bar**: Shows epoch, ETA, losses
- **Figures**: Created in `{run_dir}/figures/` if `logging.make_plots=true`
- **Logs**: Standard output shows key metrics

#### After Training

```bash
# View final config
cat outputs/runs/run_*/. hydra/config.yaml

# View training facts
cat outputs/runs/run_*/facts/events.jsonl | jq .
cat outputs/runs/run_*/facts/scalars.csv

# View figures
ls outputs/runs/run_*/figures/
```

### Output Directory Structure

```
outputs/runs/run_20251103-140953_experiment/
├── .hydra/
│   ├── config.yaml          # Full config snapshot
│   ├── overrides.yaml       # CLI overrides used
│   └── hydra.yaml           # Hydra runtime config
├── facts/
│   ├── events.jsonl         # Training events (on_start, on_epoch_end, etc.)
│   └── scalars.csv          # Per-epoch metrics (epoch, split, losses, etc.)
├── figures/                 # Training-time plots (if enabled)
│   ├── losses-on_epoch_end-e019.png
│   ├── metrics-on_epoch_end-e019.png
│   └── ...
├── model.pt                 # Symlink to best_val.pt
├── best_val.pt              # Best validation checkpoint
└── last.pt                  # Final epoch checkpoint
```

### Troubleshooting

#### Issue: CUDA out of memory

**Solutions:**
1. Reduce batch size: `data.batch_size=128`
2. Reduce model size: modify encoder/decoder hidden dims in configs
3. Use gradient accumulation (future feature)

#### Issue: Training very slow

**Solutions:**
1. Disable plots: `logging.make_plots=false`
2. Increase batch size if memory allows
3. Check GPU usage: `nvidia-smi`

#### Issue: Config validation errors

**Solutions:**
1. Check `.hydra/config.yaml` for actual resolved config
2. Ensure config group exists: `ls configs/phase1/latent_space/`
3. Validate override syntax: `phase1.trainer.epochs=20` (not `phase1.trainer.epochs 20`)

---

## Creating New Training Code

### Adding a New Training Loop

#### 1. Create Training Loop File

Create `src/thesis_ml/training_loops/my_loop.py`:

```python
from __future__ import annotations

import time
import torch
from omegaconf import DictConfig

from thesis_ml.data.h5_loader import make_dataloaders
from thesis_ml.architectures.autoencoder.base import build_from_config
from thesis_ml.facts import append_jsonl_event, append_scalars_csv, build_event_payload
from thesis_ml.monitoring.orchestrator import handle_event
from thesis_ml.utils import TrainingProgressShower
from thesis_ml.utils.seed import set_all_seeds

SUPPORTED_PLOT_FAMILIES = {"losses", "metrics"}  # Customize as needed


def train(cfg: DictConfig) -> dict:
    """My custom training loop.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration

    Returns
    -------
    dict
        Training results
    """
    # 1. Setup
    set_all_seeds(cfg.phase1.trainer.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load data
    train_dl, val_dl, test_dl, meta = make_dataloaders(cfg)

    # 3. Build model
    model = build_from_config(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.phase1.trainer.lr)

    # 4. Get run directory
    import os
    outdir = os.getcwd() if cfg.logging.save_artifacts else None

    # 5. Emit on_start event
    if outdir:
        start_payload = build_event_payload(
            moment="on_start",
            run_dir=outdir,
            cfg=cfg,
        )
        append_jsonl_event(outdir, start_payload)
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_start", start_payload)

    # 6. Training loop
    histories = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.phase1.trainer.epochs):
        # Training step
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            # ... training logic ...
            pass

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                # ... validation logic ...
                pass

        # Store histories
        histories["train_loss"].append(train_loss)
        histories["val_loss"].append(val_loss)

        # Emit on_epoch_end event
        if outdir:
            payload = build_event_payload(
                moment="on_epoch_end",
                run_dir=outdir,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                histories=histories,
                cfg=cfg,
            )
            append_jsonl_event(outdir, payload)
            append_scalars_csv(outdir, epoch=epoch, split="val",
                             train_loss=train_loss, val_loss=val_loss,
                             metrics={}, epoch_time_s=None, throughput=None,
                             max_memory_mib=None)
            handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_epoch_end", payload)

    # 7. Test evaluation
    test_loss = 0.0  # ... compute test loss ...

    # 8. Emit on_train_end event
    if outdir:
        payload_end = build_event_payload(
            moment="on_train_end",
            run_dir=outdir,
            histories=histories,
            cfg=cfg,
        )
        append_jsonl_event(outdir, payload_end)
        handle_event(cfg.logging, SUPPORTED_PLOT_FAMILIES, "on_train_end", payload_end)

    # 9. Return results
    return {
        "final_val_loss": histories["val_loss"][-1],
        "test_loss": test_loss,
    }
```

#### 2. Register in Dispatcher

Edit `src/thesis_ml/cli/train/__init__.py`:

```python
def _my_loop(cfg):
    from thesis_ml.training_loops.my_loop import train as _t
    return _t(cfg)

DISPATCH = {
    "ae": _ae,
    "gan_ae": _gan_ae,
    "diffusion_ae": _diffusion_ae,
    "test_mlp": _mlp,
    "my_loop": _my_loop,  # Add this line
}
```

#### 3. (Optional) Create Config

If your loop needs custom config, create `configs/my_loop_trainer.yaml`:

```yaml
my_loop:
  epochs: 20
  lr: 0.001
  custom_param: 42
```

#### 4. Run

```bash
thesis-train loop=my_loop
```

### Adding a New Architecture Component

#### Adding an Encoder

1. **Create encoder file**: `src/thesis_ml/architectures/autoencoder/encoders/my_encoder.py`

```python
import torch.nn as nn

class MyEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        return self.fc(x)
```

2. **Create config**: `configs/phase1/encoder/my_encoder.yaml`

```yaml
# @package _global_
phase1:
  encoder:
    _target_: thesis_ml.architectures.autoencoder.encoders.my_encoder.MyEncoder
    custom_param: 123
```

3. **Update base assembly** (if needed): `src/thesis_ml/architectures/autoencoder/base.py`

Usually not needed if you use `_target_` pattern.

4. **Run**:

```bash
thesis-train phase1/encoder=my_encoder
```

### Best Practices

#### Facts Emission

**Always emit these events:**

1. `on_start`: Beginning of training
2. `on_epoch_end`: After each epoch
3. `on_train_end`: End of training

**Include histories:**

```python
histories = {
    "train_loss": [...],      # Required
    "val_loss": [...],         # Required
    "my_metric": [...],        # Optional custom metrics
}
```

#### Error Handling

```python
try:
    # Training loop
    ...
except Exception as e:
    if outdir:
        error_payload = build_event_payload(
            moment="on_exception",
            run_dir=outdir,
            error=str(e),
            cfg=cfg,
        )
        append_jsonl_event(outdir, error_payload)
    raise
```

#### Device Management

```python
# Auto-detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Respect config (if provided)
device_pref = cfg.get("device", "auto")
if device_pref == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_pref)
```

#### Checkpoint Saving

```python
# Save best validation checkpoint
if val_loss < best_val:
    best_val = val_loss
    best_path = os.path.join(outdir, "best_val.pt")
    torch.save(model.state_dict(), best_path)

    # Create symlink for stable reference
    model_path = os.path.join(outdir, "model.pt")
    if os.path.exists(model_path):
        os.remove(model_path)
    try:
        os.symlink("best_val.pt", model_path)
    except OSError:
        # Windows fallback
        shutil.copy2(best_path, model_path)
```

### Testing New Code

```python
# tests/test_my_loop.py
import pytest
from omegaconf import OmegaConf
from thesis_ml.training_loops.my_loop import train

def test_my_loop():
    cfg = OmegaConf.create({
        "loop": "my_loop",
        "phase1": {
            "trainer": {"epochs": 2, "seed": 42},
        },
        "data": {"batch_size": 32},
        "logging": {"save_artifacts": False},
    })

    result = train(cfg)

    assert "final_val_loss" in result
    assert result["final_val_loss"] > 0
```

Run: `pytest tests/test_my_loop.py`

### Common Patterns

#### Custom Loss Functions

```python
# src/thesis_ml/architectures/autoencoder/losses/my_loss.py
import torch.nn as nn

class MyCustomLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        return self.weight * ((pred - target) ** 2).mean()
```

#### Custom Data Loaders

```python
# src/thesis_ml/data/my_dataset.py
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, cfg):
        # Load data based on cfg
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def make_my_dataloaders(cfg):
    train_ds = MyDataset(cfg)
    train_dl = DataLoader(train_ds, batch_size=cfg.data.batch_size)
    return train_dl, val_dl, test_dl, meta
```

### Documentation

When adding new training code, document:

1. **Docstring**: What the training loop does
2. **Config schema**: What config keys are required
3. **Return format**: What the function returns
4. **Example**: Example command to run

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md): System architecture
- [FACTS_SYSTEM.md](FACTS_SYSTEM.md): Facts schema and usage
- [CONFIGS_GUIDE.md](CONFIGS_GUIDE.md): Hydra configuration patterns
- [HPC_GUIDE.md](HPC_GUIDE.md): Running on Stoomboot
