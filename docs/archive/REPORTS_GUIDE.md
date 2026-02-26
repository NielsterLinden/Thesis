# Reports Guide

Complete guide for generating reports and creating new analysis types.

## 📖 Table of Contents

- [Using Existing Reports](#using-existing-reports)
- [Creating New Reports](#creating-new-reports)

---

## Using Existing Reports

### Quick Start

```bash
# Generate report from sweep
thesis-report --config-name compare_tokenizers \
    inputs.sweep_dir=outputs/multiruns/exp_20251103-140953_experiment

# With inference
thesis-report --config-name compare_tokenizers \
    inputs.sweep_dir=outputs/multiruns/exp_20251103-140953_experiment \
    inference.enabled=true
```

### Available Reports

| Report Name | File | Purpose |
|-------------|------|---------|
| `compare_tokenizers` | `compare_tokenizers.py` | Compare VQ vs non-VQ autoencoders |
| `compare_globals_heads` | `compare_globals_heads.py` | Analyze globals reconstruction weight |

### Basic Usage

#### From Sweep Directory

```bash
thesis-report --config-name compare_tokenizers \
    inputs.sweep_dir=outputs/multiruns/exp_20251103-140953_experiment
```

**Output**: `outputs/reports/report_TIMESTAMP_compare_tokenizers/`

#### From Explicit Run List

```bash
thesis-report --config-name compare_tokenizers \
    'inputs.run_dirs=["outputs/runs/run_A", "outputs/runs/run_B"]'
```

**Note**: Quote list arguments on Windows PowerShell.

### Configuration Options

#### Select Which Runs

```bash
# Filter by encoder type
thesis-report --config-name compare_tokenizers \
    inputs.sweep_dir=... \
    'inputs.select.encoder=["mlp","gnn"]'

# Filter by seed
thesis-report --config-name compare_tokenizers \
    inputs.sweep_dir=... \
    'inputs.select.seed=[42,43]'
```

#### Enable Inference

```bash
thesis-report --config-name compare_tokenizers \
    inputs.sweep_dir=... \
    inference.enabled=true \
    inference.batch_size=256 \
    inference.dataset_split=test
```

**Inference options:**
- `enabled`: Run inference (default: `false`)
- `batch_size`: Batch size for inference (default: `512`)
- `autocast`: Use mixed precision (default: `true`)
- `dataset_split`: Which split to use (`test`, `val`, `train`)
- `persist_raw_scores`: Save per-event scores (default: `false`)

#### Customize Figures

```bash
thesis-report --config-name compare_tokenizers \
    inputs.sweep_dir=... \
    'outputs.which_figures=[val_mse_vs_time,pareto_error_vs_compression]'
```

### Output Structure

```
outputs/reports/report_20251103-142813_compare_tokenizers/
├── manifest.yaml            # Report metadata
│
├── training/                # Training phase analysis
│   ├── summary.csv          # Aggregated metrics (one row per run)
│   ├── summary.json         # Metadata (sweep params, filters, etc.)
│   └── figures/
│       ├── figure-val_mse_vs_time.png
│       ├── figure-pareto_error_vs_compression.png
│       └── ...
│
└── inference/               # Inference phase (if enabled)
    ├── summary.json         # AUROC, mean errors, etc.
    └── figures/
        ├── figure-reconstruction_error_distributions.png
        ├── figure-model_comparison_bars.png
        └── ...
```

### Reading Report Outputs

#### Training Summary

```python
import pandas as pd

# Load summary
df = pd.read_csv("outputs/reports/report_*/training/summary.csv")

# View columns
print(df.columns)
# ['run_dir', 'encoder', 'tokenizer', 'seed', 'loss.total_best', ...]

# Find best run
best = df.loc[df['loss.total_best'].idxmin()]
print(f"Best run: {best['run_dir']}")
print(f"Best loss: {best['loss.total_best']}")
```

#### Inference Summary

```python
import json

# Load inference results
with open("outputs/reports/report_*/inference/summary.json") as f:
    results = json.load(f)

# View AUROC per model
for model, metrics in results["models"].items():
    print(f"{model}: AUROC = {metrics['auroc']:.3f}")
```

### Running on HPC

#### Interactive

```bash
ssh stoomboot
cd /project/atlas/users/nterlind/Thesis-Code
conda activate /data/atlas/users/nterlind/venvs/thesis-ml

thesis-report --config-name compare_tokenizers \
    inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_experiment \
    env.output_root=/data/atlas/users/nterlind/outputs
```

#### Batch Submission

```bash
condor_submit hpc/stoomboot/report.sub
```

**Edit `report.sub` for custom arguments:**

```condor
arguments = --config-name compare_tokenizers inputs.sweep_dir=/data/.../exp_* inference.enabled=true
```

### Troubleshooting

#### Issue: "No valid runs found"

**Solutions:**
1. Check `sweep_dir` path exists
2. Ensure runs have `.hydra/config.yaml` or `cfg.yaml`
3. Verify runs have `facts/events.jsonl` and `facts/scalars.csv`
4. Check for `on_train_end` event: `grep on_train_end outputs/runs/*/facts/events.jsonl`

#### Issue: "Missing data config"

**Solutions:**
1. Inference requires data config from a run
2. Check first run has `.hydra/config.yaml` with `data` section
3. Or provide explicitly: `data=h5_tokens`

#### Issue: Slow report generation

**Solutions:**
1. Disable inference: `inference.enabled=false`
2. Reduce number of runs (use `inputs.select`)
3. Reduce inference batch size: `inference.batch_size=128`

---

## Creating New Reports

### Basic Report Structure

#### 1. Create Report File

Create `src/thesis_ml/reports/analyses/my_report.py`:

```python
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from thesis_ml.facts.readers import load_runs
from thesis_ml.reports.utils.io import setup_report_environment, finalize_report, get_fig_config, save_json
from thesis_ml.monitoring.io_utils import save_figure

logger = logging.getLogger(__name__)


def run_report(cfg: DictConfig) -> None:
    """My custom report.

    Generates comparative analysis of training runs.

    Parameters
    ----------
    cfg : DictConfig
        Report configuration
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # 1. Setup environment (creates directories, discovers runs)
    (training_dir, inference_dir, training_figs_dir, inference_figs_dir,
     runs_df, per_epoch, order, report_dir, report_id, report_name) = \
        setup_report_environment(cfg)

    # 2. Analyze runs_df
    # runs_df is a DataFrame with columns: run_dir, encoder, tokenizer, seed,
    # loss.total_best, epochs, total_time_s, etc.

    logger.info(f"Loaded {len(runs_df)} runs")

    # 3. Save training summary
    runs_df.to_csv(training_dir / "summary.csv", index=False)
    save_json({"num_runs": len(runs_df)}, training_dir / "summary.json")

    # 4. Generate training figures
    fig_cfg = get_fig_config(cfg)

    # Example: Plot loss distribution
    fig, ax = plt.subplots()
    ax.hist(runs_df["loss.total_best"].dropna(), bins=20)
    ax.set_xlabel("Best validation loss")
    ax.set_ylabel("Count")
    ax.set_title("Loss distribution across runs")
    save_figure(fig, training_figs_dir, "figure-loss_distribution", fig_cfg)
    plt.close(fig)

    # Example: Plot loss vs time
    fig, ax = plt.subplots()
    for run_dir, hist_df in per_epoch.items():
        val_data = hist_df[hist_df["split"] == "val"]
        ax.plot(val_data["epoch"], val_data["val_loss"], alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Training curves")
    save_figure(fig, training_figs_dir, "figure-training_curves", fig_cfg)
    plt.close(fig)

    # 5. (Optional) Run inference
    if cfg.get("inference", {}).get("enabled", False):
        logger.info("Running inference...")
        # Import and call inference utilities
        from thesis_ml.reports.utils.inference import run_inference_on_models

        inference_results = run_inference_on_models(
            runs_df=runs_df,
            data_cfg=cfg.data,
            inference_cfg=cfg.inference,
        )

        # Save inference results
        save_json(inference_results, inference_dir / "summary.json")

        # Generate inference figures
        # ... (see example below)

    # 6. Finalize report
    finalize_report(cfg, report_dir, runs_df, Path(cfg.env.output_root), report_id, report_name)

    logger.info(f"Report complete: {report_dir}")
```

#### 2. Create Report Config

Create `configs/report/my_report.yaml`:

```yaml
# @package _global_

defaults:
  - config  # Inherit from base report config

# Inputs
inputs:
  sweep_dir: null
  run_dirs: []
  select: null  # Optional: filter runs

# Outputs
outputs:
  which_figures:
    - loss_distribution
    - training_curves

# Thresholds (if needed)
thresholds:
  val_mse: 0.1

# Inference (optional)
inference:
  enabled: false
  batch_size: 512
  autocast: true
  dataset_split: test
```

#### 3. Run

```bash
thesis-report --config-name my_report \
    inputs.sweep_dir=outputs/multiruns/exp_*
```

### Reading Facts

#### Load All Runs

```python
from thesis_ml.facts.readers import load_runs

# From sweep directory
runs_df, per_epoch, order = load_runs(
    sweep_dir="outputs/multiruns/exp_20251103-140953_experiment"
)

# From explicit list
runs_df, per_epoch, order = load_runs(
    run_dirs=["outputs/runs/run_A", "outputs/runs/run_B"]
)
```

**Returns:**
- `runs_df`: DataFrame with one row per run (aggregated metrics)
- `per_epoch`: Dict of {run_dir: DataFrame} with per-epoch metrics
- `order`: List of run_dir strings in discovery order

#### Runs DataFrame Schema

```
Columns:
  - run_dir: str (path to run)
  - encoder: str (encoder type)
  - tokenizer: str (latent space type)
  - seed: int
  - latent_dim: int
  - epochs: int
  - best_epoch: int
  - loss.total_best: float (best validation loss)
  - loss.total_final: float (final validation loss)
  - loss.recon_best: float
  - total_time_s: float
  - throughput_mean: float (samples/sec)
  - metric_perplex_final: float (if VQ)
  - [Dynamic columns from Hydra overrides]
```

#### Per-Epoch DataFrame Schema

```
Columns:
  - epoch: int
  - split: str ("train" or "val")
  - train_loss: float
  - val_loss: float
  - epoch_time_s: float
  - throughput: float
  - metric_*: float (custom metrics)
```

### Advanced Report Features

#### Filtering Runs

```python
def _filter_runs(df, select):
    """Filter runs based on select criteria."""
    if not select:
        return df

    filtered = df.copy()

    # Filter by encoder
    if "encoder" in select and select["encoder"]:
        filtered = filtered[filtered["encoder"].isin(select["encoder"])]

    # Filter by seed
    if "seed" in select and select["seed"]:
        filtered = filtered[filtered["seed"].isin(select["seed"])]

    return filtered

# Usage
selected = _filter_runs(runs_df, cfg.inputs.select)
```

#### Grouping and Aggregation

```python
# Group by tokenizer and compute mean loss
grouped = runs_df.groupby("tokenizer").agg({
    "loss.total_best": ["mean", "std", "min"],
    "total_time_s": "mean",
})

print(grouped)
```

#### Plotting Utilities

```python
from thesis_ml.reports.plots.curves import plot_loss_vs_time
from thesis_ml.reports.plots.grids import plot_metric_grid
from thesis_ml.reports.plots.scatter import plot_pareto_front

# Loss vs time curves
plot_loss_vs_time(runs_df, per_epoch, order, figs_dir, fig_cfg,
                  metric="val_loss", fname="figure-val_loss_vs_time")

# Metric grid (heatmap)
plot_metric_grid(runs_df, x_col="encoder", y_col="tokenizer",
                 value_col="loss.total_best", figs_dir, fig_cfg)
```

### Running Inference

#### Basic Inference

```python
from thesis_ml.reports.inference.forward_pass import run_forward_pass
from thesis_ml.reports.inference.metrics import compute_reconstruction_metrics

# Load model
model_path = Path(run_dir) / "model.pt"
model = load_model(model_path)  # Implement this

# Run forward pass
predictions, targets, metadata = run_forward_pass(
    model=model,
    data_loader=test_loader,
    device=device,
)

# Compute metrics
metrics = compute_reconstruction_metrics(predictions, targets)
# Returns: {"mse": ..., "mae": ..., "per_event_errors": [...]}
```

#### Anomaly Detection

```python
from thesis_ml.reports.inference.anomaly_detection import compute_auroc

# Get reconstruction errors
errors = metrics["per_event_errors"]

# Assuming you have labels (0=signal, 1=background)
labels = ...  # Load from dataset

# Compute AUROC
auroc = compute_auroc(errors, labels)
print(f"AUROC: {auroc:.3f}")
```

### Best Practices

#### Report Structure

1. **Load data** via `facts.readers`
2. **Filter/aggregate** as needed
3. **Generate training plots** (loss curves, distributions, etc.)
4. **Run inference** (optional, if enabled)
5. **Save summaries** (CSV, JSON)
6. **Finalize** with manifest and backlinks

#### Error Handling

```python
try:
    runs_df, per_epoch, order = load_runs(sweep_dir=sweep_dir)
except FileNotFoundError as e:
    logger.error(f"Sweep directory not found: {e}")
    raise
except Exception as e:
    logger.error(f"Failed to load runs: {e}")
    raise
```

#### Logging

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Processing {len(runs_df)} runs")
logger.warning(f"Skipping {skipped} incomplete runs")
logger.error(f"Failed to process run: {run_dir}")
```

### Testing

```python
# tests/test_my_report.py
import pytest
import tempfile
from pathlib import Path
from omegaconf import OmegaConf
from thesis_ml.reports.analyses.my_report import run_report

def test_my_report(tmp_path):
    # Create minimal config
    cfg = OmegaConf.create({
        "inputs": {
            "run_dirs": [str(tmp_path / "fake_run")],
        },
        "outputs": {"which_figures": []},
        "env": {"output_root": str(tmp_path)},
        "inference": {"enabled": False},
    })

    # Create fake run directory with minimal facts
    fake_run = tmp_path / "fake_run"
    fake_run.mkdir()
    (fake_run / "facts").mkdir()
    (fake_run / "facts" / "events.jsonl").write_text('{"moment":"on_train_end"}')
    (fake_run / "facts" / "scalars.csv").write_text("epoch,split,val_loss\\n0,val,0.5")
    (fake_run / ".hydra").mkdir()
    (fake_run / ".hydra" / "config.yaml").write_text("data: {}")

    # Run report
    run_report(cfg)

    # Check outputs exist
    report_dirs = list((tmp_path / "reports").glob("report_*"))
    assert len(report_dirs) == 1
    assert (report_dirs[0] / "manifest.yaml").exists()
```

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md): System architecture
- [FACTS_SYSTEM.md](FACTS_SYSTEM.md): Facts schema and reading
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md): Training runs
- [CONFIGS_GUIDE.md](CONFIGS_GUIDE.md): Hydra configuration
