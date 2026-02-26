# Facts System

The **facts system** is the foundation of reproducibility and post-hoc analysis in thesis-ml. This document explains what facts are, how they're emitted, and how to use them.

## 🎯 What Are Facts?

**Facts** are standardized records of training events and metrics, stored to disk for later analysis. They enable:

- **Reproducibility**: Every run records exact configuration and results
- **Post-hoc Analysis**: Generate reports without re-running expensive training
- **Debugging**: Inspect training history to diagnose issues
- **Comparison**: Aggregate metrics across multiple runs

## 📊 Fact Types

### 1. Events (`events.jsonl`)

**Format**: JSON Lines (one JSON object per line)

**Purpose**: Record lifecycle events with full training histories

**Event Types**:
- `on_start`: Training initialization
- `on_epoch_end`: End of each epoch
- `on_train_end`: Training completion
- `on_test_end`: Test evaluation
- `on_exception`: Error occurred

**Example**:
```json
{
  "schema_version": 1,
  "moment": "on_epoch_end",
  "run_dir": "/path/to/run",
  "epoch": 10,
  "split": "val",
  "train_loss": 0.234,
  "val_loss": 0.256,
  "metrics": {"perplex": 128.5},
  "epoch_time_s": 12.3,
  "throughput": 1500.0,
  "histories": {
    "train_loss": [0.5, 0.4, 0.3, ...],
    "val_loss": [0.6, 0.5, 0.4, ...],
    "perplex": [50.0, 75.0, 100.0, ...]
  },
  "meta": {
    "timestamp": "2025-11-03T14:09:53Z",
    "run_id": "uuid-...",
    "git_commit": "abc123",
    "hostname": "stoomboot",
    "seed": 42
  }
}
```

### 2. Scalars (`scalars.csv`)

**Format**: CSV (Comma-Separated Values)

**Purpose**: Per-epoch metrics for easy DataFrame analysis

**Columns**:
- `epoch`: Epoch number
- `split`: Data split (`train`, `val`, `test`)
- `train_loss`, `val_loss`: Losses
- `epoch_time_s`: Epoch duration
- `throughput`: Training speed (samples/sec)
- `max_memory_mib`: GPU memory usage
- `metric_*`: Custom metrics (e.g., `metric_perplex`, `metric_acc`)

**Example**:
```csv
epoch,split,train_loss,val_loss,epoch_time_s,throughput,max_memory_mib,metric_perplex
0,train,0.5,,12.3,1500.0,2048.5,
0,val,,0.6,12.3,1500.0,2048.5,50.0
1,train,0.4,,11.8,1550.0,2048.5,
1,val,,0.5,11.8,1550.0,2048.5,75.0
```

## 🏗️ Architecture

### Emission Flow

```
Training Loop
     │
     ├─ Build event payload
     │     └─ facts.builders.build_event_payload()
     │
     ├─ Write to disk
     │     ├─ facts.writers.append_jsonl_event()  → facts/events.jsonl
     │     └─ facts.writers.append_scalars_csv()  → facts/scalars.csv
     │
     └─ Optionally create plots
           └─ monitoring.orchestrator.handle_event()
```

### Consumption Flow

```
Report
     │
     ├─ Discover runs
     │     └─ facts.readers.discover_runs()
     │
     ├─ Load facts from all runs
     │     └─ facts.readers.load_runs()
     │          ├─ Read .hydra/config.yaml
     │          ├─ Read facts/events.jsonl
     │          ├─ Read facts/scalars.csv
     │          └─ Return (runs_df, per_epoch, order)
     │
     └─ Analyze and plot
```

## 📝 Emitting Facts (Training)

### Basic Pattern

```python
from thesis_ml.facts import build_event_payload, append_jsonl_event, append_scalars_csv

# During training loop
for epoch in range(num_epochs):
    # ... training step ...

    # Build event payload
    payload = build_event_payload(
        moment="on_epoch_end",
        run_dir=outdir,
        epoch=epoch,
        split="val",
        train_loss=train_loss,
        val_loss=val_loss,
        metrics={"perplex": perplex_value},
        epoch_time_s=epoch_duration,
        throughput=samples_per_sec,
        histories={
            "train_loss": history_train_loss,
            "val_loss": history_val_loss,
        },
        cfg=cfg,  # For metadata extraction
    )

    # Write facts
    append_jsonl_event(outdir, payload)
    append_scalars_csv(
        outdir,
        epoch=epoch,
        split="val",
        train_loss=train_loss,
        val_loss=val_loss,
        metrics={"perplex": perplex_value},
        epoch_time_s=epoch_duration,
        throughput=samples_per_sec,
        max_memory_mib=gpu_memory,
    )
```

### Required Events

Every training loop should emit:

1. **on_start**: Beginning of training
   ```python
   payload = build_event_payload(
       moment="on_start",
       run_dir=outdir,
       cfg=cfg,
   )
   append_jsonl_event(outdir, payload)
   ```

2. **on_epoch_end**: After each epoch
   ```python
   payload = build_event_payload(
       moment="on_epoch_end",
       run_dir=outdir,
       epoch=epoch,
       train_loss=...,
       val_loss=...,
       histories={...},
       cfg=cfg,
   )
   append_jsonl_event(outdir, payload)
   append_scalars_csv(outdir, epoch=epoch, ...)
   ```

3. **on_train_end**: End of training
   ```python
   payload = build_event_payload(
       moment="on_train_end",
       run_dir=outdir,
       total_time_s=total_time,
       histories={...},
       cfg=cfg,
   )
   append_jsonl_event(outdir, payload)
   ```

### Payload Schema

#### Standard Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Fact schema version (currently 1) |
| `moment` | str | Event type (`on_start`, `on_epoch_end`, etc.) |
| `run_dir` | str | Run output directory |
| `epoch` | int\|None | Epoch number |
| `step` | int\|None | Training step number |
| `split` | str\|None | Data split (`train`, `val`, `test`) |
| `train_loss` | float\|None | Training loss |
| `val_loss` | float\|None | Validation loss |
| `metrics` | dict | Custom metrics (e.g., `{"perplex": 128.5}`) |
| `epoch_time_s` | float\|None | Epoch duration |
| `total_time_s` | float\|None | Total training time |
| `throughput` | float\|None | Training speed (samples/sec) |
| `max_memory_mib` | float\|None | GPU memory usage |
| `histories` | dict | Full training histories |
| `meta` | dict | Metadata (timestamp, git commit, etc.) |

#### Histories Format

Histories should contain full training arrays:

```python
histories = {
    "train_loss": [0.5, 0.4, 0.3, ...],  # All epochs
    "val_loss": [0.6, 0.5, 0.4, ...],
    "perplex": [50.0, 75.0, 100.0, ...],
    # Add custom metrics
}
```

#### Metadata Block

Automatically collected by `build_event_payload`:

```python
meta = {
    "timestamp": "2025-11-03T14:09:53Z",  # UTC timestamp
    "run_id": "uuid-...",                 # Unique run ID
    "git_commit": "abc123",               # Git commit hash
    "hostname": "stoomboot",              # Machine name
    "cuda_info": {...},                   # GPU information
    "seed": 42,                           # Random seed (if in config)
    "hydra_job_id": "...",               # Hydra job ID (if available)
}
```

## 📖 Reading Facts (Reports)

### Load All Runs

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

# Skip incomplete runs
runs_df, per_epoch, order = load_runs(
    sweep_dir="...",
    require_complete=True,  # Only include runs with on_train_end event
)
```

**Returns**:
- `runs_df`: pandas DataFrame with one row per run (aggregated metrics)
- `per_epoch`: dict of {run_dir: DataFrame} with per-epoch data
- `order`: list of run_dir strings in discovery order

### Runs DataFrame Schema

```python
# Columns:
runs_df.columns
# ['run_dir', 'encoder', 'tokenizer', 'seed', 'latent_dim',
#  'epochs', 'best_epoch', 'loss.total_best', 'loss.total_final',
#  'total_time_s', 'throughput_mean', 'metric_perplex_final', ...]
```

### Per-Epoch DataFrame Schema

```python
# Access per-epoch data for a run
run_dir = runs_df.iloc[0]["run_dir"]
epoch_df = per_epoch[run_dir]

# Columns:
epoch_df.columns
# ['epoch', 'split', 'train_loss', 'val_loss', 'epoch_time_s',
#  'throughput', 'max_memory_mib', 'metric_perplex', ...]
```

### Filtering

```python
# Filter validation rows
val_df = epoch_df[epoch_df["split"] == "val"]

# Get best epoch
best_idx = val_df["val_loss"].idxmin()
best_row = val_df.loc[best_idx]
print(f"Best epoch: {best_row['epoch']} with loss {best_row['val_loss']}")
```

### Aggregating Across Runs

```python
# Group by tokenizer
grouped = runs_df.groupby("tokenizer").agg({
    "loss.total_best": ["mean", "std", "min"],
    "total_time_s": "mean",
})

print(grouped)
#              loss.total_best              total_time_s
#                         mean    std    min         mean
# tokenizer
# none                  0.123  0.01  0.115       120.5
# vq                    0.145  0.02  0.130       135.2
```

### Accessing Events

```python
from thesis_ml.facts.readers import _read_events
from pathlib import Path

# Read events from a run
run_dir = Path("outputs/runs/run_20251103-140953_experiment")
events = _read_events(run_dir)

# Find on_train_end event
for event in events:
    if event["moment"] == "on_train_end":
        print(f"Total time: {event['total_time_s']}s")
        print(f"Final val loss: {event['histories']['val_loss'][-1]}")
```

## 🔧 Advanced Usage

### Custom Metrics

Add custom metrics to payload:

```python
# In training loop
custom_metrics = {
    "my_metric": my_value,
    "another_metric": another_value,
}

payload = build_event_payload(
    moment="on_epoch_end",
    ...,
    metrics=custom_metrics,
)
```

These will appear in:
- `events.jsonl`: Under `metrics` field
- `scalars.csv`: As `metric_my_metric`, `metric_another_metric` columns

### Multiple Splits

Emit separate CSV rows for train and val:

```python
# Train split
append_scalars_csv(outdir, epoch=epoch, split="train",
                   train_loss=train_loss, val_loss=None, ...)

# Val split
append_scalars_csv(outdir, epoch=epoch, split="val",
                   train_loss=None, val_loss=val_loss, ...)
```

### Error Handling

```python
try:
    # Training loop
    ...
except Exception as e:
    # Emit error event
    error_payload = build_event_payload(
        moment="on_exception",
        run_dir=outdir,
        error=str(e),
        traceback=traceback.format_exc(),
        cfg=cfg,
    )
    append_jsonl_event(outdir, error_payload)
    raise
```

## 📁 Storage Location

Facts are always stored in `{run_dir}/facts/`:

```
outputs/runs/run_20251103-140953_experiment/
└── facts/
    ├── events.jsonl      # Lifecycle events
    └── scalars.csv       # Per-epoch metrics
```

## 🔒 Schema Versioning

Current schema version: **1**

All events include `"schema_version": 1` for forward compatibility.

Future versions may add fields but will maintain backward compatibility for reading.

## 💡 Best Practices

### DO:
✅ Emit `on_start`, `on_epoch_end`, and `on_train_end` events
✅ Include full histories in events (all epochs, not just current)
✅ Use consistent metric names across runs
✅ Add metadata via `cfg` parameter
✅ Write scalars CSV for easy DataFrame access

### DON'T:
❌ Emit events without `schema_version`
❌ Skip epochs in histories (maintain continuous arrays)
❌ Write facts manually (use `facts.writers` API)
❌ Modify facts after writing (append-only)
❌ Store large tensors in events (store to separate files)

## 🐛 Troubleshooting

### Issue: Missing facts directory

**Solution**: Ensure `cfg.logging.save_artifacts=true` or call `facts.writers.ensure_facts_dir()`.

### Issue: Incomplete runs in reports

**Solution**: Verify runs have `on_train_end` event:
```bash
grep on_train_end outputs/runs/*/facts/events.jsonl
```

### Issue: Mismatched columns in scalars.csv

**Solution**: Metrics are added dynamically. First run creates header. Subsequent runs with new metrics may have different columns. Use pandas to handle:
```python
df = pd.read_csv("scalars.csv")
df["metric_new"] = df.get("metric_new", float("nan"))
```

## 🚀 Future Enhancements

- **Parquet format**: For larger runs, use Parquet for faster loading
- **Compression**: Compress events.jsonl for storage efficiency
- **Streaming**: Stream events during training for real-time dashboards
- **Schema v2**: Add support for distributed training metadata

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md): Facts in system architecture
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md): Emitting facts in training loops
- [REPORTS_GUIDE.md](REPORTS_GUIDE.md): Reading facts in reports
