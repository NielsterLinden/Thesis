# Report Generation Guide: compare_globals_heads with 27 Models

This guide shows how to generate a report with both training plots and inference sections for all 27 migrated models.

## Prerequisites

1. **Models migrated**: All 27 runs should be migrated to:
   - Runs: `/data/atlas/users/nterlind/outputs/runs/run_20251031-152750_compare_globals_heads_job{0..26}/`
   - Multirun: `/data/atlas/users/nterlind/outputs/multiruns/exp_20251031-152750_compare_globals_heads/`

2. **Verify migration**:
   ```bash
   ls /data/atlas/users/nterlind/outputs/runs/run_20251031-152750_compare_globals_heads_job* | wc -l
   # Should show 27 runs
   ```

## Method 1: Interactive Session (Recommended for Testing)

### On Stoomboot (SSH into login node)

```bash
# Navigate to code directory
cd /project/atlas/users/nterlind/Thesis-Code

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /data/atlas/users/nterlind/venvs/thesis-ml

# Run report with inference enabled
python -m thesis_ml.reports.report_from_sweep \
  --config-name compare_globals_heads \
  inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_20251031-152750_compare_globals_heads \
  inference.enabled=true \
  env.output_root=/data/atlas/users/nterlind/outputs
```

### Expected Output

The report will be generated at:
```
/data/atlas/users/nterlind/outputs/reports/report_YYYYMMDD-HHMMSS_compare_globals_heads/
├── training/
│   ├── summary.csv
│   └── figures/
│       ├── figure-grid_best_val_loss.png
│       ├── figure-grid_best_rec_globals.png
│       ├── figure-grid_best_rec_tokens.png
│       ├── figure-tradeoff.png
│       ├── figure-all_val_curves.png
│       └── figure-all_train_curves.png
├── inference/
│   ├── summary.json
│   └── figures/
│       ├── figure-reconstruction_error_distributions_*.png (one per model)
│       ├── figure-model_comparison_bars.png
│       └── figure-auroc_comparison.png
└── manifest.yaml
```

## Method 2: Condor Submission (For Production Runs)

### Step 1: Create/Update Submit File

The existing `hpc/stoomboot/report.sub` is already configured. If you need a custom one:

```bash
# Edit if needed
nano hpc/stoomboot/report.sub
```

Ensure it has sufficient resources for inference (may need GPU if autocast is enabled):
```condor
universe      = vanilla
executable    = hpc/stoomboot/report.sh
arguments     = --config-name compare_globals_heads inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_20251031-152750_compare_globals_heads inference.enabled=true env.output_root=/data/atlas/users/nterlind/outputs

initialdir    = /project/atlas/users/nterlind/Thesis-Code

output        = /data/atlas/users/nterlind/logs/stdout/report_$(ClusterId).$(ProcId).out
error         = /data/atlas/users/nterlind/logs/stderr/report_$(ClusterId).$(ProcId).err
log           = /data/atlas/users/nterlind/logs/report_$(ClusterId).log

# Inference may benefit from GPU if autocast is enabled
request_cpus  = 4
request_memory = 16GB
request_disk  = 20GB
# Uncomment if you want GPU for inference:
# request_gpus = 1
# requirements = (CUDACapability > 0)

+UseOS = "el9"
+JobCategory = "short"
should_transfer_files = NO
getenv        = True
stream_output = True
stream_error  = True

queue 1
```

### Step 2: Submit Job

```bash
cd /project/atlas/users/nterlind/Thesis-Code

# Submit with all arguments
condor_submit hpc/stoomboot/report.sub

# Or submit with overrides
condor_submit hpc/stoomboot/report.sub \
  -append 'arguments = --config-name compare_globals_heads inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_20251031-152750_compare_globals_heads inference.enabled=true env.output_root=/data/atlas/users/nterlind/outputs'
```

### Step 3: Monitor Job

```bash
# Check job status
condor_q

# Follow output live
condor_tail -f <job_id>

# Check logs
tail -f /data/atlas/users/nterlind/logs/stdout/report_*.out
tail -f /data/atlas/users/nterlind/logs/stderr/report_*.err
```

## Configuration Options

### Enable/Disable Inference

```bash
# Enable inference (default: false)
inference.enabled=true

# Disable inference (only training plots)
inference.enabled=false
```

### Customize Inference Settings

```bash
# Change batch size (default: 512)
inference.batch_size=256

# Disable autocast (default: true)
inference.autocast=false

# Change dataset split (default: test)
inference.dataset_split=val

# Persist raw per-event scores (default: false)
inference.persist_raw_scores=true
```

### Example: Full Custom Command

```bash
python -m thesis_ml.reports.report_from_sweep \
  --config-name compare_globals_heads \
  inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_20251031-152750_compare_globals_heads \
  inference.enabled=true \
  inference.batch_size=256 \
  inference.autocast=true \
  inference.dataset_split=test \
  inference.persist_raw_scores=false \
  env.output_root=/data/atlas/users/nterlind/outputs
```

## Troubleshooting

### Issue: "No valid runs found for reporting"

**Solution**: Verify the sweep_dir path is correct:
```bash
ls /data/atlas/users/nterlind/outputs/multiruns/exp_20251031-152750_compare_globals_heads/
```

### Issue: "Missing data config"

**Solution**: The data config is automatically extracted from the first run. If it fails, ensure:
1. At least one run has `.hydra/config.yaml` or `cfg.yaml`
2. The config contains a `data` section

### Issue: "CUDA out of memory" during inference

**Solutions**:
1. Reduce batch size: `inference.batch_size=128`
2. Disable autocast: `inference.autocast=false`
3. Process fewer models at once (modify report code to subset runs)

### Issue: Report generation is slow

**Solutions**:
1. Inference runs on all 27 models - this takes time
2. Consider reducing batch size or dataset split size
3. Use GPU if available (uncomment GPU request in submit file)

## Quick Test (Single Model)

To test the report generation quickly:

```bash
# Get first run ID
FIRST_RUN=$(ls -d /data/atlas/users/nterlind/outputs/runs/run_20251031-152750_compare_globals_heads_job0)

# Run report on single run
python -m thesis_ml.reports.report_from_sweep \
  --config-name compare_globals_heads \
  inputs.run_dirs="[$FIRST_RUN]" \
  inference.enabled=true \
  env.output_root=/data/atlas/users/nterlind/outputs
```

## Output Verification

After completion, verify the report:

```bash
REPORT_DIR=$(ls -td /data/atlas/users/nterlind/outputs/reports/report_*_compare_globals_heads | head -1)

# Check training plots
ls $REPORT_DIR/training/figures/

# Check inference results
ls $REPORT_DIR/inference/figures/
cat $REPORT_DIR/inference/summary.json | head -50

# Check manifest
cat $REPORT_DIR/manifest.yaml
```
