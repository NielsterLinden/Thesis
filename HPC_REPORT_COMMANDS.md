# HPC Report Generation Commands

This document contains the corrected condor submit commands for generating reports from your multirun experiments.

## Corrected Condor Submit Commands

### Compare Positional Encodings

```bash
# For exp_20251126-150519_compare_positional_encodings
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name compare_positional_encodings inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_compare_positional_encodings inference.enabled=true'

# For exp_20251127-162910_4t_vs_background_positional
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name compare_positional_encodings inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_4t_vs_background_positional inference.enabled=true'
```

### Compare Regularization

```bash
# For exp_20251120-144239_overfitting_regularization_sweep
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name compare_regularization inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_overfitting_regularization_sweep inference.enabled=true'
```

### Compare Model Sizes

```bash
# For exp_20251208-135941_phd_exp1_4t_vs_bg_sizes_and_pe (has model_size config groups: s200k, s600k, s1500k, s3000k)
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name compare_model_sizes inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_phd_exp1_4t_vs_bg_sizes_and_pe inference.enabled=true'

# For exp_20251126-150519_compare_positional_encodings (also has model sizes)
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name compare_model_sizes inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_compare_positional_encodings inference.enabled=true'
```

### Deep Representation Analysis

```bash
# For exp_20251126-150519_compare_positional_encodings
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name analyze_representations inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_compare_positional_encodings analysis.num_batches=10'
```

### Compare Norm/Pos/Pool

```bash
# For exp_20251114-153438_compare_norm_pos_pool (100 epochs)
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name compare_norm_pos_pool inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_compare_norm_pos_pool inference.enabled=true'

# For exp_20251208-140119_phd_exp3_4t_vs_ttH_norm_policies
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name compare_norm_pos_pool inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_phd_exp3_4t_vs_ttH_norm_policies inference.enabled=true'
```

## Summary of Changes Made

### Code Updates

1. **Updated `compare_model_sizes.py`**:
   - Added extraction of `model_size` config group name from Hydra overrides (e.g., "s200k", "s600k")
   - Updated `size_label` to prefer config group name over computed labels (e.g., "64d6L")
   - Added `model_size_group` column to metadata extraction
   - Updated logging to include `model_size_group` values

### Pattern Matching

The original commands used patterns like:
- `exp_*_positional` ❌
- `exp_*_regularization` ❌
- `exp_*_model_size` ❌

But actual experiment names are:
- `exp_*_compare_positional_encodings` ✅
- `exp_*_overfitting_regularization_sweep` ✅
- `exp_*_phd_exp1_4t_vs_bg_sizes_and_pe` ✅

All commands above use the correct patterns matching your actual experiment names.

## WandB on HPC

- **WANDB_DIR**: Training and report jobs set `WANDB_DIR=/data/atlas/users/nterlind/outputs/wandb` so WandB local files are written on the large filesystem, not under the project directory (avoids disk quota). When running the migration script or any WandB-using tool manually on the login node, set the same: `export WANDB_DIR=/data/atlas/users/nterlind/outputs/wandb`.
- **Live tracking**: Add `logging=wandb_online` to the condor arguments to stream metrics to WandB in real time. Requires `wandb login` once (on login node or locally; credentials in `~/.netrc` or `WANDB_API_KEY`). If outbound HTTPS is blocked on worker nodes, use `logging=wandb_offline` and sync later with `wandb sync $WANDB_DIR`.
- **Cleanup**: Local WandB run dirs under `/data/atlas/users/nterlind/outputs/wandb` can be deleted after successful upload. Canonical data lives in Facts run dirs and in WandB cloud. After a large migration, you can remove the whole directory or old subdirs to free space.

### Binning vs Direct Experiment (PhD Presentation)

36 models: 3 pooling × 3 tokenization (direct, binned, VQ-VAE) × 2 MET × 2 vect.

```bash
# 1. Interactive preprocessing (run on login node via SSH)
cd /project/atlas/users/nterlind/Thesis-Code
conda activate thesis-ml  # or: conda activate /data/atlas/users/nterlind/venvs/thesis-ml

python scripts/create_binned_dataset.py \
  --input /data/atlas/users/nterlind/datasets/4tops_splitted.h5 \
  --output /data/atlas/users/nterlind/datasets/4tops_5bins_ours.h5 \
  --n-bins 5

python scripts/compare_binned_datasets.py \
  --ours /data/atlas/users/nterlind/datasets/4tops_5bins_ours.h5 \
  --ambres /data/atlas/users/avisive/tokens/binning/4tops/4top_5bins_binningOnBckgdEvents_train_AND_test.h5

# 2. (Optional) Pre-train VQ-VAE for tokenization=vq - run once, then copy checkpoint
#    Submit or run interactively with GPU:
python -m thesis_ml.cli.train env=stoomboot loop=ae phase1/latent_space=vq data=h5_tokens
#    After training, copy best checkpoint:
mkdir -p /data/atlas/users/nterlind/checkpoints
cp <run_dir>/best_val.pt /data/atlas/users/nterlind/checkpoints/vq_4tops_best.pt
#    The run_dir is under /data/atlas/users/nterlind/outputs/runs/run_YYYYMMDD-HHMMSS_...

# 3. Submit training sweep (36 models; omit tokenization=vq if no VQ checkpoint yet)
#    Add logging=wandb_online for live progress tracking in WandB
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=phd_presentation/exp_binning_vs_direct logging=wandb_online --multirun'

# 4. After training completes, submit report

# Exact sweep for run_20260211-141827_exp_binning_vs_direct (36 models):
# Sweep dir: /data/atlas/users/nterlind/outputs/multiruns/exp_20260211-141827_exp_binning_vs_direct

# Interactive (SSH to login node):
cd /project/atlas/users/nterlind/Thesis-Code
conda activate /data/atlas/users/nterlind/venvs/thesis-ml
export WANDB_DIR=/data/atlas/users/nterlind/outputs/wandb

thesis-report --config-name phd_summary_binning_vs_direct \
  inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_20260211-141827_exp_binning_vs_direct \
  inference.enabled=true

# Batch (Condor) - exact sweep:
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name phd_summary_binning_vs_direct inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_20260211-141827_exp_binning_vs_direct inference.enabled=true'

# Batch (Condor) - generic pattern for any exp_binning_vs_direct run:
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name phd_summary_binning_vs_direct inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_exp_binning_vs_direct inference.enabled=true'
```

Report output: `/data/atlas/users/nterlind/outputs/reports/report_<timestamp>_phd_summary_binning_vs_direct/`

**To run without VQ-VAE** (24 models): override tokenization in the sweep:
```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=phd_presentation/exp_binning_vs_direct logging=wandb_online hydra.sweeper.params.tokenization=direct,binned --multirun'
```

**WandB live tracking**: The `logging=wandb_online` override enables real-time metrics to WandB from HPC.

**Delete runs from a specific group** (e.g. failed experiment):
```bash
# Dry run first
python scripts/cleanup_wandb.py --project thesis-ml --entity nterlind-nikhef --group "exp_20260210-135259_exp_binning_vs_direct"

# Actually delete
python scripts/cleanup_wandb.py --project thesis-ml --entity nterlind-nikhef --group "exp_20260210-135259_exp_binning_vs_direct" --runs-only --execute
``` Ensure `wandb login` has been run once (e.g. locally or on a login node with `wandb login` and a valid API key). The train script sets `WANDB_DIR` and `WANDB_MODE=online`; with `logging=wandb_online` the config uses `mode: "online"` for live syncing.

## Notes

- The `compare_model_sizes` report now properly handles experiments using `+classifier/model_size: s200k,s600k,s1500k,s3000k` config groups
- Size labels will show as "s200k", "s600k", etc. when available, falling back to "64d6L" format otherwise
- All commands assume the base path `/data/atlas/users/nterlind/outputs/multiruns/` - adjust if your path differs
