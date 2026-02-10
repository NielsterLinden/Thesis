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
- **Cleanup**: Local WandB run dirs under `/data/atlas/users/nterlind/outputs/wandb` can be deleted after successful upload. Canonical data lives in Facts run dirs and in WandB cloud. After a large migration, you can remove the whole directory or old subdirs to free space.

### Binning vs Direct Experiment (PhD Presentation)

```bash
# 1. Interactive preprocessing (run on login node via SSH)
cd /project/atlas/users/nterlind/Thesis-Code
conda activate thesis-ml
python scripts/create_binned_dataset.py \
  --input /data/atlas/users/nterlind/datasets/4tops_splitted.h5 \
  --output /data/atlas/users/nterlind/datasets/4tops_5bins_ours.h5 \
  --n-bins 5

python scripts/compare_binned_datasets.py \
  --ours /data/atlas/users/nterlind/datasets/4tops_5bins_ours.h5 \
  --ambres /data/atlas/users/avisive/tokens/binning/4tops/4top_5bins_binningOnBckgdEvents_train_AND_test.h5

# 2. Submit training sweep
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=phd_presentation/exp_binning_vs_direct --multirun'

# 3. After training completes, submit report
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name phd_summary_binning_vs_direct inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_exp_binning_vs_direct inference.enabled=true'
```

## Notes

- The `compare_model_sizes` report now properly handles experiments using `+classifier/model_size: s200k,s600k,s1500k,s3000k` config groups
- Size labels will show as "s200k", "s600k", etc. when available, falling back to "64d6L" format otherwise
- All commands assume the base path `/data/atlas/users/nterlind/outputs/multiruns/` - adjust if your path differs
