# Hydra Multirun Directory Structure Fix

## Problem
Hydra multirun was creating numbered subdirectories (`0/`, `1/`, etc.) under `outputs/multiruns/exp_*/` instead of creating runs directly in `outputs/runs/run_*_jobN/` as configured.

## Root Cause
In Hydra multirun mode, when `subdir` is omitted, Hydra defaults to creating numbered subdirectories (`${hydra.job.num}`) under `sweep.dir`. The `run.dir` setting is not used for multiruns - only `sweep.dir` and `subdir` control where jobs run.

## Solution
Set `sweep.subdir` to an absolute path that matches `run.dir`. When `subdir` is an absolute path, Hydra uses it directly instead of treating it as relative to `sweep.dir`.

## Changes Made

### 1. Updated Base Multirun Config (`configs/hydra/experiment.yaml`)
- Added `subdir: ${env.output_root}/runs/run_${now:%Y%m%d-%H%M%S}_${hydra.job.name}_job${hydra.job.num}`
- This ensures each job runs in `outputs/runs/` instead of numbered subdirs under `outputs/multiruns/`

### 2. Updated Experiment Configs
- `configs/phase1/experiment/compare_globals_heads.yaml`
- `configs/phase1/experiment/compare_latent_spaces.yaml`
- Both now set `subdir` to match their `run.dir` template

### 3. Updated Documentation
- `configs/phase1/experiment/EXPERIMENT_GUIDE.md` - Updated template to show correct pattern

### 4. Added Debug Prints
- `src/thesis_ml/phase1/train/ae_loop.py` - Added debug output to inspect Hydra config resolution
- These can be removed after confirming the fix works

## Expected Behavior After Fix

When running:
```bash
python -m thesis_ml.train --multirun hydra=experiment phase1/latent_space=none,vq trainer.epochs=2 env=local
```

**Expected structure:**
```
outputs/
├── runs/
│   ├── run_20251103-134545_experiment_job0/  ← Full run artifacts here
│   │   ├── .hydra/
│   │   ├── best_val.pt
│   │   ├── facts/
│   │   └── train_figures/
│   └── run_20251103-134545_experiment_job1/  ← Full run artifacts here
│       └── ...
└── multiruns/
    └── exp_20251103-134545_experiment/  ← Metadata only
        ├── multirun.yaml
        └── (no numbered subdirectories)
```

## Testing

1. Run a test multirun:
   ```bash
   python -m thesis_ml.train --multirun hydra=experiment phase1/latent_space=none,vq trainer.epochs=2 env=local
   ```

2. Check the debug output - you should see:
   ```
   [DEBUG] cfg.hydra.run.dir: C:/Users/niels/.../outputs/runs/run_..._experiment_job0
   [DEBUG] cfg.hydra.sweep.dir: C:/Users/niels/.../outputs/multiruns/exp_...
   [DEBUG] cfg.hydra.sweep.subdir: C:/Users/niels/.../outputs/runs/run_..._experiment_job0
   [DEBUG] os.getcwd(): C:/Users/niels/.../outputs/runs/run_..._experiment_job0
   ```

3. Verify directory structure:
   - Runs should be in `outputs/runs/run_*_experiment_job0/`, `job1/`, etc.
   - `outputs/multiruns/exp_*/` should NOT contain numbered subdirectories (`0/`, `1/`)

4. After confirming it works, remove debug prints from `ae_loop.py`

## Technical Details

### Why This Works
- When `subdir` is an absolute path, Hydra treats it as the target directory directly
- When `chdir: true`, Hydra changes to `subdir` (which is now our absolute `run.dir` path)
- This bypasses Hydra's default behavior of creating numbered subdirs under `sweep.dir`

### Alternative Approaches Considered
1. Setting `subdir: ${hydra.run.dir}` - Potential resolution order issues
2. Using relative paths - Would still create nested structure
3. Post-processing to move files - More complex and error-prone

The chosen solution (explicit absolute path template) is the most reliable and clear.
