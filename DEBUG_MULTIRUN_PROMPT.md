# Debug Prompt: Hydra Multirun Directory Structure Issue

## Summary

Hydra multirun is creating numbered subdirectories (`0/`, `1/`, etc.) under `outputs/multiruns/exp_*/` with full training artifacts, instead of creating runs directly in `outputs/runs/run_*_jobN/` as configured. Despite setting `run.dir` to point to `outputs/runs/` and omitting `subdir`, Hydra seems to default to using `sweep.dir/subdir` pattern. Need to understand Hydra's multirun working directory resolution and fix the config so runs go directly to `outputs/runs/`.

**Hydra version:** 1.3.2 (from `environment.yml`)

## Goal

When running a Hydra multirun, each individual run should be created directly in `outputs/runs/` with a unique name pattern `run_TIMESTAMP_experimentname_jobN`, and the multirun directory (`outputs/multiruns/exp_TIMESTAMP_name/`) should contain **only** metadata files (`multirun.yaml`, `runs_index.json`, `runptrs/`) - **NO** numbered subdirectories (`0/`, `1/`, etc.) and **NO** training artifacts (models, facts, figures).

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
        ├── runs_index.json
        └── runptrs/
            ├── 0 → ../../runs/run_20251103-134545_experiment_job0
            └── 1 → ../../runs/run_20251103-134545_experiment_job1
```

**Current broken behavior:**
```
outputs/
├── runs/
│   └── [runs NOT here yet]
└── multiruns/
    └── exp_20251103-134545_experiment/
        ├── multirun.yaml
        ├── 0/  ← BAD: Full run artifacts nested here
        │   ├── .hydra/
        │   ├── best_val.pt
        │   ├── facts/
        │   └── train_figures/
        └── 1/  ← BAD: Full run artifacts nested here
            └── ...
```

## Problem Statement

Despite configuring `run.dir` to point to `outputs/runs/run_TIMESTAMP_name_jobN` and removing `subdir` from the sweep config, Hydra is still creating numbered subdirectories (`0/`, `1/`, etc.) under the multirun directory and placing all run artifacts there instead of in `outputs/runs/`.

## Current Configuration

### Main Hydra Config (`configs/config.yaml`)
```yaml
defaults:
  - env: stoomboot
  - hydra: run_single  # For single runs
  - logging: default
  # ... other defaults
```

### Single Run Config (`configs/hydra/run_single.yaml`)
```yaml
job:
  chdir: true
  name: run
run:
  dir: ${env.output_root}/runs/run_${now:%Y%m%d-%H%M%S}_${hydra.job.name}
```
**This works correctly** - single runs go to `outputs/runs/run_TIMESTAMP_name/`

### Multirun Config (`configs/hydra/experiment.yaml`)
```yaml
job:
  chdir: true  # Must be true for Hydra to create .hydra/ directory
  name: experiment
# Override run.dir so each job writes directly to outputs/runs/ (not nested under multirun)
run:
  dir: ${env.output_root}/runs/run_${now:%Y%m%d-%H%M%S}_${hydra.job.name}_job${hydra.job.num}
sweep:
  dir: ${env.output_root}/multiruns/exp_${now:%Y%m%d-%H%M%S}_${hydra.job.name}
  # Don't set subdir - this makes Hydra use run.dir directly instead of creating numbered subdirs
```

### Experiment-Specific Config (`configs/phase1/experiment/compare_globals_heads.yaml`)
```yaml
hydra:
  mode: MULTIRUN
  job:
    chdir: true
    name: ${experiment.name}
  run:
    dir: ${env.output_root}/runs/run_${now:%Y%m%d-%H%M%S}_${experiment.name}_job${hydra.job.num}
  sweep:
    dir: ${env.output_root}/multiruns/exp_${now:%Y%m%d-%H%M%S}_${experiment.name}
    # Don't set subdir
```

## How Training Code Gets Run Directory

### Phase1 Training Loop (`src/thesis_ml/phase1/train/ae_loop.py`)
```python
# Line ~51-58
# logging dir from Hydra (chdir=true means Hydra sets cwd to run.dir)
outdir = None
if cfg.logging.save_artifacts:
    # When chdir=true, Hydra changes to run.dir, so os.getcwd() is correct
    outdir = os.getcwd()
    os.makedirs(outdir, exist_ok=True)
```

The training code relies on `os.getcwd()` which should be set by Hydra when `chdir: true` and `run.dir` is specified.

## Command Used

```bash
python -m thesis_ml.train --multirun hydra=experiment phase1/latent_space=none,vq trainer.epochs=2 env=local
```

## Questions to Investigate

1. **Does Hydra respect `run.dir` in multirun mode?** Or does `sweep.dir` + `subdir` take precedence?
2. **When `subdir` is omitted, what does Hydra do?** Does it default to creating numbered subdirs anyway?
3. **Is there a way to force Hydra to use `run.dir` instead of `sweep.dir/subdir` in multirun mode?**
4. **Should we set `subdir` to something specific** (like `${hydra.run.dir}` or empty string) instead of omitting it?
5. **Does the order of config merging matter?** The experiment config uses `@package _global_` - does this override the base `hydra/experiment.yaml` correctly?

## Files to Examine

1. **Hydra configs:**
   - `configs/hydra/experiment.yaml` - Base multirun config
   - `configs/hydra/run_single.yaml` - Single run config (works correctly)
   - `configs/phase1/experiment/*.yaml` - Experiment-specific overrides

2. **Training code:**
   - `src/thesis_ml/phase1/train/ae_loop.py` - How it gets the run directory
   - `src/thesis_ml/general/train/test_mlp_loop.py` - Alternative training loop

3. **Entry point:**
   - `src/thesis_ml/train/__main__.py` - How Hydra is initialized

4. **Path utilities:**
   - `src/thesis_ml/utils/paths.py` - Path resolution helpers

## What We've Tried

1. ✅ Set `run.dir` to `outputs/runs/run_TIMESTAMP_name_jobN`
2. ✅ Removed `subdir` from sweep config (tried `subdir: null` and omitting it)
3. ✅ Set `chdir: true` (needed for `.hydra/` directory creation)
4. ❌ Still creates numbered subdirectories under multirun

## Expected Hydra Behavior (Documentation)

According to Hydra docs, when `chdir: true`:
- Hydra changes to the directory specified by `run.dir` (for single runs)
- For multiruns, if `subdir` is set, Hydra changes to `sweep.dir/subdir`
- If `subdir` is not set, Hydra should use `run.dir` directly

But this doesn't seem to be working. We need to understand:
- What Hydra actually does when both `run.dir` and `sweep.dir` are set but `subdir` is omitted
- Whether there's a config merge order issue
- Whether we need to use a different Hydra feature or workaround

## Success Criteria

After fixing, running:
```bash
python -m thesis_ml.train --multirun hydra=experiment phase1/latent_space=none,vq trainer.epochs=2 env=local
```

Should result in:
- ✅ Runs created in `outputs/runs/run_*_experiment_job0/`, `run_*_experiment_job1/`, etc.
- ✅ No numbered subdirectories (`0/`, `1/`) under `outputs/multiruns/exp_*/`
- ✅ Multirun directory contains only `multirun.yaml` (and later `runs_index.json` after post-processing)
- ✅ Training code's `os.getcwd()` returns the path to `outputs/runs/run_*_jobN/`

## Debugging Steps

1. **Check Hydra's actual config resolution:**
   - Print the resolved `hydra.run.dir` and `hydra.sweep.dir` values at runtime
   - Verify which config is actually being used

2. **Test Hydra behavior:**
   - Try setting `subdir: ""` (empty string) vs omitting it
   - Try using `hydra.run.dir` in `subdir` value
   - Check if there's a Hydra version-specific behavior

3. **Check config merge order:**
   - Verify that experiment-specific configs properly override base configs
   - Check if `@package _global_` is affecting the merge

4. **Alternative approaches:**
   - Use Hydra's `hydra.sweep.subdir` with a computed value that points to `outputs/runs/`
   - Post-process after each job completes to move artifacts
   - Use a custom Hydra launcher plugin

Please help debug why Hydra is ignoring `run.dir` in multirun mode and creating numbered subdirectories instead.

## Additional Context

### Entry Point (`src/thesis_ml/train/__main__.py`)
```python
@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # ... validation ...
    loop = cfg.get("loop")
    fn = DISPATCH.get(loop)
    return fn(cfg)
```

### Environment Config (`configs/env/local.yaml`)
```yaml
data_root: "C:/Users/niels/Projects/Thesis-Code/Data"
output_root: "C:/Users/niels/Projects/Thesis-Code/Code/Niels_repo/outputs"
```

### What Gets Resolved
When running `--multirun hydra=experiment`, Hydra should:
1. Load `configs/config.yaml` (which defaults to `hydra: run_single`)
2. Override with `hydra=experiment` → loads `configs/hydra/experiment.yaml`
3. Merge experiment-specific configs if using `--config-name phase1/experiment/...`
4. Resolve `${env.output_root}` → actual path from `configs/env/local.yaml` or `stoomboot.yaml`
5. Resolve `${hydra.job.num}` → job index (0, 1, 2, ...)
6. Set working directory based on `run.dir` or `sweep.dir/subdir`

**The issue:** Step 6 is not working as expected - Hydra seems to be using `sweep.dir/subdir` instead of `run.dir` even when `subdir` is omitted.

## Test Command to Reproduce

```bash
# Should create runs in outputs/runs/run_*_experiment_job0/, job1/, etc.
# But currently creates outputs/multiruns/exp_*/0/, 1/, etc.
python -m thesis_ml.train --multirun hydra=experiment phase1/latent_space=none,vq trainer.epochs=2 env=local
```

## Verification

After running, check:
```bash
# Should be empty or not exist
ls outputs/multiruns/exp_*/0/

# Should contain the runs
ls outputs/runs/run_*_experiment_job*
```

## Key Questions to Answer

1. **Does Hydra have a default `subdir` behavior in multirun mode?** Even when `subdir` is omitted, does Hydra default to `${hydra.job.num}`?

2. **What is the precedence order?** When both `run.dir` and `sweep.dir` + `subdir` are set, which takes precedence?

3. **Can we inspect the resolved Hydra config?** Add debugging to print `cfg.hydra.run.dir`, `cfg.hydra.sweep.dir`, and `cfg.hydra.sweep.subdir` at runtime to see what Hydra actually resolved.

4. **Is `chdir: true` compatible with our goal?** Does `chdir: true` force Hydra to use `sweep.dir/subdir` instead of `run.dir`?

5. **Do we need to use `hydra.sweep.subdir` differently?** Maybe set it to `${hydra.run.dir}` or use a relative path from `sweep.dir`?

## Suggested Debugging Approach

1. **Add debug prints** in the training loop to see what `os.getcwd()` actually is:
   ```python
   print(f"[DEBUG] Current working directory: {os.getcwd()}")
   print(f"[DEBUG] cfg.hydra.run.dir: {cfg.hydra.run.dir if hasattr(cfg, 'hydra') and hasattr(cfg.hydra, 'run') else 'NOT SET'}")
   print(f"[DEBUG] cfg.hydra.sweep.dir: {cfg.hydra.sweep.dir if hasattr(cfg, 'hydra') and hasattr(cfg.hydra, 'sweep') else 'NOT SET'}")
   print(f"[DEBUG] cfg.hydra.sweep.subdir: {cfg.hydra.sweep.subdir if hasattr(cfg, 'hydra') and hasattr(cfg.hydra, 'sweep') and hasattr(cfg.hydra.sweep, 'subdir') else 'NOT SET'}")
   ```

2. **Check Hydra version** - behavior might differ between versions

3. **Review Hydra documentation** for multirun working directory behavior

4. **Test if explicitly setting `subdir: null` vs omitting it makes a difference**

5. **Try setting `subdir` to a computed value** that equals `run.dir`:
   ```yaml
   sweep:
     subdir: ${hydra.run.dir}
   ```

## Expected vs Actual

**Expected:** When `run.dir` is set and `subdir` is omitted/null, Hydra should change to `run.dir` for each job.

**Actual:** Hydra is creating `sweep.dir/0/`, `sweep.dir/1/`, etc. and changing into those directories instead.

Please help identify why Hydra is not respecting `run.dir` in multirun mode and how to fix it.

## Technical Details

### Hydra Configuration Structure
- Hydra version: Check `pyproject.toml` or `environment.yml`
- Config path: `configs/` (relative to `src/thesis_ml/train/__main__.py`)
- Config name: `config` (from `configs/config.yaml`)
- Version base: `1.3`

### Config Resolution Flow
1. Base: `configs/config.yaml` → defaults to `hydra: run_single`
2. Override: `--multirun hydra=experiment` → loads `configs/hydra/experiment.yaml`
3. Additional: `--config-name phase1/experiment/...` → merges experiment-specific configs
4. Runtime: Hydra resolves all `${...}` variables and sets working directory

### Training Code Entry Points
- Single run: `python -m thesis_ml.train` → uses `hydra: run_single`
- Multirun: `python -m thesis_ml.train --multirun hydra=experiment ...` → uses `hydra: experiment`

Both eventually call the same training function (`ae_loop.train()` or `test_mlp_loop.train()`) which uses `os.getcwd()` to determine where to write artifacts.

### What to Check
1. Print the actual resolved Hydra config values at runtime
2. Check if Hydra creates a `.hydra/` directory in the wrong location
3. Verify if `subdir` has a default value when omitted
4. Test with explicit `subdir: ${hydra.run.dir}` or `subdir: ""` (empty string)
5. Check Hydra's multirun documentation for the correct way to override working directories

The core issue: **Hydra multirun mode seems to ignore `run.dir` and always uses `sweep.dir/subdir` pattern, even when `subdir` is omitted.**
