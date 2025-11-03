# Validation Commands for Outputs Refactor

Run these commands to validate the refactor implementation.

## Prerequisites

- Python environment activated
- `outputs/` directory exists and is writable

## 1. Single Training Run

### Command
```bash
python -m thesis_ml.train trainer.epochs=2 env=local
```

### Verification
```bash
# Check run directory structure
ls outputs/runs/run_*/
# Should contain: .hydra/, best_val.pt, last.pt, model.pt (symlink), facts/, train_figures/, report_pointer.txt

# Verify no cfg.yaml (use .hydra/config.yaml instead)
test ! -f outputs/runs/run_*/cfg.yaml && echo "OK: No cfg.yaml" || echo "FAIL: cfg.yaml exists"

# Verify checkpoint naming
test -f outputs/runs/run_*/best_val.pt && echo "OK: best_val.pt exists" || echo "FAIL: best_val.pt missing"
test -f outputs/runs/run_*/last.pt && echo "OK: last.pt exists" || echo "FAIL: last.pt missing"
test -L outputs/runs/run_*/model.pt && echo "OK: model.pt is symlink" || echo "WARN: model.pt not symlink (may be copy on Windows)"

# Verify train_figures/ (not figures/)
test -d outputs/runs/run_*/train_figures && echo "OK: train_figures/ exists" || echo "FAIL: train_figures/ missing"
test ! -d outputs/runs/run_*/figures && echo "OK: No old figures/ dir" || echo "WARN: Old figures/ dir exists"

# Verify no plots outside runs/
find outputs -name "*.png" -not -path "*/runs/*/train_figures/*" -not -path "*/reports/*" && echo "FAIL: Plots outside runs/reports" || echo "OK: All plots in correct locations"
```

## 2. Multirun Sweep

### Command
```bash
python -m thesis_ml.train --multirun hydra=experiment phase1/latent_space=none,vq trainer.epochs=2 env=local
```

### Verification
```bash
# Check multirun directory structure
ls outputs/multiruns/exp_*/
# Should contain: multirun.yaml (only)

# Verify runs are in outputs/runs/ (not nested under multirun)
ls outputs/runs/run_*_job*
# Should see runs like: run_20251103-134545_experiment_job0, run_20251103-134545_experiment_job1

# Verify NO nested runs in multirun directory
test -d outputs/multiruns/exp_*/0 && echo "FAIL: Nested run directory 0/ exists" || echo "OK: No nested runs"

# Verify no plots or model files in multirun directory
find outputs/multiruns/exp_* -name "*.png" -o -name "*.pt" && echo "FAIL: Artifacts in multirun dir" || echo "OK: No artifacts in multirun"
```

## 3. Report Generation

### Command
```bash
python -m thesis_ml.reports.report_from_sweep --config-name compare_tokenizers inputs.sweep_dir=outputs/multiruns/exp_* env=local
```

### Verification
```bash
# Check report directory structure
ls outputs/reports/report_*/
# Should contain: manifest.yaml, training/, inference/

# Verify training/ subdir
ls outputs/reports/report_*/training/
# Should contain: summary.csv, summary.json, figures/

# Verify inference/ subdir exists (even if empty)
test -d outputs/reports/report_*/inference && echo "OK: inference/ exists" || echo "FAIL: inference/ missing"

# Verify manifest.yaml
cat outputs/reports/report_*/manifest.yaml

# Verify report_pointer.txt in each run
for run_dir in outputs/runs/run_*; do
  if grep -q "report_" "$run_dir/report_pointer.txt" 2>/dev/null; then
    echo "OK: $run_dir has report pointer"
  else
    echo "WARN: $run_dir missing report pointer"
  fi
done

# Verify no plots outside reports/
find outputs -name "*.png" -not -path "*/runs/*/train_figures/*" -not -path "*/reports/*/training/figures/*" -not -path "*/reports/*/inference/figures/*" && echo "FAIL: Plots in wrong location" || echo "OK: All plots in correct locations"
```

## 4. Cross-Environment Test

### Local
```bash
python -m thesis_ml.train trainer.epochs=1 env=local
```

### Stoomboot (if accessible)
```bash
python -m thesis_ml.train trainer.epochs=1 env=stoomboot
```

### Verification
- Both should create runs under `outputs/runs/`
- Paths should resolve correctly regardless of `env.output_root` value
- No hardcoded paths should break

## 5. Legacy Compatibility Check

### Verify old runs still readable
```bash
# If you have old runs with cfg.yaml, they should still work
python -m thesis_ml.reports.report_from_sweep --config-name compare_tokenizers inputs.run_dirs='["outputs/runs/run_OLD_RUN"]' env=local
```

## Expected Outputs Summary

### Runs (`outputs/runs/run_*/`)
- `.hydra/config.yaml` (canonical config)
- `best_val.pt` (best validation checkpoint)
- `last.pt` (final epoch checkpoint)
- `model.pt` → symlink to `best_val.pt`
- `facts/` (scalars.csv, events.jsonl)
- `train_figures/` (quick training plots only)
- `report_pointer.txt` (append-only list of report IDs)

### Multiruns (`outputs/multiruns/exp_*/`)
- `multirun.yaml` (Hydra snapshot - metadata only)

### Reports (`outputs/reports/report_*/`)
- `manifest.yaml` (report metadata + dataset fingerprints)
- `training/` (summary.csv, summary.json, figures/)
- `inference/` (summary.json, figures/, optional raw_scores/)

## Notes

- On Windows, symlinks may fall back to copies (this is OK)
- Pointer files work cross-platform (symlinks are optional)
- Old runs with `cfg.yaml` are still readable (backward compatible)
- Reports always go to `outputs/reports/` (never under multirun directories)
