# Reports System Refactoring - Summary

## Completed: October 31, 2025

## Overview
Successfully refactored the reporting system from a phase1-specific structure to a general-purpose, modular architecture that promotes code reuse across all experiments.

## New Directory Structure

```
src/thesis_ml/reports/
├── __init__.py
├── report_from_sweep.py          # Generic CLI entry point
├── utils/
│   ├── __init__.py
│   ├── read_facts.py              # Run discovery + DataFrame building (enhanced)
│   └── io.py                      # Output resolution, directory ops, JSON save
├── plots/
│   ├── __init__.py
│   ├── curves.py                  # Training curves (loss vs time)
│   ├── grids.py                   # Heatmaps for grid experiments
│   └── scatter.py                 # Scatter plots with color grouping
└── experiments/
    ├── __init__.py
    ├── compare_tokenizers.py      # Tokenizer comparison report (refactored)
    └── compare_globals_heads.py   # Global heads experiment report (new)
```

## Key Changes

### 1. Moved from phase1-specific to general-purpose
- **Old**: `src/thesis_ml/phase1/reports/`
- **New**: `src/thesis_ml/reports/`
- Rationale: Reports aren't phase-specific and should be reusable

### 2. Extracted reusable utilities
- **`utils/io.py`**: Output directory resolution, figure config extraction
- **`utils/read_facts.py`**: Enhanced with `latent_space`, `globals_beta`, `loss.rec_globals_best` extraction

### 3. Created modular plotting functions
- **`plots/curves.py`**: Generic loss-vs-time plots
- **`plots/grids.py`**: 2D heatmaps for parameter sweeps
- **`plots/scatter.py`**: Scatter plots with color grouping and annotations

### 4. Generic CLI entry point
- **`report_from_sweep.py`**: No longer hardcoded to specific config
- Uses `--config-name` to dynamically load experiment modules
- Supports any report in `experiments/` directory

### 5. Config structure
```
configs/report/
├── compare_tokenizers.yaml    # Existing, unchanged
└── compare_globals_heads.yaml # New for globals experiment
```

## Usage Examples

### Compare Globals Heads Report
```bash
python -m thesis_ml.reports.report_from_sweep \
  --config-name compare_globals_heads \
  +sweep_dir=/path/to/exp_TIMESTAMP_compare_globals_heads
```

### Compare Tokenizers Report
```bash
python -m thesis_ml.reports.report_from_sweep \
  --config-name compare_tokenizers \
  +sweep_dir=/path/to/experiment
```

## Files Created/Modified

### New Files (11 total)
1. `src/thesis_ml/reports/__init__.py`
2. `src/thesis_ml/reports/utils/__init__.py`
3. `src/thesis_ml/reports/utils/read_facts.py` (enhanced)
4. `src/thesis_ml/reports/utils/io.py`
5. `src/thesis_ml/reports/plots/__init__.py`
6. `src/thesis_ml/reports/plots/curves.py`
7. `src/thesis_ml/reports/plots/grids.py`
8. `src/thesis_ml/reports/plots/scatter.py`
9. `src/thesis_ml/reports/experiments/__init__.py`
10. `src/thesis_ml/reports/experiments/compare_tokenizers.py` (refactored)
11. `src/thesis_ml/reports/experiments/compare_globals_heads.py`
12. `src/thesis_ml/reports/report_from_sweep.py` (generic version)
13. `configs/report/compare_globals_heads.yaml`

### Modified Files (1)
- `tests/test_reports_compare_tokenizers.py` (import path updated)

### Deleted
- `src/thesis_ml/phase1/reports/` (entire directory)

## Enhancements to read_facts.py

Added extraction for new metrics:
- `latent_space`: From `phase1/latent_space` override
- `globals_beta`: From `phase1.decoder.globals_beta` override (as float)
- `loss.rec_globals_best`: From `history_rec_globals` at best epoch

These are automatically extracted for all runs and available in the DataFrame.

## Benefits

1. **Reusability**: Plotting functions can be used across any experiment
2. **Maintainability**: Single source of truth for IO, plotting logic
3. **Extensibility**: Add new experiments by creating a module in `experiments/`
4. **Flexibility**: Generic CLI accepts any config without code changes
5. **Clarity**: Separation of concerns (utils, plots, orchestration)

## Testing

- Existing test updated and passes with new import path
- No functional changes to compare_tokenizers behavior
- New compare_globals_heads experiment ready for use

## Next Steps

To create a new experiment report:
1. Create config in `configs/report/your_experiment.yaml`
2. Create module in `src/thesis_ml/reports/experiments/your_experiment.py`
3. Implement `run_report(cfg: DictConfig)` function
4. Reuse plotting functions from `plots/` or add new ones
5. Run with: `python -m thesis_ml.reports.report_from_sweep --config-name your_experiment +sweep_dir=...`
