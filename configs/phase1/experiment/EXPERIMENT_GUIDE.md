# Phase 1 Experiment Configurations

This directory contains Hydra multirun experiment configurations for exploring different architectural and training choices in Phase 1 autoencoders.

## Available Experiments

### 1. `compare_latent_spaces.yaml`
**Purpose**: Compare different latent space types while keeping other factors constant.

**Grid**: 3 latent space configurations
- `none`: Identity bottleneck (no transformation)
- `linear`: Learned linear projection
- `vq`: Vector Quantization with discrete codebook

**Fixed Parameters**:
- Globals head: ON (`globals_beta: 1.0`)
- Epochs: 20

**Total Runs**: 3

**Usage**:
```bash
python -m thesis_ml.phase1 --config-name config phase1/experiment=compare_latent_spaces
```

---

### 2. `compare_globals_heads.yaml`
**Purpose**: Evaluate the influence of global head reconstruction (MET/METphi) across different latent space architectures.

**Grid**: 3×3 (latent spaces × globals_beta values)

**Latent Spaces**:
- `none`: Identity bottleneck
- `linear`: Learned linear projection
- `vq`: Vector Quantization

**Global Head Configurations**:
- `globals_beta: 0`: Globals head **OFF** (MET/METphi ignored)
- `globals_beta: 1`: Globals head **ON** with standard weight (equal importance to token reconstruction)
- `globals_beta: 10`: Globals head **ON** with 10× weight (MET/METphi prioritized)

**Fixed Parameters**:
- Epochs: 20

**Total Runs**: 9 (3 latent spaces × 3 beta values)

**Expected Run Matrix**:
```
                    none           linear            vq
globals_beta=0      [run_0]        [run_1]        [run_2]
globals_beta=1      [run_3]        [run_4]        [run_5]
globals_beta=10     [run_6]        [run_7]        [run_8]
```

**Usage**:
```bash
python -m thesis_ml.phase1 --config-name config phase1/experiment=compare_globals_heads
```

**Usage (locally instead of HPC)**:
```bash
python -m thesis_ml.phase1 --config-name config phase1/experiment=compare_globals_heads env=local
```

---

## How Multirun Works

When you run an experiment config:

1. **Hydra creates a sweep directory** in `outputs/experiments/`:
   ```
   exp_20251031-120000_compare_globals_heads/
   ├── 0/  (none, beta=0)
   ├── 1/  (linear, beta=0)
   ├── 2/  (vq, beta=0)
   ├── 3/  (none, beta=1)
   ├── 4/  (linear, beta=1)
   ├── 5/  (vq, beta=1)
   ├── 6/  (none, beta=10)
   ├── 7/  (linear, beta=10)
   └── 8/  (vq, beta=10)
   ```

2. **Each subdirectory contains**:
   - `cfg.yaml`: The composed config for that run
   - `model.pt`: Best model checkpoint
   - `figures/`: Generated plots
   - `facts/`: Detailed metrics and data
   - `run.log`: Training logs

3. **Sweep-level reports** can be generated post-hoc from all runs.

---

## Interpreting Results

### Key Metrics to Track

1. **`rec_tokens`**: Token reconstruction loss (should be similar across latent spaces)
2. **`rec_globals`**: Global (MET/METphi) reconstruction loss
   - Should decrease as `globals_beta` increases (more weight)
   - Should be 0 when `globals_beta=0`
3. **`total_loss`**: Combined training loss
4. **Validation performance**: How well does the model generalize?

### Expected Observations

For `compare_globals_heads`:

| Scenario | Expected Behavior |
|----------|-------------------|
| `globals_beta=0` | rec_globals should be ignored; token reconstruction dominates |
| `globals_beta=1` | Balanced training; model learns both tokens and globals |
| `globals_beta=10` | rec_globals becomes primary objective; may improve global accuracy at cost of token accuracy |
| `none` latent | No bottleneck; highest reconstruction but potentially overfitting |
| `linear` latent | Moderate bottleneck; learned transformation adds expressiveness |
| `vq` latent | Discrete bottleneck; hardest constraint but promotes interpretability |

---

## Creating New Experiments

To add a new experiment, create a YAML file following this template:

```yaml
# @package _global_

experiment:
  name: "your_experiment_name"
  description: "Brief description of what you're comparing"

hydra:
  mode: MULTIRUN
  job:
    chdir: true
    name: ${experiment.name}
  sweep:
    dir: ${env.output_root}/experiments/exp_${now:%Y%m%d-%H%M%S}_${experiment.name}
    subdir: ${hydra.job.num}
  sweeper:
    params:
      # List parameter sweeps here (comma-separated values)
      # Example: phase1/encoder: mlp,gnn
      # Example: phase1.trainer.lr: 1e-3,1e-4,1e-5
      phase1/latent_space: none,linear,vq
      phase1.trainer.epochs: 20
```

**Key points**:
- Use `@package _global_` to merge at the top level
- Sweep parameters can be:
  - Config group selections: `phase1/latent_space: none,linear,vq`
  - Scalar values: `phase1.trainer.lr: 1e-3,1e-4`
  - Multiple parameters create a cartesian product of runs
- Directory structure auto-generates based on sweep size and timestamp

---

## Reporting Results

After running an experiment, use the reporting tools:

```bash
# Generate comparison reports across all runs
python -m thesis_ml.reports.compare_experiments \
  outputs/experiments/exp_20251031-120000_compare_globals_heads/
```

This generates:
- CSV/JSON summaries of best losses per run
- Aggregated plots comparing metrics
- Recommendations for best-performing configurations
