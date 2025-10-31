# Compare Globals Heads Experiment - Complete Summary

## 📋 Overview

A new experiment configuration has been created to systematically evaluate the influence of the global head (MET/METphi reconstruction) on model performance across different latent space architectures.

**Location**: `configs/phase1/experiment/compare_globals_heads.yaml`

---

## 🎯 Experiment Design

### 3×3 Grid: 9 Total Runs

The experiment creates a Cartesian product of:
- **3 Latent Space Types**: `none`, `linear`, `vq`
- **3 Global Head Weights**: `0` (off), `1` (standard), `10` (emphasized)

```
Run Matrix:
┌──────────────┬──────────────┬──────────────┬──────────────┐
│              │    NONE      │    LINEAR    │      VQ      │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ β=0 (OFF)    │   Run 0      │   Run 1      │   Run 2      │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ β=1 (STD)    │   Run 3      │   Run 4      │   Run 5      │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ β=10 (HI)    │   Run 6      │   Run 7      │   Run 8      │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 📁 Files Created

### Experiment Configuration
- **`configs/phase1/experiment/compare_globals_heads.yaml`** - Main config (21 lines)
  - Sets up Hydra multirun sweep
  - Specifies parameter combinations
  - 20 epochs per run

### Documentation
- **`configs/phase1/experiment/EXPERIMENT_GUIDE.md`** - Complete guide
  - How to run experiments
  - Understanding multirun
  - Interpretation guide

- **`configs/phase1/experiment/COMPARE_GLOBALS_HEADS_README.md`** - Detailed analysis
  - Experiment design rationale
  - Expected behaviors and metrics
  - Prediction table for results
  - Scenario interpretation (5 scenarios)
  - Next steps after running

- **`configs/phase1/experiment/QUICK_REFERENCE.md`** - Quick reference
  - TL;DR version
  - Commands to run
  - Output structure
  - Quick analysis commands
  - Troubleshooting tips

---

## ⚙️ Configuration Details

```yaml
phase1/latent_space: none,linear,vq
# Sweeps over 3 latent space bottleneck types:
# - none: identity (z = z_e)
# - linear: learned projection (z = Linear(z_e))
# - vq: vector quantization (discrete codebook)

phase1.decoder.globals_beta: 0,1,10
# Sweeps over 3 global head loss weights:
# - 0: Globals head disabled (no MET/METphi reconstruction)
# - 1: Standard weight (equal importance as tokens)
# - 10: Emphasized (10× weight on globals)

phase1.trainer.epochs: 20
# Number of training epochs per run
# Total runtime: ~9 × 20 min = 180 min on GPU (3 hours)
```

---

## 🏃 How to Run

### Standard (HPC)
```bash
python -m thesis_ml.phase1 --config-name config phase1/experiment=compare_globals_heads
```

### Local Machine
```bash
python -m thesis_ml.phase1 --config-name config phase1/experiment=compare_globals_heads env=local
```

### Custom Settings
```bash
# Fewer epochs for quick test
python -m thesis_ml.phase1 --config-name config \
  phase1/experiment=compare_globals_heads \
  phase1.trainer.epochs=10

# Different sweep parameters
python -m thesis_ml.phase1 --config-name config \
  phase1/experiment=compare_globals_heads \
  'phase1.decoder.globals_beta=0.5,1,5'
```

---

## 📊 Expected Results

### Metrics Tracked
1. **`rec_tokens`** - Token reconstruction loss
2. **`rec_globals`** - Global reconstruction loss (MET/METphi)
3. **`total_loss`** - Sum of all losses
4. **`commit`, `codebook`, `perplex`** - VQ-specific metrics

### Expected Trends

**Column Comparison (Latent Space Effect)**:
```
reconstruction difficulty:
  none < linear < vq
  (easy) (medium) (hard)
```

**Row Comparison (Beta Effect)**:
```
globals_beta=0   → rec_globals ignored
globals_beta=1   → balanced learning
globals_beta=10  → globals prioritized (trade-off with tokens)
```

### Key Insights to Look For

1. **Does beta affect global reconstruction?**
   - If yes → globals head is learning
   - If no → investigate encoder

2. **Is there a token-globals trade-off?**
   - If rec_tokens increases at beta=10 → latent space too small
   - If rec_tokens stable → enough capacity

3. **Which latent space is best?**
   - none: highest capacity but no constraints
   - linear: good balance
   - vq: most constrained but interpretable

---

## 📁 Output Structure

```
outputs/experiments/exp_20251031-HHMMSS_compare_globals_heads/
│
├── 0/  (none, beta=0)
│   ├── cfg.yaml                    # Config used for this run
│   ├── model.pt                    # Best checkpoint
│   ├── run.log                     # Training log (final loss here)
│   ├── facts/
│   │   ├── metrics_train.csv       # Per-epoch train metrics
│   │   ├── metrics_val.csv         # Per-epoch val metrics
│   │   └── ...
│   └── figures/
│       ├── losses.png              # Loss curves
│       ├── recon.png               # Reconstruction metrics
│       └── ...
│
├── 1/  (linear, beta=0)
│   └── ... (same structure)
│
├── ...
│
└── 8/  (vq, beta=10)
    └── ... (same structure)
```

---

## 🔍 Quick Analysis

### Extract All Final Losses
```bash
for i in {0..8}; do
  echo "Run $i:"
  tail -3 outputs/experiments/exp_*compare_globals_heads/$i/run.log | grep val_loss
done
```

### Find Best Run
```bash
grep -r "val_loss" outputs/experiments/exp_*compare_globals_heads/*/run.log | \
  sort -t'=' -k3 -n | head -1
```

### View Run Configuration
```bash
cat outputs/experiments/exp_*compare_globals_heads/4/cfg.yaml  # Run 4 (linear, beta=1)
```

---

## 📈 Interpretation Scenarios

### Scenario A: Strong Beta Effect (EXPECTED ✓)
```
rec_globals:
  beta=0  = 0.500 (high, not optimized)
  beta=1  = 0.050 (moderate)
  beta=10 = 0.010 (low, well-optimized)

rec_tokens:
  beta=0  = 0.100
  beta=1  = 0.101
  beta=10 = 0.102  ← slight degradation (acceptable)

→ Conclusion: Model successfully learning globals with minimal token trade-off
```

### Scenario B: No Beta Effect (UNEXPECTED ✗)
```
rec_globals similar across all betas (e.g., all ≈ 0.200)

→ Conclusion: Something wrong
→ Investigation:
  - Does encoder pass globals to latent?
  - Is globals_head actually being built?
  - Check cfg.yaml from run to verify globals_beta is set
```

### Scenario C: Severe Trade-off (LATENT TOO SMALL)
```
rec_tokens:  beta=0=0.100, beta=10=0.150  ← 50% increase!
rec_globals: beta=0=0.500, beta=10=0.010  ← very dramatic

→ Conclusion: Zero-sum game; can't optimize both
→ Solutions:
  - Increase latent_dim: 32 → 64
  - Use globals_beta=0 (drop globals head)
  - Use globals_beta=1 (compromise)
```

### Scenario D: VQ Significantly Worse
```
all metrics for VQ >> for linear/none

→ Conclusion: Discrete codebook loses important info
→ Decision: Skip VQ for this task; use linear or none
```

---

## 🎯 Recommendations

### If Goals Are...

**"Reproduce tokens and globals equally well"**
→ Use `linear + globals_beta=1` (Run 4)
→ Balanced approach

**"Maximize token reconstruction"**
→ Use `none + globals_beta=0` (Run 0)
→ No globals constraint

**"Maximize global accuracy"**
→ Use `linear + globals_beta=10` (Run 7)
→ Strongest global optimization

**"Get discrete representation"**
→ Use `vq + globals_beta=1` (Run 5)
→ Only if VQ comparable to linear/none

**"Production baseline (safe choice)"**
→ Use `linear + globals_beta=1` (Run 4)
→ Best all-around balance

---

## 🚀 Next Steps

1. **Run the experiment** (3 hours on GPU)
   ```bash
   python -m thesis_ml.phase1 --config-name config phase1/experiment=compare_globals_heads
   ```

2. **Extract and compare metrics**
   - Open each `run.log` and note final losses
   - Or use the quick analysis commands above

3. **Analyze patterns**
   - Does beta affect globals? (should yes)
   - Is there token-globals trade-off? (should be minimal)
   - Which latent space performs best?

4. **Choose best configuration**
   - Based on your priorities
   - Update `configs/phase1/decoder/mlp.yaml` if changing default

5. **Document findings**
   - Create results table
   - Add to thesis methodology
   - Cite specific runs

6. **Continue with Phase 2**
   - Use winning configuration for downstream tasks

---

## 📚 Related Documentation

In `configs/phase1/experiment/`:
- `EXPERIMENT_GUIDE.md` - How experiments work in this project
- `COMPARE_GLOBALS_HEADS_README.md` - In-depth analysis guide
- `QUICK_REFERENCE.md` - Quick commands and patterns

Related code:
- `src/thesis_ml/phase1/train/ae_loop.py` - Training loop (loss aggregation)
- `src/thesis_ml/phase1/autoenc/decoders/mlp.py` - Globals head implementation
- `src/thesis_ml/phase1/autoenc/base.py` - AE forward pass

---

## 🛠️ Modifying the Experiment

To customize the experiment, edit `configs/phase1/experiment/compare_globals_heads.yaml`:

```yaml
sweeper:
  params:
    phase1/latent_space: none,linear,vq        # Change latent spaces
    phase1.decoder.globals_beta: 0,1,10,100    # Add more beta values
    phase1.trainer.epochs: 20                   # Change epoch count
```

Examples:
- Test 5 beta values: `0,0.5,1,5,10`
- Only compare none vs linear: `none,linear`
- Longer training: `phase1.trainer.epochs: 50`

---

## ✅ Checklist Before Running

- [ ] Data file exists: `${env.data_root}/4tops_splitted.h5`
- [ ] GPU available (or switch to `env=local` with patience)
- [ ] HPC job scheduler configured (if using `env=stoomboot`)
- [ ] Output directory writable: `${env.output_root}`
- [ ] All config files present (checked by validation)

---

## 📊 Example Results Table

| Run | Latent | Beta | rec_tokens | rec_globals | total_loss | Best For |
|-----|--------|------|-----------|------------|-----------|----------|
| 0 | none | 0 | 0.098 | N/A | 0.098 | Pure tokens |
| 1 | linear | 0 | 0.101 | N/A | 0.101 | Pure tokens |
| 2 | vq | 0 | 0.115 | N/A | 0.115 | - |
| 3 | none | 1 | 0.099 | 0.048 | 0.147 | Balanced |
| 4 | linear | 1 | 0.102 | 0.045 | 0.147 | ⭐ BEST |
| 5 | vq | 1 | 0.116 | 0.052 | 0.168 | Discrete |
| 6 | none | 10 | 0.105 | 0.008 | 0.185 | Globals |
| 7 | linear | 10 | 0.108 | 0.007 | 0.188 | Globals |
| 8 | vq | 10 | 0.122 | 0.010 | 0.212 | - |

*Example only - actual values will vary*

---

**Created**: October 31, 2025
**Experiment Config**: `configs/phase1/experiment/compare_globals_heads.yaml`
**Total Documentation**: 5 files (3 guides + config + this summary)
