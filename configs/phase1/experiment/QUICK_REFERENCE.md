# Quick Reference: Compare Globals Heads Experiment

## 🎯 What This Experiment Does

Tests how **global head importance** (MET/METphi reconstruction) affects model performance across different **latent space architectures**.

- **9 total runs** (3 latent spaces × 3 beta values)
- **20 epochs each**
- **~30-60 minutes per run** depending on hardware

---

## 📊 The Grid

```
              none              linear              vq
              ─────────────────────────────────────────
β=0  (OFF)     0                  1                2
β=1  (STD)     3                  4                5
β=10 (HI)      6                  7                8
```

---

## 🏃 Running the Experiment

### Quick Start (HPC)
```bash
python -m thesis_ml.phase1 --config-name config phase1/experiment=compare_globals_heads
```

### Quick Start (Local)
```bash
python -m thesis_ml.phase1 --config-name config phase1/experiment=compare_globals_heads env=local
```

### Custom Epochs (if needed)
```bash
python -m thesis_ml.phase1 --config-name config \
  phase1/experiment=compare_globals_heads \
  phase1.trainer.epochs=50
```

---

## 📁 Output Location

```
outputs/experiments/exp_TIMESTAMP_compare_globals_heads/
├── 0/ to 8/          ← 9 subdirectories, one per run
│   ├── cfg.yaml      ← Exact config used
│   ├── model.pt      ← Best model
│   ├── run.log       ← Training output (contains final losses)
│   ├── facts/        ← Per-epoch metrics (CSV)
│   └── figures/      ← Plots (loss curves, etc.)
```

---

## 📈 Key Metrics to Compare

### Run 0 vs Run 3 vs Run 6 (Column: `none`)
- Compare `globals_beta` effect on **same latent space**
- Best for understanding beta impact

### Run 0 vs Run 1 vs Run 2 (Row: `beta=0`)
- Compare **latent space types** without globals
- Shows pure reconstruction trade-offs

### Run 3 vs Run 4 vs Run 5 (Row: `beta=1`)
- Compare latent spaces **with balanced globals**
- Most realistic comparison for production use

### Read from `run.log`:
```
# Look for lines like:
[20/20] val_loss=0.123 rec_tokens=0.100 rec_globals=0.023
```

---

## ✅ Expected Outcomes

| Metric | At Beta=0 | At Beta=1 | At Beta=10 |
|--------|-----------|-----------|-----------|
| **rec_tokens** | ✓ Best | ≈ Slightly worse | ✗ Worst |
| **rec_globals** | ✗ Worst | ✓ Good | ✓ Best |
| **total_loss** | ✓ Best | ~ Middle | ✗ Worst |

**✓ = Good, ~ = Medium, ✗ = Bad**

---

## 💡 What Results Mean

### If `rec_globals` drops at Beta=10:
→ **Good!** Model is learning to reconstruct globals better

### If `rec_tokens` rises at Beta=10:
→ **Trade-off detected.** Latent space too small?

### If all losses similar across latent spaces:
→ **Latent space choice doesn't matter** for this task

### If VQ clearly wins:
→ **Discrete representation useful** for your data

---

## 🔍 Quick Analysis Commands

### Extract metrics for all runs:
```bash
for i in {0..8}; do
  echo "=== Run $i ==="
  tail -5 outputs/experiments/exp_*compare_globals_heads/$i/run.log | grep "val_loss"
done
```

### Find best run (by validation loss):
```bash
grep -r "val_loss" outputs/experiments/exp_*compare_globals_heads/*/run.log | sort -t= -k3 -n | head -1
```

### View full config of run N:
```bash
cat outputs/experiments/exp_*compare_globals_heads/0/cfg.yaml
```

---

## 🎓 Interpreting Patterns

### Pattern 1: Beta has no effect
```
rec_globals across beta values: similar
→ Encoder not learning globals information
```

### Pattern 2: Beta significantly affects globals (EXPECTED)
```
rec_globals values: beta=0 >> beta=1 >> beta=10
→ Model correctly optimizing globals
```

### Pattern 3: Token-Global trade-off
```
rec_tokens:  beta=0 << beta=10
rec_globals: beta=0 >> beta=10
→ Latent space might be too small
```

### Pattern 4: VQ underperforms
```
All metrics worse for VQ
→ Discrete bottleneck too restrictive
```

---

## 🎯 Decision Matrix

**Choose based on your downstream task:**

| Your Priority | Best Choice | From Grid |
|---------------|-------------|-----------|
| Pure reconstruction | none + β=0 | Run 0 |
| Balanced tokens+globals | linear + β=1 | Run 4 |
| Emphasize globals | linear + β=10 | Run 7 |
| Interpretable (discrete) | vq + β=1 | Run 5 |
| Production baseline | linear + β=1 | Run 4 |

---

## ⚠️ Troubleshooting

### Runs taking too long?
- Use `env=local` instead of HPC
- Reduce `phase1.trainer.epochs` (e.g., 10 instead of 20)

### Models not saving?
- Check `outputs/` directory permissions
- Ensure `logging.save_artifacts: true` in config

### Loss values look weird?
- Check run.log for errors
- Ensure data file exists at configured path
- Verify `globals_beta` was actually changed (check cfg.yaml)

### All runs have same results?
- Sweeper might not be working
- Run single configuration to verify: `phase1/experiment=null`

---

## 📚 Related Files

- **Experiment config**: `configs/phase1/experiment/compare_globals_heads.yaml`
- **Training code**: `src/thesis_ml/phase1/train/ae_loop.py`
- **Decoder (globals head)**: `src/thesis_ml/phase1/autoenc/decoders/mlp.py`
- **Main config**: `configs/config.yaml`
- **Data config**: `configs/data/h5_tokens.yaml`

---

## 🚀 Next Steps After Analysis

1. **Choose best config** from grid based on metrics
2. **Update default config** if needed
3. **Run Phase 2 experiments** with winning configuration
4. **Document findings** in thesis

---

## 📞 Config Parameters (if you want to tweak)

To modify experiment, edit `configs/phase1/experiment/compare_globals_heads.yaml`:

```yaml
phase1/latent_space: none,linear,vq          # Latent space types
phase1.decoder.globals_beta: 0,1,10          # Beta values to test
phase1.trainer.epochs: 20                     # Epochs per run
```

Change any of these to explore different configurations!
