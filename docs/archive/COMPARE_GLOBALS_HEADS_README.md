# Global Head Influence Experiment (`compare_globals_heads.yaml`)

## Experiment Design

This experiment systematically evaluates how the global head reconstruction loss (`globals_beta`) affects model performance across different latent space architectures.

### Grid Structure: 3×3 = 9 Runs

```
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│                 │    NONE      │    LINEAR    │      VQ      │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ globals_β=0     │   Run 0      │   Run 1      │   Run 2      │
│ (OFF)           │  Baseline    │  Baseline    │  Baseline    │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ globals_β=1     │   Run 3      │   Run 4      │   Run 5      │
│ (Standard)      │  Standard    │  Standard    │  Standard    │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ globals_β=10    │   Run 6      │   Run 7      │   Run 8      │
│ (Emphasized)    │  Emphasized  │  Emphasized  │  Emphasized  │
└─────────────────┴──────────────┴──────────────┴──────────────┘
```

---

## Parameters Explained

### Latent Space Dimension (Columns)

**`latent_space: none`**
- Identity bottleneck: `z_e = z`
- No learnable transformation
- Highest model capacity; closest to uncompressed
- Risk: overfitting, no learned compression
- Baseline for measuring the effect of constraints

**`latent_space: linear`**
- Learned linear projection: `z = Linear(z_e)`
- Learnable but continuous
- Moderate compression; adds expressiveness
- Good balance between constraint and capacity

**`latent_space: vq`**
- Vector Quantization bottleneck
- Discrete codebook with learned codes
- Strongest constraint; promotes interpretability
- Includes additional auxiliary losses (commit, codebook)

### Global Head Weight (Rows)

**`globals_beta: 0` → Globals Head Disabled**
```
Total Loss = rec_tokens + VQ_losses
```
- MET and METphi are **completely ignored**
- Model learns to compress tokens independently
- Global information discarded
- Baseline for ablation study

**`globals_beta: 1` → Standard Weighting (Default)**
```
Total Loss = rec_tokens + rec_globals + VQ_losses
```
- Balanced training on tokens and globals
- Global information encoded in latent space
- Reconstructs MET/METphi with equal importance to tokens
- Most "natural" configuration

**`globals_beta: 10` → Emphasized Globals**
```
Total Loss = rec_tokens + 10*rec_globals + VQ_losses
```
- Global reconstruction prioritized 10× over default
- Model focuses on accurately reconstructing MET/METphi
- May sacrifice token reconstruction accuracy
- Explores trade-off between global and token accuracy

---

## Expected Behavior & Metrics

### What to Monitor

1. **`rec_tokens`** (Token Reconstruction Loss)
   - Main objective; should be similar across `globals_beta` values
   - Might increase slightly as `globals_beta` increases (trade-off)
   - Will be higher for VQ (harder constraint)

2. **`rec_globals`** (Global Reconstruction Loss)
   - **At `globals_beta=0`**: Should be 0 or very small (not computed)
   - **At `globals_beta=1`**: Moderate value; balanced learning
   - **At `globals_beta=10`**: Lower value (more optimized for globals)
   - **Trend**: Should decrease as `globals_beta` increases (more weight = better optimization)

3. **`total_loss`** (Sum of All Losses)
   - Directly affected by `globals_beta` scaling
   - Use only to monitor relative convergence, not direct comparison
   - Weight by beta value for fair comparison

4. **Latent Space Metrics**
   - **For VQ**: Perplexity, codebook usage, commitment loss
   - **For Linear**: Just reconstruction (no auxiliary losses)
   - **For None**: Pure reconstruction baseline

### Predicted Results

#### Token Reconstruction (`rec_tokens`)
```
Latent Space Effect:
  none   < linear ≈ linear < vq
  ↑         ↑        ↑        ↑
least    more    constrained hardest
 loss    loss

Global Beta Effect:
  Similar across all beta values (minor trade-offs)
```

#### Global Reconstruction (`rec_globals`)
```
Global Beta Effect (STRONG):
  globals_beta=0   >> globals_beta=1 >> globals_beta=10
  (not optimized)  (moderate)        (well-optimized)

Latent Space Effect (Weak):
  Should be similar across latent spaces
  VQ might have slight advantage (discrete helps globals?)
```

#### Total Loss
```
Expected Ranking (lowest to highest):
  1. none + beta=0          (no globals constraint)
  2. linear + beta=0
  3. none + beta=1
  4. linear + beta=1
  5. none + beta=10
  6. linear + beta=10
  7. vq + beta=0            (VQ constraint)
  8. vq + beta=1
  9. vq + beta=10           (hardest problem)
```

---

## Interpretation Guide

### Scenario 1: Globals Beta Has Little Effect
**Observation**: `rec_globals` similar across all beta values

**Interpretation**:
- Global information is not encoded in latent space
- Encoder is ignoring `globals_vec`
- Model already saturating at good global accuracy
- Investigation: Check encoder design; is it really processing globals?

### Scenario 2: Strong Beta Effect on Globals, Minimal Effect on Tokens
**Observation**:
- `rec_tokens` nearly identical across beta values
- `rec_globals` improves significantly as beta increases

**Interpretation** ✓ EXPECTED:
- Latent space has capacity for both tokens and globals
- Trade-off is minimal; can optimize both simultaneously
- Globals head successfully learns to reconstruct MET/METphi

### Scenario 3: Strong Beta Effect, Notable Token-Globals Trade-off
**Observation**:
- `rec_tokens` increases notably as beta increases
- `rec_globals` decreases as beta increases
- Total loss worse at higher beta

**Interpretation**:
- Latent bottleneck is too small
- Model must choose: optimize tokens OR globals (zero-sum game)
- Recommendation: Increase latent dimension or disable globals head

### Scenario 4: VQ Significantly Worse Than Linear/None
**Observation**:
- All losses (tokens, globals) higher for VQ

**Interpretation**:
- Discrete bottleneck too restrictive for this task
- Vector quantization loses important information
- Recommendation: Use linear or none; VQ better suited for other tasks

---

## Running the Experiment

### Command

**On HPC (default)**:
```bash
cd /path/to/Niels_repo
python -m thesis_ml.phase1 --config-name config phase1/experiment=compare_globals_heads
```

**Locally**:
```bash
python -m thesis_ml.phase1 --config-name config phase1/experiment=compare_globals_heads env=local
```

### Output Structure

```
outputs/experiments/exp_20251031-120000_compare_globals_heads/
├── 0/                          # (none, beta=0)
│   ├── cfg.yaml               # Config for this run
│   ├── model.pt               # Best model
│   ├── run.log                # Training output
│   ├── facts/
│   │   └── *.csv             # Metrics per epoch
│   └── figures/
│       ├── losses.png         # Loss curves
│       └── recon.png          # Reconstruction metrics
├── 1/                          # (linear, beta=0)
│   └── ...
├── ...
└── 8/                          # (vq, beta=10)
    └── ...
```

---

## Analysis After Running

### Quick Analysis (Manual)
1. Open each run's `run.log` and check final losses
2. Compare `rec_globals` values across runs
3. Look for trade-off patterns in loss curves

### Automated Analysis
```bash
# Generate summary report across all runs
python -m thesis_ml.reports.compare_experiments \
  outputs/experiments/exp_20251031-120000_compare_globals_heads/
```

This produces:
- CSV table: best_val_loss, rec_tokens, rec_globals per run
- Visualizations: heatmap of metrics across grid
- Recommendations: best config based on criteria

### Excel/Spreadsheet Analysis
Export run metrics and create a pivot table:

| Run | Latent Space | Beta | rec_tokens | rec_globals | total_loss | notes |
|-----|--------------|------|-----------|------------|-----------|-------|
| 0   | none         | 0    |           |            |           |       |
| 1   | linear       | 0    |           |            |           |       |
| ... | ...          | ...  | ...       | ...        | ...       | ...   |

---

## Key Hypotheses to Test

1. **Does the encoder actually encode global information?**
   - Answer: If rec_globals=0 when beta=0, yes; if high, no

2. **Is the latent bottleneck sufficient for both tokens and globals?**
   - Answer: If rec_tokens stays constant as beta varies, yes

3. **Is there a sweet spot for globals_beta?**
   - Answer: Compare metrics at beta=0,1,10; pick best generalization

4. **Does latent space type affect global reconstruction?**
   - Answer: If rec_globals similar across columns, latent type doesn't matter

5. **Should we use globals head at all?**
   - Answer: Compare final model performance: beta=0 vs beta=1

---

## Recommended Next Steps

After analyzing results:

### If Globals are Well-Reconstructed at Beta=1:
→ Use `globals_beta: 1` as baseline for downstream tasks

### If Strong Trade-off Detected (Tokens vs Globals):
→ Increase latent dimension: `latent_dim: 32 → 64`
→ Re-run experiment with new config

### If Globals Barely Used (rec_globals high at beta=0):
→ Investigate encoder: why not learning globals?
→ Try: larger ID embedding, different encoder architecture

### If Linear Space Clearly Best:
→ Replace `none` with `linear` in default config
→ Use for Phase 2 experiments

### If VQ Promising Despite Higher Loss:
→ Increase codebook size
→ Adjust quantization temperature
→ Worth exploring for discrete representations
