# PhD Experiment: Binning vs Direct Embedding for 4t Classification

## Summary of Requirements

| Factor | Options |
|--------|---------|
| Pooling | CLS, mean, **max** |
| Tokenization | **Direct** (identity/raw), **Binned** (Ambre-style 5 bins), **VQ-VAE** |
| MET | With vs without MET + MET phi |
| Vector type | 4-vect (Pt, eta, phi, PID) vs 5-vect (+ E) |

**Fixed settings**: ~400k params, sinusoidal PE, normformer, no early stopping, 50 epochs.

**Model count**: 3 pooling × 3 tokenization × 2 MET × 2 vect = **36 models** (without VQ-VAE: 3×2×2×2 = 24). With VQ-VAE as an optional extension: 24 + 12 (binned variants) + VQ-VAE runs.

---

## Dataset Paths (HPC)

| Purpose | Path |
|---------|------|
| Raw data (4tops) | `/data/atlas/users/nterlind/datasets/4tops_splitted.h5` |
| Output path (our binned dataset) | `/data/atlas/users/nterlind/datasets/4tops_5bins_ours.h5` |
| Ambre's pre-binned 5bins | `/data/atlas/users/avisive/tokens/binning/4tops/4top_5bins_binningOnBckgdEvents_train_AND_test.h5` |

---

## Implementation Plan

### Phase 1: Core Data & Binning

#### 1.1 Ambre-Style Binning Module

Create `src/thesis_ml/data/binning.py`:

- **Bin edge computation** from training data:
  - `pT`, `η`, `||E_T^miss||`: quantile-based (e.g. 20% each for 5 bins)
  - `φ`, `φ_E_T^miss`: fixed width `π/5`
- **Token formula** (5 bins, 7 IDs):
  - `token_part = 125*(bin_obj−1) + 25*(bin_pT−1) + 5*(bin_η−1) + bin_φ` → [1, 875]
  - MET: 5 bins → [876, 880]; MET phi: 5 bins → [881, 885]
  - Padding: 0
- **API**: `AmbreBinning(n_bins=5, n_ids=7)` with `fit(train_cont, train_ids, train_globals)` and `transform(cont, ids, globals) -> integer_tokens`

#### 1.2 Create Binned Dataset Script (Run Interactively via SSH)

Create `scripts/create_binned_dataset.py`:

- **Input**: Raw H5 at `data_root/4tops_splitted.h5` (format: [18 ids, 2 globals, 18*4 cont], labels)
- **Output**: Binned H5 at `data_root/4tops_5bins_ours.h5` with splits:
  - `X_train`, `X_val`, `X_test` (shape [N, 20] integer tokens)
  - `Y_train`, `y_val`, `y_test` (labels)
- **Usage**: Run interactively on HPC login node:
  ```bash
  cd /project/atlas/users/nterlind/Thesis-Code
  conda activate thesis-ml
  python scripts/create_binned_dataset.py \
    --input /data/atlas/users/nterlind/datasets/4tops_splitted.h5 \
    --output /data/atlas/users/nterlind/datasets/4tops_5bins_ours.h5 \
    --n-bins 5
  ```

#### 1.3 Compare Our Binning vs Ambre's Dataset

Create `scripts/compare_binned_datasets.py`:

- **Load**: Our binned H5 + Ambre's H5 at `/data/atlas/users/avisive/tokens/binning/4tops/4top_5bins_binningOnBckgdEvents_train_AND_test.h5`
- **Inspect**: Ambre's file may use different splits (train_AND_test suggests a single split). Adapt to read her format.
- **Compare**: Token overlap, distribution, agreement statistics.
- **Usage**: Run interactively via SSH:
  ```bash
  python scripts/compare_binned_datasets.py \
    --ours /data/atlas/users/nterlind/datasets/4tops_5bins_ours.h5 \
    --ambres /data/atlas/users/avisive/tokens/binning/4tops/4top_5bins_binningOnBckgdEvents_train_AND_test.h5
  ```

---

### Phase 2: Model Changes

#### 2.1 Max Pooling

**File**: `src/thesis_ml/architectures/transformer_classifier/modules/head.py`

- Add `max` pooling: masked max over sequence (same as mean but with mask).

#### 2.2 4-vect vs 5-vect

- Add `data.cont_features` config: `[0,1,2,3]` for 5-vect, `[1,2,3]` for 4-vect.
- Modify `ClassificationSplitDataset` and binning to slice `cont` by `cont_features`.

#### 2.3 Binned Data Config

- Add `configs/data/h5_tokens_binned.yaml`:
  - `path: "${env.data_root}/4tops_5bins_ours.h5"`
  - `use_binned_tokens: true`
- Add alt config for Ambre's H5 (if format differs):
  - `configs/data/h5_tokens_binned_ambres.yaml` pointing to her path (with env override for `data_root` or explicit path).

---

### Phase 3: VQ-VAE Tokenization

#### 3.1 Pre-train VQ-VAE on Particle Data

- Extend `phase1` autoencoder config for particle sequences:
  - Encoder: MLP over [B, T, 4+id] per token → latent.
  - Bottleneck: VQ (codebook size ~256–512).
  - Decoder: latent → reconstructed [B, T, 4+id].
- Train on raw 4tops data.

#### 3.2 Implement PretrainedTokenizer

- **File**: `src/thesis_ml/architectures/transformer_classifier/modules/tokenizers/pretrained.py`
- Load checkpoint, reconstruct model, extract encoder + VQ bottleneck.
- Forward: `(tokens_cont, tokens_id, globals?)` → quantized indices → embed via lookup.
- Integrate with classifier config.

---

### Phase 4: Experiment Config

- **File**: `configs/classifier/experiment/phd_presentation/exp_binning_vs_direct.yaml`

Sweep structure (simplified):

- **Direct**: 3 pooling × 2 MET × 2 vect = 12 jobs
- **Binned (ours)**: 3 pooling × 2 MET × 2 vect = 12 jobs
- **Binned (Ambre's)**: 3 pooling × 2 MET × 2 vect = 12 jobs (optional)
- **VQ-VAE**: 3 pooling × 2 MET × 2 vect = 12 jobs (optional)

Total: 24–48 jobs depending on which tokenization variants are included.

---

### Phase 5: HPC Run Instructions

#### 5.1 Interactive Preprocessing (SSH)

```bash
# SSH to HPC
ssh nterlind@stbc-i1.nikhef.nl   # or your login node

# Activate conda
cd /project/atlas/users/nterlind/Thesis-Code
eval "$(conda shell.bash hook)"
conda activate /data/atlas/users/nterlind/venvs/thesis-ml

# Create our binned dataset
python scripts/create_binned_dataset.py \
  --input /data/atlas/users/nterlind/datasets/4tops_splitted.h5 \
  --output /data/atlas/users/nterlind/datasets/4tops_5bins_ours.h5 \
  --n-bins 5

# Compare with Ambre's
python scripts/compare_binned_datasets.py \
  --ours /data/atlas/users/nterlind/datasets/4tops_5bins_ours.h5 \
  --ambres /data/atlas/users/avisive/tokens/binning/4tops/4top_5bins_binningOnBckgdEvents_train_AND_test.h5
```

#### 5.2 Submit Training Jobs (Condor)

```bash
# 1. Edit train.sub to pass experiment arguments
#    Or use condor_submit with -append

# 2. Submit multirun sweep (example for exp_binning_vs_direct)
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=phd_presentation/exp_binning_vs_direct --multirun'

# 3. Monitor
condor_q
```

**Note**: `train.sub` currently has `queue 1` and passes no args. The `arguments` in `-append` overrides the `arguments` field in the job file. For multirun, the command becomes `python -m thesis_ml.cli.train env=stoomboot ... --multirun`.

#### 5.3 Submit Report (after training completes)

```bash
# Pattern: match the multirun directory created by the sweep
condor_submit hpc/stoomboot/report.sub -append 'arguments = --config-name phd_summary_binning_vs_direct inputs.sweep_dir=/data/atlas/users/nterlind/outputs/multiruns/exp_*_exp_binning_vs_direct inference.enabled=true'

# For WandB: use same env as training
export WANDB_DIR=/data/atlas/users/nterlind/outputs/wandb
```

---

## Files to Create or Modify

| File | Action |
|------|--------|
| `src/thesis_ml/data/binning.py` | **Create**: Ambre-style binning logic |
| `scripts/create_binned_dataset.py` | **Create**: Raw → binned H5 writer |
| `scripts/compare_binned_datasets.py` | **Create**: Compare ours vs Ambre's |
| `src/thesis_ml/data/h5_loader.py` | **Modify**: Integrate binning, cont_features |
| `src/thesis_ml/architectures/.../head.py` | **Modify**: Add max pooling |
| `src/thesis_ml/architectures/.../tokenizers/pretrained.py` | **Modify**: Implement VQ-VAE loading |
| `configs/data/h5_tokens_binned.yaml` | **Create** |
| `configs/classifier/experiment/.../exp_binning_vs_direct.yaml` | **Create** |
| `configs/report/phd_summary_binning_vs_direct.yaml` | **Create** |
| `src/thesis_ml/reports/analyses/phd_summary_binning_vs_direct.py` | **Create** |
| `HPC_REPORT_COMMANDS.md` | **Modify**: Add exp_binning_vs_direct commands |

---

## Ambre's Binning Formula (5 bins, 7 IDs)

- **Particle**: `token_part = 125*(bin_obj−1) + 25*(bin_pT−1) + 5*(bin_η−1) + bin_φ` ∈ [1, 875]
- **MET**: 5 bins → [876, 880]
- **MET phi**: 5 bins → [881, 885]
- **Vocab size**: 886 (0 = padding)

---

## Implementation Order

1. **Binning module** + **create_binned_dataset.py** → run interactively → verify output
2. **compare_binned_datasets.py** → run interactively → verify vs Ambre
3. **Max pooling** + **4/5-vect** + **data configs**
4. **Experiment config** → submit training on HPC
5. **VQ-VAE** (pre-train + PretrainedTokenizer) → optional extension
6. **Report config** + analysis script → submit report after training
