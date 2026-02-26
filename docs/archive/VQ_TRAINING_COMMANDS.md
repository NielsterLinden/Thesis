# VQ-VAE Training Commands for Dual Input Dimensions

## Background

The `exp_binning_vs_direct` experiment tests two input configurations:
- **5-vector**: `[0,1,2,3]` = E, px, py, pz (4 continuous features + 1 particle ID)
- **4-vector**: `[1,2,3]` = px, py, pz (3 continuous features + 1 particle ID)

To use VQ-VAE tokenization with both configurations, we need **two separate VQ-VAE checkpoints**, each trained on the matching input dimension.

## Step 1: Train VQ-VAE for 5-vector (E, px, py, pz)

This is already trained: `/data/atlas/users/nterlind/outputs/runs/run_20260211-083621_vq-pretrain-full/best_val.pt`

To retrain if needed:

```bash
condor_submit hpc/stoomboot/train.sub -append "arguments = env=stoomboot loop=ae phase1/latent_space=vq data=h5_tokens +data.cont_features='[0,1,2,3]' general.trainer.epochs=50 logging.wandb.mode=online hydra.job.name=vq-pretrain-5vec hydra.run.dir=\${env.output_root}/runs/run_\${now:%Y%m%d-%H%M%S}_vq-pretrain-5vec"
```

**Expected checkpoint location**: `/data/atlas/users/nterlind/outputs/runs/run_YYYYMMDD-HHMMSS_vq-pretrain-5vec/best_val.pt`

## Step 2: Train VQ-VAE for 4-vector (px, py, pz)

**NEW** - Submit this job:

```bash
condor_submit hpc/stoomboot/train.sub -append "arguments = env=stoomboot loop=ae phase1/latent_space=vq data=h5_tokens +data.cont_features='[1,2,3]' general.trainer.epochs=50 logging.wandb.mode=online hydra.job.name=vq-pretrain-4vec hydra.run.dir=\${env.output_root}/runs/run_\${now:%Y%m%d-%H%M%S}_vq-pretrain-4vec"
```

**Expected checkpoint location**: `/data/atlas/users/nterlind/outputs/runs/run_YYYYMMDD-HHMMSS_vq-pretrain-4vec/best_val.pt`

## Step 3: Update env config with checkpoint paths

After both trainings complete, update [`configs/env/stoomboot.yaml`](configs/env/stoomboot.yaml):

```yaml
# VQ-VAE checkpoints for pretrained tokenizer
vq_checkpoint_path_5vec: "/data/atlas/users/nterlind/outputs/runs/run_YYYYMMDD-HHMMSS_vq-pretrain-5vec/best_val.pt"
vq_checkpoint_path_4vec: "/data/atlas/users/nterlind/outputs/runs/run_YYYYMMDD-HHMMSS_vq-pretrain-4vec/best_val.pt"
```

Replace `YYYYMMDD-HHMMSS` with the actual run timestamps.

## Step 4: Run the full experiment

Once both VQ-VAE checkpoints exist and the config is updated:

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=phd_presentation/exp_binning_vs_direct logging=wandb_online --multirun'
```

This will now successfully run all 36 models, including the 12 VQ-tokenization variants (3 pooling × 2 MET × 2 vector types).

## How It Works

The code now automatically selects the correct VQ-VAE checkpoint based on the input dimension:

1. **[`configs/tokenization/vq.yaml`](configs/tokenization/vq.yaml)** passes both checkpoint paths:
   ```yaml
   checkpoint_path_5vec: ${env.vq_checkpoint_path_5vec}
   checkpoint_path_4vec: ${env.vq_checkpoint_path_4vec}
   ```

2. **[`tokenizers.py:get_tokenizer()`](src/thesis_ml/architectures/transformer_classifier/modules/tokenizers/tokenizers.py)** inspects `cont_dim` from meta:
   - `cont_dim == 5` (4 cont + 1 ID) → use 5vec checkpoint
   - `cont_dim == 4` (3 cont + 1 ID) → use 4vec checkpoint

3. **[`pretrained.py:PretrainedTokenizer`](src/thesis_ml/architectures/transformer_classifier/modules/tokenizers/pretrained.py)** loads the selected checkpoint with improved error messages for shape mismatches.

## Verification

To test a single VQ run before submitting the full sweep:

```bash
# Test 5-vector
python -m thesis_ml.cli.train env=stoomboot loop=transformer_classifier \
  +tokenization=vq +data.cont_features="[0,1,2,3]" \
  classifier.model.pooling=cls general.trainer.epochs=1 logging.wandb.mode=offline

# Test 4-vector (after training 4vec VQ-VAE)
python -m thesis_ml.cli.train env=stoomboot loop=transformer_classifier \
  +tokenization=vq +data.cont_features="[1,2,3]" \
  classifier.model.pooling=cls general.trainer.epochs=1 logging.wandb.mode=offline
```

Both should complete without shape mismatch errors.

## Timeline

1. Submit 4vec VQ-VAE training (~1-2 hours on GPU)
2. Wait for job completion
3. Update `stoomboot.yaml` with checkpoint path
4. Submit full `exp_binning_vs_direct` sweep
5. All 36 models should complete successfully

---

**Date Created**: 2026-02-11
**Issue**: Shape mismatch in VQ tokenization with 4-vector input
**Solution**: Dual VQ-VAE checkpoints + automatic selection
