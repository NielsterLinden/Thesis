# Quick Commands: VQ-VAE Training for exp_binning_vs_direct

## 1. Train VQ-VAE for 4-vector input (px, py, pz)

```bash
condor_submit hpc/stoomboot/train.sub -append "arguments = env=stoomboot loop=ae phase1/latent_space=vq data=h5_tokens +data.cont_features='[1,2,3]' general.trainer.epochs=50 logging.wandb.mode=online hydra.job.name=vq-pretrain-4vec hydra.run.dir=\${env.output_root}/runs/run_\${now:%Y%m%d-%H%M%S}_vq-pretrain-4vec"
```

## 2. After job completes, find the checkpoint path

```bash
# On HPC login node
ls -la /data/atlas/users/nterlind/outputs/runs/ | grep vq-pretrain-4vec
# Copy the run_YYYYMMDD-HHMMSS_vq-pretrain-4vec directory name
```

## 3. Update configs/env/stoomboot.yaml

Replace the placeholder path:
```yaml
vq_checkpoint_path_4vec: "/data/atlas/users/nterlind/outputs/runs/run_YYYYMMDD-HHMMSS_vq-pretrain-4vec/best_val.pt"
```

## 4. Commit and push the config update

```bash
git add configs/env/stoomboot.yaml
git commit -m "Update VQ 4vec checkpoint path"
git push
```

## 5. Run the full experiment (36 models)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=phd_presentation/exp_binning_vs_direct logging=wandb_online --multirun'
```

---

**Note**: The 5-vector VQ-VAE is already trained and configured. Only the 4-vector one needs training.
