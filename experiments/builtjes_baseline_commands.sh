#!/usr/bin/env bash
# =============================================================================
# Builtjes et al. (2025) Baseline Reproduction — Condor Submit Commands
#
# Experiment: configs/classifier/experiment/thesis_plan/builtjes_baseline.yaml
# Jobs: 3 variants (ParT / ParT_int / ParT_int.SM) × 5 seeds = 15 total
#
# Run from project root: c:/Users/niels/Projects/Thesis-Code/Code/Niels_repo
# =============================================================================

# -----------------------------------------------------------------------------
# TEST RUN — 1 epoch, 500 samples, W&B off (~1 min per job, 15 jobs total)
# Run this first to verify the config doesn't crash before submitting overnight.
# -----------------------------------------------------------------------------

condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/builtjes_baseline experiment.name=builtjes_baseline_test classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'

# -----------------------------------------------------------------------------
# FULL RUN — 50 epochs, full dataset (~240k), W&B on (overnight, 15 jobs)
# Submit after the test run succeeds.
# -----------------------------------------------------------------------------

condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/builtjes_baseline --multirun'

# =============================================================================
# Expected outputs
# =============================================================================
# Multirun dir: outputs/multiruns/exp_TIMESTAMP_builtjes_baseline/
# Per-run dirs: outputs/runs/run_TIMESTAMP_builtjes_baseline_jobNN/
#
# W&B group:    exp_TIMESTAMP_builtjes_baseline
# W&B tags:     level:sim_event, goal:classification, family:transformer
#
# Target AUC (Builtjes Table 7, 240k training, 4t vs background):
#   ParT        (none)                        ~0.84
#   ParT_int    (lorentz_scalar)              ~0.86
#   ParT_int.SM (lorentz_scalar+sm_int.)      ~0.87
#
# Report after completion:
#   thesis-report --config-name evaluate_classifier \
#     inputs.sweep_dir=outputs/multiruns/exp_*_builtjes_baseline \
#     inference.enabled=true
#
# =============================================================================
# Training stability grid — ParT_int only, 3×3×3 = 27 jobs (warmup × batch × lr)
# Config: configs/classifier/experiment/thesis_plan/builtjes_training_stability.yaml
# =============================================================================
#
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/builtjes_training_stability --multirun'
#
# =============================================================================
