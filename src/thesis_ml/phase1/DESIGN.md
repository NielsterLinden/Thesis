# Phase 1 Autoencoders Design

This document describes the unified assembly model and training loop interfaces for Phase 1 Standalone Autoencoders.

## Components
- Encoder: maps `(tokens_cont, tokens_id, globals_vec)` -> `z_e`.
- Bottleneck: maps `z_e` -> either `z_e` (identity) or `{z_q, indices, aux}` (VQ).
- Decoder: maps `z` -> `x_hat` with optional `rec_globals` auxiliary.

## BaseAutoencoder
- Assembles `encoder`, `bottleneck`, `decoder` from Hydra targets.
- `forward` returns:
  - `x_hat`, `z_e`, optional `z_q`, `indices`, `rec_globals`, and `aux` losses.

## Training Loops
- `phase1.train.ae_loop`: reconstruction objective; emits lifecycle events with the standardized payload consumed by `thesis_ml.plots.orchestrator`.
- `phase1.train.gan_ae_loop` and `phase1.train.diffusion_ae_loop`: skeletons for Phase 2.

## Plotting
- Orchestrator unchanged. Families extended with `adversarial` and `diffusion` stubs.

## Hydra
- Config groups under `configs/phase1` allow composition of encoder, decoder, tokenizer, trainer, and data.
- Top-level `configs/config.yaml` composes Phase 1 by default.
