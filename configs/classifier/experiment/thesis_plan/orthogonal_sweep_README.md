# Orthogonal sweep (OrthogonalSweep_A / B / C)

Single-axis comparisons from a fixed **s200k** baseline (4t vs background, full data). Each group is a **separate** `condor_submit` so you can resubmit failed slices. **Overrides must appear before `--multirun`.**

## Totals

| Batch | Config file | Groups | Runs |
|-------|-------------|--------|------|
| A | [`orthogonal_sweep_A.yaml`](orthogonal_sweep_A.yaml) | 1–6 | 54 |
| B | [`orthogonal_sweep_B.yaml`](orthogonal_sweep_B.yaml) | 7–12, 16 | 51 |
| C | [`orthogonal_sweep_C.yaml`](orthogonal_sweep_C.yaml) | 13–15, 17, baseline | 51 |
| **All** | | | **156** |

### Batch A breakdown

| Group | Axis | Values × seeds | Total |
|-------|------|----------------|-------|
| 1 | Positional | 4 × 3 | 12 |
| 2 | Norm policy | 3 × 3 | 9 |
| 3 | Norm type | 2 × 3 | 6 |
| 4 | Attention type | 2 × 3 | 6 |
| 5 | Attention norm | 3 × 3 | 9 |
| 6 | FFN (4 commands × 3 seeds) | | 12 |

### Batch B breakdown

| Group | Axis | Values × seeds | Total |
|-------|------|----------------|-------|
| 7 | Head type (3 commands × 3 seeds) | | 9 |
| 8 | Pooling | 3 × 3 | 9 |
| 9 | Tokenizer | 2 × 3 | 6 |
| 10 | PID mode | 3 × 3 | 9 |
| 11 | MET | 2 × 3 | 6 |
| 12 | Cont features | 2 × 3 | 6 |
| 16 | Causal attention | 2 × 3 | 6 |

### Batch C breakdown

| Group | Axis | Values × seeds | Total |
|-------|------|----------------|-------|
| 13 | Physics biases (7 commands × 3 seeds) | | 21 |
| 14 | Pre-encoder (2 × 3) | | 6 |
| 15 | MoE encoder (4 × 3) | | 12 |
| 17 | Token order (3 × 3) | | 9 |
| — | Baseline anchor | 3 | 3 |

## Baseline (embedded in each YAML)

- **Model (s200k):** `dim=64`, `depth=6`, `heads=8`, `mlp_dim=128`, `hidden_sizes=[448,384]`, `dropout=0.1`
- **Tokenizer:** `identity`, `pid_mode=learned`, `id_embed_dim=8`
- **PE / norm / attention / FFN / head:** `positional=none`, `positional_space=model`, `norm.policy=post`, `norm.type=layernorm`, `attention.type=standard`, `attention.norm=none`, `diff_bias_mode=shared`, `ffn.type=standard`, `head.type=linear`, `head.pooling=cls`, `causal_attention=false`
- **Physics / MoE:** `attention_biases=none`, `nodewise_mass.enabled=false`, `mia_blocks.enabled=false`, `moe.enabled=false`
- **Globals:** `include_met=false`
- **Trainer:** `lr=1e-4`, `weight_decay=0.01`, `batch_size=64`, `epochs=50`, `warmup_steps=0`
- **Task:** `signal_vs_background` signal=1, background=[2,3,4,5]; `selected_labels=null`

Within each group, the **first** value in a comma list is the baseline setting (for apples-to-apples in W&B).

## Hydra note

Each YAML uses `hydra.sweeper.params: {}`. **All sweep dimensions come from the CLI** (`--multirun` + comma-separated overrides). Verified locally with a 2×2 grid.

## GPU workflow

Condor does not assign “batch A → GPU 1”. Run three **independent** submit waves (A / B / C) when you have capacity; or stagger submissions manually.

---

## Batch A — test commands

Run from repo root on Nikhef (`initialdir` in `hpc/stoomboot/train.sub` must point at your checkout).

### Group 1 — positional (12 jobs)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_A experiment.name=OrthogonalSweep_A_g01_positional classifier.model.positional=none,sinusoidal,learned,rotary classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 2 — norm policy (9)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_A experiment.name=OrthogonalSweep_A_g02_norm_policy classifier.model.norm.policy=pre,post,normformer classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 3 — norm type (6)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_A experiment.name=OrthogonalSweep_A_g03_norm_type classifier.model.norm.type=layernorm,rmsnorm classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 4 — attention type (6)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_A experiment.name=OrthogonalSweep_A_g04_attention_type classifier.model.attention.type=standard,differential classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 5 — attention internal norm (9)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_A experiment.name=OrthogonalSweep_A_g05_attention_norm classifier.model.attention.norm=none,layernorm,rmsnorm classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 6 — FFN (four submits × 3 seeds each)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_A experiment.name=OrthogonalSweep_A_g06a_ffn_standard classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_A experiment.name=OrthogonalSweep_A_g06b_ffn_kan_hybrid classifier.model.ffn.type=kan classifier.model.ffn.kan.variant=hybrid classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_A experiment.name=OrthogonalSweep_A_g06c_ffn_kan_bottleneck classifier.model.ffn.type=kan classifier.model.ffn.kan.variant=bottleneck classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_A experiment.name=OrthogonalSweep_A_g06d_ffn_kan_pure classifier.model.ffn.type=kan classifier.model.ffn.kan.variant=pure classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

---

## Batch B — test commands

### Group 7 — head type (three submits; MoE needs encoder MoE on)

`head.type=moe` requires `classifier.model.moe.enabled=true` and `scope=head`. Do **not** combine with `linear`/`kan` in one multirun.

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_B experiment.name=OrthogonalSweep_B_g07a_head_linear classifier.model.head.type=linear classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_B experiment.name=OrthogonalSweep_B_g07b_head_kan classifier.model.head.type=kan classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_B experiment.name=OrthogonalSweep_B_g07c_head_moe classifier.model.head.type=moe classifier.model.moe.enabled=true classifier.model.moe.scope=head classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 8 — pooling (9)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_B experiment.name=OrthogonalSweep_B_g08_pooling classifier.model.head.pooling=cls,mean,max classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 9 — tokenizer (6)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_B experiment.name=OrthogonalSweep_B_g09_tokenizer classifier.model.tokenizer.name=identity,raw classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 10 — PID mode (9)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_B experiment.name=OrthogonalSweep_B_g10_pid_mode classifier.model.tokenizer.pid_mode=learned,one_hot,fixed_random classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 11 — MET (6)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_B experiment.name=OrthogonalSweep_B_g11_met classifier.globals.include_met=false,true classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 12 — continuous features (6)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_B experiment.name=OrthogonalSweep_B_g12_cont_features data.cont_features=[0,1,2,3],[1,2,3] classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 16 — causal attention (6)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_B experiment.name=OrthogonalSweep_B_g16_causal classifier.model.causal_attention=false,true classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

---

## Batch C — test commands

### Group 13 — physics biases (seven submits × 3 seeds)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g13a_lorentz_part classifier.model.attention_biases=lorentz_scalar classifier.model.bias_config.lorentz_scalar.features=[m2,deltaR] classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g13b_lorentz_mipart classifier.model.attention_biases=lorentz_scalar classifier.model.bias_config.lorentz_scalar.features=[log_kt,z,deltaR,log_m2] classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g13c_sm_binary classifier.model.attention_biases=sm_interaction classifier.model.bias_config.sm_interaction.mode=binary classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g13d_sm_running classifier.model.attention_biases=sm_interaction classifier.model.bias_config.sm_interaction.mode=running_coupling classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g13e_typepair_scratch classifier.model.attention_biases=typepair_kinematic classifier.model.bias_config.typepair_kinematic.init_from_physics=none classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g13f_typepair_physics classifier.model.attention_biases=typepair_kinematic classifier.model.bias_config.typepair_kinematic.init_from_physics=binary classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g13g_global_met classifier.model.attention_biases=global_conditioned classifier.globals.include_met=true classifier.model.bias_config.global_conditioned.mode=global_scale classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 14 — pre-encoder (6)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g14a_nodewise_mass classifier.model.nodewise_mass.enabled=true classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g14b_mia classifier.model.mia_blocks.enabled=true classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 15 — MoE in encoder (four submits × 3 seeds)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g15a_moe_token_all classifier.model.moe.enabled=true classifier.model.moe.top_k=1 classifier.model.moe.routing_level=token classifier.model.moe.scope=all_blocks classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g15b_moe_event_all classifier.model.moe.enabled=true classifier.model.moe.top_k=1 classifier.model.moe.routing_level=event classifier.model.moe.scope=all_blocks classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g15c_moe_token_middle classifier.model.moe.enabled=true classifier.model.moe.top_k=1 classifier.model.moe.routing_level=token classifier.model.moe.scope=middle_blocks classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g15d_moe_top2_token classifier.model.moe.enabled=true classifier.model.moe.top_k=2 classifier.model.moe.routing_level=token classifier.model.moe.scope=all_blocks classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Group 17 — token ordering (9)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g17a_order_default classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g17b_sort_pt data.sort_tokens_by=pt data.shuffle_tokens=false classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g17c_shuffle data.shuffle_tokens=true classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

### Baseline anchor — Batch C (3)

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_C experiment.name=OrthogonalSweep_C_g00_baseline classifier.trainer.seed=42,123,314 classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false --multirun'
```

---

## Full runs (50 epochs, full data, W&B default)

Replace each **test** command by removing `classifier.trainer.epochs=1 data.limit_samples=500 logging.use_wandb=false` from the `-append` string (keep `classifier.trainer.seed=42,123,314 --multirun`). Example for Group 1:

```bash
condor_submit hpc/stoomboot/train.sub -append 'arguments = env=stoomboot loop=transformer_classifier classifier/experiment=thesis_plan/orthogonal_sweep_A experiment.name=OrthogonalSweep_A_g01_positional classifier.model.positional=none,sinusoidal,learned,rotary classifier.trainer.seed=42,123,314 --multirun'
```

Apply the same transformation to **every** test line above for production sweeps.

---

## Local quick check

```bash
conda activate <env> && cd <repo>
thesis-train env=local classifier/experiment=thesis_plan/orthogonal_sweep_A classifier.model.positional=none,sinusoidal classifier.trainer.seed=42,123 data.limit_samples=5 classifier.trainer.epochs=1 logging.use_wandb=false --multirun
```

Expect **4** jobs (2×2) and `hydra.sweeper.params: {}` in the merged experiment.
