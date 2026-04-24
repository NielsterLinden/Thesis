# Deliverable 1 — Audit of W&B `thesis-ml` project for V2 backfill

Live audit of `nterlind-nikhef/thesis-ml` via `wandb.Api()`, captured by
[scripts/one_off/audit_v2_schema.py](../../scripts/one_off/audit_v2_schema.py).
Reference outputs written to `_reference/`:

| File | Contents |
|---|---|
| `_reference/summary.json` | Totals, sampled run IDs, top-30 keys by presence |
| `_reference/column_presence.csv` | Every W&B config key × (count, fraction) |
| `_reference/wandb_schema_sample.csv` | 367 rows × 5 sampled runs (full raw value dump) |
| `_reference/experiment_inventory.csv` | `axes/experiment_name` + `wandb.group` distribution |

**Project state at audit time:** 1002 runs, 367 unique W&B config keys across the project, 52 distinct `axes/experiment_name` buckets, 99 distinct `wandb.group` values.

---

## 1. V2 leaf checklist

Flat enumeration of every axis/sub-axis/leaf parsed from
[docs/AXES_REFERENCE_V2.md](../AXES_REFERENCE_V2.md) in document order.
The `Id` column is the V2 ID; `has_axes_key` is whether V2 says there is a dedicated `axes/<key>` mirror; `prereq` is the verbatim gate (empty = root). Every row must receive a derivation section in Deliverable 2.

### §1 G — Study framing (3 leaves)

| Id | Name | Hydra config | axes key | Prereq |
|---|---|---|---|---|
| G01 | Task type | `loop` | — (maps to `axes/experiment_name` via meta) | (root) |
| G02 | Model family | derived from `loop` | — | G01 |
| G03 | Classification task | `data.classifier.signal_vs_background`, `data.classifier.selected_labels` | — (uses `meta.process_groups_key`) | (root) |

### §2 D — Data treatment (3 leaves)

| Id | Name | Hydra config | axes key | Prereq |
|---|---|---|---|---|
| D01 | Feature set | `data.cont_features` | `cont_features` | (root) |
| D02 | MET treatment | `classifier.globals.include_met` | `include_met` | (root) |
| D03 | Token ordering | `data.sort_tokens_by`, `data.shuffle_tokens` (derived) | `token_order` | (root) |

### §3 T — Tokenizer (4 leaves)

| Id | Name | Hydra config | axes key | Prereq |
|---|---|---|---|---|
| T1 | Tokenizer family | `classifier.model.tokenizer.name` | `tokenizer_name` | (root) |
| T1-a | PID embedding mode | `classifier.model.tokenizer.pid_mode` | `pid_mode` | T1 = identity |
| T1-b | PID embedding dim | `classifier.model.tokenizer.id_embed_dim` | `id_embed_dim` | T1 = identity (override on T1-a = one_hot) |
| T1-c | Pretrained model type | `classifier.model.tokenizer.model_type` | — | T1 = pretrained |

### §4 E — Positional encoding (4 leaves)

| Id | Name | Hydra config | axes key | Prereq |
|---|---|---|---|---|
| E1 | PE type | `classifier.model.positional` | `positional` | (root) |
| E1-a | PE space | `classifier.model.positional_space` | `positional_space` | E1 ∈ {sinusoidal, learned} |
| E1-a1 | PE dim mask | `classifier.model.positional_dim_mask` | `positional_dim_mask` | E1-a = token |
| E1-b | Rotary base | `classifier.model.rotary.base` | — | E1 = rotary |

### §5 P — Pre-encoder modules (8 leaves)

| Id | Name | Hydra config | axes key | Prereq |
|---|---|---|---|---|
| P1 | Nodewise mass enabled | `classifier.model.nodewise_mass.enabled` | `nodewise_mass_enabled` | (T1 ∈ {raw, identity}) ∧ (0 ∈ D01) |
| P1-a | k values | `classifier.model.nodewise_mass.k_values` | — | P1 = true |
| P1-b | hidden dim | `classifier.model.nodewise_mass.hidden_dim` | — | P1 = true |
| P2 | MIA enabled | `classifier.model.mia_blocks.enabled` | `mia_enabled` | T1 ∈ {raw, identity} |
| P2-a | MIA placement | `classifier.model.mia_blocks.placement` | `mia_placement` | P2 = true |
| P2-b | MIA num blocks | `classifier.model.mia_blocks.num_blocks` | — | P2 = true |
| P2-c | interaction dim | `classifier.model.mia_blocks.interaction_dim` | — | P2 = true |
| P2-d | reduction dim | `classifier.model.mia_blocks.reduction_dim` | — | P2 = true |
| P2-e | MIA dropout | `classifier.model.mia_blocks.dropout` | — | P2 = true |

Note: §5 has 9 leaves once P2-e is counted; 8 if P2-e and P1-b are collapsed. We count P2-e separately → §5 = 9 leaves.

### §6 A — Attention & encoder block (6 leaves)

| Id | Name | Hydra config | axes key | Prereq |
|---|---|---|---|---|
| A1 | Norm policy | `classifier.model.norm.policy` | `norm_policy` | (root) |
| A2 | Norm type | `classifier.model.norm.type` | `norm_type` | (root) |
| A3 | Attention type | `classifier.model.attention.type` | `attention_type` | (root) |
| A3-a | Diff-bias mode | `classifier.model.attention.diff_bias_mode` | `diff_bias_mode` | A3 = differential |
| A4 | Attention internal norm | `classifier.model.attention.norm` | `attention_norm` | (root) |
| A5 | Causal masking | `classifier.model.causal_attention` | `causal_attention` | (root) |

### §7 F — Encoder FFN realization (4 leaves)

| Id | Name | Hydra config | axes key | Prereq |
|---|---|---|---|---|
| F1 | Effective FFN realization | `ffn.type`, `moe.enabled` | `ffn_type`, `moe_enabled` | (root) |
| F1-a | KAN FFN variant | `classifier.model.ffn.kan.variant` | `kan_ffn_variant` | F1-effective = kan |
| F1-a1 | KAN FFN bottleneck dim | `classifier.model.ffn.kan.bottleneck_dim` | — | F1-a = bottleneck |
| F1-b | MoE encoder scope | `classifier.model.moe.scope` | `moe_scope` | F1-effective = moe |

### §8 B — Physics biases (16 leaves)

| Id | Name | Hydra config | axes key | Prereq |
|---|---|---|---|---|
| B1 | Bias activation set | `classifier.model.attention_biases` | `attention_biases` | T1 ∈ {raw, identity} |
| B1-L1 | Lorentz features | `bias_config.lorentz_scalar.features` | `lorentz_features` | "lorentz_scalar" ∈ B1 |
| B1-L2 | Lorentz MLP type | `bias_config.lorentz_scalar.mlp_type` | `lorentz_mlp_type` | "lorentz_scalar" ∈ B1 |
| B1-L3 | Lorentz hidden dim | `bias_config.lorentz_scalar.hidden_dim` | — | "lorentz_scalar" ∈ B1 |
| B1-L4 | Lorentz per-head | `bias_config.lorentz_scalar.per_head` | `lorentz_per_head` | "lorentz_scalar" ∈ B1 |
| B1-L5 | Lorentz sparse gating | `bias_config.lorentz_scalar.sparse_gating` | `lorentz_sparse_gating` | "lorentz_scalar" ∈ B1 |
| B1-T1 | Type-pair init | `bias_config.typepair_kinematic.init_from_physics` | `typepair_init` | "typepair_kinematic" ∈ B1 |
| B1-T2 | Type-pair freeze | `bias_config.typepair_kinematic.freeze_table` | `typepair_freeze` | "typepair_kinematic" ∈ B1 |
| B1-T3 | Type-pair kinematic gate | `bias_config.typepair_kinematic.kinematic_gate` | `typepair_kinematic_gate` | "typepair_kinematic" ∈ B1 |
| B1-T4 | Type-pair kinematic feature | `bias_config.typepair_kinematic.kinematic_feature` | — | "typepair_kinematic" ∈ B1 |
| B1-T5 | Type-pair mask value | `bias_config.typepair_kinematic.mask_value` | — | "typepair_kinematic" ∈ B1 |
| B1-S1 | SM mode | `bias_config.sm_interaction.mode` | `sm_mode` | "sm_interaction" ∈ B1 |
| B1-S2 | SM mask value | `bias_config.sm_interaction.mask_value` | — | "sm_interaction" ∈ B1 |
| B1-G1 | Global-cond mode | `bias_config.global_conditioned.mode` | `global_conditioned_mode` | ("global_conditioned" ∈ B1) ∧ (met_direction → D02 = true) |
| B1-G2 | Global-cond MLP type | `bias_config.global_conditioned.mlp_type` | — | "global_conditioned" ∈ B1 |
| B1-G3 | Global-cond global dim | `bias_config.global_conditioned.global_dim` | — | "global_conditioned" ∈ B1 |

### §9 C — Classifier head (2 leaves)

| Id | Name | Hydra config | axes key | Prereq |
|---|---|---|---|---|
| C2 | Pooling strategy | `classifier.model.head.pooling` | `pooling` | (root) |
| C1 | Head realization | `classifier.model.head.type` | `head_type` | (root) |

### §10 H — Model-size hyperparameters (6 leaves)

| Id | Hydra config | axes key |
|---|---|---|
| H01 | `classifier.model.dim` | `dim` |
| H02 | `classifier.model.depth` | `depth` |
| H03 | `classifier.model.heads` | `heads` |
| H04 | `classifier.model.mlp_dim` | `mlp_dim` |
| H05 | `classifier.model.dropout` | `dropout` |
| H10 | derived `d{dim}_L{depth}` | `model_size_key` |

### §11 §K — KAN shared (5 leaves, conditional on any KAN consumer being active)

| Id | Hydra config | axes key | Consumer gate |
|---|---|---|---|
| K1 | `kan.grid_size` | `kan_grid_size` | ∃ active KAN (F1-a set, C1=kan, B1-L2=kan, B1-G2=kan) |
| K2 | `kan.spline_order` | `kan_spline_order` | same |
| K3 | `kan.grid_range` | — | same |
| K4 | `kan.spline_regularization_weight` | — | same |
| K5 | `kan.grid_update_freq` | — | same |

### §12 §M — MoE shared (5 leaves, conditional on MoE being active)

| Id | Hydra config | axes key | Gate |
|---|---|---|---|
| M1 | `moe.num_experts` | — | (F1-effective = moe) ∨ (C1 = moe) |
| M2 | `moe.top_k` | `moe_top_k` | same |
| M3 | `moe.routing_level` | `moe_routing_level` | same |
| M4 | `moe.load_balance_loss_weight` | — | same |
| M5 | `moe.noisy_gating` | — | same |

### §13 §S — Shared pairwise backbone (2 leaves, conditional)

| Id | Hydra config | axes key | Gate |
|---|---|---|---|
| S1 | `shared_backbone.enabled` | — | (∃ B1 family active) ∨ (P2 = true) |
| S2 | `shared_backbone.features` | — | same |

### §14 §R — Training protocol (17 leaves, all root)

R01–R05 have `axes/*` mirrors; R06–R17 do not.

| Id | Hydra config | axes key |
|---|---|---|
| R01 | `trainer.epochs` | `epochs` |
| R02 | `trainer.lr` | `lr` |
| R03 | `trainer.weight_decay` | `weight_decay` |
| R04 | `trainer.batch_size` | `batch_size` |
| R05 | `trainer.seed` | `seed` |
| R06 | `trainer.warmup_steps` | — |
| R07 | `trainer.lr_schedule` | — |
| R08 | `trainer.label_smoothing` | — |
| R09 | `trainer.grad_clip` | — |
| R10 | `trainer.early_stopping.enabled` | — |
| R11 | `trainer.early_stopping.patience` | — |
| R12 | `trainer.early_stopping.min_delta` | — |
| R13 | `trainer.early_stopping.restore_best_weights` | — |
| R14 | `trainer.pid_schedule.mode` | — (gate: T1 = identity ∧ T1-a = learned) |
| R15 | `trainer.pid_schedule.transition_epoch` | — (same gate) |
| R16 | `trainer.pid_schedule.reinit_mode` | — (same gate) |
| R17 | `trainer.pid_schedule.pid_lr` | — (same gate) |

### §15 §L — Logging / interpretability (7 leaves, all root)

L1–L7 all map to `raw/*` only. Not gated; always emit as stored (empty when absent).

**Grand totals**

| Group | Leaves | Gated sub-axes |
|---|---|---|
| G | 3 | 1 (G02 derived from G01) |
| D | 3 | 0 |
| T | 4 | 3 |
| E | 4 | 3 |
| P | 9 | 8 |
| A | 6 | 1 |
| F | 4 | 3 |
| B | 16 | 15 |
| C | 2 | 0 |
| H | 6 | 0 |
| §K | 5 | 5 |
| §M | 5 | 5 |
| §S | 2 | 2 |
| §R | 17 | 4 (R14–R17) |
| §L | 7 | 0 |
| **Total** | **93** | **50** |

---

## 2. Config-path presence across the 1002-run population

Across all 1002 runs, four coverage tiers emerged:

| Tier | Runs | % | Defining W&B namespaces present |
|---|---:|---:|---|
| 0 — legacy, metadata-only | ~164 | 16.4 | only `meta.*`, `data/token_order`, `source/location` |
| 1 — basic structured | 224 | 22.4 | adds `model/*`, `training/*`, `pos_enc/*`, `norm/*`, `pooling/*` |
| 2 — raw/* catch-all migration | 119 | 11.9 | adds `raw/classifier/model/*` and `raw/classifier/trainer/*` |
| 3 — full V1 axes migration | 455 | 45.4 | adds `axes/*`, `tokenizer/*`, `pid/*`, `kan/*`, `moe/*`, `bias/*` |

Cumulative: 100% meta-only, 83.6% have `model/*`, 57.3% have `raw/classifier/model/*`, 45.4% have `axes/*`. This determines a tiered lookup strategy for every V2 leaf.

### 2.1 Per-leaf config-source resolution (V2 leaf → W&B config key)

For each V2 leaf, the derivation reads the first non-empty match in this priority list. Values come from `_reference/column_presence.csv` and the 5-run sample. **"Max coverage"** is the fraction of the 1002 runs for which at least one listed lookup key returned a non-null value.

| V2 Id | Preferred namespaced key | Fallback raw key | axes/* key | Max coverage |
|---|---|---|---|---:|
| G01 | `model/loop` | `raw/loop` (absent) | — | 83.6% |
| G02 | derived from G01 | derived | — | 83.6% |
| G03 | `meta.process_groups_key`, `meta.class_def_str` | `raw/data/classifier/*` | — | 100% |
| D01 | — | `raw/data/cont_features` (0 hits) | `axes/cont_features` | 45.4% (via axes only) |
| D02 | `globals/include_met` | `raw/classifier/globals/include_met` | `axes/include_met` | 57.3% |
| D03 | `data/token_order` | `data/shuffle_tokens`, `data/sort_tokens_by` | `axes/token_order` | 100% |
| T1 | `tokenizer/type` | `raw/classifier/model/tokenizer/name` | `axes/tokenizer_name` | 57.3% |
| T1-a | `tokenizer/pid_mode` | `raw/classifier/model/tokenizer/pid_mode` | `axes/pid_mode` | 57.3% |
| T1-b | `tokenizer/id_embed_dim` | `raw/classifier/model/tokenizer/id_embed_dim` | `axes/id_embed_dim` | 57.3% |
| T1-c | `tokenizer/model_type` | `raw/classifier/model/tokenizer/model_type` | — | very low (<5%) |
| E1 | `pos_enc/type` | `raw/classifier/model/positional` | `axes/positional` | 78.9% |
| E1-a | `pos_enc/space` | `raw/classifier/model/positional_space` | `axes/positional_space` | 61.5% |
| E1-a1 | `pos_enc/dim_mask` | `raw/classifier/model/positional_dim_mask` | `axes/positional_dim_mask` | ≤57.3% |
| E1-b | `pos_enc/rotary_base` | `raw/classifier/model/rotary/base` | — | 64.1% |
| P1 | `nodewise_mass/enabled` | `raw/classifier/model/nodewise_mass/enabled` | `axes/nodewise_mass_enabled` | 57.3% |
| P1-a | `nodewise_mass/k_values` | `raw/classifier/model/nodewise_mass/k_values` | — | 57.3% |
| P1-b | — | `raw/classifier/model/nodewise_mass/hidden_dim` | — | 57.3% |
| P2 | `mia/enabled` | `raw/classifier/model/mia_blocks/enabled` | `axes/mia_enabled` | 57.3% |
| P2-a | `mia/placement` | `raw/classifier/model/mia_blocks/placement` | `axes/mia_placement` | 57.3% |
| P2-b | `mia/num_blocks` | `raw/classifier/model/mia_blocks/num_blocks` | — | 57.3% |
| P2-c | — | `raw/classifier/model/mia_blocks/interaction_dim` | — | 57.3% |
| P2-d | — | `raw/classifier/model/mia_blocks/reduction_dim` | — | 57.3% |
| P2-e | — | `raw/classifier/model/mia_blocks/dropout` | — | 57.3% |
| A1 | `norm/policy` | `raw/classifier/model/norm/policy` | `axes/norm_policy` | 78.9% |
| A2 | `norm/type` | `raw/classifier/model/norm/type` | `axes/norm_type` | 78.9% |
| A3 | `attention/type` | `raw/classifier/model/attention/type` | `axes/attention_type` | 45.4% |
| A3-a | `attention/diff_bias_mode` | `raw/classifier/model/attention/diff_bias_mode` | `axes/diff_bias_mode` | 45.4% |
| A4 | `attention/norm` | `raw/classifier/model/attention/norm` | `axes/attention_norm` | 45.4% |
| A5 | `model/causal_attention` | `raw/classifier/model/causal_attention` | `axes/causal_attention` | 57.3% |
| F1 | `ffn/type`, `moe/enabled` | `raw/classifier/model/ffn/type`, `raw/.../moe/enabled` | `axes/ffn_type`, `axes/moe_enabled` | 45.4% |
| F1-a | `ffn/kan_variant` | `raw/classifier/model/ffn/kan/variant` | `axes/kan_ffn_variant` | 45.4% |
| F1-a1 | — | `raw/classifier/model/ffn/kan/bottleneck_dim` | — | 45.4% |
| F1-b | `moe/scope` | `raw/classifier/model/moe/scope` | `axes/moe_scope` | 45.4% |
| B1 | `bias/selector` | `raw/classifier/model/attention_biases` | `axes/attention_biases` | 57.3% |
| B1-L1 | `bias/lorentz_features` | `raw/.../lorentz_scalar/features` | `axes/lorentz_features` | 57.3% |
| B1-L2 | `kan/bias_lorentz_mlp_type` | `raw/.../lorentz_scalar/mlp_type` | `axes/lorentz_mlp_type` | 45.4% |
| B1-L3 | — | `raw/.../lorentz_scalar/hidden_dim` | — | 57.3% |
| B1-L4 | `bias/lorentz_per_head` | `raw/.../lorentz_scalar/per_head` | `axes/lorentz_per_head` | 57.3% |
| B1-L5 | `bias/lorentz_sparse_gating` | `raw/.../lorentz_scalar/sparse_gating` | `axes/lorentz_sparse_gating` | 57.3% |
| B1-T1 | `bias/typepair_init` | `raw/.../typepair_kinematic/init_from_physics` | `axes/typepair_init` | 57.3% |
| B1-T2 | `bias/typepair_freeze` | `raw/.../typepair_kinematic/freeze_table` | `axes/typepair_freeze` | 57.3% |
| B1-T3 | `bias/typepair_kinematic_gate` | `raw/.../typepair_kinematic/kinematic_gate` | `axes/typepair_kinematic_gate` | 57.3% |
| B1-T4 | — | `raw/.../typepair_kinematic/kinematic_feature` | — | 57.3% |
| B1-T5 | — | `raw/.../typepair_kinematic/mask_value` | — | 57.3% |
| B1-S1 | `bias/sm_mode` | `raw/.../sm_interaction/mode` | `axes/sm_mode` | 57.3% |
| B1-S2 | — | `raw/.../sm_interaction/mask_value` | — | 57.3% |
| B1-G1 | `bias/global_mode` | `raw/.../global_conditioned/mode` | `axes/global_conditioned_mode` | 57.3% |
| B1-G2 | `kan/bias_global_mlp_type` | `raw/.../global_conditioned/mlp_type` | — | 45.4% |
| B1-G3 | — | `raw/.../global_conditioned/global_dim` | — | 57.3% |
| C2 | `pooling/type` | `raw/classifier/model/head/pooling` | `axes/pooling` | 78.9% |
| C1 | `head/type` | `raw/classifier/model/head/type` | `axes/head_type` | 45.4% |
| H01 | `model/dim` | `raw/classifier/model/dim` | `axes/dim` | 79.3% |
| H02 | `model/depth` | `raw/classifier/model/depth` | `axes/depth` | 79.3% |
| H03 | `model/heads` | `raw/classifier/model/heads` | `axes/heads` | 79.3% |
| H04 | `model/mlp_dim` | `raw/classifier/model/mlp_dim` | `axes/mlp_dim` | 57.3% |
| H05 | `model/dropout` | `raw/classifier/model/dropout` | `axes/dropout` | 79.3% |
| H10 | `model/size_key` | derived from H01, H02 | `axes/model_size_key` | 79.3% |
| K1 | `kan/grid_size` | `raw/classifier/model/kan/grid_size` | `axes/kan_grid_size` | 45.4% |
| K2 | `kan/spline_order` | `raw/classifier/model/kan/spline_order` | `axes/kan_spline_order` | 45.4% |
| K3 | `kan/grid_range` | `raw/classifier/model/kan/grid_range` | — | 45.4% |
| K4 | `kan/spline_reg_weight` | `raw/classifier/model/kan/spline_regularization_weight` | — | 45.4% |
| K5 | `kan/grid_update_freq` | `raw/classifier/model/kan/grid_update_freq` | — | 45.4% |
| M1 | `moe/num_experts` | `raw/classifier/model/moe/num_experts` | — | 45.4% |
| M2 | `moe/top_k` | `raw/classifier/model/moe/top_k` | `axes/moe_top_k` | 45.4% |
| M3 | `moe/routing_level` | `raw/classifier/model/moe/routing_level` | `axes/moe_routing_level` | 45.4% |
| M4 | `moe/lb_weight` | `raw/.../moe/load_balance_loss_weight` | — | 45.4% |
| M5 | `moe/noisy_gating` | `raw/.../moe/noisy_gating` | — | 45.4% |
| S1 | — | `raw/classifier/model/shared_backbone/enabled` | — | 57.3% |
| S2 | — | `raw/classifier/model/shared_backbone/features` | — | 57.3% |
| R01 | `training/epochs` | `raw/classifier/trainer/epochs` | `axes/epochs` | 79.7% |
| R02 | `training/lr` | `raw/classifier/trainer/lr` | `axes/lr` | 79.7% |
| R03 | `training/weight_decay` | `raw/classifier/trainer/weight_decay` | `axes/weight_decay` | 79.7% |
| R04 | `training/batch_size` | `raw/classifier/trainer/batch_size` | `axes/batch_size` | 79.7% |
| R05 | `training/seed` | `raw/classifier/trainer/seed` | `axes/seed` | 79.7% |
| R06 | `training/warmup_steps` | `raw/classifier/trainer/warmup_steps` | — | 79.7% |
| R07 | `training/lr_schedule` | `raw/classifier/trainer/lr_schedule` | — | 79.7% |
| R08 | `training/label_smoothing` | `raw/classifier/trainer/label_smoothing` | — | 79.7% |
| R09 | `training/grad_clip` | `raw/classifier/trainer/grad_clip` | — | 79.7% |
| R10 | `early_stop/enabled` | `raw/classifier/trainer/early_stopping/enabled` | — | 64.9% |
| R11 | `early_stop/patience` | `raw/classifier/trainer/early_stopping/patience` | — | 64.9% |
| R12 | — | `raw/classifier/trainer/early_stopping/min_delta` | — | 57.3% |
| R13 | — | `raw/classifier/trainer/early_stopping/restore_best_weights` | — | 57.3% |
| R14 | `pid/schedule_mode` | `raw/classifier/trainer/pid_schedule/mode` | — | 57.3% |
| R15 | `pid/transition_epoch` | `raw/classifier/trainer/pid_schedule/transition_epoch` | — | 57.3% |
| R16 | — | `raw/classifier/trainer/pid_schedule/reinit_mode` | — | 57.3% |
| R17 | — | `raw/classifier/trainer/pid_schedule/pid_lr` | — | 57.3% |
| L1 | — | `raw/classifier/trainer/log_pid_embeddings` | — | 57.3% |
| L2 | — | `raw/classifier/trainer/interpretability/enabled` | — | 57.3% |
| L3 | — | `raw/classifier/trainer/interpretability/save_attention_maps` | — | 57.3% |
| L4 | — | `raw/classifier/trainer/interpretability/save_kan_splines` | — | 57.3% |
| L5 | — | `raw/classifier/trainer/interpretability/save_moe_routing` | — | 57.3% |
| L6 | — | `raw/classifier/trainer/interpretability/save_gradient_norms` | — | 57.3% |
| L7 | — | `raw/classifier/trainer/interpretability/checkpoint_epochs` | — | 57.3% |

**Flags — V2 axes with effectively zero non-raw presence in live sample:**

- **T1-c** (pretrained tokenizer model_type): `tokenizer/model_type` appears on ≤5% of runs. Consistent with `T1 = pretrained` being a rarely used mode. Confirmed as correct, not a typo.
- **D01 `axes/cont_features`** only exists on 45.4% of runs despite V2 claiming it as a root axis. On the 119 tier-2 runs the fallback to `raw/data/cont_features` is absent too — `data.cont_features` sits at the top level of the Hydra tree and `extract_wandb_config` chose to store it as `data/cont_features` not under `raw/data/*`. Action: add `data/cont_features` (fraction 0.37 in live audit) as an additional preferred key in Deliverable 2.
- **F1-a1 bottleneck_dim**: present only on runs where KAN FFN is actually used, not a bug.
- All V2 leaves are either reachable on the ~455-574 run populations or are documented as empty-by-design for the legacy tiers.

---

## 3. Prerequisite graph — flat parent→children table

Re-emission of V2 §0.2 as child → parent list + verbatim gate. A dependency is discharged only when all listed conditions are satisfied.

| Child | Parent(s) | Gate |
|---|---|---|
| G02 | G01 | infer model_family from loop string |
| T1-a | T1 | T1 == "identity" |
| T1-b | T1, T1-a | T1 == "identity" (override to "num_types" when T1-a == "one_hot") |
| T1-c | T1 | T1 == "pretrained" |
| E1-a | E1 | E1 ∈ {"sinusoidal", "learned"} |
| E1-a1 | E1-a | E1-a == "token" |
| E1-b | E1 | E1 == "rotary" |
| P1 | T1, D01 | (T1 ∈ {"raw", "identity"}) ∧ (0 ∈ D01) |
| P1-a | P1 | P1 == true |
| P1-b | P1 | P1 == true |
| P2 | T1 | T1 ∈ {"raw", "identity"} |
| P2-a | P2 | P2 == true |
| P2-b | P2 | P2 == true |
| P2-c | P2 | P2 == true |
| P2-d | P2 | P2 == true |
| P2-e | P2 | P2 == true |
| A3-a | A3 | A3 == "differential" |
| F1-a | F1 | F1-effective == "kan" |
| F1-a1 | F1-a | F1-a == "bottleneck" |
| F1-b | F1 | F1-effective == "moe" |
| B1 | T1 | T1 ∈ {"raw", "identity"} |
| B1-L1..L5 | B1 | "lorentz_scalar" ∈ B1 |
| B1-T1..T5 | B1 | "typepair_kinematic" ∈ B1 |
| B1-S1..S2 | B1 | "sm_interaction" ∈ B1 |
| B1-G1 | B1, D02 | ("global_conditioned" ∈ B1) ∧ (mode == "met_direction" ⇒ D02 == true) |
| B1-G2..G3 | B1 | "global_conditioned" ∈ B1 |
| K1..K5 | F1-a, C1, B1-L2, B1-G2 | any of (F1-a != ""), (C1 == "kan"), (B1-L2 == "kan"), (B1-G2 == "kan") |
| M1..M5 | F1, C1 | (F1-effective == "moe") ∨ (C1 == "moe") |
| S1, S2 | B1, P2 | (∃ B1 family active) ∨ (P2 == true) |
| R14..R17 | T1, T1-a | (T1 == "identity") ∧ (T1-a == "learned") |

Topological-sort order (for Deliverable 3's derive function):
**G01 → G02 → G03 → D01 → D02 → D03 → T1 → T1-a → T1-b → T1-c → E1 → E1-a → E1-a1 → E1-b → A1..A5 → A3-a → F1 (effective) → F1-a → F1-a1 → F1-b → C2 → C1 → P1 → P1-a → P1-b → P2 → P2-a..e → B1 → B1-L1..L5 → B1-T1..T5 → B1-S1..S2 → B1-G1..G3 → K1..K5 → M1..M5 → S1..S2 → H01..H05 → H10 → R01..R13 → R14..R17 → L1..L7**

---

## 4. Experiment inventory

From `_reference/experiment_inventory.csv`. Two groupings: `axes/experiment_name` (populated only on tier-3 runs, 455/1002) and `wandb.group` (populated on all runs that went through migration, ≥838/1002).

- **52 distinct `axes/experiment_name` values** across the 455 tier-3 runs. Top buckets include the recent sweeps `phd_exp1_4t_vs_bg_sizes_and_pe`, `OrthogonalSweep_A_g04_attention_type`, `compare_positional_encodings`, `builtjes_baseline_test`. The 547 tier-0/1/2 runs show `axes/experiment_name` as empty.
- **99 distinct `wandb.group` values** covering the full 1002 runs. These are the `exp_<timestamp>_<name>` groupings set by `_extract_group_from_run` in [scripts/wandb/migrate_runs_to_wandb.py](../../scripts/wandb/migrate_runs_to_wandb.py). This is the stable grouping key.
- The V2 backfill preserves both columns untouched.
- Full distribution is in `_reference/experiment_inventory.csv` (one row per `(dimension, value)` pair with run counts).

---

## 5. Five sampled runs — prerequisite-graph coverage

From `_reference/summary.json`. Each sample is one of the representative points requested in Deliverable 4 Phase B. The T1=binned case and the T1-not-identity-no-bias case collapsed onto the same run (7ipzizh6) because that run is `binned` and has `attention_biases = none`. Phase B therefore runs on four distinct runs.

| Bucket | Run ID | Run name | Group |
|---|---|---|---|
| T1 = identity | `1iuxis2l` | `run_20260320-141518_phd_exp1_4t_vs_bg_sizes_and_pe_job000` | `exp_20260320-141518_phd_exp1_4t_vs_bg_sizes_and_pe` |
| T1 = binned | `7ipzizh6` | `run_20260320-123722_builtjes_baseline_test_job000` | `exp_20260320-123722_builtjes_baseline_test` |
| E1 = rotary | `wqo0y6mw` | `run_20251126-150519_compare_positional_encodings_job009` | `exp_20251126-150519_compare_positional_encodings` |
| A3 = differential | `17ovy6fy` | `run_20260320-163524_OrthogonalSweep_A_g04_attention_type_job003` | `exp_20260320-163524_OrthogonalSweep_A_g04_attention_type` |
| T1 ≠ identity, no biases | `7ipzizh6` | (same as row 2) | (same) |

Concrete prerequisite-behaviour predictions based on the sample values:
- `1iuxis2l` (T1 = identity): T1-a, T1-b must be non-empty; B1-L*/T*/S*/G* empty depending on B1; P1 eligible.
- `7ipzizh6` (T1 = binned, no biases): T1-a, T1-b, P1, P2, B1-L1..G3, R14..R17 all empty.
- `wqo0y6mw` (E1 = rotary): E1-a empty, E1-a1 empty, E1-b populated.
- `17ovy6fy` (A3 = differential): A3-a populated; other A3 gates unaffected.

These predictions form the Phase B visual-inspection checklist in Deliverable 4.

---

## 6. Cross-cutting findings

- Only 45.4% of runs currently have any `axes/*` keys. V2 backfill will therefore write new `axes/*` keys for 547 runs, not overwrite existing ones — the `--overwrite` default of `true` changes no behaviour for these runs. For the remaining 455 tier-3 runs, `--overwrite` does change behaviour: V2 values replace legacy `axes/*` values, which is the explicit intent per the planning clarification.
- 428 tier-0/1 runs lack all `raw/classifier/model/*` keys. For these runs, almost every architectural V2 leaf will emit `""` (not because of prerequisites, but because the config source is unavailable). These cases land in the `keys_empty_missing_config` report column in Deliverable 3. They are not pollution — empty cells simply remove these runs from V2 column value counts, which is the intended W&B semantics.
- `meta.*` fields are available on 100% of runs and carry a parallel axis taxonomy that V2 does not displace (`meta.row_key`, `meta.process_groups_key`, `meta.class_def_str`). Leave untouched.
- The legacy `attn_pairwise.*` family (574 runs have `raw/classifier/model/attn_pairwise/enabled`) maps to the V2 B1 family per V2 §9 — it must be handled in Deliverable 2's B1 derivation rule as a fallback when `attention_biases` is `"none"` but `attn_pairwise.enabled == true`.

Doc-level ambiguities and typos discovered are flagged in [00_doc_issues.md](00_doc_issues.md).
