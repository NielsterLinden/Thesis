# Orthogonal Axes Reference

This document lists experiment **axes** for the **transformer classifier** (`loop=transformer_classifier`): Hydra **config keys**, canonical **Weights & Biases** config keys, and (where applicable) `**axes/*`** names from `[src/thesis_ml/facts/axes.py](../src/thesis_ml/facts/axes.py)` (mirrored into W&B for filtering).

**Sources of truth:**

- Flat W&B config: `[src/thesis_ml/utils/wandb_utils.py](../src/thesis_ml/utils/wandb_utils.py)` — `extract_wandb_config()`
- Canonical thesis axes JSON / `axes/*`: `[src/thesis_ml/facts/axes.py](../src/thesis_ml/facts/axes.py)` — `build_axes_metadata()`

Sub-axes only matter when the parent axis takes a specific value. **Total: 33 numbered axes** across **5 groups**, plus **hyperparameter table H01–H10**, plus **appendices** for shared infrastructure and training-only switches.

---

## How to read Config vs W&B vs axes


| Column     | Meaning                                                                               |
| ---------- | ------------------------------------------------------------------------------------- |
| **Config** | Hydra key (often override as `classifier.model.foo=...`)                              |
| **W&B**    | Key under W&B run `config` from `extract_wandb_config()` (namespace `like/this`)      |
| **axes**   | Key in `facts/axes.json` and W&B `axes/<key>` when present in `build_axes_metadata()` |


If **W&B** is omitted for an axis, logging may still expose it via the `**raw/*`** catch-all flatten in `extract_wandb_config()`.

---

## 1. Input representation (how events are encoded before the transformer)

**Chapter summary — filter in W&B**


| Topic               | Config prefix / keys                         | W&B prefix / keys                                                                        | axes/*                                                            |
| ------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Tokenizer & PID     | `classifier.model.tokenizer.*`               | `tokenizer/type`, `tokenizer/id_embed_dim`, `tokenizer/pid_mode`, `tokenizer/model_type` | `tokenizer_name`, `pid_mode`, `id_embed_dim`                      |
| Continuous features | `data.cont_features`                         | `data/cont_features`                                                                     | `cont_features`                                                   |
| MET as globals      | `classifier.globals.include_met`             | `globals/include_met`                                                                    | `include_met`                                                     |
| Token order         | `data.shuffle_tokens`, `data.sort_tokens_by` | `data/shuffle_tokens`, `data/sort_tokens_by`                                             | `token_order` (derived: `input_order` · `pt_sorted` · `shuffled`) |


### D01. Tokenizer

- **Config:** `classifier.model.tokenizer.name`
- **W&B:** `tokenizer/type`
- **axes:** `tokenizer_name`
- **Settings:** `raw` · `identity` · `binned` · `pretrained`
- **What it tests:** Discrete vs continuous input, binning, vs VQ-VAE embeddings.
- **Experiments:** `emb_pe_4tbg`, `exp_binning_vs_direct`

### D02. PID embedding mode

- **Config:** `classifier.model.tokenizer.pid_mode`
- **W&B:** `tokenizer/pid_mode`
- **axes:** `pid_mode`
- **Settings:** `learned` · `one_hot` · `fixed_random`
- **Prerequisite:** Meaningful for `**identity`** (and experiments that use PID embeddings). `**raw**` passes through continuous features only — PID is ignored; `**binned**` / `**pretrained**` use different ID handling.
- **What it tests:** Learned vs fixed particle-type representations.
- **Experiments:** `pid_deepdive`

### D03. PID embedding dimension

- **Config:** `classifier.model.tokenizer.id_embed_dim`
- **W&B:** `tokenizer/id_embed_dim`
- **axes:** `id_embed_dim`
- **Settings:** e.g. `8` · `16` · `32`
- **Prerequisite:** Same as D02 (`identity` tokenizer for PID embedding studies).
- **What it tests:** Capacity of the type embedding.
- **Experiments:** `pid_deepdive`

### D04. Continuous feature set

- **Config:** `data.cont_features`
- **W&B:** `data/cont_features`
- **axes:** `cont_features`
- **Settings:** `[0,1,2,3]` (E, pT, η, φ) · `[1,2,3]` (pT, η, φ)
- **What it tests:** Whether explicit energy E helps vs 4-vector without E.
- **Experiments:** `exp_binning_vs_direct`
- **Note:** May be omitted from default `[configs/data/h5_tokens.yaml](../configs/data/h5_tokens.yaml)`; loader defaults apply (see `[src/thesis_ml/data/h5_loader.py](../src/thesis_ml/data/h5_loader.py)`).

### D05. MET treatment

- **Config:** `classifier.globals.include_met`
- **W&B:** `globals/include_met`
- **axes:** `include_met`
- **Settings:** `false` · `true`
- **What it tests:** MET / METφ as global-conditioned inputs (distinct from MET-conditioned **attention bias** B07).
- **Experiments:** `emb_pe_4tbg`, `met_treatment`

### D06. Token ordering

- **Config:** `data.sort_tokens_by`, `data.shuffle_tokens`
- **W&B:** `data/sort_tokens_by`, `data/shuffle_tokens`
- **axes:** `token_order` (derived)
- **Settings:** original order · `sort_tokens_by: pt` · `shuffle_tokens: true`
- **What it tests:** Sensitivity to ordering vs permutation invariance.
- **Experiments:** `order_pe_attention_4t_vs_bg`
- **Hydra presets:** `[configs/data/order/pt_ordered.yaml](../configs/data/order/pt_ordered.yaml)`, `[configs/data/order/shuffled.yaml](../configs/data/order/shuffled.yaml)`

---

## 2. Positional encoding (what the model knows about token position)

**Chapter summary**


| Topic                  | Config                                                                   | W&B                                                 | axes                                                    |
| ---------------------- | ------------------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------- |
| PE type / space / mask | `classifier.model.positional`, `positional_space`, `positional_dim_mask` | `pos_enc/type`, `pos_enc/space`, `pos_enc/dim_mask` | `positional`, `positional_space`, `positional_dim_mask` |
| RoPE                   | `classifier.model.rotary.base`                                           | `pos_enc/rotary_base`                               | —                                                       |


### D07. Positional encoding type

- **Config:** `classifier.model.positional`
- **W&B:** `pos_enc/type`
- **axes:** `positional`
- **Settings:** `none` · `sinusoidal` · `learned` · `rotary`
- **What it tests:** Whether explicit position signal helps; `none` is the strict permutation baseline if input order is uninformative.
- **Experiments:** `exp1_*_sizes_and_pe`, `emb_pe_4tbg`, `order_pe_attention_4t_vs_bg`

### D07b. Rotary base (sub-axis of D07)

- **Config:** `classifier.model.rotary.base`
- **W&B:** `pos_enc/rotary_base`
- **Prerequisite:** D07 = `rotary`
- **What it tests:** RoPE frequency schedule.

### D08. Positional encoding space

- **Config:** `classifier.model.positional_space`
- **W&B:** `pos_enc/space`
- **axes:** `positional_space`
- **Settings:** `model` (after projection) · `token` (before projection)
- **Prerequisite:** Relevant when D07 ≠ `none` and D07 ≠ `rotary` (RoPE applies in model space by design in this stack).
- **What it tests:** PE on raw semantic features vs projected embeddings.
- **Experiments:** `exp2_*_selective_masks`, `emb_pe_4tbg`

### D09. Selective PE dimension mask

- **Config:** `classifier.model.positional_dim_mask`
- **W&B:** `pos_enc/dim_mask`
- **axes:** `positional_dim_mask`
- **Settings:** `null` (all) · feature groups per `[configs/classifier/positional_dim_mask/](../configs/classifier/positional_dim_mask/)`
- **Prerequisite:** D08 = `token` (and D07 active)
- **What it tests:** Which continuous / ID channels should carry PE.
- **Experiments:** `exp2_*_selective_masks`, `selective_positional_encoding`

---

## 3. Transformer architecture (encoder structure)

**Chapter summary**


| Topic                 | Config                                             | W&B                                                                                                                  | axes                                                                     |
| --------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Block norm            | `classifier.model.norm.policy`, `norm.type`        | `norm/policy`, `norm/type`                                                                                           | `norm_policy`, `norm_type`                                               |
| Attention             | `classifier.model.attention.*`, `causal_attention` | `attention/type`, `attention/norm`, `attention/diff_bias_mode`, `model/causal_attention`                             | `attention_type`, `attention_norm`, `diff_bias_mode`, `causal_attention` |
| Pooling / head kind   | `head.pooling`, `head.type`                        | `pooling/type`, `head/type`                                                                                          | `pooling`, `head_type`                                                   |
| FFN                   | `ffn.type`, `ffn.kan.*`                            | `ffn/type`, `ffn/kan_variant`                                                                                        | `ffn_type`, `kan_ffn_variant`                                            |
| MoE (FFN or head)     | `classifier.model.moe.*`                           | `moe/enabled`, `moe/num_experts`, `moe/top_k`, `moe/routing_level`, `moe/scope`, `moe/lb_weight`, `moe/noisy_gating` | `moe_enabled`, `moe_top_k`, `moe_routing_level`, `moe_scope`             |
| Global KAN params     | `classifier.model.kan.*`                           | `kan/grid_size`, `kan/spline_order`, `kan/grid_range`, `kan/spline_reg_weight`, `kan/grid_update_freq`               | `kan_grid_size`, `kan_spline_order`                                      |
| Size / regularization | `dim`, `depth`, `heads`, `mlp_dim`, `dropout`      | `model/dim`, `model/depth`, `model/heads`, `model/mlp_dim`, `model/dropout`                                          | `dim`, `depth`, `heads`, `mlp_dim`, `dropout`                            |


### A01. Normalization policy

- **Config:** `classifier.model.norm.policy`
- **W&B:** `norm/policy`
- **axes:** `norm_policy`
- **Settings:** `pre` · `post` · `normformer`
- **Experiments:** `compare_norm_pos_pool`, `exp3_*_norm_policies`

### A02. Normalization type

- **Config:** `classifier.model.norm.type`
- **W&B:** `norm/type`
- **axes:** `norm_type`
- **Settings:** `layernorm` · `rmsnorm`
- **Experiments:** `exp_diff_attn_blocknorm`

### A03. Attention type

- **Config:** `classifier.model.attention.type`
- **W&B:** `attention/type`
- **axes:** `attention_type`
- **Settings:** `standard` · `differential`
- **Experiments:** `exp_diff_attn_core`, `exp_diff_attn_blocknorm`, `exp_diff_attn_normformer`

### A04. Attention-internal normalization

- **Config:** `classifier.model.attention.norm`
- **W&B:** `attention/norm`
- **axes:** `attention_norm`
- **Settings:** `none` · `layernorm` · `rmsnorm`
- **Experiments:** `exp_diff_attn_core`

### A05. Differential attention bias mode

- **Config:** `classifier.model.attention.diff_bias_mode`
- **W&B:** `attention/diff_bias_mode`
- **axes:** `diff_bias_mode`
- **Settings:** `none` · `shared` · `split`
- **Prerequisite:** A03 = `differential`

### A06. Pooling strategy

- **Config:** `classifier.model.head.pooling`
- **W&B:** `pooling/type`
- **axes:** `pooling`
- **Settings:** `cls` · `mean` · `max`
- **Experiments:** `compare_norm_pos_pool`, `exp_binning_vs_direct`

### A07. Causal attention

- **Config:** `classifier.model.causal_attention`
- **W&B:** `model/causal_attention`
- **axes:** `causal_attention`
- **Settings:** `false` · `true`
- **Experiments:** `order_pe_attention_4t_vs_bg`

### A08. FFN type

- **Config:** `classifier.model.ffn.type`
- **W&B:** `ffn/type`
- **axes:** `ffn_type`
- **Settings:** `standard` · `kan`
- **Experiments:** `exp_kan_ffn`

### A09. KAN FFN variant

- **Config:** `classifier.model.ffn.kan.variant`, `classifier.model.ffn.kan.bottleneck_dim`
- **W&B:** `ffn/kan_variant` (bottleneck_dim not in primary extract — see `raw/*`)
- **axes:** `kan_ffn_variant`
- **Prerequisite:** A08 = `kan`
- **Settings:** `hybrid` · `bottleneck` · `pure`
- **Experiments:** `exp_kan_ffn`

### A10. Classifier head type

- **Config:** `classifier.model.head.type`
- **W&B:** `head/type`
- **axes:** `head_type`
- **Settings:** `linear` · `kan` · `moe`
- **Note:** `moe` head uses the **same** `classifier.model.moe` block with `moe.scope: head` (shared experts config as FFN MoE).
- **Experiments:** `exp_kan_head`, `exp_moe_first_pass` (when head uses MoE)

### A11. MoE: enabled

- **Config:** `classifier.model.moe.enabled`
- **W&B:** `moe/enabled`
- **axes:** `moe_enabled`
- **Settings:** `false` · `true`
- **What it tests:** Sparse experts in **FFN** (`scope` ≠ `head`) or **head** (`scope: head`).
- **Experiments:** `exp_moe_first_pass`

### A12. MoE: top-k

- **Config:** `classifier.model.moe.top_k`
- **W&B:** `moe/top_k`
- **axes:** `moe_top_k`
- **Prerequisite:** A11 = `true`
- **Settings:** typically `1`–`3` (see `[configs/classifier/model/transformer.yaml](../configs/classifier/model/transformer.yaml)`)

### A13. MoE: routing level

- **Config:** `classifier.model.moe.routing_level`
- **W&B:** `moe/routing_level`
- **axes:** `moe_routing_level`
- **Settings:** `token` · `event`
- **Prerequisite:** A11 = `true`

### A14. MoE: scope

- **Config:** `classifier.model.moe.scope`
- **W&B:** `moe/scope`
- **axes:** `moe_scope`
- **Settings:** `head` · `middle_blocks` · `all_blocks`
- **Prerequisite:** A11 = `true`

### A15. MoE: number of experts

- **Config:** `classifier.model.moe.num_experts`
- **W&B:** `moe/num_experts`
- **Prerequisite:** A11 = `true`

### A16. MoE: load-balancing loss weight

- **Config:** `classifier.model.moe.load_balance_loss_weight`
- **W&B:** `moe/lb_weight`
- **Prerequisite:** A11 = `true`

### A17. MoE: noisy gating

- **Config:** `classifier.model.moe.noisy_gating`
- **W&B:** `moe/noisy_gating`
- **Prerequisite:** A11 = `true`

### K01–K05. Global KAN hyperparameters (shared)

Used when **any** of: FFN KAN (A08–A09), head KAN (A10), Lorentz bias KAN (B04), or global-conditioned bias MLP type uses KAN.


| Config                                              | W&B                     |
| --------------------------------------------------- | ----------------------- |
| `classifier.model.kan.grid_size`                    | `kan/grid_size`         |
| `classifier.model.kan.spline_order`                 | `kan/spline_order`      |
| `classifier.model.kan.grid_range`                   | `kan/grid_range`        |
| `classifier.model.kan.spline_regularization_weight` | `kan/spline_reg_weight` |
| `classifier.model.kan.grid_update_freq`             | `kan/grid_update_freq`  |


**axes:** `kan_grid_size`, `kan_spline_order` (subset in `build_axes_metadata`)

---

## 4. Physics-informed attention biases

**Chapter summary**


| Topic                            | Config                              | W&B                                                                                                                                                                                                                                                       | axes                                                                                                                                                                                     |
| -------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Selector                         | `classifier.model.attention_biases` | `bias/selector`                                                                                                                                                                                                                                           | `attention_biases`                                                                                                                                                                       |
| SM / Lorentz / typepair / global | `classifier.model.bias_config.*`    | `bias/sm_mode`, `bias/lorentz_features`, `bias/lorentz_per_head`, `bias/lorentz_sparse_gating`, `bias/typepair_init`, `bias/typepair_freeze`, `bias/typepair_kinematic_gate`, `bias/global_mode`, `kan/bias_lorentz_mlp_type`, `kan/bias_global_mlp_type` | `sm_mode`, `lorentz_features`, `lorentz_mlp_type`, `lorentz_per_head`, `lorentz_sparse_gating`, `typepair_init`, `typepair_freeze`, `typepair_kinematic_gate`, `global_conditioned_mode` |


### B01. Bias selector

- **Config:** `classifier.model.attention_biases`
- **W&B:** `bias/selector`
- **axes:** `attention_biases`
- **Settings:** `"none"` · `"lorentz_scalar"` · `"sm_interaction"` · `"typepair_kinematic"` · `"global_conditioned"` · `"+"` combinations
- **Experiments:** `bias_experiments/*`

### B02. SM interaction mode

- **Config:** `classifier.model.bias_config.sm_interaction.mode`
- **W&B:** `bias/sm_mode`
- **axes:** `sm_mode`
- **Prerequisite:** B01 includes `sm_interaction`
- **Experiments:** `sm_progression`

### B03. Lorentz scalar features

- **Config:** `classifier.model.bias_config.lorentz_scalar.features`
- **W&B:** `bias/lorentz_features`
- **axes:** `lorentz_features`
- **Prerequisite:** B01 includes `lorentz_scalar`
- **Experiments:** `lorentz_features`

### B04. Lorentz MLP type

- **Config:** `classifier.model.bias_config.lorentz_scalar.mlp_type`
- **W&B:** `kan/bias_lorentz_mlp_type`
- **axes:** `lorentz_mlp_type`
- **Prerequisite:** B01 includes `lorentz_scalar`
- **Experiments:** `exp_kan_bias`

**Additional Lorentz sub-keys (config-only in table; logged in W&B):**

- `classifier.model.bias_config.lorentz_scalar.per_head` → `bias/lorentz_per_head` → axes `lorentz_per_head`
- `classifier.model.bias_config.lorentz_scalar.sparse_gating` → `bias/lorentz_sparse_gating` → axes `lorentz_sparse_gating`
- `hidden_dim`, `features` list — see YAML

### B05. Type-pair initialization

- **Config:** `classifier.model.bias_config.typepair_kinematic.init_from_physics`
- **W&B:** `bias/typepair_init`
- **axes:** `typepair_init`
- **Prerequisite:** B01 includes `typepair_kinematic`
- **Experiments:** `interpretability`

### B06. Type-pair freeze

- **Config:** `classifier.model.bias_config.typepair_kinematic.freeze_table`
- **W&B:** `bias/typepair_freeze`
- **axes:** `typepair_freeze`
- **Prerequisite:** B01 includes `typepair_kinematic`

**Additional typepair keys:**

- `kinematic_gate`, `kinematic_feature`, `mask_value` — see `[transformer.yaml](../configs/classifier/model/transformer.yaml)`; `kinematic_gate` → `bias/typepair_kinematic_gate` → axes `typepair_kinematic_gate`

### B07. Global-conditioned bias mode

- **Config:** `classifier.model.bias_config.global_conditioned.mode`
- **W&B:** `bias/global_mode`
- **axes:** `global_conditioned_mode`
- **Settings:** `global_scale` · `met_direction`
- **Prerequisite:** B01 includes `global_conditioned`; `**met_direction`** is intended when MET signal is available (align with **D05**).
- **Experiments:** `met_treatment`

**Global-conditioned MLP type:**

- `classifier.model.bias_config.global_conditioned.mlp_type` → `kan/bias_global_mlp_type`

---

## 5. Pre-encoder modules (before the main transformer encoder)

**Chapter summary**


| Topic              | Config                             | W&B                                               | axes                           |
| ------------------ | ---------------------------------- | ------------------------------------------------- | ------------------------------ |
| Nodewise mass      | `classifier.model.nodewise_mass.*` | `nodewise_mass/enabled`, `nodewise_mass/k_values` | `nodewise_mass_enabled`        |
| MIA (MIParT-style) | `classifier.model.mia_blocks.*`    | `mia/enabled`, `mia/num_blocks`, `mia/placement`  | `mia_enabled`, `mia_placement` |


### P01. Nodewise mass patch

- **Config:** `classifier.model.nodewise_mass.enabled` (+ `k_values`, `hidden_dim`)
- **W&B:** `nodewise_mass/enabled`, `nodewise_mass/k_values`
- **axes:** `nodewise_mass_enabled`
- **Prerequisite:** `cont_dim ≥ 4` (energy **E** in continuous features)
- **Experiments:** `single_module_sweep` (CLI)

### P02. MIA encoder

- **Config:** `classifier.model.mia_blocks.enabled` (+ `num_blocks`, `interaction_dim`, `reduction_dim`, `dropout`)
- **W&B:** `mia/enabled`, `mia/num_blocks`
- **axes:** `mia_enabled`
- **Experiments:** `single_module_sweep` (CLI)

### P03. MIA placement

- **Config:** `classifier.model.mia_blocks.placement`
- **W&B:** `mia/placement`
- **axes:** `mia_placement`
- **Settings:** `prepend` · `append` · `interleave` (config accepts `interleave`; **implementation currently warns and falls back to `prepend`** — see `[src/thesis_ml/architectures/transformer_classifier/base.py](../src/thesis_ml/architectures/transformer_classifier/base.py)`)
- **Prerequisite:** P02 = `true`

---

## S. Shared pairwise backbone (bridge: pairwise features for biases / MIA)

Not a numbered axis in the original 33-count; **pairwise computation** for physics biases.

- **Config:** `classifier.model.shared_backbone.enabled`, `classifier.model.shared_backbone.features`
- **W&B:** Not in primary `extract_wandb_config()` keys; may appear under `**raw/classifier.model.shared_backbone/*`** via flatten catch-all.
- **axes:** Not in `build_axes_metadata()` (consider adding if you need stable filters).

---

## L. Legacy attention pairwise

- **Config:** `classifier.model.attn_pairwise.*`
- **W&B:** `raw/...` unless logged elsewhere
- **Behavior:** If `attention_biases` is `"none"` and legacy flags match, maps toward Lorentz-style bias (see comments in `[transformer.yaml](../configs/classifier/model/transformer.yaml)`).

---

## Hyperparameter axes (often fixed or swept separately)

Studied in: `model_size_sweep_4t_vs_bg` (H01–H04), `overfitting_regularization_sweep` (H05–H07).


| ID  | Axis             | Config key                        | W&B key                              | axes             |
| --- | ---------------- | --------------------------------- | ------------------------------------ | ---------------- |
| H01 | Model dimension  | `classifier.model.dim`            | `model/dim`                          | `dim`            |
| H02 | Encoder depth    | `classifier.model.depth`          | `model/depth`                        | `depth`          |
| H03 | Attention heads  | `classifier.model.heads`          | `model/heads`                        | `heads`          |
| H04 | FFN hidden dim   | `classifier.model.mlp_dim`        | `model/mlp_dim`                      | `mlp_dim`        |
| H05 | Dropout          | `classifier.model.dropout`        | `model/dropout`                      | `dropout`        |
| H06 | Learning rate    | `classifier.trainer.lr`           | `training/lr`                        | `lr`             |
| H07 | Weight decay     | `classifier.trainer.weight_decay` | `training/weight_decay`              | `weight_decay`   |
| H08 | Batch size       | `classifier.trainer.batch_size`   | `training/batch_size`                | `batch_size`     |
| H09 | Random seed      | `classifier.trainer.seed`         | `training/seed`                      | `seed`           |
| H10 | Model size label | derived `d{dim}_L{depth}`         | `model/size_key`, `model/size_label` | `model_size_key` |


**Additional training keys (W&B):** `training/epochs`, `training/label_smoothing`, `training/warmup_steps`, `training/lr_schedule`, `training/grad_clip`, `early_stop/enabled`, `early_stop/patience`

---

## T. Training protocol — PID embedding schedule (not in the 33 axes)

Used for **PID embedding** studies (`pid_deepdive`).


| Config                                             | W&B                              |
| -------------------------------------------------- | -------------------------------- |
| `classifier.trainer.pid_schedule.mode`             | `pid/schedule_mode`              |
| `classifier.trainer.pid_schedule.transition_epoch` | `pid/transition_epoch`           |
| `classifier.trainer.pid_schedule.reinit_mode`      | (raw / add to extract if needed) |
| `classifier.trainer.pid_schedule.pid_lr`           | (raw)                            |
| `classifier.trainer.log_pid_embeddings`            | (raw)                            |


**axes:** Not in `build_axes_metadata()` today.

---

## Interpretability artifacts (training loop)

- **Config:** `classifier.trainer.interpretability.*`
- **W&B:** Primarily under `raw/*` unless extended in `extract_wandb_config()`

---

## Changelog (vs original slide-era doc)

- Added **chapter summary tables** with **Config / W&B / axes** columns.
- Aligned **D02/D03** prerequisites with code: `**raw`** tokenizer does not use PID embeddings.
- Documented **D07b** RoPE base, **A15–A17** MoE extras, **K01–K05** global KAN, **S** shared backbone, **L** legacy pairwise, **T** PID schedule.
- **P03:** documented `interleave` **not implemented** (falls back to `prepend`).
- Cross-linked implementation files for W&B and facts.
