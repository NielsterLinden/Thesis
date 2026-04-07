# AXES Audit Log

This log records the audit used to freeze `docs/AXES_REFERENCE.md`.

## Authority Order
1. `configs/classifier/model/transformer.yaml`
2. `configs/classifier/trainer/default.yaml`
3. `src/thesis_ml/architectures/transformer_classifier/base.py`
4. `src/thesis_ml/facts/axes.py`
5. `src/thesis_ml/utils/wandb_utils.py`
6. Experiment configs under `configs/classifier/experiment/`
7. March 31 slides and the legacy attached `AXES_REFERENCE.md`

## Final Decisions

### Core Count
- The frozen thesis core remains **33 numbered axes**:
  `D01-D09`, `A01-A14`, `B01-B07`, `P01-P03`.
- `H01-H10` stay documented as hyperparameter axes, but outside the 33-axis architectural count.
- KAN globals, PID schedule, shared backbone, and legacy pairwise paths stay documented as appendices or sub-settings, not numbered axes.

### Resolved Conflicts
- `D03` stays **PID embedding dimension** because it has a real config key, a canonical `axes/id_embed_dim`, and appears in the W&B export. PID schedule is logged separately under `pid/*` and remains a training-protocol appendix.
- `B05` and `B06` stay split as **type-pair initialization** and **type-pair freeze** because they are separate config keys and separate `axes/*` keys.
- `typepair_kinematic_gate`, `lorentz_per_head`, and `lorentz_sparse_gating` remain **documented sub-settings**, not numbered axes. They are logged and configurable, but not the stable thesis core.
- `classifier.model.shared_backbone.*` remains **infrastructure**, not a numbered axis. It is consumed by code but not surfaced as a canonical `axes/*` selector.
- `classifier.model.attn_pairwise.*` remains **legacy compatibility**. Code can still map it into Lorentz bias behavior, but it should not define the frozen design space.
- `A15-A17` from the previous repo doc were demoted from numbered axes to **MoE sub-settings**. The numbered architecture block is restored to `A01-A14`.

### Slide Mismatches
- The March 31 slides use `D03` for PID schedule, while the frozen reference uses `D03` for PID embedding dimension.
- The slides bundle type-pair initialization, freezing, and kinematic gate differently from the frozen reference.
- The slides mention `25 orthogonal axes`, which is stale relative to the current code and logging surface. The frozen repo reference is `33 core axes` plus hyperparameters and appendices.

## Core Cross-Reference Matrix

Legend:
- `CSV`: present in the audited CSV export
- `Slides`: reflected in the March 31 deck, even if grouped or numbered differently
- `Status`: `Consistent`, `Mismatch`, or `Needs note`

### Input Representation
| Axis | Config | `axes/*` | Structured W&B | CSV | Slides | Status | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D01 | `classifier.model.tokenizer.name` | `tokenizer_name` | `tokenizer/type` | Yes | Yes | Consistent | Slides and code agree on the concept |
| D02 | `classifier.model.tokenizer.pid_mode` | `pid_mode` | `tokenizer/pid_mode` | Yes | Yes | Consistent | Only meaningful with `identity` tokenizer |
| D03 | `classifier.model.tokenizer.id_embed_dim` | `id_embed_dim` | `tokenizer/id_embed_dim` | Yes | No | Needs note | Slides reused D03 for PID schedule |
| D04 | `data.cont_features` | `cont_features` | `data/cont_features` | Yes | Yes | Consistent | Loader defaults still apply when omitted |
| D05 | `classifier.globals.include_met` | `include_met` | `globals/include_met` | Yes | Yes | Consistent | Separate from global-conditioned bias mode |
| D06 | `data.sort_tokens_by`, `data.shuffle_tokens` | `token_order` | `data/sort_tokens_by`, `data/shuffle_tokens` | Yes | Yes | Consistent | `axes/token_order` is derived |

### Positional Encoding
| Axis | Config | `axes/*` | Structured W&B | CSV | Slides | Status | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D07 | `classifier.model.positional` | `positional` | `pos_enc/type` | Yes | Yes | Consistent | Includes `rotary` |
| D08 | `classifier.model.positional_space` | `positional_space` | `pos_enc/space` | Yes | Yes | Consistent | Mostly relevant for additive PE |
| D09 | `classifier.model.positional_dim_mask` | `positional_dim_mask` | `pos_enc/dim_mask` | No | Yes | Needs note | W&B export had no `axes/positional_dim_mask` column in the audited CSV |

### Transformer Architecture
| Axis | Config | `axes/*` | Structured W&B | CSV | Slides | Status | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A01 | `classifier.model.norm.policy` | `norm_policy` | `norm/policy` | Yes | Yes | Consistent | — |
| A02 | `classifier.model.norm.type` | `norm_type` | `norm/type` | Yes | Yes | Consistent | — |
| A03 | `classifier.model.attention.type` | `attention_type` | `attention/type` | Yes | Yes | Consistent | — |
| A04 | `classifier.model.attention.norm` | `attention_norm` | `attention/norm` | Yes | Yes | Consistent | Slides used A04/A05 differently across pages |
| A05 | `classifier.model.attention.diff_bias_mode` | `diff_bias_mode` | `attention/diff_bias_mode` | Yes | Yes | Needs note | Stable in code, but slide phrasing varies |
| A06 | `classifier.model.head.pooling` | `pooling` | `pooling/type` | Yes | Yes | Consistent | Code still supports legacy `classifier.model.pooling` |
| A07 | `classifier.model.causal_attention` | `causal_attention` | `model/causal_attention` | Yes | Yes | Consistent | — |
| A08 | `classifier.model.ffn.type` | `ffn_type` | `ffn/type` | Yes | Yes | Consistent | — |
| A09 | `classifier.model.ffn.kan.variant` | `kan_ffn_variant` | `ffn/kan_variant` | Yes | Yes | Consistent | `bottleneck_dim` remains a sub-setting |
| A10 | `classifier.model.head.type` | `head_type` | `head/type` | Yes | Yes | Consistent | `moe` head still uses shared `moe.*` block |
| A11 | `classifier.model.moe.enabled` | `moe_enabled` | `moe/enabled` | Yes | Yes | Consistent | — |
| A12 | `classifier.model.moe.top_k` | `moe_top_k` | `moe/top_k` | Yes | Yes | Consistent | — |
| A13 | `classifier.model.moe.routing_level` | `moe_routing_level` | `moe/routing_level` | Yes | Yes | Consistent | — |
| A14 | `classifier.model.moe.scope` | `moe_scope` | `moe/scope` | Yes | Yes | Consistent | Previous repo doc over-promoted extra MoE settings to A15-A17 |

### Physics-Informed Attention Biases
| Axis | Config | `axes/*` | Structured W&B | CSV | Slides | Status | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B01 | `classifier.model.attention_biases` | `attention_biases` | `bias/selector` | Yes | Yes | Consistent | String selector is canonical |
| B02 | `classifier.model.bias_config.sm_interaction.mode` | `sm_mode` | `bias/sm_mode` | Yes | Yes | Consistent | Slides grouped this differently on some pages |
| B03 | `classifier.model.bias_config.lorentz_scalar.features` | `lorentz_features` | `bias/lorentz_features` | Yes | Yes | Consistent | Lorentz extras remain sub-settings |
| B04 | `classifier.model.bias_config.lorentz_scalar.mlp_type` | `lorentz_mlp_type` | `kan/bias_lorentz_mlp_type` | Yes | Yes | Needs note | Structured key uses `kan/` prefix, not `bias/` |
| B05 | `classifier.model.bias_config.typepair_kinematic.init_from_physics` | `typepair_init` | `bias/typepair_init` | Yes | Yes | Consistent | Kept split from B06 |
| B06 | `classifier.model.bias_config.typepair_kinematic.freeze_table` | `typepair_freeze` | `bias/typepair_freeze` | Yes | Yes | Consistent | Kept split from B05 |
| B07 | `classifier.model.bias_config.global_conditioned.mode` | `global_conditioned_mode` | `bias/global_mode` | Yes | Yes | Needs note | Slide numbering for globals differs |

### Pre-Encoder Modules
| Axis | Config | `axes/*` | Structured W&B | CSV | Slides | Status | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P01 | `classifier.model.nodewise_mass.enabled` | `nodewise_mass_enabled` | `nodewise_mass/enabled` | Yes | Yes | Consistent | Requires energy in continuous features |
| P02 | `classifier.model.mia_blocks.enabled` | `mia_enabled` | `mia/enabled` | Yes | Yes | Consistent | — |
| P03 | `classifier.model.mia_blocks.placement` | `mia_placement` | `mia/placement` | Yes | Yes | Needs note | `interleave` falls back to `prepend` in code |

## Logged Sub-Settings Matrix

| Item | Config | `axes/*` | Structured W&B | CSV | Decision |
| --- | --- | --- | --- | --- | --- |
| Rotary base | `classifier.model.rotary.base` | No | `pos_enc/rotary_base` | No | Keep as D07 sub-setting |
| KAN grid size | `classifier.model.kan.grid_size` | `kan_grid_size` | `kan/grid_size` | Yes | Keep as KAN appendix |
| KAN spline order | `classifier.model.kan.spline_order` | `kan_spline_order` | `kan/spline_order` | Yes | Keep as KAN appendix |
| MoE num experts | `classifier.model.moe.num_experts` | No | `moe/num_experts` | No | Keep as A11-A14 sub-setting |
| MoE load-balance weight | `classifier.model.moe.load_balance_loss_weight` | No | `moe/lb_weight` | No | Keep as A11-A14 sub-setting |
| MoE noisy gating | `classifier.model.moe.noisy_gating` | No | `moe/noisy_gating` | No | Keep as A11-A14 sub-setting |
| Lorentz per-head | `classifier.model.bias_config.lorentz_scalar.per_head` | `lorentz_per_head` | `bias/lorentz_per_head` | Yes | Keep as B03/B04 sub-setting |
| Lorentz sparse gating | `classifier.model.bias_config.lorentz_scalar.sparse_gating` | `lorentz_sparse_gating` | `bias/lorentz_sparse_gating` | Yes | Keep as B03/B04 sub-setting |
| Typepair kinematic gate | `classifier.model.bias_config.typepair_kinematic.kinematic_gate` | `typepair_kinematic_gate` | `bias/typepair_kinematic_gate` | Yes | Keep as B05/B06 sub-setting |
| Nodewise mass details | `classifier.model.nodewise_mass.k_values`, `hidden_dim` | No | `nodewise_mass/k_values` | No | Keep as P01 sub-settings |
| MIA details | `classifier.model.mia_blocks.num_blocks`, `interaction_dim`, `reduction_dim`, `dropout` | No | `mia/num_blocks` | No | Keep as P02/P03 sub-settings |
| PID schedule mode | `classifier.trainer.pid_schedule.mode` | No | `pid/schedule_mode` | Yes | Keep as training appendix |
| PID transition epoch | `classifier.trainer.pid_schedule.transition_epoch` | No | `pid/transition_epoch` | Yes | Keep as training appendix |
| PID reinit and PID LR | `classifier.trainer.pid_schedule.reinit_mode`, `pid_lr` | No | raw only | No | Keep as training appendix |
| Shared backbone | `classifier.model.shared_backbone.enabled`, `features` | No | raw only | No | Keep as infrastructure appendix |
| Legacy pairwise | `classifier.model.attn_pairwise.*` | No | raw only | No | Keep as legacy appendix |

## Follow-Up Plan

These items are intentionally out of scope for the documentation freeze, but they are the concrete next steps if richer W&B or CSV tracking is desired.

### Logging And Export Follow-Ups
- Decide whether the CSV export should mirror the full W&B config surface or remain a curated subset.
- If the export should become richer, add first-class columns for:
  `tokenizer/*`, `pooling/type`, `head/type`, `globals/include_met`, `data/cont_features`, `data/shuffle_tokens`, `data/sort_tokens_by`, `moe/num_experts`, `moe/lb_weight`, `moe/noisy_gating`, `kan/*`, `nodewise_mass/k_values`, `mia/num_blocks`, `pos_enc/rotary_base`, `kan/bias_lorentz_mlp_type`, and `kan/bias_global_mlp_type`.
- Consider adding explicit extraction for `pid/reinit_mode` and `pid/pid_lr` if PID schedule comparisons need to be filterable without relying on `raw/*`.
- Decide whether `axes/positional_dim_mask` should be exported consistently in CSV when present, since it exists in `build_axes_metadata()` but was absent from the audited export header.

### Code-Level Follow-Ups
- If shared backbone should become a user-facing axis, add canonical `axes/*` coverage for `classifier.model.shared_backbone.*` before promoting it in the docs.
- If `interleave` for MIA placement should remain documented as a real setting, implement it in `src/thesis_ml/architectures/transformer_classifier/base.py`; otherwise remove it from the accepted config values.
- If legacy `attn_pairwise` should eventually disappear, deprecate it more aggressively and remove the compatibility path from `bias_composer.py` and `base.py` after old checkpoints are migrated.

### Slide Follow-Ups
- Update the slide deck so `D03` no longer conflicts with the frozen reference.
- Update the bias-group slides so type-pair initialization and freezing are shown as separate numbered axes, with kinematic gating shown as a sub-setting.
- Replace the stale `25 orthogonal axes` statement with the frozen count used in the repo documentation.
