# Deliverable 2 — Per-axis V2 derivation specification

One section per V2 leaf in document order, following the T1-b template from the planning prompt. Every rule below is derived from [docs/AXES_REFERENCE_V2.md](../AXES_REFERENCE_V2.md) and the config-path lookup table in [01_audit.md](01_audit.md) §2.1. The 10 doc ambiguities are discharged with the assumptions stated in [00_doc_issues.md](00_doc_issues.md).

## 0. Global rules (apply to every section)

1. **Missing representation.** The only value emitted for "not applicable" or "missing config" is the empty string `""`. Never `None`, `NaN`, `"NA"`, `null`, `0`, `-1`, or the axis default.
2. **Short-circuit prerequisite.** The parent gate is evaluated first using V2-derived parent values. If any gate condition is false, emit `""` immediately and do not read any config path for this leaf.
3. **V2 defaults are not filler.** An axis default is emitted only when the prerequisite is satisfied and the config path is absent from `run.config`. This case is rare (<5% of gated runs per audit) and must be flagged under `missing_but_expected` in the report CSV.
4. **Topological evaluation.** Execute in this order:
   `G01 → G02 → G03 → D01 → D02 → D03 → T1 → T1-a → T1-b → T1-c → E1 → E1-a → E1-a1 → E1-b → A1..A5 → A3-a → F1(effective) → F1-a → F1-a1 → F1-b → C2 → C1 → P1 → P1-a..b → P2 → P2-a..e → B1 → B1-L1..L5 → B1-T1..T5 → B1-S1..S2 → B1-G1..G3 → K1..K5 → M1..M5 → S1..S2 → H01..H05 → H10 → R01..R13 → R14..R17 → L1..L7`.
5. **Config-source chain (RAW-FIRST).** Per thesis-author direction (see [00_doc_issues.md](00_doc_issues.md) Global Resolution), every leaf reads from the first non-empty match in this priority list: *(a)* `raw/classifier/...` catch-all key (authoritative when present), *(b)* namespaced slice key (`tokenizer/...`, `model/...`, `training/...`, etc.), *(c)* legacy `axes/<key>` mirror. If none yield a non-empty value, treat the config path as absent. **This priority supersedes any per-section ordering shown below** — per-section bullets preserve the V2-doc order for traceability but the implementation must read raw-first.
6. **Value normalisation.** All V2 `axes/*` values are written as **strings** (possibly empty). Integers are emitted as canonical base-10 strings; booleans as lowercase `"true"`/`"false"`; lists as the literal `str()` form with spaces removed (e.g. `"[0,1,2,3]"`); floats passed through; enums as their lowercase string form.
7. **Write semantics.** V2 overwrites any existing `axes/*` value. Legacy non-V2 keys (`meta.*`, `raw/*`, `tokenizer/*`, `model/*`, etc.) are never touched.

---

## §1 G — Study framing

### G01 · Task type
**Target W&B key(s):** `axes/task_type` (new; V2 doc has no explicit `axes_key` for G01 but the leaf needs a column).
**Config source:** `model/loop` → fallback `raw/loop` → fallback `meta.model_family` (derived).
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:**
&nbsp;&nbsp;1. Read `model/loop`. If absent, read fallback chain.
&nbsp;&nbsp;2. If all sources empty, emit `""`.
&nbsp;&nbsp;3. Otherwise emit the stored string lowercased (e.g. `"transformer_classifier"`).
**Value normalisation:** lowercase string, no trimming beyond stripping whitespace.
**Coverage expectation:** ~83.6% non-empty (tier-1 and above).

### G02 · Model family
**Target W&B key(s):** `axes/model_family` (new; mirrors `model/type`).
**Config source:** `model/type` → else derived from G01 via the inference rules in [wandb_utils.py](../../src/thesis_ml/utils/wandb_utils.py) L114–131.
**Parent:** G01.
**Prerequisite:** G01 non-empty.
**Emit rule:**
&nbsp;&nbsp;1. If G01 is `""`, emit `""`.
&nbsp;&nbsp;2. Read `model/type`. If present, emit lowercased.
&nbsp;&nbsp;3. Otherwise derive from G01: `"transformer"` if G01 contains `"transformer"`, `"mlp"` if `"mlp"`, `"bdt"` if `"bdt"`, `"autoencoder"` if G01 contains `"ae"`/`"autoencoder"`/`"gan"`/`"diffusion"`; else emit G01 unchanged.
**Value normalisation:** lowercase string ∈ {`transformer`, `mlp`, `bdt`, `autoencoder`}.
**Coverage expectation:** ~83.6%.

### G03 · Classification task
**Target W&B key(s):** `axes/class_def` (new; mirrors `meta.class_def_str`).
**Config source:** `meta.class_def_str` → fallback `meta.process_groups_key` → fallback `meta.row_key`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:**
&nbsp;&nbsp;1. Emit `meta.class_def_str` if present (100% of runs). Else chain down.
&nbsp;&nbsp;2. If all empty, emit `""`.
**Value normalisation:** string pass-through.
**Coverage expectation:** 100%.

---

## §2 D — Data treatment

### D01 · Feature set
**Target W&B key(s):** `axes/cont_features`.
**Config source:** `data/cont_features` → `raw/data/cont_features` → legacy `axes/cont_features`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:**
&nbsp;&nbsp;1. Read first non-empty source in chain.
&nbsp;&nbsp;2. If all empty, emit `""`.
&nbsp;&nbsp;3. Normalise to `"[0,1,2,3]"` form (strip spaces) so set-membership tests in P1 are deterministic.
**Value normalisation:** string form of sorted-or-preserved int list, no spaces.
**Coverage expectation:** ~60% (union of tier-2 and tier-3).

### D02 · MET treatment
**Target W&B key(s):** `axes/include_met`.
**Config source:** `globals/include_met` → `raw/classifier/globals/include_met` → legacy `axes/include_met`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:**
&nbsp;&nbsp;1. Read first non-empty source.
&nbsp;&nbsp;2. Cast to lowercase boolean string.
**Value normalisation:** `"true"` / `"false"`. Non-boolean raw values (e.g. `1`, `"True"`) are mapped.
**Coverage expectation:** ~57.3%.

### D03 · Token ordering
**Target W&B key(s):** `axes/token_order`.
**Config source:** `data/token_order` (preferred, 100% presence) → derive from `data/shuffle_tokens` + `data/sort_tokens_by` per `_infer_token_order` in [facts/axes.py](../../src/thesis_ml/facts/axes.py) L35–42.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:**
&nbsp;&nbsp;1. If `data/token_order` present, emit it.
&nbsp;&nbsp;2. Else: if `data/shuffle_tokens == true` → `"shuffled"`; elif `data/sort_tokens_by == "pt"` → `"pt_sorted"`; else `"input_order"`.
**Value normalisation:** enum string ∈ {`input_order`, `pt_sorted`, `shuffled`}.
**Coverage expectation:** 100%.

---

## §3 T — Tokenizer

### T1 · Tokenizer family
**Target W&B key(s):** `axes/tokenizer_name`.
**Config source:** `tokenizer/type` → `raw/classifier/model/tokenizer/name` → legacy `axes/tokenizer_name`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:**
&nbsp;&nbsp;1. Read first non-empty source, lowercase.
&nbsp;&nbsp;2. If none, emit `""`.
**Value normalisation:** enum string ∈ {`raw`, `identity`, `binned`, `pretrained`}.
**Coverage expectation:** ~57.3%.

### T1-a · PID embedding mode
**Target W&B key(s):** `axes/pid_mode`.
**Config source:** `tokenizer/pid_mode` → `raw/classifier/model/tokenizer/pid_mode` → legacy `axes/pid_mode`.
**Parent:** T1.
**Prerequisite:** `T1 == "identity"`.
**Emit rule:**
&nbsp;&nbsp;1. Compute T1 first.
&nbsp;&nbsp;2. If `T1 != "identity"` → emit `""`. Stop.
&nbsp;&nbsp;3. Read config chain. If present, emit lowercase string ∈ {`learned`, `one_hot`, `fixed_random`}.
&nbsp;&nbsp;4. If absent, emit `""` and flag `missing_but_expected`.
**Value normalisation:** lowercase enum.
**Coverage expectation:** subset of T1=identity runs (audit: est. ~50–60% of tier ≥1 runs).

### T1-b · PID embedding dimension
**Target W&B key(s):** `axes/id_embed_dim`.
**Config source:** `raw/classifier/model/tokenizer/id_embed_dim` → `tokenizer/id_embed_dim` → legacy `axes/id_embed_dim`.
**Parent:** T1 (secondary: T1-a).
**Prerequisite:** `T1 == "identity"`.
**Emit rule:**
&nbsp;&nbsp;1. Compute T1 first.
&nbsp;&nbsp;2. If `T1 != "identity"` → emit `""`. Stop.
&nbsp;&nbsp;3. Override: if V2 `T1-a == "one_hot"`, emit the literal string `"num_types"` (see I-1). The stored `id_embed_dim` is runtime-overridden to `num_types` by the tokenizer code and does not reflect the trained model.
&nbsp;&nbsp;4. Otherwise read config chain. If present, emit integer as string (e.g. `"8"`).
&nbsp;&nbsp;5. If prereq satisfied but config absent and no override, emit `""` and flag `missing_but_expected`.
**Value normalisation:** string digit-form integer, or the literal `"num_types"` under override.
**Coverage expectation:** subset of T1=identity runs, with `"num_types"` override on the T1-a=one_hot subset.

### T1-c · Pretrained model type
**Target W&B key(s):** `axes/pretrained_model_type` (new; per I-5).
**Config source:** `raw/classifier/model/tokenizer/model_type` → `tokenizer/model_type`.
**Parent:** T1.
**Prerequisite:** `T1 == "pretrained"`.
**Emit rule:**
&nbsp;&nbsp;1. If `T1 != "pretrained"` → emit `""`.
&nbsp;&nbsp;2. Read chain. Emit stored value as lowercase string.
&nbsp;&nbsp;3. If prereq satisfied and config absent, emit default `"vq"` (per I-5).
**Value normalisation:** enum ∈ {`vq`, ...} (string).
**Coverage expectation:** <5%.

---

## §4 E — Positional encoding

### E1 · PE type
**Target W&B key(s):** `axes/positional`.
**Config source:** `pos_enc/type` → `raw/classifier/model/positional` → legacy `axes/positional`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:** Read chain, emit lowercase string ∈ {`none`, `sinusoidal`, `learned`, `rotary`}. If all empty, `""`.
**Value normalisation:** lowercase enum.
**Coverage expectation:** ~78.9%.

### E1-a · PE space
**Target W&B key(s):** `axes/positional_space`.
**Config source:** `pos_enc/space` → `raw/classifier/model/positional_space` → legacy `axes/positional_space`.
**Parent:** E1.
**Prerequisite:** `E1 ∈ {"sinusoidal", "learned"}`.
**Emit rule:**
&nbsp;&nbsp;1. If E1 not in the gate set, emit `""`.
&nbsp;&nbsp;2. Read chain; emit `"model"` or `"token"`.
&nbsp;&nbsp;3. If prereq satisfied but config absent, emit the default `"model"` and flag `missing_but_expected`.
**Value normalisation:** enum ∈ {`model`, `token`}.
**Coverage expectation:** subset of E1 ∈ {sinusoidal, learned} runs, approx 40–50%.

### E1-a1 · PE dimension mask
**Target W&B key(s):** `axes/positional_dim_mask`.
**Config source:** `pos_enc/dim_mask` → `raw/classifier/model/positional_dim_mask` → legacy `axes/positional_dim_mask`.
**Parent:** E1-a.
**Prerequisite:** `E1-a == "token"`.
**Emit rule:**
&nbsp;&nbsp;1. If E1-a != `"token"`, emit `""`.
&nbsp;&nbsp;2. Read chain. If stored is None/null → emit literal `"null"` (per I-8 — "null" means "all dims", a meaningful axis value, not missing).
&nbsp;&nbsp;3. Else emit stored list form without spaces.
**Value normalisation:** string. Either `"null"`, or a list-form like `"[eta]"`, or a preset name.
**Coverage expectation:** small subset of E1-a=token runs.

### E1-b · Rotary base frequency
**Target W&B key(s):** `axes/rotary_base` (new; V2 has no `axes_key` but the column is needed per I-5-style addition).
**Config source:** `pos_enc/rotary_base` → `raw/classifier/model/rotary/base`.
**Parent:** E1.
**Prerequisite:** `E1 == "rotary"`.
**Emit rule:**
&nbsp;&nbsp;1. If E1 != `"rotary"`, emit `""`.
&nbsp;&nbsp;2. Read chain, emit float as string (e.g. `"10000.0"`).
&nbsp;&nbsp;3. If prereq satisfied but config absent, emit default `"10000.0"` and flag.
**Value normalisation:** float-as-string.
**Coverage expectation:** rotary-only subset (~5–10% per audit).

---

## §5 P — Pre-encoder modules

### P1 · Nodewise mass enabled
**Target W&B key(s):** `axes/nodewise_mass_enabled`.
**Config source:** `nodewise_mass/enabled` → `raw/classifier/model/nodewise_mass/enabled` → legacy `axes/nodewise_mass_enabled`.
**Parent:** T1, D01.
**Prerequisite:** `(T1 ∈ {"raw", "identity"}) ∧ (0 ∈ parsed D01)`.
**Emit rule:**
&nbsp;&nbsp;1. Evaluate gate: parse D01 as JSON int list (tolerant of `str(list)` form per I-2); gate false if `0` not in set, or T1 outside allowed set, or D01 parse fails.
&nbsp;&nbsp;2. If gate false, emit `""`.
&nbsp;&nbsp;3. Else read chain. Emit `"true"`/`"false"`.
&nbsp;&nbsp;4. If prereq satisfied but config absent, emit default `"false"`.
**Value normalisation:** lowercase boolean string.
**Coverage expectation:** subset of raw/identity runs with energy feature.

### P1-a · Nodewise neighbourhood sizes
**Target W&B key(s):** `axes/nodewise_mass_k_values` (new).
**Config source:** `nodewise_mass/k_values` → `raw/classifier/model/nodewise_mass/k_values`.
**Parent:** P1.
**Prerequisite:** `P1 == "true"`.
**Emit rule:**
&nbsp;&nbsp;1. If P1 != `"true"`, emit `""`.
&nbsp;&nbsp;2. Read chain; emit list-as-string (`"[2,4,8]"`).
&nbsp;&nbsp;3. If prereq satisfied and config absent, emit default `"[2,4,8]"` and flag.
**Value normalisation:** list-as-string.
**Coverage expectation:** P1=true subset.

### P1-b · Nodewise hidden dimension
**Target W&B key(s):** `axes/nodewise_mass_hidden_dim` (new).
**Config source:** `raw/classifier/model/nodewise_mass/hidden_dim` only (no namespaced mirror exists).
**Parent:** P1.
**Prerequisite:** `P1 == "true"`.
**Emit rule:**
&nbsp;&nbsp;1. If P1 != `"true"`, emit `""`.
&nbsp;&nbsp;2. Read raw key; emit int-as-string.
&nbsp;&nbsp;3. If prereq satisfied and absent, emit default `"64"` and flag.
**Value normalisation:** int-as-string.
**Coverage expectation:** P1=true subset.

### P2 · MIA pre-encoder enabled
**Target W&B key(s):** `axes/mia_enabled`.
**Config source:** `mia/enabled` → `raw/classifier/model/mia_blocks/enabled` → legacy `axes/mia_enabled`.
**Parent:** T1.
**Prerequisite:** `T1 ∈ {"raw", "identity"}`.
**Emit rule:**
&nbsp;&nbsp;1. If T1 outside gate set, emit `""`.
&nbsp;&nbsp;2. Read chain; emit `"true"`/`"false"`.
&nbsp;&nbsp;3. If absent, emit default `"false"`.
**Value normalisation:** boolean string.
**Coverage expectation:** raw/identity subset.

### P2-a · MIA placement
**Target W&B key(s):** `axes/mia_placement`.
**Config source:** `mia/placement` → `raw/classifier/model/mia_blocks/placement` → legacy `axes/mia_placement`.
**Parent:** P2.
**Prerequisite:** `P2 == "true"`.
**Emit rule:**
&nbsp;&nbsp;1. If P2 != `"true"`, emit `""`.
&nbsp;&nbsp;2. Read chain. Normalise `"interleave"` → `"prepend"` (V2 §9: interleave falls back to prepend).
&nbsp;&nbsp;3. If absent, emit default `"prepend"`.
**Value normalisation:** enum ∈ {`prepend`, `append`}.
**Coverage expectation:** P2=true subset.

### P2-b · MIA number of blocks
**Target W&B key(s):** `axes/mia_num_blocks` (new).
**Config source:** `mia/num_blocks` → `raw/classifier/model/mia_blocks/num_blocks`.
**Parent:** P2.
**Prerequisite:** `P2 == "true"`.
**Emit rule:** Standard gated-int; default `"5"` on missing-with-prereq.
**Value normalisation:** int-as-string.
**Coverage expectation:** P2=true subset.

### P2-c · MIA interaction dimension
**Target W&B key(s):** `axes/mia_interaction_dim` (new).
**Config source:** `raw/classifier/model/mia_blocks/interaction_dim` only.
**Parent:** P2.
**Prerequisite:** `P2 == "true"`.
**Emit rule:** Gated-int; default `"64"` on missing-with-prereq.
**Value normalisation:** int-as-string.
**Coverage expectation:** P2=true subset.

### P2-d · MIA reduction dimension
**Target W&B key(s):** `axes/mia_reduction_dim` (new).
**Config source:** `raw/classifier/model/mia_blocks/reduction_dim` only.
**Parent:** P2.
**Prerequisite:** `P2 == "true"`.
**Emit rule:** Gated-int; default `"8"` on missing-with-prereq.
**Value normalisation:** int-as-string.
**Coverage expectation:** P2=true subset.

### P2-e · MIA dropout
**Target W&B key(s):** `axes/mia_dropout` (new).
**Config source:** `raw/classifier/model/mia_blocks/dropout` only.
**Parent:** P2.
**Prerequisite:** `P2 == "true"`.
**Emit rule:** Gated-float; default `"0.0"` on missing-with-prereq.
**Value normalisation:** float-as-string.
**Coverage expectation:** P2=true subset.

---

## §6 A — Attention & encoder block

### A1 · Normalization policy
**Target W&B key(s):** `axes/norm_policy`.
**Config source:** `norm/policy` → `raw/classifier/model/norm/policy` → legacy `axes/norm_policy`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:** Read chain, emit enum string ∈ {`pre`, `post`, `normformer`}; `""` if all empty.
**Value normalisation:** lowercase enum.
**Coverage expectation:** ~78.9%.

### A2 · Normalization type
**Target W&B key(s):** `axes/norm_type`.
**Config source:** `norm/type` → `raw/classifier/model/norm/type` → legacy `axes/norm_type`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:** Read chain, emit ∈ {`layernorm`, `rmsnorm`}; `""` if all empty.
**Value normalisation:** lowercase enum.
**Coverage expectation:** ~78.9% (assumed co-present with A1).

### A3 · Attention type
**Target W&B key(s):** `axes/attention_type`.
**Config source:** `attention/type` → `raw/classifier/model/attention/type` → legacy `axes/attention_type`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:** Read chain, emit ∈ {`standard`, `differential`}; `""` if all empty.
**Value normalisation:** lowercase enum.
**Coverage expectation:** ~45.4%.

### A3-a · Differential attention bias mode
**Target W&B key(s):** `axes/diff_bias_mode`.
**Config source:** `attention/diff_bias_mode` → `raw/classifier/model/attention/diff_bias_mode` → legacy `axes/diff_bias_mode`.
**Parent:** A3.
**Prerequisite:** `A3 == "differential"`.
**Emit rule:**
&nbsp;&nbsp;1. If A3 != `"differential"`, emit `""`.
&nbsp;&nbsp;2. Read chain, emit ∈ {`none`, `shared`, `split`}.
&nbsp;&nbsp;3. If absent, emit default `"shared"` and flag.
**Value normalisation:** lowercase enum.
**Coverage expectation:** A3=differential subset.

### A4 · Attention-internal normalization
**Target W&B key(s):** `axes/attention_norm`.
**Config source:** `attention/norm` → `raw/classifier/model/attention/norm` → legacy `axes/attention_norm`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:** Read chain, emit ∈ {`none`, `layernorm`, `rmsnorm`}; `""` if all empty. Default is `"none"` but only emit default when a non-axis config source confirms the architecture has attention — if config absent across the board we have no evidence, emit `""`.
**Value normalisation:** lowercase enum.
**Coverage expectation:** ~45.4%.

### A5 · Causal masking
**Target W&B key(s):** `axes/causal_attention`.
**Config source:** `model/causal_attention` → `raw/classifier/model/causal_attention` → legacy `axes/causal_attention`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:** Read chain, emit boolean string.
**Value normalisation:** `"true"`/`"false"`.
**Coverage expectation:** ~57.3%.

---

## §7 F — Encoder FFN realization

### F1 · Encoder FFN (raw + effective)
**Target W&B key(s):** `axes/ffn_type`, `axes/moe_enabled`, plus a new `axes/ffn_realization` per I-4.
**Config source:**
- `ffn_type`: `ffn/type` → `raw/classifier/model/ffn/type` → legacy `axes/ffn_type`.
- `moe_enabled`: `moe/enabled` → `raw/classifier/model/moe/enabled` → legacy `axes/moe_enabled`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:**
&nbsp;&nbsp;1. Read `ffn_type` stored value lowercase (`"standard"`/`"kan"` or `""`).
&nbsp;&nbsp;2. Read `moe_enabled` stored value as boolean string.
&nbsp;&nbsp;3. Derive `axes/ffn_realization`:
&nbsp;&nbsp;&nbsp;&nbsp;- if `moe_enabled == "true"` → `"moe"`;
&nbsp;&nbsp;&nbsp;&nbsp;- elif `ffn_type == "kan"` → `"kan"`;
&nbsp;&nbsp;&nbsp;&nbsp;- elif either is non-empty → `"standard"`;
&nbsp;&nbsp;&nbsp;&nbsp;- else `""`.
**Value normalisation:** `axes/ffn_type` string, `axes/moe_enabled` boolean string, `axes/ffn_realization` enum string.
**Coverage expectation:** ~45.4% (requires tier-3 or raw-layer data).

### F1-a · KAN FFN variant
**Target W&B key(s):** `axes/kan_ffn_variant`.
**Config source:** `ffn/kan_variant` → `raw/classifier/model/ffn/kan/variant` → legacy `axes/kan_ffn_variant`.
**Parent:** F1.
**Prerequisite:** `axes/ffn_realization == "kan"`.
**Emit rule:**
&nbsp;&nbsp;1. If `ffn_realization != "kan"`, emit `""`.
&nbsp;&nbsp;2. Read chain, emit ∈ {`hybrid`, `bottleneck`, `pure`}.
&nbsp;&nbsp;3. If absent, emit default `"hybrid"` and flag.
**Value normalisation:** lowercase enum.
**Coverage expectation:** KAN-FFN subset.

### F1-a1 · KAN FFN bottleneck dimension
**Target W&B key(s):** `axes/kan_ffn_bottleneck_dim` (new).
**Config source:** `raw/classifier/model/ffn/kan/bottleneck_dim` only.
**Parent:** F1-a.
**Prerequisite:** `F1-a == "bottleneck"`.
**Emit rule:**
&nbsp;&nbsp;1. If F1-a != `"bottleneck"`, emit `""`.
&nbsp;&nbsp;2. Read raw; emit int-as-string or literal `"null"` (means auto = mlp_dim//4).
**Value normalisation:** int-as-string or `"null"`.
**Coverage expectation:** F1-a=bottleneck subset.

### F1-b · MoE encoder scope
**Target W&B key(s):** `axes/moe_scope`.
**Config source:** `moe/scope` → `raw/classifier/model/moe/scope` → legacy `axes/moe_scope`.
**Parent:** F1.
**Prerequisite:** `axes/ffn_realization == "moe"`.
**Emit rule:**
&nbsp;&nbsp;1. If `ffn_realization != "moe"`, emit `""`.
&nbsp;&nbsp;2. Read chain, emit ∈ {`all_blocks`, `middle_blocks`, `head`}.
&nbsp;&nbsp;3. If absent, emit default `"all_blocks"`.
**Value normalisation:** lowercase enum.
**Coverage expectation:** MoE-FFN subset (small).

---

## §8 C — Classifier head (C2 before C1 per V2 §3 ordering)

### C2 · Pooling strategy
**Target W&B key(s):** `axes/pooling`.
**Config source:** `pooling/type` → `raw/classifier/model/head/pooling` → fallback `raw/classifier/model/pooling` (legacy top-level key) → legacy `axes/pooling`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:** Read chain, emit ∈ {`cls`, `mean`, `max`}.
**Value normalisation:** lowercase enum.
**Coverage expectation:** ~78.9%.

### C1 · Head realization
**Target W&B key(s):** `axes/head_type`.
**Config source:** `head/type` → `raw/classifier/model/head/type` → legacy `axes/head_type`.
**Parent:** (root).
**Prerequisite:** None.
**Emit rule:** Read chain, emit ∈ {`linear`, `kan`, `moe`}.
**Value normalisation:** lowercase enum.
**Coverage expectation:** ~45.4%.

---

## §9 B — Physics-informed attention biases

### B1 · Bias activation set
**Target W&B key(s):** `axes/attention_biases`.
**Config source:** `raw/classifier/model/attention_biases` → `bias/selector` → legacy `axes/attention_biases`. **No legacy attn_pairwise remap** — per I-3 (RESOLVED), zero runs have `attn_pairwise.enabled=true`, so the remap is unreachable and is dropped.
**Parent:** T1.
**Prerequisite:** `T1 ∈ {"raw", "identity"}`.
**Emit rule:**
&nbsp;&nbsp;1. If T1 outside gate set, emit `""`.
&nbsp;&nbsp;2. Read chain.
&nbsp;&nbsp;3. If non-empty, emit raw `+`-combined string lowercase (e.g. `"lorentz_scalar+typepair_kinematic"`).
&nbsp;&nbsp;4. If absent, emit `""` (pre-bias-era runs — 428 of them — have no bias system; empty is the correct V2 value, not the default `"none"`).
**Value normalisation:** lowercase `+`-joined string.
**Coverage expectation:** ~574 raw/identity runs with bias config; 428 pre-bias-era runs emit `""`.

### B1-L1 · Lorentz feature set
**Target W&B key(s):** `axes/lorentz_features`.
**Config source:** `raw/classifier/model/bias_config/lorentz_scalar/features` → `bias/lorentz_features` → legacy `axes/lorentz_features`. No attn_pairwise fallback (I-3 RESOLVED).
**Parent:** B1.
**Prerequisite:** `"lorentz_scalar" ∈ B1`.
**Emit rule:**
&nbsp;&nbsp;1. If substring test fails, emit `""`.
&nbsp;&nbsp;2. Read chain, emit list-as-string without spaces.
&nbsp;&nbsp;3. If absent, emit default `"[m2,deltaR]"` and flag.
**Value normalisation:** list-as-string, space-stripped.
**Coverage expectation:** lorentz-on subset.

### B1-L2 · Lorentz MLP type
**Target W&B key(s):** `axes/lorentz_mlp_type`.
**Config source:** `kan/bias_lorentz_mlp_type` → `raw/classifier/model/bias_config/lorentz_scalar/mlp_type` → legacy `axes/lorentz_mlp_type`.
**Parent:** B1.
**Prerequisite:** `"lorentz_scalar" ∈ B1`.
**Emit rule:** Gated, read chain, emit ∈ {`standard`, `kan`}; default `"standard"` on missing-with-prereq.
**Value normalisation:** lowercase enum.
**Coverage expectation:** lorentz-on subset.

### B1-L3 · Lorentz hidden dimension
**Target W&B key(s):** `axes/lorentz_hidden_dim` (new).
**Config source:** `raw/classifier/model/bias_config/lorentz_scalar/hidden_dim` only.
**Parent:** B1.
**Prerequisite:** `"lorentz_scalar" ∈ B1`.
**Emit rule:** Gated-int; default `"8"` on missing-with-prereq.
**Value normalisation:** int-as-string.
**Coverage expectation:** lorentz-on subset.

### B1-L4 · Lorentz per-head mode
**Target W&B key(s):** `axes/lorentz_per_head`.
**Config source:** `bias/lorentz_per_head` → `raw/classifier/model/bias_config/lorentz_scalar/per_head` → legacy `axes/lorentz_per_head`.
**Parent:** B1.
**Prerequisite:** `"lorentz_scalar" ∈ B1`.
**Emit rule:** Gated-bool; default `"false"` on missing-with-prereq.
**Value normalisation:** `"true"`/`"false"`.
**Coverage expectation:** lorentz-on subset.

### B1-L5 · Lorentz sparse gating
**Target W&B key(s):** `axes/lorentz_sparse_gating`.
**Config source:** `bias/lorentz_sparse_gating` → `raw/classifier/model/bias_config/lorentz_scalar/sparse_gating` → legacy `axes/lorentz_sparse_gating`.
**Parent:** B1.
**Prerequisite:** `"lorentz_scalar" ∈ B1`.
**Emit rule:** Gated-bool; default `"false"` on missing-with-prereq.
**Value normalisation:** `"true"`/`"false"`.
**Coverage expectation:** lorentz-on subset.

### B1-T1 · Type-pair initialization
**Target W&B key(s):** `axes/typepair_init`.
**Config source:** `bias/typepair_init` → `raw/.../typepair_kinematic/init_from_physics` → legacy `axes/typepair_init`.
**Parent:** B1.
**Prerequisite:** `"typepair_kinematic" ∈ B1`.
**Emit rule:** Gated; enum ∈ {`none`, `binary`, `fixed_coupling`}; default `"none"` on missing-with-prereq.
**Value normalisation:** lowercase enum.
**Coverage expectation:** typepair-on subset.

### B1-T2 · Type-pair freeze table
**Target W&B key(s):** `axes/typepair_freeze`.
**Config source:** `bias/typepair_freeze` → `raw/.../typepair_kinematic/freeze_table` → legacy `axes/typepair_freeze`.
**Parent:** B1.
**Prerequisite:** `"typepair_kinematic" ∈ B1`.
**Emit rule:** Gated-bool; default `"false"`.
**Value normalisation:** boolean string.
**Coverage expectation:** typepair-on subset.

### B1-T3 · Type-pair kinematic gate
**Target W&B key(s):** `axes/typepair_kinematic_gate`.
**Config source:** `bias/typepair_kinematic_gate` → `raw/.../typepair_kinematic/kinematic_gate` → legacy `axes/typepair_kinematic_gate`.
**Parent:** B1.
**Prerequisite:** `"typepair_kinematic" ∈ B1`.
**Emit rule:** Gated-bool; default `"true"`.
**Value normalisation:** boolean string.
**Coverage expectation:** typepair-on subset.

### B1-T4 · Type-pair kinematic feature
**Target W&B key(s):** `axes/typepair_kinematic_feature` (new).
**Config source:** `raw/.../typepair_kinematic/kinematic_feature` only.
**Parent:** B1.
**Prerequisite:** `"typepair_kinematic" ∈ B1`.
**Emit rule:** Gated-string; default `"log_m2"`.
**Value normalisation:** lowercase string.
**Coverage expectation:** typepair-on subset.

### B1-T5 · Type-pair mask value
**Target W&B key(s):** `axes/typepair_mask_value` (new).
**Config source:** `raw/.../typepair_kinematic/mask_value` only.
**Parent:** B1.
**Prerequisite:** `"typepair_kinematic" ∈ B1`.
**Emit rule:** Gated-float; default `"-5.0"`.
**Value normalisation:** float-as-string.
**Coverage expectation:** typepair-on subset.

### B1-S1 · SM interaction mode
**Target W&B key(s):** `axes/sm_mode`.
**Config source:** `bias/sm_mode` → `raw/.../sm_interaction/mode` → legacy `axes/sm_mode`.
**Parent:** B1.
**Prerequisite:** `"sm_interaction" ∈ B1`.
**Emit rule:** Gated; enum ∈ {`binary`, `fixed_coupling`, `running_coupling`}; default `"binary"`.
**Value normalisation:** lowercase enum.
**Coverage expectation:** sm-on subset.

### B1-S2 · SM mask value
**Target W&B key(s):** `axes/sm_mask_value` (new).
**Config source:** `raw/.../sm_interaction/mask_value` only.
**Parent:** B1.
**Prerequisite:** `"sm_interaction" ∈ B1`.
**Emit rule:** Gated-float; default `"-100.0"`.
**Value normalisation:** float-as-string.
**Coverage expectation:** sm-on subset.

### B1-G1 · Global-conditioned mode
**Target W&B key(s):** `axes/global_conditioned_mode`.
**Config source:** `bias/global_mode` → `raw/.../global_conditioned/mode` → legacy `axes/global_conditioned_mode`.
**Parent:** B1, D02.
**Prerequisite:** `("global_conditioned" ∈ B1)` and, if stored mode is `"met_direction"`, additionally `D02 == "true"`.
**Emit rule:**
&nbsp;&nbsp;1. If `"global_conditioned"` not in B1, emit `""`.
&nbsp;&nbsp;2. Read chain. If stored is `"met_direction"` and D02 != `"true"`, this run is misconfigured per V2; emit `""` and flag `unresolved_flags = "B1-G1: met_direction without MET"`.
&nbsp;&nbsp;3. Otherwise emit stored value; default `"global_scale"`.
**Value normalisation:** enum ∈ {`global_scale`, `met_direction`}.
**Coverage expectation:** global-cond-on subset (small).

### B1-G2 · Global-conditioned MLP type
**Target W&B key(s):** `axes/global_conditioned_mlp_type` (new).
**Config source:** `kan/bias_global_mlp_type` → `raw/.../global_conditioned/mlp_type`.
**Parent:** B1.
**Prerequisite:** `"global_conditioned" ∈ B1`.
**Emit rule:** Gated; enum ∈ {`standard`, `kan`}; default `"standard"`.
**Value normalisation:** lowercase enum.
**Coverage expectation:** global-cond-on subset.

### B1-G3 · Global-conditioned global dimension
**Target W&B key(s):** `axes/global_conditioned_global_dim` (new).
**Config source:** `raw/.../global_conditioned/global_dim` only.
**Parent:** B1.
**Prerequisite:** `"global_conditioned" ∈ B1`.
**Emit rule:** Gated-int; default `"16"`.
**Value normalisation:** int-as-string.
**Coverage expectation:** global-cond-on subset.

---

## §10 H — Model-size hyperparameters (all root, no prereqs)

### H01 · Model dimension
**Target W&B key(s):** `axes/dim`. **Config source:** `model/dim` → `raw/classifier/model/dim` → legacy `axes/dim`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** Read chain, emit int-as-string. **Value normalisation:** int-as-string. **Coverage expectation:** ~79.3%.

### H02 · Encoder depth
**Target W&B key(s):** `axes/depth`. **Config source:** `model/depth` → `raw/classifier/model/depth` → legacy `axes/depth`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** Read chain, emit int-as-string. **Value normalisation:** int-as-string. **Coverage expectation:** ~79.3%.

### H03 · Attention heads
**Target W&B key(s):** `axes/heads`. **Config source:** `model/heads` → `raw/classifier/model/heads` → legacy `axes/heads`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** Read chain, emit int-as-string. **Value normalisation:** int-as-string. **Coverage expectation:** ~79.3%.

### H04 · FFN hidden dimension
**Target W&B key(s):** `axes/mlp_dim`. **Config source:** `model/mlp_dim` → `raw/classifier/model/mlp_dim` → legacy `axes/mlp_dim`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** Read chain, emit int-as-string. **Value normalisation:** int-as-string. **Coverage expectation:** ~57.3–79.3% (varies by tier — `model/mlp_dim` is only on 57.3%).

### H05 · Dropout
**Target W&B key(s):** `axes/dropout`. **Config source:** `model/dropout` → `raw/classifier/model/dropout` → legacy `axes/dropout`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** Read chain, emit float-as-string. **Value normalisation:** float-as-string. **Coverage expectation:** ~79.3%.

### H10 · Model size label
**Target W&B key(s):** `axes/model_size_key`.
**Config source:** derived from H01 and H02.
**Parent:** H01, H02.
**Prerequisite:** H01 non-empty ∧ H02 non-empty.
**Emit rule:**
&nbsp;&nbsp;1. If H01 or H02 empty, emit `""`.
&nbsp;&nbsp;2. Coerce to `int` via `int(float(x))`; if coercion fails, emit `""` and flag.
&nbsp;&nbsp;3. Emit `f"d{dim}_L{depth}"`.
**Value normalisation:** string literal form, e.g. `"d128_L4"`.
**Coverage expectation:** ~79.3%.

---

## §11 §K — KAN shared hyperparameters

**Shared KAN-consumer gate (reused in K1..K5):** any of F1-a non-empty, C1 == `"kan"`, B1-L2 == `"kan"`, B1-G2 == `"kan"`.

### K1 · KAN grid size
**Target W&B key(s):** `axes/kan_grid_size`. **Config source:** `kan/grid_size` → `raw/classifier/model/kan/grid_size` → legacy `axes/kan_grid_size`. **Parent:** F1-a, C1, B1-L2, B1-G2. **Prerequisite:** shared KAN-consumer gate. **Emit rule:** Gated-int; `""` if gate fails. **Value normalisation:** int-as-string. **Coverage expectation:** KAN-consumer subset.

### K2 · KAN spline order
**Target W&B key(s):** `axes/kan_spline_order`. **Config source:** `kan/spline_order` → `raw/classifier/model/kan/spline_order` → legacy `axes/kan_spline_order`. **Parent:** same. **Prerequisite:** same. **Emit rule:** Gated-int. **Value normalisation:** int-as-string. **Coverage expectation:** KAN-consumer subset.

### K3 · KAN grid range
**Target W&B key(s):** `axes/kan_grid_range` (new). **Config source:** `kan/grid_range` → `raw/classifier/model/kan/grid_range`. **Parent:** same. **Prerequisite:** same. **Emit rule:** Gated; list-as-string. **Value normalisation:** list-as-string (space-stripped). **Coverage expectation:** KAN-consumer subset.

### K4 · KAN spline regularization weight
**Target W&B key(s):** `axes/kan_spline_reg_weight` (new). **Config source:** `kan/spline_reg_weight` → `raw/classifier/model/kan/spline_regularization_weight`. **Parent:** same. **Prerequisite:** same. **Emit rule:** Gated-float. **Value normalisation:** float-as-string. **Coverage expectation:** KAN-consumer subset.

### K5 · KAN grid update frequency
**Target W&B key(s):** `axes/kan_grid_update_freq` (new). **Config source:** `kan/grid_update_freq` → `raw/classifier/model/kan/grid_update_freq`. **Parent:** same. **Prerequisite:** same. **Emit rule:** Gated-int. **Value normalisation:** int-as-string. **Coverage expectation:** KAN-consumer subset.

---

## §12 §M — MoE shared hyperparameters

**Shared MoE-consumer gate (reused in M1..M5):** (`axes/ffn_realization == "moe"`) ∨ (`axes/head_type == "moe"`).

### M1 · Number of experts
**Target W&B key(s):** `axes/moe_num_experts` (new). **Config source:** `moe/num_experts` → `raw/classifier/model/moe/num_experts`. **Parent:** F1, C1. **Prerequisite:** MoE-consumer gate. **Emit rule:** Gated-int. **Value normalisation:** int-as-string. **Coverage expectation:** MoE subset.

### M2 · Top-k
**Target W&B key(s):** `axes/moe_top_k`. **Config source:** `moe/top_k` → `raw/classifier/model/moe/top_k` → legacy `axes/moe_top_k`. **Parent:** same. **Prerequisite:** same. **Emit rule:** Gated-int. **Value normalisation:** int-as-string. **Coverage expectation:** MoE subset.

### M3 · Routing level
**Target W&B key(s):** `axes/moe_routing_level`. **Config source:** `moe/routing_level` → `raw/classifier/model/moe/routing_level` → legacy `axes/moe_routing_level`. **Parent:** same. **Prerequisite:** same. **Emit rule:** Gated-enum. **Value normalisation:** lowercase string. **Coverage expectation:** MoE subset.

### M4 · Load-balance weight
**Target W&B key(s):** `axes/moe_lb_weight` (new). **Config source:** `moe/lb_weight` → `raw/classifier/model/moe/load_balance_loss_weight`. **Parent:** same. **Prerequisite:** same. **Emit rule:** Gated-float. **Value normalisation:** float-as-string. **Coverage expectation:** MoE subset.

### M5 · Noisy gating
**Target W&B key(s):** `axes/moe_noisy_gating` (new). **Config source:** `moe/noisy_gating` → `raw/classifier/model/moe/noisy_gating`. **Parent:** same. **Prerequisite:** same. **Emit rule:** Gated-bool. **Value normalisation:** boolean string. **Coverage expectation:** MoE subset.

---

## §13 §S — Shared pairwise backbone

**Shared backbone-consumer gate (reused in S1..S2):** (∃ B1 family active ⇒ B1 non-empty and non-`"none"`) ∨ (P2 == `"true"`).

### S1 · Backbone enabled
**Target W&B key(s):** `axes/shared_backbone_enabled` (new). **Config source:** `raw/classifier/model/shared_backbone/enabled` only. **Parent:** B1, P2. **Prerequisite:** shared backbone-consumer gate. **Emit rule:** Gated-bool; default `"false"` only when gate true and config missing. **Value normalisation:** boolean string. **Coverage expectation:** B1/P2 subset.

### S2 · Backbone features
**Target W&B key(s):** `axes/shared_backbone_features` (new). **Config source:** `raw/classifier/model/shared_backbone/features` only. **Parent:** same. **Prerequisite:** same. **Emit rule:** Gated; list-as-string. **Value normalisation:** list-as-string. **Coverage expectation:** B1/P2 subset.

---

## §14 §R — Training protocol

### R01 · Epochs
**Target W&B key(s):** `axes/epochs`. **Config source:** `training/epochs` → `raw/classifier/trainer/epochs` → legacy `axes/epochs`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** int-as-string. **Value normalisation:** int-as-string. **Coverage expectation:** ~79.7%.

### R02 · Learning rate
**Target W&B key(s):** `axes/lr`. **Config source:** `training/lr` → `raw/classifier/trainer/lr` → legacy `axes/lr`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** float-as-string. **Value normalisation:** float-as-string. **Coverage expectation:** ~79.7%.

### R03 · Weight decay
**Target W&B key(s):** `axes/weight_decay`. **Config source:** `training/weight_decay` → `raw/classifier/trainer/weight_decay` → legacy `axes/weight_decay`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** float-as-string. **Value normalisation:** float-as-string. **Coverage expectation:** ~79.7%.

### R04 · Batch size
**Target W&B key(s):** `axes/batch_size`. **Config source:** `training/batch_size` → `raw/classifier/trainer/batch_size` → legacy `axes/batch_size`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** int-as-string. **Value normalisation:** int-as-string. **Coverage expectation:** ~79.7%.

### R05 · Seed
**Target W&B key(s):** `axes/seed`. **Config source:** `training/seed` → `raw/classifier/trainer/seed` → legacy `axes/seed`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** int-as-string. **Value normalisation:** int-as-string. **Coverage expectation:** ~79.7%.

### R06 · Warmup steps
**Target W&B key(s):** `axes/warmup_steps` (new per I-6). **Config source:** `training/warmup_steps` → `raw/classifier/trainer/warmup_steps`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** int-as-string. **Value normalisation:** int-as-string. **Coverage expectation:** ~79.7%.

### R07 · LR schedule
**Target W&B key(s):** `axes/lr_schedule` (new). **Config source:** `training/lr_schedule` → `raw/classifier/trainer/lr_schedule`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** string pass-through, lowercased. **Value normalisation:** lowercase string. **Coverage expectation:** ~79.7%.

### R08 · Label smoothing
**Target W&B key(s):** `axes/label_smoothing` (new). **Config source:** `training/label_smoothing` → `raw/classifier/trainer/label_smoothing`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** float-as-string. **Value normalisation:** float-as-string. **Coverage expectation:** ~79.7%.

### R09 · Gradient clipping
**Target W&B key(s):** `axes/grad_clip` (new). **Config source:** `training/grad_clip` → `raw/classifier/trainer/grad_clip`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** float-as-string. **Value normalisation:** float-as-string. **Coverage expectation:** ~79.7%.

### R10 · Early stopping enabled
**Target W&B key(s):** `axes/early_stop_enabled` (new). **Config source:** `early_stop/enabled` → `raw/classifier/trainer/early_stopping/enabled`. **Parent:** (root). **Prerequisite:** None. **Emit rule:** boolean string. **Value normalisation:** `"true"`/`"false"`. **Coverage expectation:** ~64.9%.

### R11 · Early stopping patience
**Target W&B key(s):** `axes/early_stop_patience` (new). **Config source:** `early_stop/patience` → `raw/classifier/trainer/early_stopping/patience`. **Parent:** R10. **Prerequisite:** `R10 == "true"`. **Emit rule:** Gated-int; default per V2 on missing-with-prereq. **Value normalisation:** int-as-string. **Coverage expectation:** early-stop-on subset.

### R12 · Early stopping min delta
**Target W&B key(s):** `axes/early_stop_min_delta` (new). **Config source:** `raw/classifier/trainer/early_stopping/min_delta` only. **Parent:** R10. **Prerequisite:** `R10 == "true"`. **Emit rule:** Gated-float. **Value normalisation:** float-as-string. **Coverage expectation:** early-stop-on subset.

### R13 · Restore best weights
**Target W&B key(s):** `axes/early_stop_restore_best` (new). **Config source:** `raw/classifier/trainer/early_stopping/restore_best_weights` only. **Parent:** R10. **Prerequisite:** `R10 == "true"`. **Emit rule:** Gated-bool. **Value normalisation:** boolean string. **Coverage expectation:** early-stop-on subset.

### R14 · PID schedule mode
**Target W&B key(s):** `axes/pid_schedule_mode` (new). **Config source:** `pid/schedule_mode` → `raw/classifier/trainer/pid_schedule/mode`. **Parent:** T1, T1-a. **Prerequisite:** `(T1 == "identity") ∧ (T1-a == "learned")`. **Emit rule:** Gated-string. **Value normalisation:** lowercase string. **Coverage expectation:** PID-learned subset (small).

### R15 · Transition epoch
**Target W&B key(s):** `axes/pid_transition_epoch` (new). **Config source:** `pid/transition_epoch` → `raw/classifier/trainer/pid_schedule/transition_epoch`. **Parent:** same. **Prerequisite:** same. **Emit rule:** Gated-int. **Value normalisation:** int-as-string. **Coverage expectation:** PID-learned subset.

### R16 · Reinit mode
**Target W&B key(s):** `axes/pid_reinit_mode` (new). **Config source:** `raw/classifier/trainer/pid_schedule/reinit_mode` only. **Parent:** same. **Prerequisite:** same. **Emit rule:** Gated-string. **Value normalisation:** lowercase string. **Coverage expectation:** PID-learned subset.

### R17 · PID separate LR
**Target W&B key(s):** `axes/pid_lr` (new). **Config source:** `raw/classifier/trainer/pid_schedule/pid_lr` only. **Parent:** same. **Prerequisite:** same. **Emit rule:** Gated-float. **Value normalisation:** float-as-string. **Coverage expectation:** PID-learned subset.

---

## §15 §L — Logging / interpretability

All §L leaves are root (no prereqs). Stored in `raw/*` only. New V2 target columns per I-6.

### L1 · Log PID embeddings
**Target W&B key(s):** `axes/log_pid_embeddings` (new). **Config source:** `raw/classifier/trainer/log_pid_embeddings` only. **Parent:** (root). **Prerequisite:** None. **Emit rule:** boolean string. **Value normalisation:** `"true"`/`"false"`. **Coverage expectation:** ~57.3%.

### L2 · Interpretability enabled
**Target W&B key(s):** `axes/interp_enabled` (new). **Config source:** `raw/classifier/trainer/interpretability/enabled` only. **Parent:** (root). **Prerequisite:** None. **Emit rule:** boolean string. **Value normalisation:** `"true"`/`"false"`. **Coverage expectation:** ~57.3%.

### L3 · Save attention maps
**Target W&B key(s):** `axes/interp_save_attention_maps` (new). **Config source:** `raw/classifier/trainer/interpretability/save_attention_maps` only. **Parent:** L2. **Prerequisite:** `L2 == "true"`. **Emit rule:** Gated-bool. **Value normalisation:** `"true"`/`"false"`. **Coverage expectation:** L2-on subset.

### L4 · Save KAN splines
**Target W&B key(s):** `axes/interp_save_kan_splines` (new). **Config source:** `raw/classifier/trainer/interpretability/save_kan_splines` only. **Parent:** L2. **Prerequisite:** `L2 == "true"`. **Emit rule:** Gated-bool. **Value normalisation:** `"true"`/`"false"`. **Coverage expectation:** L2-on subset.

### L5 · Save MoE routing
**Target W&B key(s):** `axes/interp_save_moe_routing` (new). **Config source:** `raw/classifier/trainer/interpretability/save_moe_routing` only. **Parent:** L2. **Prerequisite:** `L2 == "true"`. **Emit rule:** Gated-bool. **Value normalisation:** `"true"`/`"false"`. **Coverage expectation:** L2-on subset.

### L6 · Save gradient norms
**Target W&B key(s):** `axes/interp_save_gradient_norms` (new). **Config source:** `raw/classifier/trainer/interpretability/save_gradient_norms` only. **Parent:** L2. **Prerequisite:** `L2 == "true"`. **Emit rule:** Gated-bool. **Value normalisation:** `"true"`/`"false"`. **Coverage expectation:** L2-on subset.

### L7 · Checkpoint epochs
**Target W&B key(s):** `axes/interp_checkpoint_epochs` (new). **Config source:** `raw/classifier/trainer/interpretability/checkpoint_epochs` only. **Parent:** L2. **Prerequisite:** `L2 == "true"`. **Emit rule:** Gated; list-as-string. **Value normalisation:** list-as-string. **Coverage expectation:** L2-on subset.

---

## 16. Unresolved rules

All 10 doc-issue items (I-1 through I-10) are resolved — see [00_doc_issues.md](00_doc_issues.md) for per-item decisions. Zero blocking items remain.

Material changes from the prior plan-assumption draft:

1. **I-1 T1-b override:** confirmed literal `"num_types"` (the stored `id_embed_dim` is overridden at runtime by the tokenizer constructor).
2. **I-3 legacy attn_pairwise:** **remap dropped.** Zero runs in W&B have `attn_pairwise.enabled=true`, making the backward-compat path unreachable. B1 becomes a pure read, and 428 pre-bias-era runs emit `""` for B1.
3. **I-4 F1 effective realization:** confirmed — sources from raw/* `moe/enabled` and `ffn/type`, MoE > KAN > standard. Both legacy `axes/ffn_type`/`axes/moe_enabled` values are kept plus a new `axes/ffn_realization` column for the effective value.
4. **Config-source priority flipped to raw-first project-wide** (see §0 Rule 5 and Global Resolution in 00_doc_issues.md).

All 93 axes have a single deterministic derivation rule.
