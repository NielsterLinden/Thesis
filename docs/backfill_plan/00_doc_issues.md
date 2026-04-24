# V2 doc issues discovered during Deliverable 1 audit

Each item below is a case where [docs/AXES_REFERENCE_V2.md](../AXES_REFERENCE_V2.md) either contradicts observed W&B data, leaves a derivation genuinely ambiguous, or describes a convention that does not match what `extract_wandb_config` / `build_axes_metadata` actually write. All 10 items have been reviewed by the thesis author; resolutions below supersede the earlier "plan assumptions."

## Global resolution: prefer `raw/*` config sources

Per thesis-author direction, the V2 derivation **prefers the `raw/classifier/...` catch-all** over the legacy `axes/*` mirrors and namespaced sliders like `tokenizer/*`, `pos_enc/*`, `pid/*`. Rationale: the original Hydra → W&B mapping in `extract_wandb_config` dropped detail for some leaves, while `raw/*` preserves the unfiltered Hydra dict under a flattened key. The new per-axis config-source priority order is:

1. `raw/classifier/...` catch-all key (authoritative when present).
2. Namespaced slice key (`tokenizer/...`, `model/...`, `training/...`, etc.) — used when `raw/*` is absent but the namespaced slice is present.
3. Legacy `axes/<key>` — last resort.

This priority applies to every section in [02_mapping_spec.md](02_mapping_spec.md) regardless of the listed order of sources in that section.

---

## I-1 (RESOLVED) — T1-b override semantics ("num_types")

**V2 §3 T1-b:** *"Overridden to `num_types` when T1-a = `one_hot`."*

Verified in [src/thesis_ml/architectures/transformer_classifier/modules/tokenizers/identity.py](../../src/thesis_ml/architectures/transformer_classifier/modules/tokenizers/identity.py) L43-55: when `pid_mode == "one_hot"`, the constructor silently forces `id_embed_dim = num_types` regardless of what the configured `id_embed_dim` says. The stored `tokenizer/id_embed_dim` value in such runs is therefore **ignored at runtime** — it's a misleading config artefact, not the effective value.

**Resolution:** Emit the literal string `"num_types"` for T1-b when T1-a = `"one_hot"`. The stored numeric `id_embed_dim` is dropped because it does not reflect the model that actually trained. An analysis consumer needing the effective dimension should look up `meta.num_types` (or equivalent) for that run.

## I-2 (RESOLVED) — D01 energy-membership test

**V2 §2 D01 + §5 P1 prerequisite:** *"D01 includes energy (index 0)."*

**Resolution:** Parse the stored `cont_features` value as JSON (tolerant of Python-list `str(...)` output — strip spaces, then `json.loads`), coerce to `set[int]`, test `0 in set`. If parse fails, emit `""` for P1 and record the run id under `unresolved_flags`.

## I-3 (RESOLVED — REMAP DROPPED) — Legacy `attn_pairwise` ignored

**Code check:** `bias_composer.py` L373-376 contains a backward-compat hook that remaps `attn_pairwise.enabled=true` + `attention_biases="none"` → `"lorentz_scalar"` with features `[m2, deltaR]`. The legacy `PairwiseBiasNet` in `modules/pairwise_bias.py` is mathematically equivalent to `LorentzScalarBias(features=["m2","deltaR"])`.

**Live data check (all 1002 runs):**

| bucket | count |
|---|---:|
| `attn_pairwise` key absent (pre-bias-era runs) | 428 |
| `attn_pairwise.enabled=false` + new `attention_biases` set (lorentz/typepair/etc.) | 293 |
| `attn_pairwise.enabled=false` + `attention_biases="none"` | 281 |
| **`attn_pairwise.enabled=true`** | **0** |

Earliest run with non-`none` `attention_biases` is 2026-03-06 — matches the bias system's ~6-week age.

**Resolution:** No run in the project exercises the backward-compat path. The V2 derivation **completely ignores** `raw/classifier/model/attn_pairwise/*`. B1 becomes a pure read of `raw/classifier/model/attention_biases`:

- Non-empty stored value → emit as-is.
- Stored `"none"` → emit `"none"`.
- `attention_biases` key absent → emit `""` (means "bias system did not exist in this run's code").

This drops the previously-planned legacy feature inference for B1-L1 hidden_dim/features/per_head/etc.; pre-bias-era runs have none of those sub-axes and all B1-* leaves are `""` for them.

## I-4 (RESOLVED) — F1 effective realization from raw configs

**Rule (direct from thesis author):**

```
if raw/classifier/model/moe/enabled == true         → axes/ffn_realization = "moe"
elif raw/classifier/model/ffn/type == "kan"         → axes/ffn_realization = "kan"
else                                                 → axes/ffn_realization = "standard"
```

Also keep `axes/ffn_type` = raw `ffn.type` literal (standard/kan) and `axes/moe_enabled` = raw boolean, so the distinction "config had `ffn.type=kan` but MoE won" is preserved. F1-a, F1-a1, F1-b gate on `axes/ffn_realization`, not on the raw fields.

Source for both reads is `raw/classifier/model/moe/enabled` and `raw/classifier/model/ffn/type`. Namespaced `moe/enabled` and `ffn/type` are used only as fallbacks.

## I-5 (RESOLVED) — T1-c pretrained model_type

**Resolution:** Read `raw/classifier/model/tokenizer/model_type`. When `T1 != "pretrained"`, emit `""`. When `T1 == "pretrained"` and the config path is absent, emit the default `"vq"`. Target column: `axes/pretrained_model_type` (new).

## I-6 (RESOLVED) — §L and §R derive from raw configs

**Resolution:** For every §L and §R leaf that lacks an explicit `axes/*` key in the V2 doc, V2 creates one. Source is the `raw/classifier/trainer/...` catch-all key. No namespaced fallback (per I-6 thesis-author direction: "derive from raw configs"). If the raw key is absent, the V2 target is `""`. Mirror column names follow a consistent convention: `axes/log_<leaf>` for §L, `axes/training_<leaf>` or `axes/early_stop_<leaf>` or `axes/pid_<leaf>` for §R by topic.

## I-7 (RESOLVED) — H10 model_size_key derivation

**Resolution:** Use the `build_axes_metadata`-style form: coerce H01 and H02 to `int` via `int(float(x))`, fall back to stringifying on coercion failure, emit `f"d{dim}_L{depth}"`. Inconsistencies with stored legacy `axes/model_size_key` (which sometimes has `d128.0_L4.0`) are fine — V2 overwrites the legacy column.

## I-8 (RESOLVED) — E1-a1 dim_mask null/empty semantics

**Resolution:** When prereq (E1-a == `"token"`) is satisfied and stored value is `None`/`null`/`""`, emit literal string `"null"`. Empty string `""` is reserved for prerequisite failure.

## I-9 (RESOLVED) — D03 options match `_infer_token_order`

**Resolution:** Follow [src/thesis_ml/facts/axes.py](../../src/thesis_ml/facts/axes.py) L35-42 exactly. Prefer reading `data/token_order` directly (100% presence); fall back to inference from `data/shuffle_tokens` + `data/sort_tokens_by` for edge cases. Output enum is `{input_order, pt_sorted, shuffled}`.

## I-10 (RESOLVED) — G02 model family

**Resolution:** Mirror the inference logic from [src/thesis_ml/utils/wandb_utils.py](../../src/thesis_ml/utils/wandb_utils.py) L114-131 verbatim. Read `model/type` first (raw-config equivalent is `raw/model/type` if present); else infer from G01 (`model/loop`) using the substring heuristics: transformer / mlp / bdt / autoencoder.

---

## Summary

All 10 items resolved. 0 blocking items remain. Three material changes from the previous plan assumptions:

1. **Config-source priority flipped:** `raw/classifier/...` is preferred first, not last. (Applies project-wide.)
2. **I-3 legacy `attn_pairwise` remap dropped:** zero runs use it, so the remap code path is unreachable. B1 is a pure read.
3. **I-4 F1 realization sourced from raw:** no longer reads legacy `axes/ffn_type`/`axes/moe_enabled` unless raw is missing.
