# Deliverable 3 — Extension plan for existing W&B scripts

Diff-level plan for making the V2 backfill a first-class mode of the existing `scripts/wandb/` infrastructure. Target files:

- [scripts/wandb/backfill_labels.py](../../scripts/wandb/backfill_labels.py) (208 lines) — the current stamp-a-constant-label path.
- [scripts/wandb/cleanup_wandb.py](../../scripts/wandb/cleanup_wandb.py) (242 lines) — the current delete-runs-and-artifacts path.

**Filename note.** The planning prompt referred to `scripts/wandb/backfill.py` and `scripts/wandb/cleanupwandb.py`. These do not exist as named. The actual files are `backfill_labels.py` and `cleanup_wandb.py`. All diffs below target the actual filenames.

## 1. Split vs extend decision

Based on the audit ([01_audit.md](01_audit.md) §6), the default "extend" choice is viable but risks muddying a small, well-scoped file. The existing `backfill_labels` takes a single global `{key: value}` dict and writes it to every matching run — a pure stamp operation. The V2 backfill computes a per-run dict via `derive_v2_axes(run.config)` with topological ordering and prerequisite gating. These are semantically different operations.

**Decision: extend, not split.** `backfill_labels.py` becomes a thin dispatcher between two subcommands:

- `stamp` (default, current behaviour) — kept backward-compatible via a positional-less CLI where `--labels` is required.
- `v2` (new) — the V2 derivation path; required mode for V2 backfill.

Rationale: a second file duplicates imports, argparse boilerplate, logging, and the `api.runs(...)` iteration. Extending keeps all W&B iteration code in one place. The V2 path imports its derivation logic from a new `scripts/wandb/v2_axes.py`, which is pure and testable in isolation.

If during Phase A the extended script exceeds ~500 lines or tests become entangled, fall back to `scripts/wandb/backfill_v2.py` as a sibling file that imports the shared iteration helpers. Make that call at the Phase A review gate, not earlier.

## 2. New module — `scripts/wandb/v2_axes.py`

Pure-Python module. No W&B imports. Importable from tests.

### 2.1 Public surface

- `V2_AXES: list[V2Axis]` — ordered list of all 93 axes from [02_mapping_spec.md](02_mapping_spec.md). Each `V2Axis` is a lightweight dataclass with fields `id`, `target_key`, `config_sources`, `parent_ids`, `prereq_fn`, `emit_fn`, `normalise_fn`. `prereq_fn` and `emit_fn` are small lambdas / named functions, not strings.
- `topological_sort(axes: list[V2Axis]) -> list[V2Axis]` — Kahn-algorithm sort using `parent_ids`. Idempotent. Raises on cycle (V2 tree is acyclic).
- `derive_v2_axes(run_config: dict, *, flag_bucket: dict | None = None) -> dict[str, str]` — pure. Returns `{target_key: str_value}` where every value is a string and empty strings are allowed. If `flag_bucket` is provided, fills in three subkeys: `keys_left_empty_by_prereq` (list of ids), `keys_empty_missing_config` (list), `unresolved_flags` (list of `"<id>: <reason>"`).

### 2.2 V2 axis registry form

Pseudocode for the per-leaf entries (conceptual, not literal code):

```
V2Axis(
  id="T1-b",
  target_key="axes/id_embed_dim",
  config_sources=[
    "tokenizer/id_embed_dim",
    "raw/classifier/model/tokenizer/id_embed_dim",
    "axes/id_embed_dim",  # legacy tier-3 fallback
  ],
  parent_ids=["T1", "T1-a"],
  prereq_fn=lambda derived: derived["T1"] == "identity",
  emit_fn=lambda cfg, derived, sources: (
    "num_types" if derived["T1-a"] == "one_hot"
    else _read_chain_as_int_str(cfg, sources)
  ),
  normalise_fn=None,  # baked into emit_fn
)
```

One entry per row of [02_mapping_spec.md](02_mapping_spec.md). For root axes, `parent_ids=[]` and `prereq_fn` returns `True`. The registry is built at module import time from a single list literal — no dynamic dispatch, easy to grep.

### 2.3 Derivation algorithm

`derive_v2_axes(cfg, flag_bucket=None)`:

1. `derived: dict[str, str] = {}` — `{axis_id: value_string}`.
2. For each axis in topological order:
   1. If `prereq_fn(derived)` is false:
      - `derived[axis.id] = ""`.
      - Append axis.id to `flag_bucket["keys_left_empty_by_prereq"]`.
      - Continue.
   2. Otherwise run `emit_fn(cfg, derived, axis.config_sources)`.
   3. If the result is `""` but the prereq was satisfied:
      - Append axis.id to `flag_bucket["keys_empty_missing_config"]`.
3. Produce output dict `{axis.target_key: derived[axis.id] for axis in V2_AXES}`.

Helper utilities lowered to module scope: `_read_chain(cfg, sources)` (first non-None/non-empty), `_as_bool_str`, `_as_int_str`, `_as_float_str`, `_as_list_str`, `_parse_cont_features(s) -> set[int]`, `_contains_substring(haystack, needle)` for B1 family tests.

### 2.4 Tests

Add `tests/wandb/test_v2_axes.py` (new directory). Small unit tests, no W&B:

- T1-b returns `""` when `T1=binned`.
- T1-b returns `"num_types"` when `T1=identity` and `T1-a=one_hot`.
- T1-b returns `"8"` when `T1=identity` and `T1-a=learned` and config has `tokenizer/id_embed_dim=8`.
- E1-a1 returns `"null"` when prereq satisfied and config has no dim_mask.
- P1 gate evaluates `0 ∈ cont_features` correctly for list, `str(list)`, and JSON list inputs.
- F1 effective realization priority (MoE > KAN > standard).
- Idempotency: `derive(derive(cfg) | cfg) == derive(cfg)`.
- B1 legacy-mapping: run with `attention_biases="none"` and `attn_pairwise.enabled=true` emits `"lorentz_scalar"`.

## 3. Extensions to `scripts/wandb/backfill_labels.py`

### 3.1 CLI surface

Before:

```
--labels JSON_STR (required)
--project (default thesis-ml)
--entity (default nterlind-nikhef)
--dry-run (default true)
--execute (disables dry-run)
--force (overwrite existing)
--group REGEX
--tag TAG
```

After (new flags marked ▲, changed flags marked ◐):

```
--mode {stamp, v2}                           ▲  default "stamp" (back-compat)
--labels JSON_STR                            ◐  required only when mode=stamp
--project (default thesis-ml)
--entity (default nterlind-nikhef)
--dry-run (default true in both modes)
--execute
--overwrite                                  ▲  alias for --force; in v2 mode defaults to TRUE
--no-overwrite                               ▲  explicit opt-out in v2 mode
--group REGEX
--experiment-filter REGEX                    ▲  alias for --group for thesis-facing docs
--tag TAG
--limit N                                    ▲
--run-ids-file PATH                          ▲  newline-delimited wandb run ids
--report-path PATH                           ▲  writes CSV report; required in v2 mode
--rate-limit-batch-size N                    ▲  default 50
--rate-limit-sleep-seconds F                 ▲  default 2.0
--resume-from-report PATH                    ▲  skip run_ids already present in a prior report
--snapshot-before-write                      ▲  v2 only; logs pre-state of V2 keys as artifact
```

### 3.2 Dispatcher pattern

```
main():
  args = parse()
  if args.mode == "stamp":
      return _run_stamp_mode(args)   # existing `backfill_labels(...)` body
  if args.mode == "v2":
      return _run_v2_mode(args)
```

`_run_stamp_mode` is the current 80-line loop extracted into a function. `_run_v2_mode` is new.

### 3.3 `_run_v2_mode` pseudocode

```
fetch_runs(api, entity, project)                     # api.runs(path)
apply_filters(--limit, --run-ids-file, --group/--experiment-filter, --tag)
if args.snapshot_before_write and not dry_run:
    log pre-backfill snapshot artifact (see §5)
report_rows = []
for run in batched(filtered_runs, --rate-limit-batch-size):
    cfg = dict(run.config)
    flag_bucket = {keys_left_empty_by_prereq: [], keys_empty_missing_config: [], unresolved_flags: []}
    v2_values = derive_v2_axes(cfg, flag_bucket=flag_bucket)
    to_write = {}
    skipped = []
    for key, val in v2_values.items():
        current = cfg.get(key, None)
        if args.no_overwrite and current not in (None, ""):
            skipped.append(key); continue
        if current == val:
            continue                                  # no-op
        to_write[key] = val
    if dry_run:
        log [DRY RUN] summary
    else:
        cfg.update(to_write)
        run.config.update(to_write)
        run.update(config=cfg)
    report_rows.append({
        run_id, run_name, experiment_name,
        keys_written=len(to_write),
        keys_left_empty_by_prereq=len(flag_bucket[...]),
        keys_empty_missing_config=len(flag_bucket[...]),
        keys_skipped_already_set=len(skipped),
        unresolved_flags=";".join(flag_bucket["unresolved_flags"]),
        error="",
    })
    sleep(args.rate_limit_sleep_seconds)
write_report(args.report_path, report_rows)
log_report_artifact(args.report_path, job_type="backfill_v2")  # see §5
```

### 3.4 Report CSV format

Headers (exact):

```
run_id,run_name,experiment_name,keys_written,keys_left_empty_by_prereq,keys_empty_missing_config,keys_skipped_already_set,unresolved_flags,error
```

Invariants:

- One row per processed run (never zero, never multiple).
- `keys_written + keys_left_empty_by_prereq + keys_empty_missing_config + keys_skipped_already_set == len(V2_AXES) == 93`.
- The `unresolved_flags` cell is the semicolon-joined list of `"<axis_id>: <reason>"` strings (for I-3 mappings, B1-G1 met-direction mismatches, H10 cast failures).
- Empty `error` on success; exception repr on failure.
- Total file size is ~150 KB for 1000 runs — easily opened in Excel.

### 3.5 Idempotency

Second pass semantics:

- `v2_values` is deterministic from `cfg`.
- If `current == val` for every key, `to_write` is empty and `run.update` is not called.
- Report rows on the second pass show `keys_written=0` for every run.
- Verified by the validation step in Phase C.

## 4. Extensions to `scripts/wandb/cleanup_wandb.py`

### 4.1 New mode

Add `--mode {delete,undo_backfill_v2}`; default `delete` (current behaviour).

- `delete` — unchanged.
- `undo_backfill_v2` — non-destructive reset of V2-written keys.

### 4.2 `undo_backfill_v2` semantics

For each run in the project:

1. Fetch `cfg = dict(run.config)`.
2. Determine the V2 key list by importing `V2_AXES` from `scripts/wandb/v2_axes.py` and collecting `target_key` values (~93 keys).
3. If the pre-backfill snapshot artifact exists for this run (produced by Phase C `--snapshot-before-write`):
   - Restore those keys from the snapshot.
4. Else:
   - Emit a warning and either (a) remove the V2 keys from `cfg` entirely (the "strip" path), or (b) skip the run, controlled by `--strip-if-no-snapshot` flag (default false → skip).
5. Write `cfg` back via `run.update(config=cfg)`.

### 4.3 W&B delete-key primitive

W&B's Public API has no per-key delete. The only primitive is `run.update(config=full_dict)`, which writes the whole config back — keys not present are left in W&B's storage but vanish from the UI. The implementation therefore:

- Reads current `run.config`.
- Produces a new dict with V2 keys replaced (from snapshot) or removed.
- Calls `run.update(config=new_dict)`.

This is the exact pattern `backfill_labels.py` already uses.

### 4.4 New CLI

```
--mode {delete, undo_backfill_v2}        default delete
--strip-if-no-snapshot                   (undo_backfill_v2 only; default false)
--snapshot-artifact-name STR             default "v2_backfill_snapshot"
--dry-run                                default true
--execute
```

Keep all existing `delete` flags unchanged.

## 5. Snapshot & report artifacts

The V2 backfill produces two kinds of W&B artifacts:

1. **Pre-write snapshot** — only when `--snapshot-before-write` is passed. Written as a single `wandb.Artifact(name="v2_backfill_snapshot", type="v2_backfill_snapshot")` in a *new* wandb run of `job_type="backfill_v2_snapshot"`. The artifact file is a single `snapshot.json` of the form `{run_id: {axes/xxx: prior_value, ...}, ...}` restricted to the V2 key set for every touched run. One snapshot per full backfill pass.
2. **Report artifact** — always. A new wandb run of `job_type="backfill_v2"` with the report CSV attached and key summary metrics logged as scalar values: `n_runs_written`, `total_keys_written`, `runs_with_unresolved_flags`, `runs_with_errors`. This is how the thesis author opens the report in the W&B UI alongside the project.

Both artifacts are logged to the same `thesis-ml` project so they appear in the Runs list for context. Tag both with `backfill_v2` for easy filtering.

## 6. Scope changes summary

| File | LOC before | LOC after | Change type |
|---|---:|---:|---|
| `scripts/wandb/backfill_labels.py` | 208 | ~430 | extended |
| `scripts/wandb/cleanup_wandb.py` | 242 | ~360 | extended |
| `scripts/wandb/v2_axes.py` | 0 | ~550 | new |
| `tests/wandb/test_v2_axes.py` | 0 | ~150 | new |

No other files are modified. `src/thesis_ml/facts/axes.py`, `src/thesis_ml/utils/wandb_utils.py`, and `src/thesis_ml/facts/meta.py` are consulted as references only. Nothing in `src/` changes for this backfill.

## 7. Rollback

Three levels of rollback are available, ordered from cheap to expensive:

1. **Skip the run** — every write call is wrapped in try/except; failures are logged and surface in the `error` column of the report. No retry inside the backfill; retries are via `--resume-from-report`.
2. **Undo V2 for a subset of runs** — `cleanup_wandb.py --mode undo_backfill_v2 --run-ids-file failed_ids.txt --execute`.
3. **Undo V2 for the entire project** — same as (2) without `--run-ids-file`. Requires the Phase C snapshot artifact to be present; otherwise fall through to `--strip-if-no-snapshot` which leaves legacy `axes/*` absent (acceptable since those runs never had V2-correct values anyway).
