# Deliverable 4 — Three-phase gated runbook

Executes the V2 backfill in three gated phases: full read-only dry run, five-run live pilot, full 1002-run write. Each phase has (a) prerequisites, (b) exact shell commands, (c) expected runtime, (d) success criteria, (e) sign-off, and (f) failure recovery.

All commands assume the `WANDB_API_KEY` is exported and the working directory is the repo root:

```bash
source hpc/stoomboot/.wandb_env        # exports WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT
cd /c/Users/niels/Projects/Thesis-Code/Code/Niels_repo
```

---

## Phase A — Full dry run (read-only, 1002 runs)

### A.1 Prerequisites

- Code changes from [03_extension_plan.md](03_extension_plan.md) merged.
- `tests/test_v2_axes.py` passing locally.
- `docs/backfill_plan/_reference/wandb_schema_sample.csv` exists. If absent, rerun the audit first:
  ```bash
  python scripts/one_off/audit_v2_schema.py
  ```
- [00_doc_issues.md](00_doc_issues.md) discharged (see its status block).

### A.2 Commands (two-step, dump-first)

Phase A is now a **two-step offline workflow**. Step A.2a pulls every run's
config once from W&B; step A.2b derives the V2 axes from the local dump. This
decouples the slow API I/O from the fast derivation logic so you can iterate
on the derivation without re-hitting the API.

#### A.2a — Dump the project once (~3–5 min)

```bash
python scripts/wandb/dump_runs.py \
    --out wandb_cleanup/wandb_dump.jsonl
```

Produces `wandb_cleanup/wandb_dump.jsonl` — one JSON object per run with
`id`, `name`, `group`, `tags`, `state`, `created_at`, full `config` dict,
and `summary` dict. The dumper transparently JSON-decodes the handful of
legacy runs whose `run.config` is served as a double-encoded string.

Optional `--csv` flag produces a flat `wandb_dump.csv` alongside for Excel
spot checks (lossy on nested values — keep JSONL as the source of truth).

#### A.2b — Derive from the dump (~1–3 s)

```bash
python scripts/wandb/backfill_labels.py \
    --mode v2 \
    --from-dump wandb_cleanup/wandb_dump.jsonl \
    --report-path wandb_cleanup/backfill_dryrun.csv
```

`--from-dump` implies dry-run; it never writes to W&B. Re-run as often as
you like while iterating on `v2_axes.py` — no API calls are made.

### A.3 Expected runtime

| Step | Time | Network |
|---|---|---|
| A.2a dump | 3–5 min | 1 × project listing + 1002 × lazy `run.config` fetches |
| A.2b derive | 1–3 s | none (pure local read) |

Compare to the old single-step flow (~5 min wall time per iteration); the
dump is a one-time cost, and every subsequent derivation pass is near-instant.

### A.4 Success criteria

- Exit code 0.
- `wandb_cleanup/backfill_dryrun.csv` has 1002 rows.
- Row-wise invariant: `keys_written + keys_left_empty_by_prereq + keys_empty_missing_config + keys_skipped_already_set == N` for every row, where `N = len(V2_AXES)` (currently 94: the axis set includes three F-family entries — `F1` raw FFN type, `F1-moe` raw MoE flag, `F1-eff` effective realization).
- Aggregate expectations (from audit, [01_audit.md](01_audit.md) §2):
  - ≥455 rows with `keys_written ≥ 85` (tier-3 runs — gate-triggered gaps and missing rare axes pull the number from the per-run `N`).
  - ~120 rows with `40 ≤ keys_written ≤ 85` (tier-2).
  - ~220 rows with `10 ≤ keys_written ≤ 40` (tier-1 basic-structured).
  - ~170 rows with `keys_written ≤ 10` (tier-0 metadata-only).
- `unresolved_flags` column is non-empty for ≤5% of rows; the expected cases are B1-G1 met_direction mismatches and legacy attn_pairwise remappings.

### A.5 Thesis-author review gate

Stop here. Review the CSV manually:

1. Open `wandb_cleanup/backfill_dryrun.csv` in Excel / a CSV viewer.
2. Sort by `keys_empty_missing_config` descending. The top rows should all be tier-0 legacy runs (recognisable by `experiment_name==""`).
3. Sort by `unresolved_flags` descending. Inspect any value that isn't `""` — each should match one of the four blocking doc issues, with a row count matching the audit's legacy-pairwise estimate.
4. Spot-check 3 runs from the list by opening their W&B pages and comparing the derived V2 keys reported in the CSV against the source `raw/*` or `tokenizer/*` fields shown in the W&B config panel.
5. Confirm no `error` cells.

Sign-off unblocks Phase B.

### A.6 Failure recovery

| Failure | Symptom | Recovery |
|---|---|---|
| W&B API 429 rate limit | `wandb.errors.CommError` with 429 | Script already honours `--rate-limit-sleep-seconds`. Raise to 5 s, rerun. |
| Auth failure | `wandb.errors.AuthenticationError` | `echo $WANDB_API_KEY` — reopen shell if empty. |
| Module import error | `ImportError: v2_axes` | Confirm `scripts/wandb/v2_axes.py` exists and `sys.path` is repo root. |
| Some runs missing config entirely | `dict(run.config)` returns `{}` | Expected for ~5 legacy runs; they show `keys_written=0` and `keys_empty_missing_config=N` (the per-run axis count). |
| Report directory missing | `FileNotFoundError: wandb_cleanup/...` | `mkdir -p wandb_cleanup` before rerun. |

No W&B state is written in Phase A, so there is nothing to roll back on failure — always safe to rerun.

---

## Phase B — Five-run live pilot

### B.1 Prerequisites

- Phase A sign-off.
- Pilot run list prepared at `pilot_runs.txt` using the five sampled runs from [01_audit.md](01_audit.md) §5:
  ```
  1iuxis2l
  7ipzizh6
  wqo0y6mw
  17ovy6fy
  ```
  (Only four distinct IDs because `7ipzizh6` covered two buckets — see audit §5.)

### B.2 Command

```bash
python scripts/wandb/backfill_labels.py \
    --mode v2 \
    --execute \
    --run-ids-file pilot_runs.txt \
    --snapshot-before-write \
    --overwrite \
    --report-path wandb_cleanup/backfill_pilot.csv \
    --rate-limit-batch-size 2 \
    --rate-limit-sleep-seconds 3.0
```

`--execute` disables the dry-run default. `--overwrite` (the v2-mode default; stated explicitly here for unambiguous behaviour) replaces existing `axes/*` values. `--snapshot-before-write` logs a pre-backfill snapshot artifact scoped to these four runs.

### B.3 Expected runtime

~1 minute. 4 runs × (~3 s read + ~5 s write + 3 s sleep) + snapshot log (~10 s).

### B.4 Success criteria

- Exit code 0.
- Report has 4 rows (one per unique run ID).
- Snapshot artifact `v2_backfill_snapshot` created in `thesis-ml`, referencing a new run of `job_type="backfill_v2_snapshot"`.
- Per-run predicted behaviour (from [01_audit.md](01_audit.md) §5), each verified in the W&B UI:
  - `1iuxis2l` (T1=identity): T1-a, T1-b non-empty. P1 (if `0 ∈ D1`) non-empty. B1 and its sub-axes populated per stored `attention_biases`.
  - `7ipzizh6` (T1=binned, no biases): T1-a, T1-b, P1, P2, all B1-* leaves, R14–R17 all empty (rendered as blank cells in the W&B Runs table).
  - `wqo0y6mw` (E1=rotary): E1-a, E1-a1 empty; E1-b populated.
  - `17ovy6fy` (A3=differential): A3-a populated.

### B.4.1 V2 column naming (W&B)

Per the 2026-04-22 naming directive, every V2 column in W&B is written as:

```
axes/<ID>_<Canonical Name>
```

- `<ID>` uses no leading zeros (`G1` not `G01`, `D2` not `D02`, `H5` not `H05`,
  `R1` not `R01`). Two-digit IDs (`H10`, `R10`…`R17`) are unchanged.
- `<Canonical Name>` is the Title-Case section heading from
  `docs/AXES_REFERENCE_V2.md`, with acronyms preserved (`PID`, `MET`, `MIA`,
  `MoE`, `KAN`, `SM`, `LR`, `PE`, `FFN`). Spaces are kept.

Concrete examples:

| Target key (W&B column) | Meaning |
|---|---|
| `axes/G1_Task Type` | Study loop (classifier / autoencoder / …) |
| `axes/T1-b_PID Embedding Dimension` | ID-embedding dim; `"num_types"` when `pid_mode=one_hot` |
| `axes/D2_MET Treatment` | Whether MET is included |
| `axes/F1_FFN Type` | Raw `classifier.model.ffn.type` |
| `axes/F1-moe_MoE Enabled` | Raw `classifier.model.moe.enabled` |
| `axes/F1-eff_FFN Realization` | Derived `moe` / `kan` / `standard` |
| `axes/H10_Model Size Label` | `d{dim}_L{depth}` |

The full 94-entry mapping is `{a.target_key for a in v2_axes.V2_AXES}`.
Legacy `axes/<old_slug>` keys (e.g. `axes/pid_mode`) are **untouched** by this
backfill — they coexist with the new columns until a later cleanup pass.

### B.5 Thesis-author review gate

Stop here. Open the W&B UI:

1. Load `https://wandb.ai/nterlind-nikhef/thesis-ml/runs/` filtered to the four pilot run IDs.
2. In the Runs table, add every column from the V2 target key list (see §B.4.1; currently 94 `axes/<ID>_<Name>` columns).
3. For each pilot run, visually confirm:
   - Gated-empty cells render as blank (not `"None"`, not `"NA"`, not `"0"`).
   - Non-empty cells show values that match the predicted per-run behaviour in §B.4.
4. Confirm legacy non-V2 columns (`meta.*`, `raw/*`, `model/dim`) are unchanged — grab the run page for one of the four runs pre-backfill (browser tab history) and compare counts.
5. Confirm the `keys_written` number in the report matches the number of non-blank V2 cells in the W&B table (per run).

Sign-off unblocks Phase C. **If any gated cell renders as non-empty, stop — this indicates empty-string serialisation is wrong and Phase C must not proceed.** See §B.6.

### B.6 Failure recovery

| Failure | Symptom | Recovery |
|---|---|---|
| Gated cell renders as `0` or `"None"` | W&B UI shows the cell filled but with a wrong value | Bug in `_as_bool_str` / `_as_int_str` — they must not convert `""` inputs. Fix code, then: `python scripts/wandb/cleanup_wandb.py --mode undo_backfill_v2 --run-ids-file pilot_runs.txt --execute` to restore from snapshot, then retry pilot. |
| Write fails for one run | Error row in CSV | Rerun only that run via `--run-ids-file` with the single ID. |
| Snapshot artifact missing | Artifact not in project | Re-invoke snapshot step manually: `python scripts/wandb/backfill_labels.py --mode v2 --execute --run-ids-file pilot_runs.txt --snapshot-before-write --dry-run=false --skip-backfill` (new flag to just take snapshot). |
| Overwrite skipped in error | `keys_skipped_already_set` unexpectedly high | Confirm `--overwrite` was passed; the v2-mode default is true but only when `--no-overwrite` is not passed. |

Rollback just the four pilot runs:

```bash
python scripts/wandb/cleanup_wandb.py \
    --mode undo_backfill_v2 \
    --run-ids-file pilot_runs.txt \
    --execute
```

This restores the four runs from their snapshot artifact. Verify in UI. Then iterate on the bug and re-run Phase B.

---

## Phase C — Full backfill (1002 runs, write)

### C.1 Prerequisites

- Phase B sign-off with successful UI visual inspection.
- No code changes since Phase B. (If code changed, repeat Phase A then B.)
- `wandb_cleanup/backfill_dryrun.csv` and `wandb_cleanup/backfill_pilot.csv` both present for reference.

### C.2 Command

```bash
python scripts/wandb/backfill_labels.py \
    --mode v2 \
    --execute \
    --overwrite \
    --snapshot-before-write \
    --report-path wandb_cleanup/backfill_full.csv \
    --rate-limit-batch-size 50 \
    --rate-limit-sleep-seconds 2.0
```

### C.3 Expected runtime

~30–45 minutes for 1002 runs at (50 runs per batch × ~4 s per run for read+write) + 2 s inter-batch sleep = 200 s × 20 batches = ~70 min worst case. Faster if W&B API is responsive (observed ~250 ms/write during pilot).

Single-run timing breakdown: ~250 ms `dict(cfg)` read + ~3 ms derivation + ~1 s `run.update(config=...)` write + ~0.04 s sleep share = ~1.3 s per run. 1002 × 1.3 s = ~22 minutes best case, ~45 minutes expected.

### C.4 Progress monitoring

Tail the process's stdout in one terminal. In another, watch the report grow:

```bash
watch -n 30 'wc -l wandb_cleanup/backfill_full.csv'
```

Row count should increase monotonically. If stalled for >2 minutes, suspect API rate limit or network drop — see §C.6.

### C.5 Validation pass

Immediately after Phase C finishes and writes `wandb_cleanup/backfill_full.csv`, run the read-back validation:

```bash
python scripts/wandb/backfill_labels.py \
    --mode v2 \
    --dry-run \
    --report-path wandb_cleanup/backfill_validation.csv
```

Expected: every row has `keys_written=0` (idempotent). Any row with `keys_written>0` identifies a post-write drift — inspect and decide whether to rerun those runs.

### C.6 Success criteria

- Exit code 0.
- `wandb_cleanup/backfill_full.csv` has 1002 rows.
- `keys_written` distribution matches Phase A prediction (§A.4) — confirm histogram visually.
- `error` column is empty for ≥999 rows.
- Second-pass validation (`backfill_validation.csv`) has `keys_written=0` for ≥999 rows.
- Snapshot artifact `v2_backfill_snapshot` present in the project.
- W&B UI sanity check on 10 randomly sampled runs confirms `axes/*` columns populated; legacy columns untouched.

### C.7 Failure recovery

| Failure | Symptom | Recovery |
|---|---|---|
| W&B API 429 mid-batch | Some runs processed, crash with 429 | Use `--resume-from-report wandb_cleanup/backfill_full.csv` on a rerun — the new invocation will skip IDs already reported. |
| Network drop | TCP reset, incomplete report | Same as above: resume from the partial report. |
| Disk full on report path | `OSError: No space left on device` | Free disk, then resume. The report file is append-compatible — any in-memory buffered rows lost on crash are re-derived on resume. |
| Config write rejected by W&B | Row has non-empty `error` column | Fetch error details via `grep ",<run_id>," wandb_cleanup/backfill_full.csv`. Common cause: stale W&B run that was soft-deleted — skip via `--skip-ids-file`. |
| Snapshot artifact corrupt | Validation pass shows drift; undo cannot restore | Re-derive V2 from current live config (idempotent) — the canonical state is the derivation, not the snapshot. If the original legacy values are needed, consult W&B's per-run config-history endpoint (`/runs/<id>/config-updates`) — accessible only via wandb-internal API. |

Full rollback (worst case, undo all V2 writes):

```bash
python scripts/wandb/cleanup_wandb.py \
    --mode undo_backfill_v2 \
    --execute \
    --rate-limit-batch-size 50 \
    --rate-limit-sleep-seconds 2.0
```

Runtime: ~20 minutes. Restores every run from the Phase C snapshot. Legacy `axes/*` are restored for the 455 tier-3 runs; the 547 runs that never had V2 keys are effectively no-ops (the V2 target keys are removed from the config write-back since no snapshot contained them).

### C.8 Sign-off

Phase C success criteria met + validation pass green + W&B UI sanity check pass = backfill complete. The thesis analysis scripts can now query the project assuming every run has the full V2 `axes/*` key set (with empty strings for inapplicable axes, which the analysis code must treat as "excluded from value counts" per the user's design intent).

---

## Timing summary

| Phase | Reads | Writes | Wall time (est.) | Sign-off required |
|---|---:|---:|---|:---:|
| A | 1002 | 0 | 5 min | yes — thesis author |
| B | 4 | 4 (plus snapshot) | 1 min | yes — thesis author + UI check |
| C | 1002 | 1002 (plus snapshot) | 30–45 min | yes — thesis author + validation |
| C validation | 1002 | 0 | 5 min | (automatic) |

Total: ~50 min of execution wall time, gated by two manual reviews.
