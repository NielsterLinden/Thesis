# V2 Axis Backfill — Plan Directory

Planning-only deliverables for retroactively backfilling every V2 axis and sub-axis onto every run in the `thesis-ml` W&B project. No code changes yet, no W&B writes yet — all gated by the three-phase runbook in `04_runbook.md`.

## Deliverables

| File | One-line summary |
|---|---|
| [00_doc_issues.md](00_doc_issues.md) | 10 discovered ambiguities in `AXES_REFERENCE_V2.md` (4 blocking, 6 non-blocking); plan assumptions stated for each. |
| [01_audit.md](01_audit.md) | V2 leaf checklist (93 rows), per-leaf W&B config-key presence across 1002 runs, prerequisite graph as a flat table, experiment-name inventory. |
| [02_mapping_spec.md](02_mapping_spec.md) | One derivation section per V2 leaf using the T1-b template: target W&B key, config source chain, parent, prerequisite, emit rule, value normalisation, coverage expectation. Unresolved-rules appendix with 4 blocking items. |
| [03_extension_plan.md](03_extension_plan.md) | Diff-level plan for `scripts/wandb/backfill_labels.py` (add `--mode v2` dispatcher + new flags), new `scripts/wandb/v2_axes.py` module with pure `derive_v2_axes`, report CSV format, `cleanup_wandb.py --mode undo_backfill_v2`, snapshot + report artifact scheme. |
| [04_runbook.md](04_runbook.md) | Three-phase gated runbook: Phase A dry run (5 min) → Phase B 4-run live pilot (1 min) → Phase C full backfill (30–45 min) + validation pass. |

## Reference data (from live audit on 2026-04-22)

Stored in [`_reference/`](_reference/):

- `summary.json` — totals and top-30 keys by presence.
- `column_presence.csv` — all 367 W&B config keys × run counts × fractions.
- `wandb_schema_sample.csv` — 367 keys × 5 sampled runs of diverse prerequisite coverage.
- `experiment_inventory.csv` — distribution over `axes/experiment_name` + `wandb.group`.

Audit helper script: [`scripts/one_off/audit_v2_schema.py`](../../scripts/one_off/audit_v2_schema.py).

## Key numbers at a glance

| Metric | Value |
|---|---:|
| Total runs | 1002 |
| Unique W&B config keys | 367 |
| Distinct `axes/experiment_name` values | 52 |
| Distinct `wandb.group` values | 99 |
| Sampled runs (used by Phase B) | 4 distinct (one covers two buckets) |
| V2 leaves to derive | 93 |
| Of which gated (have a prerequisite) | 50 |
| Tier-3 runs (have `axes/*`) | 455 (45.4%) |
| Tier-2 runs (have `raw/classifier/model/*`) | 574 (57.3%) |
| Tier-1 runs (have `model/*`, `training/*`) | 799 (79.7%) |
| Tier-0 legacy runs (only `meta.*`) | ~164 (16.4%) |
| Blocking doc issues (I-1 to I-4) | 0 (all resolved) |
| Non-blocking doc notes (I-5 to I-10) | 0 (all resolved) |
| Unresolved derivation rules | 0 |

## Per-group V2 leaf counts

| Group | Leaves | Gated | Root |
|---|---:|---:|---:|
| §1 G (framing) | 3 | 1 | 2 |
| §2 D (data) | 3 | 0 | 3 |
| §3 T (tokenizer) | 4 | 3 | 1 |
| §4 E (positional) | 4 | 3 | 1 |
| §5 P (pre-encoder) | 9 | 8 | 1 |
| §6 A (attention) | 6 | 1 | 5 |
| §7 F (FFN) | 4 | 3 | 1 |
| §8 B (biases) | 16 | 15 | 1 |
| §9 C (head) | 2 | 0 | 2 |
| §10 H (size) | 6 | 0 | 6 |
| §11 §K (KAN) | 5 | 5 | 0 |
| §12 §M (MoE) | 5 | 5 | 0 |
| §13 §S (backbone) | 2 | 2 | 0 |
| §14 §R (training) | 17 | 4 | 13 |
| §15 §L (logging) | 7 | 0 (*L2 gates 5) | 7 |
| **Total** | **93** | **50** | **43** |

\* §15 §L — L3 through L7 are only written when L2 is enabled. Counted as "root" above because L2 itself is root and all §L leaves can be populated; "gated" counts only those whose prerequisite gate (not `always`) could short-circuit to `""`. L3–L7 also short-circuit, bringing effective gated total to 55 with §L folded in.

## Copy-pasteable commands (from `04_runbook.md`)

```bash
source hpc/stoomboot/.wandb_env
cd /c/Users/niels/Projects/Thesis-Code/Code/Niels_repo

# Phase A — dry run
python scripts/wandb/backfill_labels.py \
    --mode v2 --dry-run \
    --report-path wandb_cleanup/backfill_dryrun.csv

# Phase B — 4-run live pilot
python scripts/wandb/backfill_labels.py \
    --mode v2 --execute --overwrite \
    --run-ids-file pilot_runs.txt \
    --snapshot-before-write \
    --report-path wandb_cleanup/backfill_pilot.csv \
    --rate-limit-batch-size 2 --rate-limit-sleep-seconds 3.0

# Phase C — full backfill
python scripts/wandb/backfill_labels.py \
    --mode v2 --execute --overwrite \
    --snapshot-before-write \
    --report-path wandb_cleanup/backfill_full.csv \
    --rate-limit-batch-size 50 --rate-limit-sleep-seconds 2.0

# Phase C validation (idempotency check)
python scripts/wandb/backfill_labels.py \
    --mode v2 --dry-run \
    --report-path wandb_cleanup/backfill_validation.csv

# Rollback (undo V2 writes for pilot runs)
python scripts/wandb/cleanup_wandb.py \
    --mode undo_backfill_v2 --execute \
    --run-ids-file pilot_runs.txt
```

## Gate ordering

```
Deliverable 1 audit
  → thesis-author confirmation of I-1..I-4
    → Deliverable 2 derivation rules
      → Deliverable 3 extension code written & unit-tested
        → Phase A dry run + sign-off
          → Phase B pilot + UI inspection sign-off
            → Phase C full backfill + validation pass
              → analysis proceeds against clean `axes/*` CSV
```
