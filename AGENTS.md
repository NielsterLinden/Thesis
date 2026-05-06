# Agent Operating Manual

Cross-tool contract for working in this repo. The repo holds two domains — an ML codebase and a LaTeX thesis — and the agents that work on them are deliberately separated.

This document is the human-readable summary. The enforced rules live in `.claude/agents/*.md` and `.claude/skills/*/SKILL.md`.

## Domains

1. **ML codebase**: `src/thesis_ml/`, `configs/`, `hpc/`, `scripts/`, `docs/`.
2. **LaTeX thesis**: `thesis_report/`.

The thesis is about a configurable transformer workbench for rare multi-top-quark classification at ATLAS.

## Subagents

| Agent | May edit | May not edit | Bash |
|-------|----------|--------------|------|
| `thesis-writer` | `thesis_report/**`, `docs/thesis_evidence_notes/**` (annotate only) | `src/`, `configs/`, `hpc/`, `scripts/` | LaTeX build only |
| `experiment-coder` | `src/`, `configs/`, `hpc/`, `scripts/`, `docs/thesis_evidence_notes/**` | `thesis_report/**` | Read-only inspection + local sanity checks; **never `condor_submit`** |
| `reviewer` (optional) | nothing | everything | none |

The boundary is enforced by tool/path allowlists in each agent's frontmatter, not by prose alone.

## Hand-off

The only structured channel between coder and writer is an evidence note at:

```
docs/thesis_evidence_notes/<chN.S>_<axis-group>.md
```

Status state machine: `triaged → run-pending → run-complete → interpreted`. Coder writes everything except the final `interpreted` flip and figure confirmation, which the writer owns.

## Workflow

Triage first, never assume new training is needed:

1. `/extract-evidence <chN.S> <axis-group>` — coder runs `data-inventory`, picks an entry point, stages the action.
2. User executes the staged action (`condor_submit ...` or local cmd).
3. `/extract-evidence ...` again — coder ingests results, imports plots, fills in the note.
4. `/draft-section <chN.S> <axis-group>` — writer drafts/revises the `.tex`.
5. (Optional) `/review-section <path>` — reviewer comments.

## Entry points (lowest cost first)

| Code | When | Tool |
|------|------|------|
| D | Existing runs need a uniform report bundle | `report.sh` + `configs/report/...` |
| C | Model + `03_analysis_ready.csv` already exist | `src/thesis_ml/reports/` modules, local |
| E | Bespoke analysis on `model.pt` | new module under `src/thesis_ml/reports/analyses/` or `scripts/` |
| B | Model exists, inference missing | inference job (rare) |
| A | No checkpoints — new training | `train.sh` + Hydra sweep |

Default expectation for the current thesis batch is **C**, not A.

## Source-of-truth hierarchy

1. Hydra config + implementation in `src/thesis_ml/`.
2. `facts/axes.json` per run + W&B `axes/*`.
3. W&B CSV exports.
4. `docs/AXES_REFERENCE_V2.md` (v1 only as fallback).
5. Thesis prose — never a source of truth for code.

## Forbidden habits

- No invented metric numbers in `.tex`. Use `\todo{...}` when pending.
- No renaming of axis IDs, config keys, W&B keys, figure paths, or LaTeX labels without a deliberate cross-cutting pass.
- No generic transformer / ML-101 background in the thesis.
- No `condor_submit` from any agent.
- No long-running training/inference on the head node — submit through Condor.
- No writes outside the per-agent allowlists, even when convenient.
- No deletes or overwrites under `/data/atlas/users/nterlind/...`. Additive only.

## Paths

- Repo / code / thesis: `/project/atlas/users/nterlind/Thesis-Code` (also accessible as `/project/atlas/Users/nterlind/Thesis-Code`).
- Run outputs / checkpoints / large artifacts: `/data/atlas/users/nterlind`.
- Claude Code config: `/project/atlas/users/nterlind/.claude-config` (`CLAUDE_CONFIG_DIR`). Out of repo.

See root `CLAUDE.md` for the Stoomboot / tmux operational rules.
