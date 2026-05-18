# CLAUDE.md

This repository is the MSc thesis codebase for transformer-based event classification in ATLAS / top-physics studies.

## Important paths

- **Repository, code, and thesis text:** `/project/atlas/users/nterlind/Thesis-Code`
- **Data, runs, outputs, checkpoints, plots, large experiment artifacts:** `/data/atlas/users/nterlind`
- **Thesis analysis-ready table (cleaned runs × columns):** `/project/atlas/users/nterlind/Thesis-Code/thesis_results/04_cleaned_backfilled_analysis_ready.csv` — canonical for metrics and axes. Legacy W&B export snapshot: `thesis_results/03_analysis_ready.csv` (same schema; includes unevaluated / unfixed-cohort rows).
- **Full combined export + eval (optional copy on data disk):** `/data/atlas/users/nterlind/thesis_artifacts/2026-04-29_phase2_02_eval_combined.csv`
- **Home directory:** `/user/nterlind` — do **not** use for project code, experiment outputs, model checkpoints, or large files (home quota is tight on Stoomboot).
- **Local papers / reference for agents (gitignored contents):** `agent_reference/` — drop PDFs and notes here; only `agent_reference/README.md` is tracked.

Claude Code configuration lives outside the repo: `/project/atlas/users/nterlind/.claude-config` (`CLAUDE_CONFIG_DIR`). OAuth/token files stay there, not in git.

## Multi-agent contract

The thesis writing and code/experiment work are split across separate Claude Code subagents with enforced boundaries. See **`AGENTS.md`** at repo root for the contract; the enforced rules live in `.claude/agents/*.md` and `.claude/skills/*/SKILL.md`. Do not inline thesis writing-style rules in this file — they belong in `.claude/agents/thesis-writer.md`.

## Working rules

- Plan before editing; prefer small, inspectable changes.
- Never commit secrets, credentials, tokens, API keys, W&B keys, or local machine paths.
- Do not write model checkpoints, run outputs, plots, caches, or datasets into the repo or home directory.
- Use `/data/atlas/users/nterlind` for generated outputs.
- Before destructive commands, explain what will be changed.
- Do **not** run long training or large sweeps on interactive CPU/GPU nodes.
- For production training or evaluation, prepare or submit Condor / HPC batch jobs.
- When changing code, run the smallest relevant tests first.
- Respect the existing Hydra / W&B / reporting structure.

## Nikhef Stoomboot

- **Interactive CPU nodes:** `stbc-i1.nikhef.nl`, `stbc-i2.nikhef.nl`, `stbc-i3.nikhef.nl` — pick **one** node for interactive Claude + tmux (recommended default: **stbc-i2**).
- **`/project` and `/data` are shared** across nodes, but **processes and tmux sessions are node-local.** Always SSH to the **same** hostname you used when starting tmux to reattach (otherwise `tmux attach` will show no session).

### tmux workflow (recommended)

```bash
ssh nterlind@stbc-i2.nikhef.nl
cd /project/atlas/users/nterlind/Thesis-Code
tmux new -s claude-thesis
# inside tmux:
source ~/.bashrc
cd /project/atlas/users/nterlind/Thesis-Code
claude --rc "Thesis-Code Stoomboot"
# detach: Ctrl-b d
# later, same node only:
ssh nterlind@stbc-i2.nikhef.nl
tmux attach -t claude-thesis
```

## Thesis-specific hints

- Configs live under `configs/`.
- Reporting / inference utilities live under `src/thesis_ml/reports/`.
- Report configs are under `configs/report/`.
- HPC scripts live under `hpc/`.
- Primary thesis run table: `thesis_results/04_cleaned_backfilled_analysis_ready.csv` (see `docs/thesis_evidence_notes/ch8_analysis_ready_cleanup.md`). Raw W&B export before cleanup: `thesis_results/03_analysis_ready.csv`. W&B re-eval pipeline: `scripts/wandb/eval_pipeline/`.
- Prefer reproducible commands; record config keys and W&B run identifiers when relevant.

## Authentication (operator)

- **Terminal SSH:** set `CLAUDE_CODE_OAUTH_TOKEN` in `/project/atlas/users/nterlind/.claude-config/claude_env` (from local `claude setup-token`). Do not commit.
- **Mobile Remote Control:** use full `.credentials.json` in that directory and **do not** use the setup-token OAuth export (see project setup docs).
