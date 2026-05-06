# thesis_report scope rules

This directory holds the LaTeX MSc thesis. Code, configs, and HPC scripts live elsewhere in the repo.

## Boundary

- Edits in this directory must remain inside `thesis_report/`.
- Do **not** edit `src/`, `configs/`, `hpc/`, or `scripts/` from a thesis-writing context. If a thesis claim requires new code or a new experiment, hand off to `/extract-evidence`.
- Figures under `figures/` enter via the `figure-import` skill, which records provenance back into the corresponding evidence note in `docs/thesis_evidence_notes/`.

## Structure (do not reorganize without reason)

- `report.tex` — main entry point.
- `report.bib` — bibliography.
- `tudelft-report.cls` — TU Delft class file. Do not modify.
- `frontmatter/` — preface, summary, title pages, nomenclature.
- `mainmatter/01..11_*.tex` — chapter files. Note: there is intentionally no `05_*` at the moment.
- `appendix/appendix-{a,b,c}.tex`.
- `figures/ch<N>/` — figures grouped by chapter.

## Build

> TODO — pin a build command (latexmk recipe or a `build.sh`) once chosen. For now, manual:
>
> ```bash
> xelatex report.tex
> biber report
> xelatex report.tex
> xelatex report.tex
> ```

## Writing rules

The full style guide lives in `.claude/agents/thesis-writer.md`. The boundary-relevant rules:

- Every architecture claim references its axis ID (see `docs/AXES_REFERENCE_V2.md`) and Hydra config key.
- Result claims must come from an evidence note in `docs/thesis_evidence_notes/`. If the note's `Status` is not at least `run-complete`, do not draft results prose.
- Never invent metric numbers. Use `\todo{...}` (or a clearly marked comment) when a value is pending.
