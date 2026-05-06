# thesis_evidence_notes

Structured hand-off artifacts between `experiment-coder` (author) and `thesis-writer` (reader). One markdown per (thesis section, axis group). The full template and status state machine live in `.claude/skills/evidence-note/SKILL.md`.

## Naming

```
<chN.S>_<axis-group>.md
# e.g. ch4.2_D02.md, ch6.3_T1c.md, ch8.1_H07.md
```

If a section ultimately needs a regenerated run, version the file as `<chN.S>_<axis-group>-v2.md` rather than overwriting.

## Lifecycle

```
triaged  →  run-pending  →  run-complete  →  interpreted
```

- `experiment-coder` creates the file at `triaged` after running `data-inventory`, advances it through `run-pending` and `run-complete`.
- `thesis-writer` flips it to `interpreted` once the corresponding `.tex` is updated.

## Authoring rules

- Do not invent metrics. Leave `<TBD>` until a run produces the number.
- Always record `Reused vs new code` so we can audit which sections rely on shared plotting (entry point D) vs bespoke modules (entry point E).
- Do not delete a note. If invalidated, supersede with a `-vN` file and link the old one in the `Confounders / limitations` section.
