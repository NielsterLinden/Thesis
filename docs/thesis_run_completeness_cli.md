# Thesis Run Completeness CLI Workflow

Use this workflow on Atlas from:

`/project/atlas/users/nterlind/Thesis-Code`

## 1) Check all required thesis experiments (Ch 4, 5, 6, 8)

```bash
python scripts/check/check_thesis_experiment_completeness.py \
  --configs-dir configs/classifier/experiment/thesis_experiments \
  --multiruns-dir /data/atlas/users/nterlind/outputs/multiruns \
  --runs-dir /data/atlas/users/nterlind/outputs/runs \
  --chapters 4,5,6,8
```

## 2) Optional targeted checks

Per chapter:

```bash
python scripts/check/check_thesis_experiment_completeness.py --chapters 5
```

Per specific experiments:

```bash
python scripts/check/check_thesis_experiment_completeness.py \
  --experiments ch5_bias_families_exp5a,ch6_attention_type_exp6a
```

## 3) Paste-back contract (for chat handoff)

Copy and paste the full stdout sections delimited by:

- `===THESIS_COMPLETENESS_SUMMARY_START===`
- `===THESIS_COMPLETENESS_SUMMARY_END===`
- `===THESIS_CANONICAL_MAPPING_START===`
- `===THESIS_CANONICAL_MAPPING_END===`

Do not trim lines in those blocks. With those blocks, the canonical mapping section for
`docs/thesis_execution_plan_066bfb22.plan.md` can be generated directly.
