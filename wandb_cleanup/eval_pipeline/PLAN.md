# Phase 2 eval pipeline — plan pointer

Implementation for the **Unified inference / re-evaluation pipeline** lives in this directory:

- **Stages:** `stage_a_manifest.py`, `stage_b_inference.py`, `stage_c_aggregate.py`, `stage_d_push.py`
- **Policy / data:** `config/eval_spec.yaml`, `config/test_splits.yaml`, `config/schema.yaml`
- **HTCondor:** `hpc/stoomboot/eval_stage_b.{sub,sh}` + `hpc/stoomboot/thesis_inference.sh`
- **Shared loader:** `src/thesis_ml/reports/utils/inference.py` (`load_classifier_from_run_dir`, `resolve_classifier_weights_path`)

Operational steps and CLI examples: [README.md](README.md).

The original design rationale and pre-reading notes remain in the Cursor plan artifact for this feature (not duplicated here to avoid drift).
