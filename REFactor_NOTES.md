Refactor summary

- Unified training entrypoint at `thesis_ml.train` with lazy dispatch and cfg.loop source of truth (fallback to cfg.trainer.loop).
- Removed dead references to `vq_ae_loop`.
- Moved smoke test loop to `thesis_ml/general/test_loop.py`.
- Moved Phase-1 reports to `thesis_ml/phase1/reports/` and added compatibility shims under `thesis_ml/reports/`.
- Relocated generic models to `thesis_ml/general/models/` and added a shim in `thesis_ml/models/__init__.py`.

New CLI examples
- Phase‑1 AE: `python -m thesis_ml.train loop=ae phase1/encoder=mlp phase1/decoder=mlp phase1/tokenizer=none logging.save_artifacts=false trainer.epochs=1`
- Smoke test: `python -m thesis_ml.train loop=mlp logging.save_artifacts=false`

Deprecations
- `thesis_ml.reports.*` is deprecated; use `thesis_ml.phase1.reports.*`. Shims warn once and will be removed after v0.3.
- `thesis_ml.models` is deprecated; use `thesis_ml.general.models`. Shim warns once and will be removed after v0.3.

Notes
- Dispatch is lazy and imports loop modules only when invoked, reducing import-time failures.
- Facts payload schema produced by loops remains unchanged.
