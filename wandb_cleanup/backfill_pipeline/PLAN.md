# Stage 0: Faithful W&B export to CSV

Pipeline folder layout:

```
wandb_cleanup/backfill_pipeline/
  PLAN.md
  stage0_export.py
  snapshots/
    <yyyy-mm-dd>_raw/
      00_raw_export.csv
```

Scope: **Stage 0 only** — one project, full dump, no backfill/push/validation. Later stages are planned after inspecting `00_raw_export.csv`.

---

## 1. Existing code review (reuse)

| Location | Role | Reuse |
|----------|------|--------|
| [`scripts/wandb/dump_runs.py`](../../scripts/wandb/dump_runs.py) | `wandb.Api().runs(...)`, `_unwrap_wandb_value`, `_coerce_config`, `_coerce_summary`, `_strip_private`, skip-and-continue on per-run errors, `per_page` | **Reuse** coercion helpers via `importlib` load of that module from repo root (same idea as [`src/thesis_ml/utils/wandb_utils.py`](../../src/thesis_ml/utils/wandb_utils.py) loading `v2_axes.py`). |
| [`src/thesis_ml/utils/wandb_utils.py`](../../src/thesis_ml/utils/wandb_utils.py) | `extract_wandb_config`, `_flatten_dict` for `raw/*`, metric key shapes | **Reference only** for flattening/naming alignment (`/` namespaces, `meta.*` dots, `axes/...`); not imported at export time. |
| [`wandb_cleanup/_export_wandb_columns.py`](../_export_wandb_columns.py) | Scans all runs and writes `wandb_mcp_columns_thesis-ml.csv` (`column_name`, `kind`) — **same key granularity W&B exposes** on `run.config` / `run.summary._json_dict` (each key string is one column, including paths like `raw/classifier/...`). | **Regenerate** that CSV when new keys appear project-wide; Stage 0 reads it as the **fixed column list**. |

**W&B project (verified):** `thesis-ml` from `env.wandb_project` in [`configs/env/local.yaml`](../../configs/env/local.yaml) / [`stoomboot.yaml`](../../configs/env/stoomboot.yaml) and [`configs/logging/default.yaml`](../../configs/logging/default.yaml).

---

## 2. Design decisions Q1–Q9

**Q1 — Key paths (schema-driven)**
Columns are defined by [`wandb_cleanup/wandb_mcp_columns_thesis-ml.csv`](../wandb_mcp_columns_thesis-ml.csv) (from [`_export_wandb_columns.py`](../_export_wandb_columns.py)): each row is a **top-level** key on `run.config` or `run.summary._json_dict` (same as the MCP scan). Values are read with **`.get(key)`** — no extra flattening; nested structures appear as a **single cell** encoded per Q2 if the stored value is a dict/list. Underscore-prefixed keys (e.g. `_wandb`) are **not** stripped from config so schema columns stay faithful to W&B.

**Q2 — Non-scalar cells**
`dict` / `list` / `tuple` → `json.dumps(..., sort_keys=True, separators=(",", ":"), default=str)`.

**Q3 — Missing keys**
Empty string `""` when that run has no value for a schema key.

**Q4 — Run order**
Sort by `created_at` ascending; tie-breaker: `meta_run/id` lexicographic.

**Q5 — Auth**
`WANDB_API_KEY`; optional read of `hpc/stoomboot/.wandb_env` when the key is still unset (same discovery pattern as [`_load_wandb_env_if_needed`](../../src/thesis_ml/utils/wandb_utils.py)); otherwise `wandb login` / `.netrc`. On failure: clear message to **stderr**, non-zero exit (no partial CSV if the failure happens before any successful write — delete incomplete file on total failure where practical).

**Q6 — Resilience**
Per-run API/coercion errors: **skip**, log a **warning to stderr**, continue. No `run_log.txt`; the CSV is the artifact when the script completes successfully.

**Q7 — Pagination**
`wandb.Api().runs(entity/project, per_page=...)` paginates transparently; use explicit `per_page=500` and a generous `timeout` (e.g. 120–180s) like existing scripts.

**Q8 — Run metadata columns**
Prefix `meta_run/` (no `meta_run/entity` — single-project export):
`meta_run/id`, `meta_run/name`, `meta_run/created_at`, `meta_run/state`, `meta_run/tags`, `meta_run/group`, `meta_run/project`.

**Q9 — Determinism (cell encoding)**
Final rules only:

- `None` → `""`
- `dict` / `list` / `tuple` → `json.dumps(..., sort_keys=True, separators=(",", ":"), default=str)`
- `bool` / `int` / `float` → `json.dumps(...)` (scalar JSON)
- `str` → write as-is (CSV writer handles escaping)

**Columns:** `meta_run/*` in fixed Q8 order, then every schema row as **`{kind}/{column_name}`** (e.g. `summary/train/loss`, `config/axes/D1_Feature Set`) so `summary` vs `config` is explicit; only `_wandb` overlaps by name across kinds.

**Rows:** Q4 order.

---

## 3. Script structure (`stage0_export.py`)

- `_repo_root()` — Find directory containing `pyproject.toml`.
- `_load_wandb_env_if_needed()` — Optional `.wandb_env` load (mirrors training helper scope).
- `_import_dump_runs()` — `importlib` load `scripts/wandb/dump_runs.py`; return coercion module.
- `_load_schema(path)` / `_schema_headers(...)` — Read `column_name,kind` rows; build `summary/...` and `config/...` headers.
- `_cell_str(v)` — Q9 encoding.
- `_created_at_sort_key(run)` — Parse `created_at` + id for stable row ordering.
- `_meta_row(run, project)` — Build the seven `meta_run/*` string cells.
- `export_stage0(...)` — List runs, sort, coerce config/summary, fill schema columns, write CSV; stderr warnings on skip.
- `main()` — Argparse (`--schema-csv`, `--out-csv`, `--per-page`, `--limit`), default schema `wandb_cleanup/wandb_mcp_columns_thesis-ml.csv`.

---

## 4. Files

| Path | Purpose |
|------|---------|
| `wandb_cleanup/backfill_pipeline/PLAN.md` | This document |
| `wandb_cleanup/backfill_pipeline/stage0_export.py` | Stage 0 exporter |
| `snapshots/<date>_raw/00_raw_export.csv` | Produced at run time |

---

## 5. How to run

From repo root (conda env per project rules):

```bash
conda activate "C:\Users\niels\mambaforge\Library\envs\thesis-niels_repo"
cd "c:\Users\niels\Projects\Thesis-Code\Code\Niels_repo"
python wandb_cleanup/backfill_pipeline/stage0_export.py \
  --entity nterlind-nikhef --project thesis-ml
```

Defaults: `--entity` from `WANDB_ENTITY` else `nterlind-nikhef`; `--project` from `WANDB_PROJECT` else `thesis-ml`. `--schema-csv` defaults to `wandb_cleanup/wandb_mcp_columns_thesis-ml.csv` (re-run `_export_wandb_columns.py` to refresh). Optional `--out-csv`, `--per-page`, `--limit` for smoke tests.

**Output:** `wandb_cleanup/backfill_pipeline/snapshots/<yyyy-mm-dd>_raw/00_raw_export.csv`. Progress and skips on stderr.
