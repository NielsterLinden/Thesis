# Metadata Schema for Run Classification

This document defines the canonical metadata schema stored in `facts/meta.json` for each training run. This metadata enables reliable slicing, filtering, and comparison of runs in W&B dashboards and reports.

## Design Principles

1. **`process_groups` is the canonical slicer** - replaces brittle "task names" with explicit physics process composition
2. **Facts is source of truth** - W&B is a mirror that can be rebuilt from Facts at any time
3. **Never guess** - if metadata cannot be determined reliably, use `null` with `needs_review: true`
4. **Canonical ordering** - ensures identical tasks hash identically regardless of config order
5. **Derived fields are cache** - can be regenerated at any time; don't treat as primary truth

## Schema Version

Current schema version: **1**

When adding new fields:
1. Increment `schema_version`
2. Write a backfill script to populate the new field for old runs
3. Use `"unknown"` for values that cannot be determined

---

## Canonical Fields (Primary Truth)

These fields define the semantic identity of a run and are used for hashing/grouping.

### `schema_version` (int)
Schema version number. Currently `1`.

### `level` (string)
Data abstraction level.
- `"sim_event"` - Simulated event-level data (current)
- Future: `"reco_event"`, `"jet_level"`, etc.

### `goal` (string)
Training objective.
- `"classification"` - Supervised classification task
- `"anomaly_detection"` - Unsupervised/semi-supervised anomaly detection

**Important:** This should be set **explicitly** in Hydra config for new runs, not inferred.

### `dataset_name` (string)
Dataset identifier, extracted from `data.path`.
- `"4topsplitted"` - Current 4-tops dataset
- Future datasets will have distinct names

### `model_family` (string)
Model architecture family.
- `"transformer"` - Transformer-based classifier
- `"mlp"` - Multi-layer perceptron classifier
- `"bdt"` - Boosted Decision Tree (XGBoost)
- `"ae"` - Autoencoder family (AE, VAE, VQ-VAE)

### `process_groups` (list[list[str]] | null)
**The primary slicer.** Defines class composition as a list of lists of physics process names.

Example for binary "4t vs background":
```json
[["4t"], ["ttH", "ttW", "ttWW", "ttZ"]]
```

Example for 3-way classification:
```json
[["4t"], ["ttH"], ["ttW", "ttWW", "ttZ"]]
```

**Canonicalization rules:**
1. Sort processes within each class **lexicographically**
2. Sort classes by their **string signature**: `sorted(groups, key=lambda cls: "+".join(cls))`
3. Exception: if config explicitly declares `signal_vs_background`, preserve signal class position

**If uncertain:** Set to `null` with `needs_review: true`. Never guess.

### `datatreatment` (object)
Structured representation of data preprocessing choices.

```json
{
  "token_order": "input_order",
  "tokenization": "direct",
  "pid_encoding": "embedded",
  "id_embed_dim": 8,
  "met_rep": "met_metphi",
  "globals_included": true,
  "feature_set_version": "v1",
  "normalization": "zscore"
}
```

Field definitions:
- `token_order`: `"pt_sorted"` | `"shuffled"` | `"input_order"` | `"unknown"`
- `tokenization`: `"direct"` | `"binned"` | `"vq"` – High-level tokenization: direct (identity/raw), binned (Ambre-style integer tokens), or vq (pretrained VQ-VAE). Used for W&B grouping in binning-vs-direct experiments.
- `pid_encoding`: `"embedded"` | `"onehot"` | `"raw"` | `"none"` | `"unknown"`
- `id_embed_dim`: integer or `null` (only if pid_encoding=embedded)
- `met_rep`: `"met_metphi"` | `"met_vec"` | `"none"` | `"unknown"`
- `globals_included`: `true` | `false` | `"unknown"`
- `feature_set_version`: `"v1"` | `"v2"` | `"unknown"` (from `FEATURE_SET_VERSION` constant)
- `normalization`: `"zscore"` | `"minmax"` | `"none"` | `"unknown"`

**Important:** Never bake in guesses. Use `"unknown"` if not provable from config.

### `meta_hash` (string)
SHA1 hash of canonical fields for deduplication and grouping.
Format: `"sha1:abc123def456"`

Computed from: `schema_version`, `level`, `goal`, `dataset_name`, `model_family`, `process_groups`, `datatreatment`

### `meta_source` (string)
How this metadata was generated.
- `"live"` - Generated during training
- `"backfill"` - Inferred from existing run
- `"override"` - Manually corrected via override file

### `meta_confidence` (string)
Overall confidence level (minimum of per-field confidences).
- `"high"` - All fields determined reliably
- `"medium"` - Some fields inferred with moderate certainty
- `"low"` - One or more fields uncertain

### `meta_confidence_fields` (object)
Per-field confidence for debugging.

```json
{
  "dataset_name": "high",
  "process_groups": "medium",
  "datatreatment": "low"
}
```

### `needs_review` (boolean)
Whether this run needs manual review before appearing in coverage grids.

### `needs_review_reason` (list[string])
Specific reasons for needing review.
- `"missing_process_groups"`
- `"missing_goal"`
- `"missing_dataset_name"`
- `"ambiguous_datatreatment"`

---

## Derived Fields (Cache, Regeneratable)

These fields are computed from canonical fields and may be regenerated at any time.
**Do not treat as primary truth.**

### `n_classes` (int | null)
Number of classes. Derived from `len(process_groups)`.

### `processes_all` (list[str] | null)
Flat sorted list of all processes across all classes.
Example: `["4t", "ttH", "ttW", "ttWW", "ttZ"]`

### `class_def_str` (string | null)
Human-readable class definition for display.
Format: `"4t | ttH+ttW+ttWW+ttZ"` (with spaces)

### `process_groups_key` (string | null)
Compact key for W&B grouping (no spaces).
Format: `"4t|ttH+ttW+ttWW+ttZ"`

This is the **preferred W&B "group by" key**.

### `row_key` (string | null)
Full key for coverage grid rows (dataset + process groups).
Format: `"4topsplitted::4t|ttH+ttW+ttWW+ttZ"` (no spaces)

---

## Process ID to Name Mapping

```python
PROCESS_ID_NAMES = {
    1: "4t",    # tttt
    2: "ttH",
    3: "ttW",
    4: "ttWW",
    5: "ttZ",
}
```

---

## Example `facts/meta.json`

```json
{
  "schema_version": 1,
  "level": "sim_event",
  "goal": "classification",
  "dataset_name": "4topsplitted",
  "model_family": "transformer",
  "process_groups": [["4t"], ["ttH", "ttW", "ttWW", "ttZ"]],
  "datatreatment": {
    "token_order": "input_order",
    "tokenization": "direct",
    "pid_encoding": "embedded",
    "id_embed_dim": 8,
    "met_rep": "met_metphi",
    "globals_included": true,
    "feature_set_version": "v1",
    "normalization": "zscore"
  },
  "meta_hash": "sha1:a1b2c3d4e5f6",
  "meta_source": "live",
  "meta_confidence": "high",
  "meta_confidence_fields": {
    "dataset_name": "high",
    "goal": "high",
    "model_family": "high",
    "process_groups": "high",
    "datatreatment": "high"
  },
  "needs_review": false,
  "needs_review_reason": [],
  "n_classes": 2,
  "processes_all": ["4t", "ttH", "ttW", "ttWW", "ttZ"],
  "class_def_str": "4t | ttH+ttW+ttWW+ttZ",
  "process_groups_key": "4t|ttH+ttW+ttWW+ttZ",
  "row_key": "4topsplitted::4t|ttH+ttW+ttWW+ttZ"
}
```

---

## W&B Integration

### Config Fields

All meta fields are uploaded to W&B config with `meta.*` prefix:
- `meta.schema_version`
- `meta.level`
- `meta.goal`
- `meta.model_family`
- `meta.dataset_name`
- `meta.process_groups` (JSON string)
- `meta.process_groups_key` (best for grouping)
- `meta.class_def_str` (for display)
- `meta.row_key`
- `meta.n_classes`
- `meta.datatreatment` (JSON string)
- `meta.datatreatment_tokenization` (direct | binned | vq – for W&B filtering)
- `meta.meta_hash`
- `meta.meta_confidence`
- `meta.needs_review`

### Tags (Keep Minimal)

Only these tags are added:
- `level:sim_event`
- `goal:classification` or `goal:anomaly_detection`
- `family:transformer` / `family:mlp` / `family:bdt` / `family:ae`
- `dataset:4topsplitted`
- `needs_review` (if true)

Everything else should be config fields, not tags.

---

## Manual Override Workflow

For runs with `needs_review: true`:

1. Create `facts/meta_override.json` in the run directory:
```json
{
  "process_groups": [["4t"], ["ttH", "ttW", "ttWW", "ttZ"]],
  "goal": "classification"
}
```

2. Run backfill with `--apply-overrides`:
```bash
python scripts/backfill_meta.py --runs-dir ... --apply-overrides
```

**Merge semantics:**
- Override values **win** over inferred values
- Fields from override get `meta_source: "override"`
- Provides clear pathway to 100% coverage grid

---

## Coverage Grid Structure

**Rows** = `row_key` (`dataset_name::process_groups_key`)

**Columns** = `model_family` (transformer, mlp, bdt, ae)

**Cells** = Three values:
- Best AUROC
- Median AUROC (guards against cherry-picking)
- n_runs (sample size)

**Default filter:** `meta.needs_review = false`
