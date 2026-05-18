# Evidence Note: ch2 — Data Treatment (Gaps 2, 3, 4)

**Status: interpreted**
**Chapter:** 2
**Section:** 2.x — Data Treatment (dataset description)
**Created:** 2026-05-15
**Last updated:** 2026-05-15

All three gaps (2, 3, 4) are resolved. Gap 2 was confirmed by direct h5py audit of the HDF5
file. Gaps 3 and 4 were resolved from source-code inspection and the canonical CSV.

---

## 1. Inventory Snapshot

| Item | Status |
|---|---|
| HDF5 file | `/data/atlas/users/nterlind/datasets/4tops_splitted.h5` — present |
| H5 loader source | `/project/atlas/users/nterlind/Thesis-Code/src/thesis_ml/data/h5_loader.py` — read |
| Data configs | `configs/data/h5_tokens.yaml`, `configs/data/order/pt_ordered.yaml`, `configs/data/order/shuffled.yaml` — read |
| Canonical CSV | `thesis_results/04_cleaned_backfilled_analysis_ready.csv` — queried (1253 rows) |
| W&B group | N/A — this note concerns static dataset properties, not training runs |
| model.pt | N/A |
| report directory | N/A |

**Entry point: E** (bespoke inspection). This note does not produce training or inference outputs. It records factual properties of the dataset and data-loading code for use in thesis Chapter 2 prose.

Higher-cost entry points not applicable: no new training, inference, or report run is needed.

---

## 2. Axes Covered

| Axis ID | Name | Config Key | W&B key |
|---|---|---|---|
| D3 | Token Ordering | `data.sort_tokens_by`, `data.shuffle_tokens` | `config/axes/D3_Token Ordering` |
| D1 | Feature Set | `data.cont_features` | `config/axes/D1_Feature Set` |

No other axes are swept; this note covers dataset-level constants, not model-axis comparisons.

---

## 3. Gap 2 — Sample Counts per Split and per Class

### Status: RESOLVED

Confirmed by direct h5py audit of `/data/atlas/users/nterlind/datasets/4tops_splitted.h5`.

### What we know from source code

The config file `configs/data/h5_tokens.yaml` sets:

```yaml
datasets:
  x_train: "X_train"
  x_val:   "X_val"
  x_test:  "X_test"
  y_train: "Y_train"   # capital Y — file is inconsistently cased
  y_val:   "y_val"     # lowercase y
  y_test:  "y_test"    # lowercase y
n_tokens: 18
globals:
  present: true
  size: 2
```

The comment at `h5_loader.py` line 85 shows `# shapes: [N, 92]` for the X splits.
With T=18 tokens, 2 globals, and T×4=72 continuous features, the flat event dimension is
18 + 2 + 72 = 92 columns. This is consistent.

From `ch4.1_tokenizer_family.md` (evidence note, Section 5): with the binary task filter
(selected_labels=[1,2], 4tops vs. ttH only), the test split contains **30,207 events**.
The full file will contain more events (labels 1–5 are present based on the process-ID name
mapping in `h5_loader.py` lines 463–464).

### Confirmed sample counts

HDF5 keys present: `X_train`, `Y_train`, `X_val`, `y_val`, `X_test`, `y_test`.
Feature dimension: 92 columns per event (flat row layout, consistent with T=18 tokens,
2 globals, 4 continuous features per token: 18×4 + 18 + 2 = 92).

| Split | X shape | Y shape | Label 1 (4t) | Label 2 (ttH) | Label 3 (ttW) | Label 4 (ttWW) | Label 5 (ttZ) | Total |
|---|---|---|---|---|---|---|---|---|
| train | `[241658, 92]` | `[241658]` | 121,359 | 28,854 | 30,511 | 30,461 | 30,473 | 241,658 |
| val   | `[30207, 92]`  | `[30207]`  | 15,266  | 3,666  | 3,779  | 3,760  | 3,736  | 30,207  |
| test  | `[30207, 92]`  | `[30207]`  | 15,375  | 3,552  | 3,710  | 3,779  | 3,791  | 30,207  |
| **all** | — | — | **152,000** | **36,072** | **38,000** | **37,000** | **37,000** (approx) | **302,072** |

Split ratio: 80 / 10 / 10 (241,658 / 30,207 / 30,207).

Label 1 (signal, tttt) comprises approximately 50% of the dataset; the four background
processes (ttH, ttW, ttWW, ttZ) are roughly balanced at 10–12% each.

Note: label codes are ProcessID integers. Mapping from `h5_loader.py` line 463:
`{1: "4t", 2: "ttH", 3: "ttW", 4: "ttWW", 5: "ttZ"}`.

---

## 4. Gap 3 — Z-Score Normalisation Scope

### Status: RESOLVED

**Finding: normalisation collapses both the sample and token-position dimensions, giving one scalar
(mu and sigma) per continuous feature, shared across all tokens and all events.**

This applies to both active dataset classes:

#### H5ClassificationDataset (used for all thesis classifier runs)

Source: `src/thesis_ml/data/h5_loader.py`, lines 551–553:

```python
train_cont = self.splits["train"][0][:, self.T + 2 :].view(-1, self.T, 4)
self.mu = train_cont.mean(dim=(0, 1), keepdim=True)   # shape: [1, 1, 4]
self.sd = train_cont.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)
```

`dim=(0, 1)` collapses axis-0 (sample N) and axis-1 (token position T), leaving axis-2
(continuous feature index). Result shapes are `[1, 1, 4]`, i.e. one scalar per feature.

If `data.cont_features` selects a subset (e.g. `[1, 2, 3]` for the 4-vector ablation),
the slice is applied *before* computing stats, so `mu` and `sd` have shape `[1, 1, cont_dim]`
with `cont_dim = len(cont_features)`.

#### H5TokenDataset (legacy loader, pre-classifier refactor)

Source: `src/thesis_ml/data/h5_loader.py`, lines 112–116:

```python
train_cont_all = self.splits["train"][:, self.T + 2 :].view(-1, self.T, 4)
train_cont = train_cont_all[:, :, self.cont_features]  # [N, T, cont_dim]
self.mu = train_cont.mean(dim=(0, 1), keepdim=True)  # [1, 1, cont_dim]
self.sd = train_cont.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)
```

Identical scope: `dim=(0,1)`, giving `[1, 1, cont_dim]`.

#### Summary

| Property | Value |
|---|---|
| Scope | Global across all token positions AND all training events |
| Computed from | Training split only (val and test are NOT used) |
| Result shape | `[1, 1, cont_dim]` — one scalar per continuous feature |
| `cont_dim` (default) | 4 (features E, pT, eta, phi; indices 0,1,2,3) |
| `cont_dim` (4-vector ablation) | 3 (pT, eta, phi; indices 1,2,3) |
| Applied at | `get_split()` call time; normalisation is broadcast per `[N, T, cont_dim]` batch |
| Clamp | `sd.clamp_min(1e-8)` — prevents divide-by-zero for degenerate features |

**Interpretation for thesis:** The normalisation is per-feature-type, not per-token-position. All
18 token slots share the same mu/sigma for a given feature (e.g. pT), estimated over the
full training distribution. This is the standard "global z-score" approach; it does not
treat different token positions (e.g. leading jet vs. 18th jet) differently.

---

## 5. Gap 4 — Default Token Ordering

### Status: RESOLVED

#### From canonical CSV (`thesis_results/04_cleaned_backfilled_analysis_ready.csv`)

Query: value counts on column `config/axes/D3_Token Ordering` across all 1253 rows.

| D3 value | Count | Fraction |
|---|---|---|
| `input_order` | 1236 | 98.6% |
| `shuffled` | 14 | 1.1% |
| `pt_sorted` | 3 | 0.2% |

**The overwhelming default across all thesis runs is `input_order`.**

#### From config files

`configs/data/h5_tokens.yaml` does NOT set `sort_tokens_by` or `shuffle_tokens`,
so neither key is present in the default data config. The loader code interprets this as:

- `shuffle_tokens = False` (default, `h5_loader.py` line 601)
- `sort_tokens_by = None` (default, `h5_loader.py` line 602)

With both absent/False/None, `_token_order_permutation()` at line 76 falls through to:
```python
return torch.arange(T, dtype=torch.long)
```
— the identity permutation, preserving the original order in the HDF5 file.

The `order/` sub-configs (`configs/data/order/pt_ordered.yaml` and `shuffled.yaml`) are
override fragments that must be explicitly composed in the Hydra CLI or experiment YAML to
activate non-default ordering:

```yaml
# configs/data/order/pt_ordered.yaml
# @package data
sort_tokens_by: pt
shuffle_tokens: false
```

```yaml
# configs/data/order/shuffled.yaml
# @package data
shuffle_tokens: true
```

#### Interpretation

"Input order" means the token sequence is presented in the same order as the events are
stored in the HDF5 file. The provenance of that ordering within the file is unknown (see
confounders, Section 7). It is NOT pt-sorted by the data loader unless `order/pt_ordered`
is composed.

---

## 6. What Was Held Fixed (Confounders)

- **Split provenance:** The file is labelled `4tops_splitted.h5`. The splitting criterion
  (stratified vs. random, split fractions, random seed) is unknown — the file was received
  pre-split. This is a provenance gap. No split-resampling is performed at load time.
- **All five process labels (1–5) are present** in the file; the default config filters to
  `selected_labels: [1, 2]` (4tops vs. ttH binary task). Per-label counts for all five
  labels are now confirmed (see Section 3).
- **Token ordering in the file:** The HDF5 ordering within each event (which physical object
  appears at position 0, 1, …, 17) is unknown. If the file was created with objects sorted
  by some criterion (e.g. pT), then `input_order` is effectively pT-sorted, and the
  `pt_ordered.yaml` override would be a no-op. This cannot be determined without examining
  the file-creation script; no such script is available in the repo.
- **Normalisation uses training stats only.** Val and test events are normalised using mu/sd
  computed from training events. This is the standard ML practice; it prevents look-ahead
  leakage from test data.

---

## 7. Staged Action

No Condor job needed. All three gaps are resolved.

**Full output path:** None — this evidence note contains no generated artifacts. All
results are confirmed from direct file inspection and source-code reading.

---

## 8. Thesis-Safe Interpretation

### Gap 3 — Normalisation

The continuous kinematic features (E, pT, eta, phi) are z-score normalised using statistics
computed over the entire training set, pooled across all 18 token positions. This produces
four global scalars (one mean and one standard deviation per feature type) that are applied
uniformly to every token in every event, regardless of particle rank or identity. The choice
to pool across token positions avoids inducing spurious positional meaning from the
normalisation step itself; it does not, however, account for the different kinematic
distributions of different particle types (jets vs. leptons vs. MET), which may have very
different pT ranges. A per-type normalisation was not used.

### Gap 4 — Token Ordering

Across 98.6% of all thesis training runs (1236/1253), tokens are presented in the order
they appear in the HDF5 file (`input_order`). No reordering is applied at load time for
these runs. A small subset of runs (14 shuffled, 3 pt-sorted) explored alternative orderings
for the Chapter 4 data-treatment experiment. The default `input_order` behaviour is
therefore the operative assumption for all Chapter 5–8 results. If the HDF5 file stores
objects in pT-descending order (which is common in ATLAS analysis ntuples), `input_order`
is functionally equivalent to `pt_sorted`; this cannot be confirmed without the
file-creation provenance.

---

## 9. Open Questions

- **HDF5 file provenance is unknown.** Split fractions, stratification criterion, and
  within-event object ordering are not documented in the repo. The thesis should state that
  the file was received pre-split and that the splitting methodology is inherited from the
  upstream analysis.
- **Within-event object ordering.** If the HDF5 file was created with objects pre-sorted by
  pT, then `input_order` is functionally equivalent to `pt_sorted`. This cannot be confirmed
  without the file-creation script, which is not available in the repo.
