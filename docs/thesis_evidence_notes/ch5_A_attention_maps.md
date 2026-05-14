# Evidence Note: ch5 — 5A Attention Maps (none vs full-combination)

**Status: triaged**
**Chapter:** 5
**Section:** 5A — Bias Activation Set
**Created:** 2026-05-14
**Last updated:** 2026-05-14

> Companion to `ch5_B1_bias_families.md`. Scope: post-hoc attention-weight
> inspection of the `none` baseline and `full`-combination checkpoints from
> Exp 5A, extracted at inference time.

---

## 1. Method

**Entry point: B** (full forward pass required). Attention weights are
extracted via `model.encoder(..., capture_attention=True)` and returned as
`[batch, heads, layers, seq_len, seq_len]` tensors.

**Sequence layout:** `[CLS] + [18 physical tokens] + [MET] + [MET-phi]` = 21
tokens total. `PHYS_SLICE = slice(1, 19)` isolates the 18 physical particle
tokens. Pad tokens (type 0) are excluded from all aggregation and display.

**Aggregation:** For each event, attention is averaged over all heads and
layers. Each `(type_i → type_j)` cell is the mean weight from all tokens of
type `i` attending to all tokens of type `j`, then averaged over 2000
validation events and all 3 seeds per family.

**Script:** `scripts/one_off/ch5_attention_maps.py`

**Figures:** `/data/atlas/users/nterlind/outputs/reports/report_ch5_attention_maps/`

| File | What it shows |
|---|---|
| `figure-attn_typepair_mean_none_vs_full.pdf` | Two 7×7 type-pair heatmaps: `none` baseline (left) vs `full` combination (right), averaged over all heads/layers/seeds/events |
| `figure-attn_typepair_per_head_none.pdf` | 4 heads × 3 layers grid for `none` baseline (seed 42) |
| `figure-attn_typepair_per_head_full.pdf` | 4 heads × 3 layers grid for `full` combination (seed 42) |

Companion CSV: `exp5a_typepair_mean_attn.csv`

---

## 2. Quantitative findings

### 2.1 None baseline — diffuse attention

The `none` baseline shows near-uniform type-pair attention in the range
0.059–0.113. There is no dominant pair; the slight elevations are:

| Pair (i → j) | mean_attn |
|---|---|
| mu+ → e- | 0.113 |
| photon → mu- | 0.112 |
| e- → mu+ | 0.111 |
| mu- → photon | 0.109 |

These are modest fluctuations around the uniform-attention baseline of ≈ 1/7
≈ 0.143 (before bias correction); without physics-informed biases the model
distributes attention roughly evenly across particle types.

### 2.2 Full combination — physics-structured attention

Adding all four bias families sharpens attention onto physically meaningful
pairs. Strongest elevated cells (mean_attn ≥ 0.13):

| Pair (i → j) | mean_attn | Physics interpretation |
|---|---|---|
| mu- → photon | **0.178** | Lepton–photon association (FSR / ee→μγ) |
| photon → photon | **0.164** | Photon self-attention (isolation / cluster) |
| e- → e+ | 0.139 | Opposite-sign lepton pair (Z/W decay) |
| jet → mu+ | 0.145 | b-decay muon inside jet cone |
| b-jet → mu+ | 0.142 | Same — direct b→μ association |
| b-jet → mu- | 0.133 | Same |
| jet → mu- | 0.129 | b-decay muon |
| e+ → e- | 0.129 | Opposite-sign lepton pair |
| b-jet → e- | 0.130 | b→e semi-leptonic decay |

Depressed cells (mean_attn ≤ 0.055) are inter-lepton same-charge pairs
(e.g. mu- → e+: 0.039, e+ → mu-: 0.049) and lepton self-attention.

### 2.3 Summary contrast

| Metric | none | full |
|---|---|---|
| max type-pair attention | 0.113 | 0.178 |
| min type-pair attention | 0.059 | 0.039 |
| range (max − min) | 0.054 | 0.139 |

The full-combination model's attention range is ≈ 2.6× wider than the
baseline, reflecting structured physical priors learned through the bias
families.

---

## 3. Thesis-safe interpretation

> In the `none` baseline, type-pair attention weights are diffuse (range ≈
> 0.05), with no pair exceeding 0.11. When all four physics-informed bias
> families are active (`full`), the model sharpens attention onto pairs with
> clear physical interpretations: lepton–photon pairs (mu-→photon: 0.178),
> photon self-attention (0.164), opposite-sign lepton pairs consistent with
> Z/W-boson decay (e-→e+: 0.139, e+→e-: 0.129), and jet/b-jet→muon pairs
> consistent with semi-leptonic b-decay (jet→mu+: 0.145). The total
> attention range widens by a factor of ≈ 2.6 relative to the unbiased
> baseline. This demonstrates that the physics-informed biases do not merely
> shift global scale — they restructure which particle-type relationships
> the model attends to.

---

## 4. Caveats

- **Aggregated signal only.** Type-pair averaging collapses per-head and
  per-event structure. The per-head grids (figure B/C) show the spread across
  heads and layers; some heads are more uniform than others.
- **3 seeds, seed 42 for per-head figures.** Seed-mean is the primary
  quantity; per-head figures use seed 42 as representative.
- **Input normalization.** Attention weights are extracted from the model
  operating on z-score normalized inputs; the bias modules (where active)
  compute pairwise features in that normalized space (see `ch5_B1L_lorentz_interpretability.md` §1 for the normalization caveat specific to m²).
- **Attended tokens only.** MET and MET-phi tokens are in the sequence but
  excluded from this type-pair display; their attention contributions are not
  shown.

---

## Imported figures

| Destination (thesis_report/figures/ch5/) | Source (/data/atlas/users/nterlind/outputs/reports/) | LaTeX label |
|---|---|---|
| `figure-attn_typepair_mean_none_vs_full.pdf` | `report_ch5_attention_maps/figure-attn_typepair_mean_none_vs_full.pdf` | `fig:5a_attn_mean` |
| `figure-attn_typepair_per_head_full.pdf` | `report_ch5_attention_maps/figure-attn_typepair_per_head_full.pdf` | `fig:5a_attn_per_head_full` |
| `figure-attn_typepair_per_head_none.pdf` | `report_ch5_attention_maps/figure-attn_typepair_per_head_none.pdf` | (not yet imported — supplementary) |
