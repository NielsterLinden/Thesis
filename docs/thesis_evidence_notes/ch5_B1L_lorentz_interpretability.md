# Evidence Note: ch5 — B1-L Lorentz Bias Interpretability (Exp 5B, entry E)

**Status: triaged**
**Chapter:** 5
**Section:** 5 — Physics-Informed Attention Biases (Lorentz sub-axes, interpretability)
**Created:** 2026-05-13
**Last updated:** 2026-05-13

> Companion to `ch5_B1L_lorentz.md`. Scope: post-hoc inspection of the 60 5B
> `model.pt` checkpoints — KAN spline shapes, per-module gate convergence,
> and sparse-gating feature selection.

---

## 1. Critical context: inputs are normalized

`H5TokenDataset` z-score normalizes (E, pT, η, φ) at the model input using
TRAIN-split statistics. `LorentzScalarBias.forward` then computes pairwise
features from those *normalized* four-vectors, so every value the bias
module learns on is on the **normalized input scale** — not in physics
units. Two consequences:

- m² and log m² collapse: the pairwise invariant computed from normalized
  four-vectors clamps near zero. Empirical validation percentiles on the
  normalized scale: `m²` p50 = 0.000, p95 = 2.47; `log m²` p50 = log(1e-6)
  ≈ –13.8 for the bulk of pairs. The KAN spline grid adapts to that
  distribution, but the splines have almost no signal to learn from.
  *Conclusion:* "the KAN learns physical mass peaks at W/top scale" is
  **not** an honest reading of these checkpoints; that argument would
  require either training on un-normalized inputs or de-normalizing the
  feature axis.
- ΔR, log kT, z, ΔR·pT remain well-behaved on the normalized scale (ΔR p5
  = 0.79, p95 = 4.17; log kT p5 = –0.43, p95 = 1.27). These are the
  features for which spline-shape interpretability is meaningful.

All figures use normalized inputs and the captions say so explicitly.

---

## 2. Entry point and tooling

**Entry point: E (load checkpoints, no inference).** The KAN sub-module
is reconstructed forward-only from each `model.pt` state dict; no full
model is built and no validation forward pass is required for the spline
plots themselves. Validation features are replayed once
(`scripts/one_off/_ch5_lorentz_interp.py::replay_validation_features`) to
recover empirical feature ranges and the medians used as
partial-dependence baselines.

| Script | Purpose |
|---|---|
| `scripts/one_off/_ch5_lorentz_interp.py` | Run index, vendored `KANLinear` forward, `LorentzScalarBias` reconstruction, validation-feature replay, 1-D / 2-D sweep helpers. |
| `scripts/one_off/ch5_lorentz_interp_figA.py` | Figure A — 1-D bias curves. |
| `scripts/one_off/ch5_lorentz_interp_figB.py` | Figure B — 2-D bias surface. |
| `scripts/one_off/ch5_lorentz_interp_figC.py` | Figure C — sparse gates + module-gate strip. |

Inventory: 60/60 `model.pt` present (verified in `ch5_B1L_lorentz.md`).

---

## 3. Figures staged

All under `/data/atlas/users/nterlind/outputs/reports/report_ch5_lorentz_interp/`.

| File | What it shows |
|---|---|
| `figure-1d_bias_kan_vs_standard.pdf` | 1-D bias(ΔR) (1-feature config, gate-off) and partial-dependence bias(log kT) (4-feature config, others at validation median, gate-off). Three seeds light; seed-mean bold. Standard (grey) vs KAN (teal). |
| `figure-2d_bias_surface_deltaR_logkt.pdf` | 2-D bias(ΔR, log kT) surfaces for the best-AUROC standard and KAN checkpoints in the 4-feature gate-off cohort, with validation-density contours overlaid. |
| `figure-feature_gates_and_module_gate.pdf` | Left: σ(feature_gates) heatmap for the 15 sparse-on KAN runs. Right: tanh(per-module gate) strip across all 60 runs. |

Companion CSVs (each next to its PDF): `exp5b_1d_bias_curves.csv`,
`exp5b_module_gate_values.csv`, `exp5b_feature_gates.csv`.

---

## 4. Quantitative findings

### 4.1 KAN extracts substantially more structure than standard MLP

Range of the seed-mean bias function on the swept feature, gate-off cohort:

| Panel | Standard | KAN |
|---|---|---|
| bias(ΔR) — 1-feature config | 0.46 | **0.99** |
| bias(log kT) — partial dependence, 4-feature config | 0.05 | **0.65** |

The standard MLP barely learns any log kT dependence in the 4-feature
config (range 0.05, max\|y\| = 0.10), whereas the KAN learns a curve with
range 0.65. With a single input feature (ΔR) the gap shrinks but stays
≈ 2× in favour of the KAN. This is consistent with the KAN being a much
more expressive univariate-edge model on small `(F, hidden)` budgets.

### 4.2 Per-module gate

`tanh(gate)` converged values, mean ± std across the 30 runs per MLP type:

| MLP type | Gate-off | Gate-on |
|---|---|---|
| standard | 0.220 ± 0.095 | 0.217 ± 0.099 |
| **kan** | **0.280 ± 0.077** | **0.298 ± 0.079** |

All 60 runs land at `tanh(gate) > 0` (min = 0.068 for standard, 0.136 for
KAN). The bias module is non-trivially used in every checkpoint; KAN runs
let through ~30 % more of the bias signal than standard runs on average,
echoing 4.1. Sparse gating has essentially no effect on the module gate.

### 4.3 Sparse feature-gating is a no-op

σ(feature_gates) averaged across the 15 sparse-on KAN checkpoints:

| Feature | n | mean | std |
|---|---|---|---|
| log kT | 6 | 0.620 | 0.026 |
| deltaR | 12 | 0.533 | 0.016 |
| m² | 9 | 0.521 | 0.019 |
| deltaR_ptw | 3 | 0.516 | 0.026 |
| z | 6 | 0.512 | 0.037 |
| log m² | 6 | 0.494 | 0.022 |

`feature_gates` is initialised at 0 (`σ(0) = 0.5`) and barely moves: every
feature in every cohort lands inside 0.49–0.62. The only mildly elevated
feature is log kT; log m² (the feature whose input distribution is
clamped to the floor) is the only one whose mean drifts slightly below
0.5. The sparse-gating mechanism does **not** prune features at this seed
budget and training-time configuration — which is consistent with the
AUROC bars in `ch5_B1L_lorentz.md` showing gate-on cells slightly *below*
gate-off cells (the σ ≈ 0.5 attenuation only halves each feature's
amplitude without offering compensating signal).

---

## 5. Thesis-safe interpretation

> Across the 30 KAN checkpoints in Exp 5B, the learned attention-bias
> functions show substantially richer shape than the matched standard-MLP
> checkpoints: the bias(ΔR) curve in the single-feature configuration
> spans roughly twice the dynamic range under a KAN MLP, and in the
> 4-feature partial-dependence view the standard MLP barely learns any
> log kT dependence (range ≈ 0.05 in arbitrary units) while the KAN
> learns a structured curve (range ≈ 0.65). The per-module gate
> `tanh(g)` converged to a positive value in every one of the 60 5B
> runs, with KAN modules averaging ≈ 0.29 versus ≈ 0.22 for standard
> modules — the bias is non-trivially active throughout the cohort. The
> sparse feature-gating mechanism, however, does not learn meaningful
> per-feature selection at this seed budget: σ(feature_gates) stays
> within 0.49–0.62 of its 0.5 initialisation across all 15 sparse-on
> KAN runs, and the only feature whose mean drifts slightly below 0.5 is
> the one (log m²) whose input is mostly clamped to its floor under the
> z-score input normalisation. A separate but important caveat applies
> to the m²/log m² features themselves: under the model's z-score
> normalisation the pairwise invariant collapses near zero (validation
> p50 of m² is 0.000 on the normalised scale), so claims about
> "learning the W or top mass peak" are not supported by these
> checkpoints; any such analysis would require training the bias on
> un-normalised four-vectors or de-normalising the feature axis.

---

## 6. Confounders / Limitations

- **Normalized-input scale.** All bias-vs-feature plots are on the
  z-score input axis; physics units are not directly recoverable
  without re-derivation through the normalization constants.
- **3 seeds per cell.** Curve-shape uncertainty is shown by the
  per-seed overlay; treat the seed-mean line as representative, not
  significant.
- **`update_grid` was not called during training (verified via state-dict
  inspection of `mlp.0.grid`).** The KAN B-spline grid stays at the
  init range; spline expressivity is therefore bounded by that fixed
  basis and *not* by adaptive grid relocation.
- **`include_met = true` for every 5B run** — pad MET tokens contribute
  pairs that compute_pairwise_feature_set zeroes out via mask, so this
  doesn't affect the spline analysis, but it remains the case that the
  full bias matrix is over the 18-token sequence including padding.

---

## 7. Pending follow-ups (not in scope for this turn)

- Per-edge spline atlas (small multiples of every univariate edge in a
  representative best-AUROC KAN checkpoint) — appendix material if the
  chapter wants it.
- A "de-normalized" version of bias(ΔR) showing the physical ΔR axis on
  the secondary scale (recoverable for ΔR; not trivially for m²).
- Standard-vs-KAN comparison on the 6-feature config — present
  checkpoints exist, only an extension of the same scripts is needed.
- Promote evidence-note status to `final` after the user reviews the
  three PDFs.

---

## Imported figures

| Destination (thesis_report/figures/ch5/) | Source (/data/atlas/users/nterlind/outputs/reports/) | LaTeX label |
|---|---|---|
| `figure-1d_bias_kan_vs_standard.pdf` | `report_ch5_lorentz_interp/figure-1d_bias_kan_vs_standard.pdf` | `fig:5b_bias_1d` |
| `figure-feature_gates_and_module_gate.pdf` | `report_ch5_lorentz_interp/figure-feature_gates_and_module_gate.pdf` | `fig:5b_gates` |
| `figure-2d_bias_surface_deltaR_logkt.pdf` | `report_ch5_lorentz_interp/figure-2d_bias_surface_deltaR_logkt.pdf` | (not yet imported — available) |
