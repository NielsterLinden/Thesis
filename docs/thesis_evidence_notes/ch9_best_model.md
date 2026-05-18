# Evidence Note: ch9 — Best Model Physics Reach

**Status:** interpreted (chapter drafted in thesis_report/mainmatter/09_best_model.tex; figures used: thesis_report/figures/ch9/ch9_mia_roc_comparison.pdf, ch9_mia_sig_eff_vs_bkg_rej.pdf, ch9_mia_score_distributions.pdf, ch9_mia_significance_vs_threshold.pdf)
**Date:** 2026-05-17
**Chapter:** 9 — The Best Model and Its Physics Reach
**Script:** `src/thesis_ml/reports/analyses/ch9_best_model.py`

---

## Best model decision

**MIA (Message-Interaction Attention) is the best model for ch9.**

MIA (`mia_od_followup_seeds`, 5 seeds) outperforms or matches all other candidates on the
correct task (ttH+ttW+ttWW+ttZ | 4t) and is the natural endpoint of the full design
exploration in chapters 5–8. Cand01 (surrogate-selected, 3 seeds, AUROC 0.8495) and Optuna
best trial (0.8485) are both marginally weaker and are shown only as reference points.

---

## Entry point chosen

**Entry point D — all data in canonical CSV.**

All MIA and baseline rows (ROC curves, score histograms, scalar metrics) are fully populated
in `thesis_results/04_cleaned_backfilled_analysis_ready.csv`. No `.pt` files or secondary
exports are needed for the primary analysis.

---

## Inventory snapshot

| Model | Group key | Seeds | ROC in CSV | score_hist in CSV |
|-------|-----------|-------|------------|------------------|
| MIA (best model) | `exp_20260511-113827_mia_od_followup_seeds` | 5 | ✓ all 5 | ✓ all 5 |
| Physics baseline | `exp_20260306-190512_4tbg_physics_baseline` | 9 | ✓ | ✓ |

---

## Data sources

| Source | Path | Used for |
|--------|------|----------|
| Main CSV | `thesis_results/04_cleaned_backfilled_analysis_ready.csv` | All MIA + baseline ROC, histograms, scalars |
| Optuna W&B export | `agent_reference/wandb_export_2026-05-17T22_02_28.058+02_00.csv` | Optuna best-trial AUROC annotation only |

---

## Confirmed metrics

### MIA (5 seeds, group exp_20260511-113827_mia_od_followup_seeds)

| Seed | AUROC | εS(1/B=50) | εS(1/B=100) | εS(1/B=1000) |
|------|-------|-----------|------------|-------------|
| job000 | 0.8500 | 0.2952 | 0.2051 | 0.0482 |
| job001 | 0.8510 | 0.3079 | 0.2039 | 0.0529 |
| job002 | 0.8500 | 0.2968 | 0.1977 | 0.0465 |
| job003 | 0.8511 | 0.2980 | 0.1981 | 0.0399 |
| job004 | 0.8500 | 0.2957 | 0.1941 | 0.0414 |
| **mean ± std** | **0.8504 ± 0.0005** | **0.2987 ± 0.0053** | **0.1998 ± 0.0046** | **0.0458 ± 0.0052** |

### Physics baseline (9 seeds, group exp_20260306-190512_4tbg_physics_baseline)

| Metric | Mean ± Std |
|--------|-----------|
| AUROC | 0.8063 ± 0.0013 |
| εS(1/B=50) | 0.1721 ± 0.003 |
| εS(1/B=100) | 0.1037 ± 0.002 |
| εS(1/B=1000) | 0.0163 ± 0.003 |

### Optuna best trial (150 trials)

| Metric | Value |
|--------|-------|
| Best AUROC | 0.8485 |

---

## Plots produced

| LaTeX path (`thesis_report/figures/ch9/`) | Source (`/data/atlas/.../plots/ch9/`) | Description |
|------------------------------------------|--------------------------------------|-------------|
| `ch9_mia_roc_comparison.pdf` | `roc_comparison.pdf` | ROC curves: MIA mean ± std band (5 seeds) vs. physics_baseline (9 seeds); Optuna AUROC annotation; working points 1/B = 50, 100, 1000 marked |
| `ch9_mia_sig_eff_vs_bkg_rej.pdf` | `sig_eff_vs_bkg_rej.pdf` | Signal efficiency (TPR) vs. background rejection (1/FPR, log x-axis); εS values labelled at working points |
| `ch9_mia_score_distributions.pdf` | `score_distributions.pdf` | Normalised signal/background discriminant histograms; MIA vs. physics_baseline |
| `ch9_mia_significance_vs_threshold.pdf` | `significance_vs_threshold.pdf` | Illustrative Z vs. score cut; σ(4t)=12 fb, σ(bkg)=800 fb, L=300 fb⁻¹, 10% syst; Z=2/5 reference lines; caveat annotation on plot |

---

## Physics assumptions for significance plot

These are explicitly stated in the plot annotation:

- σ(4t) = 12.0 fb — ATLAS measurement, arXiv:2303.15061
- σ(bkg combined) = 800 fb — order-of-magnitude placeholder for ttH+ttW+ttWW+ttZ combined
- L = 300 fb⁻¹ — HL-LHC benchmark
- N_MC_signal = 15375, N_MC_bkg = 14832 — test set counts from test_scores.pt (cand01 seed 0)
- Systematic on background = 10% (flat, uncorrelated)
- Significance formula: Z = S / sqrt(B + (0.1 B)^2)

**These yields are illustrative. The plot is annotated: "Not a full profile-likelihood fit."**

---

## Confounders and limitations

1. Cand01 ROC curves are derived from raw probabilities in test_scores.pt rather than the standardised eval pipeline, so there may be small numerical differences vs. the W&B-logged AUROC. Both sources are internally consistent.
2. The physics_baseline uses 9 seeds (all available); cand01 uses 3 seeds. The baseline std band therefore understates seed variance.
3. The significance plot uses a single representative set of MC event counts (from cand01 seed 0 test split) for the signal/background histograms. All three seeds see the same test split, so this is stable.
4. The background cross-section (800 fb) is a rough order-of-magnitude estimate. The plot annotation makes this explicit.
5. Optuna best trial: the curve is absent (score_hist/roc columns are empty in the Optuna export); only the AUROC scalar is shown as a legend annotation.

---

## Run command

```bash
cd /project/atlas/users/nterlind/Thesis-Code
source ~/.bashrc && thesis
python src/thesis_ml/reports/analyses/ch9_best_model.py
```

Expected output: 4 PDF files in `/data/atlas/users/nterlind/outputs/plots/ch9/` plus a summary table printed to stdout.

---

## Reused vs. new code

- **Reused:** `thesis_ml.reports.plots.style` (apply_thesis_style, axis_color, figure_size, CATEGORICAL_COLORS)
- **New:** `src/thesis_ml/reports/analyses/ch9_best_model.py` — standalone script, no Hydra config required
- No new modules were added to `src/thesis_ml/reports/analyses/` beyond the script itself.

---

## Thesis-safe interpretation paragraph

The MIA model achieves a mean test AUROC of 0.8504 ± 0.0005 across five independent
training seeds, compared with 0.8063 ± 0.0013 for the simple physics baseline — an
improvement of Δ AUROC ≈ 0.044. The gain is most pronounced at high background-rejection
working points: at 1/B = 100, MIA achieves signal efficiency εS ≈ 0.200 versus ≈ 0.104
for the baseline, a factor-of-two improvement in signal yield at the same background
contamination. These working-point gains are physically meaningful: in a cut-based analysis
at L = 300 fb⁻¹, the improved classifier approximately doubles signal retention at fixed
background rate, directly increasing expected significance. The illustrative significance
calculation (σ(4t) = 12 fb, σ(bkg) = 800 fb, 10% flat systematic) shows that MIA reaches
Z = 2 at a substantially looser score cut than the baseline, extending the useful operating
range of the discriminant. These results are approximate — a proper sensitivity study
requires a profile-likelihood fit over all signal regions — but they demonstrate that the
architectural improvements identified in chapters 5–8 translate into tangible physics reach.
