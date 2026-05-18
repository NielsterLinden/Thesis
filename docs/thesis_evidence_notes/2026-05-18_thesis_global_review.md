# Thesis-wide global review and staging audit

**Date:** 2026-05-18
**Author:** reviewer agent (read-only)
**Scope:** All thesis source files under `thesis_report/`, all evidence notes under `docs/thesis_evidence_notes/`, plotting modules under `src/thesis_ml/reports/analyses/`, the axes canonical reference `docs/AXES_REFERENCE_V2.md`, and the analysis-ready table `thesis_results/04_cleaned_backfilled_analysis_ready.csv`.
**Status:** STAGING — diagnostic only, no `.tex`, code, config, or figure was modified. The chapter-by-chapter rewrite has not been started; this document tells the writer what must be fixed first.

> Severity legend: `BLOCKER` (rewrite cannot proceed until resolved), `HIGH` (must be fixed before submission), `MEDIUM` (should be fixed; minor scientific or structural risk), `LOW` (cosmetic / stylistic).

---

## 1. Executive diagnosis

The thesis is, structurally, in the second half of a healthy production cycle: nine of ten main-matter chapters are drafted, the evidence-note ledger is in place for chapters 2–9, all per-chapter figures live under `thesis_report/figures/ch4/` … `ch9/`, the canonical analysis table (`thesis_results/04_cleaned_backfilled_analysis_ready.csv`) is frozen, and a working plotting style module (`src/thesis_ml/reports/plots/style.py`) exists. The main scientific arc — workbench → axis ablations → global surrogate → recommended model → physics reach — is recognisable from the table of contents alone.

The risks fall into three groups.

**Group 1 — scientific integrity (BLOCKER / HIGH).** Two large inconsistencies stand out. First, the introduction (`01_introduction.tex:217`) and the conclusion (`10_conclusion_and_outlook.tex:6`) describe the workbench as "33 orthogonal axes", a framing that contradicts `docs/AXES_REFERENCE_V2.md` Section 7 (~90 configurable dimensions; explicitly described as a *dependency tree with activation sets, not a fully orthogonal parameter space*). Second, Chapter 8 promotes `cand01` (AUROC 0.8495, identity tokeniser + Lorentz-scalar bias + MIA blocks, **standard attention** per `08_global_surrogate.tex:516–523`), but Chapter 9 then evaluates a different model named "MIA" (group `mia_od_followup_seeds`, AUROC 0.8504, **differential attention + split bias mode** per `09_best_model.tex:22–30`). The handover claim that "MIA accumulates all architectural recommendations from Chapters 5–8" is not the same model as the one Chapter 8 promotes. This must be reconciled before any rewrite touches Chapters 8, 9, or 10.

**Group 2 — chapter-to-chapter alignment (HIGH / MEDIUM).** The chapter numbering shifted during writing and the text still carries fossils: Chapter 5 refers to attention-bias analysis as "Ch.\,6" in the bias-family carry-forward (`05_physics_informed_attention_biases.tex:734`), Chapter 5 also calls best-model selection "Ch.\,8" (`:679`), the introduction reader's guide says Part II is Chapters 4–9 and Part III is Chapters 10–11 (`01_introduction.tex:258`) but the actual `report.tex` ends mainmatter at Chapter 10, and `10_conclusion_and_outlook.tex:39` cites `\ref{ch:preenc}` which is not defined anywhere in the source — confirmed broken cross-reference. A duplicate Chapter 8 file `08_what_does_not_matter.tex` exists in `mainmatter/` but is **not** included by `report.tex`; it must be archived or deleted to avoid confusion during the rewrite. Chapter 4 reports an AUROC of "approximately 0.842" for the identity tokeniser (`04_best_input_representation.tex:103`) while Chapter 5 reports the `none`-baseline at 0.84125 ± 0.00068; these are consistent in magnitude but the writer must verify the cohorts before cross-citing them.

**Group 3 — cosmetic / readability (MEDIUM / LOW).** The introduction still has six `\todo{Figure: …}` placeholders (SM table, Feynman diagrams, LHC, ATLAS detector, transformer diagram, four-top event diagram) and two `\todo{ask supervisor: …}` items in Chapter 2 (HDF5 splitting methodology, within-event object ordering). Every `\wandbaxis{}` macro currently expands to a placeholder URL (`WANDB_AXIS_LINK_PLACEHOLDER_*` in `report.tex:18`), which is fine for draft but must be set before the final compile. The figure inventory is otherwise visually rough: Chapter 7 Pareto panels are functional but un-styled; Chapter 8 SHAP beeswarms are minimally captioned; Chapter 9 has tasteful colour use but mixes ms / MiB / pp units in adjacent panels.

**What is strong.** The axes reference V2 is clean, complete, and authoritative; the evidence notes for Ch. 6.1, 6.2, 6.3, 7, 8, and 9 are detailed, source-traceable, and explicitly enumerate confounders; the plotting style module exists and is consistent; Chapter 4 has the cleanest scientific narrative (clear baseline, controlled ablation, explicit limitations); Chapter 6 cleanly states the differential-attention-by-bias interaction. The analysis-ready CSV and the evidence-note ledger together make every quantitative claim traceable in principle.

**What is risky.** The two BLOCKER items above (axis-count framing, Ch8↔Ch9 model identity), the broken cross-reference in the conclusion, three "Ch.\," fossils in Chapter 5, the missing Chapter 3 axis ID consistency with V2 (Section 3 only lists a subset of B1 sub-axes), and the conflation in Chapter 8 of "BDT/XGBoost as surrogate" vs. the wider G02 = `bdt_classifier` family.

**What is cosmetic.** Plot styling, caption uniformity, figure ordering, removal of duplicate Chapter 8 file, axis-link button population.

**What blocks readability.** Inconsistent chapter numbering in cross-references, undefined labels (`ch:preenc`), `\todo{}` placeholders rendered red in compiled PDF.

**What must be fixed before rewriting.** Items 1–9 in §10.BLOCKER + 10.HIGH.

---

## 2. Global storyline and inter-chapter alignment

The intended scientific arc (read top-to-bottom in `report.tex`):

| Ch | File | Intended question | Lens | Axes in scope (V2 IDs) | Handover to next |
|----|------|-------------------|------|------------------------|------------------|
| 1 | `01_introduction.tex` | Why do this thesis? | framing | none | sets G, D, T, A, B, F, C, H vocabulary |
| 2 | `02_dataset_and_task_definition.tex` | What is the dataset and the classification task? | framing | G01–G03, D01–D03 | establishes 4t-vs-bg primary task |
| 3 | `03_workbench_architecture_and_axes.tex` | What are the design knobs? | framing | enumerates G, D, T, E, P, A, F, B, C, H, §K, §M, §S | baseline architecture for ablation chapters |
| 4 | `04_best_input_representation.tex` | Which tokenizer / data treatment? | performance | T1, T1-a, T1-b, D01–D03 | identity + 4-vec + input_order baseline |
| 5 | `05_physics_informed_attention_biases.tex` | Do physics biases help? | interpretability | B1, B1-L, B1-T, B1-S, B1-G, §S | bias families characterised |
| 6 | `06_(differential)_attention.tex` | Which attention type / norm? | performance | A3, A3-a, A4, B1 as co-factor | recommends A3=diff, A4=none, A3-a=split |
| 7 | `07_model_scaling_and_efficiency.tex` | What model size is Pareto-optimal? | efficiency | H01, H02, H10, G03 | `d64_L6` baseline |
| 8 | `08_global_surrogate.tex` | Globally, what matters? | observational analysis | all axes | promotes `cand01` to Ch9 |
| 9 | `09_best_model.tex` | What is the physics reach of the best model? | physics | recommended config | numerical sensitivity |
| 10 | `10_conclusion_and_outlook.tex` | What did we learn? | synthesis | none | future work |

**Misalignments to flag:**

- **`HIGH` — Single best model vs. design-space workbench.** Chapter 1 (`01_introduction.tex:40–42`) explicitly says the thesis "does not propose a single optimised architecture" but develops a workbench, while Chapter 9 (`09_best_model.tex:1–16`) is titled "The Best Model and Its Physics Reach" and reports a single MIA model. The framing tension is real; the writer should pick one stance (e.g. "the workbench enables a principled choice of a best configuration, which is then evaluated against the physics baseline") and apply it consistently in 01, 08, 09, and 10.

- **`BLOCKER` — Orthogonal axes vs. dependency tree.** `01_introduction.tex:217` and `10_conclusion_and_outlook.tex:6` claim "33 orthogonal axes". Authority document `docs/AXES_REFERENCE_V2.md` Section 7 explicitly states "**dependency tree with activation sets** — not a fully orthogonal parameter space" and counts ~90 configurable dimensions. Chapter 3 (`03_workbench_architecture_and_axes.tex:294`) and Chapter 8 (`08_global_surrogate.tex:88`) say "approximately orthogonal", which is correct. The introduction and conclusion must be brought into line.

- **`BLOCKER` — Local ablations vs. global surrogate vs. final model.** Chapter 8 (`08_global_surrogate.tex:516–523`) promotes `cand01` with: identity tokeniser, learned PID embedding dim 8, Lorentz-scalar bias, standard FFN, post-LN, dim 64, depth 6, MIA blocks enabled, **standard attention** (no mention of differential), cosine LR. Chapter 9 (`09_best_model.tex:22–30`) declares MIA = group `mia_od_followup_seeds` with: **differential attention (A3), no internal norm (A4), Lorentz-scalar split (A3-a=split), recommended scale/depth**. These are not the same model. Either Chapter 9 must justify the substitution (cand01 was promoted, but a follow-up "MIA" run actually used the Ch6-recommended differential attention) or Chapter 8 must promote a configuration that matches what Chapter 9 evaluates. The evidence note `docs/thesis_evidence_notes/ch9_best_model.md` confirms the Ch9 model is `mia_od_followup_seeds` and does not reference `cand01` at all. Without reconciliation, the thesis tells two stories.

- **`HIGH` — AUROC-only conclusions vs. physics-reach claims.** Chapters 5, 6, 7, 8 are evaluated almost exclusively on test AUROC. Chapter 9 introduces signal efficiency at fixed background rejection and an illustrative significance estimate. The writer must either lift εS into the earlier chapters (Chapter 7 already reports εS at 1/B = 50; Chapters 5, 6, 8 do not) or explicitly justify why AUROC is sufficient for ranking architectures while εS is reserved for the final-model comparison.

- **`HIGH` — Old chapter / experiment numbering fossils.**
  - `05_physics_informed_attention_biases.tex:30` says "as motivated by the Ch.\,4 findings" — D02 carry-forward from Ch.4 is correct, OK as-is. **VERIFIED OK.**
  - `05_physics_informed_attention_biases.tex:679` says "for the downstream best-model selection in Ch.\,8" — best-model is now Ch.9, since Ch.8 is global surrogate.
  - `05_physics_informed_attention_biases.tex:734` says "For downstream experiments in Ch.\,6 and Ch.\,8" — Ch.6 is correct, "Ch.\,8" should be "Ch.\,9" (best model) or "Ch.\,8" (global surrogate) depending on intent.
  - `08_what_does_not_matter.tex:65,74` references "Ch.\,9" but this file is NOT included in `report.tex` (the included Ch.8 file is `08_global_surrogate.tex`); the duplicate file should be archived.
  - `10_conclusion_and_outlook.tex:28` references "Chapter~10" for the significance discussion, which is in Chapter 9 (`09_best_model.tex:158`).
  - `10_conclusion_and_outlook.tex:39` references `\ref{ch:preenc}`, a label that does not exist in any `.tex` file — broken cross-reference.

- **`HIGH` — Duplicate Chapter 8 source files.** `thesis_report/mainmatter/` contains both `08_global_surrogate.tex` (included in `report.tex:71`) and `08_what_does_not_matter.tex` (NOT included). The two files cover overlapping material with different chapter titles ("Global Surrogate-Driven Analysis" vs. "What Matters: A Global Analysis"). The unused file must be deleted or moved out of `mainmatter/` before any rewrite; otherwise the next agent will accidentally edit the wrong one.

- **`MEDIUM` — Part II/III mismatch in reader's guide.** `01_introduction.tex:258` states Part II = Chapters 4–9 and Part III = Chapters 10–11, but `report.tex` defines Part II = Chapters 4–7 and Part III = Chapters 8–10. The reader's guide must be updated.

---

## 3. Axis-framework consistency audit

Authority: `docs/AXES_REFERENCE_V2.md` (canonical); Hydra configs under `configs/classifier/` (source of truth). All claims below were checked against the V2 file.

### 3.1 Wrong axis count and framing

| Severity | Location | Current wording | Proposed correction |
|---|---|---|---|
| `BLOCKER` | `01_introduction.tex:217` | "organises 33 architectural design choices into orthogonal axes" | "organises ~90 configurable design dimensions into a dependency tree with seven architectural branches (T, E, P, A, F, B, C) plus model-size H, study-framing G, and data-treatment D groups; see Chapter 3 and AXES_REFERENCE_V2." |
| `BLOCKER` | `10_conclusion_and_outlook.tex:6` | "parameterised along 33 orthogonal axes" | same correction; clarify "configurable axes" and use V2-consistent count. |
| `HIGH` | `01_introduction.tex:41` | "architectural choices are exposed as orthogonal axes" | "exposed as named axes in a dependency tree with controlled activation sets" |
| `HIGH` | `01_introduction.tex:219` | "axes are grouped into ten thematic groups" | clarify which ten (G, D, T, E, P, A, F, B, C, H — confirm against V2 §0.1); §K, §M, §S are cross-cutting, not thematic groups, and §R, §L are explicitly not architectural axes. |

### 3.2 Axis IDs and labels — confirmed present in V2

The thesis uses `\wandbaxis{T1}`, `T1-a`, `T1-b`, `D01–D03`, `E1`, `A1`, `A2`, `A3`, `A3-a`, `A4`, `A5`, `F1`, `B1`, `B1-L1`, `B1-L2`, `B1-L5`, `B1-T1`, `B1-T2`, `B1-T3`, `B1-S1`, `B1-G1`, `C1`, `C2`, `H01–H10`, `R5`. All of these IDs appear in V2 §6 master tables. **No wrong axis IDs found** in chapters 3–9 main bodies. Note: Chapter 8 uses informal forms `R1`, `R2`, `R4`, `R5`, `D1`, `D2`, `H1`, `H2` (e.g. `08_global_surrogate.tex:531–536`) where V2 uses `R01`, `R02`, `R04`, `R05`, `D01`, `D02`, `H01`, `H02`. **Severity `MEDIUM`** — apply the V2 zero-padded form consistently.

### 3.3 Wrong / missing config keys

| Severity | Location | Issue | Fix |
|---|---|---|---|
| `MEDIUM` | `04_best_input_representation.tex:400` | `data.sort_tokens_by` is cited as the config key for D03 but V2 §2 (`D03`) lists both `data.sort_tokens_by` and `data.shuffle_tokens` and notes D03 is an *inferred* axis. | Reference both keys, or use `axes/token_order` and cite the inference rule. |
| `MEDIUM` | `04_best_input_representation.tex:399` | `classifier.globals.include_met` for D02 — matches V2 §2 (`D02`). **OK.** | — |
| `MEDIUM` | `08_global_surrogate.tex:534–537` | Optuna search description uses informal names `H1` / `H2` and refers to `differential attention with shared bias mode` as fixed, but the table of fixed axes does not give Hydra keys. | Add the V2 IDs for each fixed axis and the corresponding `classifier.model.*` Hydra key. |
| `MEDIUM` | `09_best_model.tex:23` | Refers to MIA group `mia_od_followup_seeds` but does not state which Hydra config or W&B export was used to launch those runs. | Cite the originating Hydra config (`configs/classifier/experiment/thesis_experiments/...`) and the seed list. |

### 3.4 B1, A, F group naming consistency

- `B` group: in V2 the active sub-families are `B1-L` (Lorentz), `B1-T` (typepair), `B1-S` (SM interaction), `B1-G` (global-conditioned). Chapter 5 uses these correctly. **OK.**
- `A` group: V2 has A1, A2, A3, A3-a, A4, A5 (no A6). `08_what_does_not_matter.tex:20` mentions "A1--A6" — that file is the unused duplicate; if any final-form chapter references A6, that is wrong. **MEDIUM if surviving.**
- `F` group: V2 lists F1, F1-a, F1-a1, F1-b. Chapter 3 mentions F1, F1-a, F1-b. **OK.**

### 3.5 Cross-cutting blocks §K, §M, §S, §R, §L

- Chapter 3 (`:279–289`) correctly distinguishes architectural axes from §K/§M/§S cross-cutting blocks. **OK.**
- Chapter 8 (`08_global_surrogate.tex:160–167`) correctly notes R1, R2, R4, R5 are training-protocol axes whose large AUROC range reflects under-training, not architectural sensitivity. **OK** — strong example of correct §R framing.
- Chapter 8 does not explicitly mark the L (logging / interpretability) axes as out-of-scope from the SHAP attribution; the SHAP family bar (`figure-surrogate_shap_family_bar.pdf`) may be averaging L-family contributions that are pure logging side-effects. **HIGH** — re-state which families are admissible for "importance" before aggregating.

### 3.6 BDT/XGBoost ambiguity

- V2 §1 (G01, G02) lists `bdt_classifier` as one G01 task type and `bdt` as one G02 model family.
- Chapter 8 (`08_global_surrogate.tex:196`) uses XGBoost as a **surrogate over the run database**, not as a classifier of physics events.
- Chapter 3 (`03_workbench_architecture_and_axes.tex:57`) correctly clarifies "the BDT appears only in Ch.\,8 as a surrogate model for configuration prediction, not as a classifier" — but it points to Ch.\,8 in the old numbering. Verify chapter pointer.

Severity for these BDT/XGBoost framing items: `MEDIUM`. The writer must state once, near the first XGBoost mention in Ch.8, that XGBoost here is a surrogate over the experiment database and is unrelated to G02 = `bdt`.

---

## 4. Claim-evidence registry

Claims are sampled across all main-matter chapters. Status codes: `S` = supported by an evidence note + CSV / figure; `W` = weakly supported (within seed noise or not directly cited); `U` = unsupported (claim made without traceable evidence); `C` = contradicted by another part of the thesis; `R` = needs rewording for tone or precision.

| # | Claim | Ch / Section | Axis / experiment | Evidence source | Metric | Status | Action |
|---|-------|--------------|-------------------|-----------------|--------|--------|--------|
| 1 | "33 orthogonal axes" | 1 / `:217`, 10 / `:6` | meta | none — contradicted by V2 §7 | n/a | `C` | rewrite per §3.1 |
| 2 | "identity tokeniser dominates other choices by a large margin" | 4 / `:362` | T1 | ch4.1 note; Fig `figure-auroc_bar_by_tokenizer.pdf` | AUROC +0.038 vs raw, +0.117 vs binned | `S` | OK |
| 3 | "T1-a, T1-b, D01, D02, D03 are null at this scale" | 4 / `:364–366` | T1-a, T1-b, D01–D03 | ch4.2 note; Fig 4-bars + heatmap | ΔAUROC ≤ 0.003 | `S` | OK |
| 4 | "D03 ordering effect of −0.0026 attributable to sinusoidal PE" | 4 / `:344–349` | D03 × E1 | none — E1=none control not run, explicitly noted | n/a | `R` | already correctly hedged; flag as hypothesis |
| 5 | "Lorentz bias provides modest +0.0025 AUROC over 45 ch6 runs" | 6 / `:262` | B1 | ch6.3 note; Fig `ch6_B1_auroc_seedspread_bias.pdf` | +0.0025 | `S` | OK |
| 6 | "Differential attention gives +0.0036 AUROC" | 6 / `:114–116` | A3 | ch6.1 note; Fig `ch6_A3_auroc_seedspread_attn_type.pdf` | +0.0036 | `S` | OK |
| 7 | "Lorentz-scalar bias effective only with differential attention" | 6 / `:463–471` | A3 × B1 interaction | Fig `ch6_A3xB1_auroc_heatmap.pdf`; Table `tab:ch6_A3xB1_clean` | +0.0033 vs +0.0002 | `S` | OK |
| 8 | "Attention-internal normalisation is detrimental" | 6 / `:458` | A4 | Fig `ch6_A4_auroc_seedspread_attn_norm.pdf` | −0.005 | `S` — strong | OK |
| 9 | "AUROC saturates at d64_L6 (~202k params)" | 7 / `:84` | H10 | Fig `ch7_auroc_vs_model_size.pdf`; Table `tab:ch7_auroc` | mean AUROC 0.852 ± 0.025 | `S` — note std reflects cross-task variance, already disclosed | OK |
| 10 | "d64_L6 is Pareto-optimal across FLOPs/latency/throughput/memory" | 7 / `:224` | H10 | Figs `ch7_pareto_*.pdf`; Table `tab:ch7_inference` | latency 0.552 ms, mem 1968 MiB | `S` | OK |
| 11 | "Largest preset is d192_L12" | 7 / `:22, :44` | H10 | evidence note `ch7_scaling.md:73` notes thesis text previously said d256_L12 — already corrected to d192_L12 | n/a | `S` | OK if d192_L12 throughout |
| 12 | "983 runs in primary cohort" | 8 / `:46` | meta | ch8 evidence note | row count | `S` | OK |
| 13 | "T family accounts for 37.9% of SHAP" | 8 / `:300` | meta | Fig `surrogate_shap_family_bar.pdf` | normalised mean |SHAP| | `S` with caveat already stated about cohort correlation | OK |
| 14 | "cand01 promoted to Chapter 9" | 8 / `:639` | derived | Table `tab:ch8_validation` | cand01 mean AUROC 0.8495 | `C` — Chapter 9 evaluates `mia_od_followup_seeds`, not cand01 | **see §1; reconcile** |
| 15 | "MIA accumulates all architectural recommendations from Chapters 5–8" | 9 / `:30` | derived | ch9 evidence note | qualitative | `W` — claim is not directly traced to the cand01 promotion; the configuration listed in 9 (`:22–30`) overlaps with but is not identical to the cand01 spec in 8 (`:516–523`) | rewrite to match |
| 16 | "MIA AUROC 0.8504 ± 0.0005 over 5 seeds" | 9 / `:43, :55` | meta | ch9 evidence note table; CSV `04_cleaned_backfilled_analysis_ready.csv` | mean ± std | `S` | OK |
| 17 | "MIA εS at 1/B=100 is ~2× baseline" | 9 / `:117–119` | meta | Table `tab:ch9_wp_comparison` | 0.200 vs 0.104 | `S` | OK; numerical ratio matches |
| 18 | "Significance plot Z = 2 reached at substantially looser score cut" | 9 / `:227` | physics | Fig `ch9_mia_significance_vs_threshold.pdf` | qualitative; labelled "illustrative only" | `R` | already hedged; verify the figure annotation "Not a full profile-likelihood fit" is rendered |
| 19 | "Reduction in luminosity required to reach Z=5 is directly quantifiable" | 10 / `:20–21` | physics | none — Chapter 9 explicitly says "Z=5 cannot be read reliably from this simplified estimate" (`09_best_model.tex:235`) | n/a | `C` | soften the conclusion claim |
| 20 | "Workbench is process-agnostic by construction" | 10 / `:46` | meta | none — depends on `data.classifier.*` and dataset loader being swappable; not demonstrated in the thesis | n/a | `W` | reword to "designed to be process-agnostic; demonstrating transfer to other processes is future work" |
| 21 | "Anomaly detection loops available but not studied here" | 3 / `:51–53` | G01 | configs include `ae`, `gan_ae`, `diffusion_ae` per V2 §1 G01 | n/a | `S` | OK; check whether any AE artefacts make it into Ch8 cohort and are filtered |
| 22 | "RMSNorm efficiency claim" | 3 / `:162–164` | A2 | not evaluated in any ablation chapter; only normalisation policy A1 and attention-internal norm A4 are studied | n/a | `U` | either run an A2 ablation, drop the wording "improves efficiency without degrading performance", or label it as a design hypothesis not tested in this thesis |
| 23 | "Pre-norm stability claim" | not made directly in main chapters | A1 | A1 ablation is absent from chapters 4–7 | n/a | n/a | OK; do not claim what was not measured |
| 24 | "KAN extracts more functional structure" | 5 / `:382–384` | B1-L2 | Fig `figure-1d_bias_kan_vs_standard.pdf` + interpretive paragraph | range 0.99 vs 0.46 (arbitrary units) | `S` for the range claim; `R` for "more functional structure" (anthropomorphises) | soften to "the KAN-implemented bias function has roughly 2× larger output range than the standard MLP, consistent with greater expressivity at the same parameter budget" |
| 25 | "Type-pair table broadly preserves SM topology through training" | 5 / `:558–564, :721–727` | B1-T1 (fixed_coupling) | Fig `figure-typepair_learned_tables.pdf`, `figure-typepair_diff_from_init.pdf`; Table `tab:5c_table_stats` | Frobenius drift 0.93, structure intact | `S` | strong interpretability finding; keep |
| 26 | "Surrogate Spearman ρ = 0.739 ± 0.078 across 5 folds" | 8 / `:253` | meta | Table `tab:ch8_cv`; ch8 evidence note | metric | `S` | OK |
| 27 | "Surrogate-guided cand01 beats marginal-greedy candidates by 0.012 AUROC" | 8 / `:506–509` | meta | Table `tab:ch8_validation` | 0.8495 vs 0.8372 | `S` | OK |
| 28 | "Optuna best trial (0.8485) below cand01 mean" | 8 / `:556, :583–587` | meta | Table `tab:ch8_optuna`; Fig `optuna_auroc_histogram.pdf` | 0.8485 (1 seed) vs 0.8495 ± 0.0007 (3 seeds) | `S` — properly hedged | OK |
| 29 | "Dark Machines dataset 302,072 events, 80/10/10 split" | 2 / `:98–117` | data | Table `tab:sample_counts`; ch2 evidence note | counts | `S` | OK |
| 30 | "T = 18 token slots = largest event" | 2 / `:64–66` | data | citation `Visive:2026mjd` | n/a | `S` if citation correct | verify bib entry |
| 31 | "AUROC is primary metric throughout" | 3 / `:300–305`, 9 / `:43` | meta | consistent across chapters | n/a | `S` | OK |
| 32 | "Three seeds per condition unless stated otherwise" | 3 / `:298–300` | R05 | consistent; some chapters use more (e.g. Ch.7 has n=9 = 3 tasks × 3 seeds aggregated, Ch.9 uses 5 for MIA and 9 for baseline) | n/a | `S` | clarify per-chapter table |
| 33 | "Single-event latency exhibits OS jitter on interactive node" | 7 / `:161` | meta | not measured in CSV explicitly, claim is interpretive | n/a | `R` | mark as interpretation, cite the run host |
| 34 | "Anti-kT jets" | 1 / `:128` | physics | textbook reference, no citation | n/a | `R` | add citation |

Physics vs ML claim separation: the claim-evidence table makes the split explicit. Items 5, 6, 7, 8, 22, 24 are ML-motivated; items 4 (D03×E1), 25 (SM topology preservation), 17 (εS ratio), 18 (Z vs threshold) are physics-motivated. Mixed claims (e.g. "physics-informed biases restructure attention") in Chapter 5 must be split into two adjacent sentences as instructed by the writing rules.

---

## 5. Figure and plotting audit

### 5.1 Inventory

| Chapter | Figure file (relative to `thesis_report/figures/`) | Supports | Source | Caption quality | Pub-ready? | Verdict |
|---|---|---|---|---|---|---|
| 1 | (none) — six `\todo{Figure: …}` placeholders | physics framing | external sources | n/a | N | source supervisor diagrams or remove |
| 2 | (none in `figures/ch2/`) | dataset description | data loader | n/a | N — chapter has no figures | add at least one event-display or token-count histogram |
| 4 | `ch4/figure-auroc_bar_by_tokenizer.pdf` | T1 main result | `ch4_best_input_repr.py` | strong | Y | keep |
| 4 | `ch4/figure-roc_curves_by_tokenizer.pdf` | T1 ROC | same | OK | Y | keep |
| 4 | `ch4/figure-val_auroc_by_tokenizer.pdf` | T1 training dynamics | same | strong | Y | keep |
| 4 | `ch4/figure-auroc_heatmap_pid_mode_x_embed_dim.pdf` | T1-a×T1-b | same | OK | Y | keep |
| 4 | `ch4/figure-auroc_bar_by_pid_mode.pdf` | T1-a confound | same | OK; caption already explains confound | Y | keep |
| 4 | `ch4/figure-failure_analysis_raw_vs_identity.pdf` | tokenizer outcome categories | same | strong | Y | keep |
| 4 | `ch4/figure-auroc_bar_by_features.pdf`, `…_by_met.pdf`, `…_by_shuffle.pdf`, val_auroc variants | D01, D02, D03 nulls | same | OK | Y | keep; consider consolidating into a single 3-panel "null-result panel" to save space |
| 5 | `ch5/figure-auroc_bar_by_bias_family.pdf` | B1 family comparison | `ch5_*` | OK | Y | keep |
| 5 | `ch5/figure-auroc_bar_by_lorentz.pdf` | B1-L1 × B1-L2 × B1-L5 | same | dense; 20 bars × 3 seeds | partial | consider grouped layout or split into two panels |
| 5 | `ch5/figure-auroc_bar_by_typepair.pdf`, `…_by_sm_mode.pdf` | B1-T, B1-S | same | OK | Y | keep |
| 5 | `ch5/figure-typepair_learned_tables.pdf`, `figure-typepair_diff_from_init.pdf`, `figure-typepair_reference_tables.pdf` | B1-T interpretability | same | strong; physics interp in caption | Y | keep — flagship interpretability figures of Ch.5 |
| 5 | `ch5/figure-1d_bias_kan_vs_standard.pdf`, `figure-2d_bias_surface_deltaR_logkt.pdf` | B1-L2 KAN vs MLP | same | OK | Y | keep |
| 5 | `ch5/figure-feature_gates_and_module_gate.pdf` | B1-L5 sparse-gating null | same | OK | Y | keep |
| 5 | `ch5/figure-attn_typepair_mean_none_vs_full.pdf`, `…per_head_none.pdf`, `…per_head_full.pdf` | B1 attention restructuring | same | partial — small text on 12-panel grids | partial | regenerate with larger panel labels or move per-head grids to appendix |
| 6 | `ch6/ch6_A3_auroc_seedspread_attn_type.pdf` | A3 main result | `ch6_attention_mechanisms.py` | strong | Y | keep |
| 6 | `ch6/ch6_A3_per_class_auroc.pdf` | A3 per-class | same | OK | Y | keep |
| 6 | `ch6/ch6_A3_val_auroc_curves.pdf` | A3 training dynamics | same | OK | Y | keep |
| 6 | `ch6/ch6_A3_lambda_evolution.pdf` | differential λ_h | same | OK | Y | keep |
| 6 | `ch6/ch6_A3_attention_entropy_by_layer.pdf` | diff attention interp | same | OK; needs clearer statistical wording | Y | keep with caption tightening |
| 6 | `ch6/ch6_A4_auroc_seedspread_attn_norm.pdf`, `ch6_A4_val_auroc_curves.pdf` | A4 results | same | OK | Y | keep |
| 6 | `ch6/ch6_A3xA4_auroc_heatmap.pdf`, `ch6_A3xB1_auroc_heatmap.pdf`, `ch6_A4xB1_auroc_heatmap.pdf` | interactions | same | strong | Y | keep — use `RdBu_r` to make signs of effects visible |
| 6 | `ch6/ch6_B1_auroc_seedspread_bias.pdf`, `ch6_B1_val_auroc_curves.pdf` | B1 effect | same | OK | Y | keep |
| 6 | `ch6/ch6_A3a_auroc_seedspread_diff_bias_mode.pdf` | A3-a result | same | OK; the n imbalance between cells (3 vs 21) must be flagged in caption | Y | keep with caption fix |
| 7 | `ch7/ch7_auroc_vs_model_size.pdf`, `ch7_flops_vs_model_size.pdf`, `ch7_wallclock_vs_model_size.pdf`, `ch7_epoch_vs_model_size.pdf` | scaling | `ch7_scaling.py` | OK; log axes need labels stating "log scale" | partial | re-render with clearer log-axis tick labelling and consistent x-axis ordering |
| 7 | `ch7/ch7_pareto_auroc_vs_flops.pdf`, `…_latency.pdf`, `…_throughput.pdf`, `…_memory.pdf` | Pareto | same | functional | partial | annotate the Pareto frontier with a dashed step line; reuse a single shared legend across the 4 panels |
| 8 | `ch8/audit_run_counts_by_spec.pdf`, `audit_missingness_heatmap.pdf`, `audit_cramer_v_heatmap.pdf` | audit | scripts/one_off/ch8_final_plots.py | OK; Cramér's V heatmap mixes Cramér's V (categorical) with axis-family grouping — caption needs strengthening | partial | tighten captions, especially "mean Cramér's V across columns" interpretation |
| 8 | `ch8/marginals_ranked_range_bar.pdf` | marginal main result | same | strong; family color coding present | Y | keep; consider a second panel without R-family axes to expose architectural effects |
| 8 | `ch8/marginals_top5_boxdot.pdf`, `marginals_seed_noise_floor.pdf` | seed noise | same | OK | Y | keep |
| 8 | `ch8/surrogate_cv_scatter.pdf`, `surrogate_shap_family_bar.pdf`, `surrogate_shap_top5_beeswarm.pdf`, `surrogate_shap_dependence_top3.pdf` | XGBoost surrogate + SHAP | scripts/one_off/ch8_surrogate_fit.py | partial — beeswarm captions need explicit description of cyan-vs-grey encoding (active vs inactive level) | partial | rewrite captions; consider replacing the beeswarm with a stripplot with axis labels |
| 8 | `ch8/top10_scatter.pdf`, `optuna_auroc_histogram.pdf`, `validation_predicted_vs_actual.pdf` | candidate validation | same / ch8_optuna scripts | OK | Y | keep |
| 9 | `ch9/ch9_mia_roc_comparison.pdf`, `ch9_mia_sig_eff_vs_bkg_rej.pdf`, `ch9_mia_score_distributions.pdf`, `ch9_mia_significance_vs_threshold.pdf` | best model vs baseline | `ch9_best_model.py` | strong; significance plot has on-figure caveat | Y | keep; verify the "Not a full profile-likelihood fit" annotation is visible in the rendered PDF |

### 5.2 Plotting style audit

The style module `src/thesis_ml/reports/plots/style.py` defines `apply_thesis_style()`, `figure_size(...)`, `axis_color(...)`, `AXIS_GROUP_COLORS`, and a categorical cycle. It is well-designed.

- **`HIGH`** — color/axis-group consistency. `AXIS_GROUP_COLORS` defines stable hex colours per group: D=dark blue, T=sky blue, E=green, B=teal, A=pink, F=amber, C=vermillion, H=gray, plus `baseline` (near-black) and `recommended` (TU Delft cyan). The Chapter 8 marginal ranked bar (`marginals_ranked_range_bar.pdf`) already uses these. **Verify** that the same colour map is used in Chapters 4, 6, 7, 9 figures. If different colour maps are used in different chapters (likely the case for Ch.4 / Ch.5, which were generated earlier), recolour to the canonical map.
- **`HIGH`** — `plt.cm.tab10` must not be used for axis-group comparisons. Grep across `src/thesis_ml/reports/analyses/` for `tab10` or `tab20` and replace with `AXIS_GROUP_COLORS` / the categorical cycle.
- **`MEDIUM`** — hardcoded `fontsize=` arguments. Found in `ch9_best_model.py:205,245,327,338` (fontsizes 7 and 8 used for annotations). For inline annotations a hardcoded fontsize is acceptable, but axis labels / tick labels / legends must come from `THESIS_RC`. **Verify** none of the chapter scripts override `axes.labelsize`, `xtick.labelsize`, `ytick.labelsize`, `legend.fontsize`.
- **`MEDIUM`** — no `set_title` / `suptitle`. Confirmed absent in `ch7_scaling.py` and `ch9_best_model.py`. **Verify** for `ch4_best_input_repr.py`, `ch6_attention_mechanisms.py`, `scripts/one_off/ch8_*.py`.
- **`MEDIUM`** — ROC plots. `ch9_best_model.py:188` correctly draws the gray dashed diagonal, mean curves at `lw=2.5`, fill bands at `alpha=0.15`. **OK.** Ch.7 ROC curves should be re-checked for the same conventions.
- **`MEDIUM`** — heatmap colour conventions. V2-aligned: `RdBu_r` for diverging (AUROC differences should be diverging around the baseline; check `ch6_A3xB1_auroc_heatmap.pdf` and `audit_cramer_v_heatmap.pdf`), `viridis` for non-negative magnitudes, `Blues` for confusion matrices. Confirm in the chapter scripts.
- **`MEDIUM`** — spines top and right absent. `THESIS_RC` sets `axes.spines.top=False, axes.spines.right=False`. Confirm none of the scripts re-enable them.
- **`MEDIUM`** — bar charts should plot individual seed points behind bars when n_seeds > 1. Confirm in Chapter 6 figures (e.g. `ch6_A3_auroc_seedspread_attn_type.pdf` — the name suggests seed-spread, OK).
- **`NOTE`** — scatter plots with > 5k points should set `rasterized=True`. Ch.8 surrogate scatter (`surrogate_cv_scatter.pdf`) has ~948 points — OK. Ch.4 failure analysis scatter has 30,207 points — must be rasterized in the PDF, verify file size.
- **`NOTE`** — all figures are PDF. **OK.**

### 5.3 Recommended approach

Given the volume of figures (~50) already produced, a **central refresh pass via a dedicated plotting agent** is more efficient than per-chapter local fixes. The pass should:

1. Confirm every chapter plot module imports `apply_thesis_style()` at module level.
2. Replace any `plt.cm.tab10` / `tab20` with `CATEGORICAL_COLORS` / `axis_color()`.
3. Remove any `set_title` / `suptitle` in plotting modules (titles belong in LaTeX captions only).
4. Regenerate all figures to a staging directory and only commit those that visibly improved.
5. Update captions in `.tex` files so that figure interpretation lives in the body and the caption is self-contained.

Colour-encoding rules to apply consistently:
- Where panels compare axis groups (e.g. Ch.8 family SHAP bar), use `AXIS_GROUP_COLORS`.
- Where panels compare baseline vs. recommended model (Ch.9), use the `baseline` / `recommended` keys.
- Where panels show different settings within one axis (e.g. Ch.6 `A3 = standard | differential`), use a small distinct palette within that axis's family colour family (e.g. for A: shades of pink), not the per-group palette.

Captions needing stronger statistical wording: Ch.5 `tab:5b_auroc` and surrounding figures (state n explicitly; n=3 is small and the writer must state per-cell std); Ch.6 `ch6_A3a_auroc_seedspread_diff_bias_mode.pdf` (the cell sizes 3 vs 21 must be stated in the caption); Ch.7 `tab:ch7_auroc` ("std 0.025 reflects cross-task variance" already disclosed in the table caption — keep).

---

## 6. Chapter-by-chapter review

### Chapter 1 — Introduction (`01_introduction.tex`)

- **Role:** Framing — physics motivation, ML challenge, axes vocabulary, thesis statement.
- **Status:** Drafted; six `\todo{Figure: …}` placeholders.
- **Main claims:** Standard Model background, top-quark special role, four-top + backgrounds, transformers as a structurally compatible architecture, controlled-ablation methodology, 33-axis workbench.
- **Evidence:** none directly — all evidence is forward-referenced.
- **Missing / weak figures:** Standard Model table, Feynman diagrams, LHC schematic, ATLAS cutaway, supervisor four-top diagram, transformer architecture diagram. All flagged with `\todo`.
- **Outdated wording / wrong numbering:**
  - `:217` "33 architectural design choices into orthogonal axes" — **BLOCKER**, see §3.1.
  - `:258` "Part II Chapters 4–9; Part III Chapters 10–11" — wrong, see §2.
  - `:265` "Axis groups are referred to by their labels (D, A, B, P, H)" — incomplete list, should match V2 §0.1 enumeration (G, D, T, E, P, A, F, B, C, H).
- **Interchapter dependencies:** sets vocabulary used by Ch.3–9.
- **Recommended rewrite strategy:** First reconcile the "33 orthogonal axes" framing with V2, then fill in placeholder figures (most are supervisor-provided), then tighten reader's guide to match actual chapter numbering.
- **Priority:** `BLOCKER` items 1, 2.

### Chapter 2 — Dataset and task definition (`02_dataset_and_task_definition.tex`)

- **Role:** Framing — what is the data and the supervised task.
- **Status:** Drafted; two `\todo{ask supervisor: …}` items.
- **Main claims:** 302,072 events, 80/10/10 split, fixed token length T=18, MET as global, z-score normalisation pooled over training events, 4t-vs-bg as the primary task.
- **Evidence:** `ch2_data_treatment.md` (new evidence note); Table `tab:sample_counts`; citations.
- **Missing / weak figures:** chapter has no figures at all. Consider one of: token-count histogram, per-class εs preview, simple event-display sketch.
- **Outdated wording:** `\todo` items must be resolved with the supervisor or noted as deferred.
- **Recommended rewrite strategy:** add one figure; resolve `\todo`; verify bib entry for `Visive:2026mjd` is correct.
- **Priority:** `MEDIUM`.

### Chapter 3 — Workbench, architecture, axes (`03_workbench_architecture_and_axes.tex`)

- **Role:** Framing — enumerates all axes.
- **Status:** Drafted; consistent with V2 in axis IDs.
- **Main claims:** axis taxonomy, base transformer skeleton, controlled-ablation methodology.
- **Evidence:** none — definitional.
- **Missing / weak figures:** would benefit from a single high-level diagram of the dependency tree (§K, §M, §S as cross-cutting boxes) — taken from V2 §0.2 mermaid graph and rendered as a clean figure.
- **Outdated wording:**
  - `:273` "primary axis of variation in Ch.\,9" — Ch.9 is "best model"; H is the primary axis of variation in Ch.7. **HIGH** fix.
  - `:57` "BDT appears only in Ch.\,8 as a surrogate" — Ch.8 is correct.
  - `:294` "orthogonal sweep" — fine if read as "one-axis-at-a-time", but in light of §3.1 the introduction needs to be reworded too.
  - F group only lists F1, F1-a, F1-b — F1-a1 (KAN bottleneck dim) is missing. **MEDIUM.**
  - A group lists A1, A2, A3, A3-a, A4, A5 — complete and correct.
  - B group: many sub-axes are listed (B1-L1, B1-L2, B1-L5, B1-T2, B1-T3, B1-S1, B1-G1) but B1-L3, B1-L4, B1-T1, B1-T4, B1-T5, B1-S2, B1-G2, B1-G3 are absent. Decide whether to enumerate all or refer the reader to V2.
- **Interchapter dependencies:** axis IDs are referenced by every Part II chapter.
- **Recommended rewrite strategy:** add dependency-tree diagram, fix chapter-number cross-references, decide whether the full V2 enumeration of B-sub-axes belongs here or in an appendix.
- **Priority:** `HIGH`.

### Chapter 4 — Best input representation (`04_best_input_representation.tex`)

- **Role:** First ablation; sets T1/D baseline.
- **Status:** Drafted; figures present; cleanest scientific narrative in Part II.
- **Main claims:** identity tokeniser dominates; PID sub-choices and data-treatment axes are null at this scale.
- **Evidence:** ch4.1 + ch4.2 evidence notes; all figures in `figures/ch4/`.
- **Missing / weak figures:** none.
- **Outdated wording:** internal — minor wording (`:43` "with $n=3$ seeds per condition, these figures are indicative" is correctly hedged).
- **Recommended rewrite strategy:** light polish only.
- **Priority:** `LOW`.

### Chapter 5 — Physics-informed attention biases (`05_physics_informed_attention_biases.tex`)

- **Role:** Interpretability-centric ablation; sets B baseline.
- **Status:** Drafted; rich interpretability content; many figures and tables.
- **Main claims:** no bias family individually improves AUROC beyond seed noise, but bias families restructure attention towards physics-motivated patterns; KAN bias networks have 2× larger range than standard MLP at the same parameter budget; SM `fixed_coupling` initialisation preserves SM topology through training.
- **Evidence:** five evidence notes (`ch5_A_attention_maps.md`, `ch5_B1_bias_families.md`, `ch5_B1L_lorentz.md`, `ch5_B1L_lorentz_interpretability.md`, `ch5_B1T_typepair.md`, `ch5_B1S_sm_mode.md`).
- **Missing / weak figures:** the per-head, per-layer attention grids (`figure-attn_typepair_per_head_none.pdf`, `…_full.pdf`) are 12-panel grids that may be too small to read in the printed PDF; consider moving to an appendix with a single overview figure in the main text.
- **Outdated wording:**
  - `:679` "for the downstream best-model selection in Ch.\,8" — fix to Ch.9 if best model is in Ch.9; OR re-confirm that the carry-forward target is the global surrogate in Ch.8.
  - `:734` "For downstream experiments in Ch.\,6 and Ch.\,8" — Ch.6 OK, Ch.8 OK (since Ch.8 is global surrogate). Verify intent.
  - `:228` "moderate overtraining over 50 epochs" — anthropomorphises; rephrase to "validation AUROC drifts slightly below the optimum over the 50 epoch budget".
- **Interchapter dependencies:** carries forward `lorentz_scalar` to Ch.6 and (according to §5 synthesis) to Ch.8.
- **Recommended rewrite strategy:** fix chapter-number cross-references; soften anthropomorphisations; verify large attention grids render at acceptable size.
- **Priority:** `MEDIUM`.

### Chapter 6 — Attention mechanisms (`06_(differential)_attention.tex`)

- **Role:** Performance ablation on A axis.
- **Status:** Drafted; clean experimental design; correctly hedged.
- **Main claims:** A3=differential gives +0.0036 AUROC; A4=none is best; B1 effective only with differential attention; A3-a=split has no measurable advantage but is adopted as default.
- **Evidence:** ch6.1, ch6.2, ch6.3 evidence notes; all figures in `figures/ch6/`.
- **Missing / weak figures:** none.
- **Outdated wording:** filename `06_(differential)_attention.tex` contains parentheses, which is unusual; consider renaming to `06_attention_mechanisms.tex` to match the chapter title.
- **Recommended rewrite strategy:** confirm Ch.6 recommended settings (differential + none + lorentz_scalar + split) feed directly into Ch.9's MIA spec; this is the cleanest handover into Ch.9 and should be highlighted in §6.5.
- **Priority:** `LOW` (content) / `MEDIUM` (filename).

### Chapter 7 — Model scaling and efficiency (`07_model_scaling_and_efficiency.tex`)

- **Role:** Efficiency analysis on H axis.
- **Status:** Drafted; full factorial design.
- **Main claims:** AUROC saturates at d64_L6 across all three tasks; d64_L6 is Pareto-optimal across FLOPs, latency, throughput, memory.
- **Evidence:** `ch7_scaling.md`; all figures in `figures/ch7/`.
- **Missing / weak figures:** Pareto frontiers would benefit from an explicit dashed step line connecting frontier points.
- **Outdated wording:** evidence note (`ch7_scaling.md:73`) flagged a previous d256_L12 / d192_L12 mismatch — verify the current `.tex` uses d192_L12 throughout. **VERIFIED** at `:22, :44–47, :100–104`. OK.
- **Recommended rewrite strategy:** add a single panel showing AUROC and inference latency on the same axes (twin y-axis) to make the "no AUROC gain at higher latency" argument visually direct.
- **Priority:** `LOW`.

### Chapter 8 — Global surrogate (`08_global_surrogate.tex`)

- **Role:** Observational global analysis; promotes a candidate to Ch.9.
- **Status:** Drafted; substantial; surrogate fit + SHAP + Optuna comparison.
- **Main claims:** 983 runs; seed noise floor 0.000856; T-family 37.9% SHAP; cand01 wins over marginal-greedy and Optuna; cand01 promoted to Ch.9.
- **Evidence:** `ch8_global_surrogate.md` (richest evidence note in the ledger); audit / SHAP / candidate figures in `figures/ch8/`.
- **Missing / weak figures:** the marginal ranked bar mixes R-family training-protocol axes with architectural axes; a second panel with R-family axes excluded would make the architectural ranking visible.
- **Outdated wording / wrong numbering:**
  - `:163–167` "training-regime rather than genuine architectural sensitivity" — correctly stated; this disclosure should also appear under the SHAP family bar, where R = 18.4% is still attributed to "training protocol" (`:301–302`).
  - `:631–647` promotes cand01 to Ch.9 — but Ch.9 evaluates `mia_od_followup_seeds` (see §1, §2). **BLOCKER**.
  - `:534–537` Optuna axis names `D1`, `D2`, `H1`, `H2` should be `D01`, `D02`, `H01`, `H02` per V2.
- **Interchapter dependencies:** outputs cand01 spec; Ch.9 must consume that exact spec, or Ch.8 must promote the actual MIA spec.
- **Recommended rewrite strategy:** Resolve the cand01 / MIA identity question first (decision required from the writer); then add a R-excluded marginal panel; then update Optuna axis ID forms.
- **Priority:** `BLOCKER` (cand01↔MIA reconciliation), `HIGH` (R-excluded panel + axis IDs).

### Chapter 9 — Best model and physics reach (`09_best_model.tex`)

- **Role:** Final synthesis on the best architecture; physics-reach evaluation.
- **Status:** Drafted; consistent with `ch9_best_model.md` evidence note.
- **Main claims:** MIA AUROC 0.8504 ± 0.0005; physics baseline 0.8063 ± 0.0013; εS at 1/B=100 is ~2× baseline; illustrative Z-vs-threshold curve reaches Z=2 at looser cut than baseline.
- **Evidence:** `ch9_best_model.md`; all four figures in `figures/ch9/`.
- **Missing / weak figures:** none.
- **Outdated wording / wrong numbering:**
  - `:5–9` "MIA … is the natural endpoint of the design exploration" — must be reconciled with Ch.8's promotion of cand01 (§1).
  - `:9` "Message-Interaction Attention (MIA)" — verify this is the correct expansion of the acronym MIA; V2 §3.P uses "MIA pre-encoder" with MIA standing for "MI-attention". Be consistent.
  - `:11` "Section~\ref{sec:ch9_selection}" — verify label exists. **VERIFIED** at `:20`.
  - Physics-baseline group `exp_20260306-190512_4tbg_physics_baseline` is described as "raw input features, standard multi-head self-attention, no Lorentz-scalar bias, and the smallest architecture preset" (`:35–37`) — confirm this matches the actual config; "smallest preset" should resolve to a specific V2 H10 value (e.g. `d32_L3`).
  - `:31` "The model was trained over five independent random seeds" — for reproducibility, list the seed values explicitly.
- **Interchapter dependencies:** is the synthesis target of Ch.4–8.
- **Recommended rewrite strategy:** make MIA spec match the cand01 promotion verbatim, OR justify the substitution as a follow-up experiment that incorporated Ch.6's recommended differential-attention settings on top of cand01.
- **Priority:** `BLOCKER` (model identity).

### Chapter 10 — Conclusion and outlook (`10_conclusion_and_outlook.tex`)

- **Role:** Synthesis; future work.
- **Status:** Drafted but short (52 lines); no new results.
- **Main claims:** small set of input-representation choices dominates; rest are null; recommended configuration improves expected significance; transferable to other multi-object processes.
- **Evidence:** synthetic — should cite Ch.4–9 explicitly.
- **Missing / weak figures:** none required; this is a synthesis chapter.
- **Outdated wording:**
  - `:6` "33 orthogonal axes" — **BLOCKER** (§3.1).
  - `:28` "Chapter~10" for significance discussion — actually in Chapter 9. **HIGH.**
  - `:39` `\ref{ch:preenc}` — undefined label. **HIGH.**
  - `:18–21` "reduction in the luminosity required to reach $Z = 5$ that is directly quantifiable in units of LHC run time" — contradicts Ch.9 caveat that Z=5 cannot be reliably read from the simplified estimate. **HIGH.**
- **Recommended rewrite strategy:** rewrite from scratch after Ch.4–9 are finalised; the conclusion is too short relative to the body and currently contains the most wrong-numbering / wrong-claim density.
- **Priority:** `BLOCKER` (axis count, Z=5 claim, broken ref).

---

## 7. Reproducibility and evidence audit

| Major result | Script | Config / Hydra override | W&B run / export | Figure file | Table / CSV |
|---|---|---|---|---|---|
| Ch.4 tokenizer | `src/thesis_ml/reports/analyses/ch4_best_input_repr.py` | tokenizer experiments in `configs/classifier/experiment/thesis_experiments/` | run IDs cited in `04_best_input_representation.tex:154` footnote | `figures/ch4/*` | `thesis_results/04_cleaned_backfilled_analysis_ready.csv` |
| Ch.5 biases | not yet a single canonical module; evidence notes `ch5_*.md` cite scripts | bias-family configs in `configs/` | groups cited in evidence notes | `figures/ch5/*` | same CSV |
| Ch.6 attention | `src/thesis_ml/reports/analyses/ch6_attention_mechanisms.py` | ch6 experiment configs | groups cited in `ch6.1_attention_type.md` etc. | `figures/ch6/*` | same CSV |
| Ch.7 scaling | `src/thesis_ml/reports/analyses/ch7_scaling.py` | `configs/report/thesis_experiments_reports/ch7_scaling.yaml` | groups cited in `ch7_scaling.md:38–40` | `figures/ch7/*` | same CSV |
| Ch.8 surrogate | `scripts/one_off/ch8_final_plots.py`, `scripts/one_off/ch8_surrogate_fit.py`, `scripts/one_off/ch8_optuna_reeval.py`, `scripts/one_off/ch8_validation_summary.py` | candidate configs in `configs/classifier/experiment/thesis_experiments/ch8_candidates/` | primary cohort in `thesis_results/ch8_streamlined/05_ch8_streamlined_primary.csv` | `figures/ch8/*` | same CSV |
| Ch.9 best model | `src/thesis_ml/reports/analyses/ch9_best_model.py` | not yet cited in chapter text | group `exp_20260511-113827_mia_od_followup_seeds`; baseline group `exp_20260306-190512_4tbg_physics_baseline` | `figures/ch9/*` | same CSV; Optuna export at `agent_reference/wandb_export_2026-05-17T22_02_28.058+02_00.csv` |

**Findings:**

- **`HIGH`** — Chapter 5 has the weakest reproducibility chain. The evidence ledger has five separate `ch5_*` notes, but no single Hydra module aggregates the Ch.5 analyses. A `ch5_*.py` script analogous to `ch7_scaling.py` would close the loop.
- **`HIGH`** — Chapter 9 does not cite the Hydra config that produced the MIA group. The evidence note (`ch9_best_model.md:23`) cites the W&B group only. Add the originating Hydra config path.
- **`MEDIUM`** — Several scripts live under `scripts/one_off/` (e.g. `ch8_final_plots.py`, `ch8_surrogate_fit.py`, `ch8_optuna_reeval.py`). These should be promoted to `src/thesis_ml/reports/analyses/ch8_*.py` for long-term traceability, or each `one_off` script must be locked at a specific commit hash referenced in the corresponding figure caption.
- **`MEDIUM`** — Chapter 8 (`:43–45`) says "three legacy runs on an earlier hash are discarded". Confirm the legacy hash and the discard predicate are reproducible; the evidence note should record the SHA.
- **`HIGH`** — Verify that no Ch.4 / Ch.5 result depends on the pre-cleanup raw export (`03_analysis_ready.csv`). The cleanup that produced `04_*` fixed 49 runs with wrong G3 and backfilled 170 runs without G2 (per `git log`). Any chapter result computed before that cleanup must be re-run.
- **`MEDIUM`** — Ch.4 §4.5 explicitly traces a `−0.026` AUROC discrepancy to a data-loading bug in the re-inference pipeline (`04_best_input_representation.tex:286–289`). This is a strong example of reproducibility hygiene that should be highlighted in the conclusion and replicated for any other chapter where re-inference was involved.

"We trained / we compare" claims with unclear evidence paths:

- Ch.5 — "All experiments in this chapter hold D02=`include_met` at `true`" (`05_…:29`). Evidence note `ch5_B1_bias_families.md` must confirm.
- Ch.7 — "All 45 runs in this chapter share a common fixed baseline configuration" (`07_…:11`). Evidence note `ch7_scaling.md:36–46` confirms ✓.
- Ch.9 — "Each of these choices was supported by the ablation evidence from the corresponding chapter; MIA accumulates all recommended settings simultaneously" (`09_…:28–30`). The "all recommended settings simultaneously" claim is **not** verifiable from the evidence ledger; the MIA spec listed in `09_…:22–28` mixes Ch.6 recommendations (A3=differential, A4=none, A3-a=split, B1=lorentz_scalar) with Ch.7 recommendation (d64_L6) but does not match the cand01 promotion from Ch.8. See §1 BLOCKER.

---

## 8. Scope control — what NOT to do in the next 5–10 days

The thesis is in the "freeze and polish" phase, not the "explore and add" phase. Avoid:

1. **No new axes.** Even if a Ch.6 hypothesis (e.g. RMSNorm on its own) is interesting, do not add an A2 ablation now; mark it as future work.
2. **No codebase redesign.** The Hydra / W&B / report-config tree is stable. Do not refactor `src/thesis_ml/` packages now.
3. **No new experiments unless they unblock a central claim.** The single legitimate exception is the cand01↔MIA reconciliation in §1: if Ch.9 must run the actual cand01 (rather than `mia_od_followup_seeds`), then a 3-seed cand01 run is justified. Otherwise no new training jobs.
4. **No literature expansion** beyond what is needed to support physics framing in Ch.1 and Ch.2 and the differential-attention citation in Ch.6 (already present: `\cite{YE2025}`, `\cite{vaswani2017attention}`).
5. **No appendix polish** before the main argument coherence (i.e. before BLOCKER + HIGH items are resolved). The three appendix files exist but are not read.
6. **No figure beautification before survival decisions.** Decide which figures live in the main text vs. appendix vs. cut, then refresh those that survive once and only once.
7. **No bibliography re-import.** The bib file has 33 entries (per its header) and is currently usable; only add missing citations for outstanding `\cite` calls that produce `?` in the PDF.
8. **No \wandbaxis URL population** until the second-to-last day; this is mechanical and risks merge conflicts.
9. **No rewrite of Chapter 4.** It is the cleanest chapter; touch only for cross-reference fixes.
10. **No new Optuna runs**, no new SHAP sweeps, no new physics-reach extensions. The illustrative Z-vs-threshold in Ch.9 is sufficient — explicitly hedged as "illustrative only".

---

## 9. Recommended execution plan

### 5-day compressed plan

| Day | Objective | Chapters / files touched | Figures / scripts touched | Expected output | Risk |
|---|---|---|---|---|---|
| 1 | Freeze storyline + axis framing | `01_introduction.tex`, `10_conclusion_and_outlook.tex`, `03_workbench_architecture_and_axes.tex` | none | Axis-count claim corrected everywhere; broken `\ref{ch:preenc}` removed; reader's guide aligned with actual chapter structure; deletion of `08_what_does_not_matter.tex` | low |
| 2 | Reconcile Ch.8 ↔ Ch.9 | `08_global_surrogate.tex`, `09_best_model.tex`, `ch9_best_model.md` | none (no new training) | Either Ch.8 promotes the MIA spec verbatim and Ch.9 cites the cand01 → MIA evolution, OR Ch.9 evaluates cand01 directly with a 3-seed run already in the database | **BLOCKER risk if no existing cand01 multi-seed run** — verify in CSV before committing to the rewrite |
| 3 | Freeze figure list + recolour pass | all Ch.4–9 figures | `src/thesis_ml/reports/analyses/*.py`, `scripts/one_off/ch8_*.py` | All figures regenerated with `apply_thesis_style()`; consistent axis-group colours; no `set_title`; PDFs re-committed | medium — file diffs in `figures/` will be noisy |
| 4 | Rewrite Ch.8 and Ch.9 main text | `08_global_surrogate.tex`, `09_best_model.tex` | none | Coherent Ch.8 → Ch.9 handover; physics-reach caveats consistent with main claims | medium — depends on Day-2 decision |
| 5 | Polish Ch.1, Ch.2, Ch.10 | `01_introduction.tex`, `02_dataset_and_task_definition.tex`, `10_conclusion_and_outlook.tex` | placeholder figures (supervisor diagrams) | Compiles without `\todo`; bib resolved; `\wandbaxis` URL placeholders populated; final PDF inspection | low |

### 10-day safer plan

| Day | Objective | Chapters / files touched | Figures / scripts | Expected output | Risk |
|---|---|---|---|---|---|
| 1 | Storyline freeze + delete `08_what_does_not_matter.tex` | meta | none | clean chapter file list | low |
| 2 | Ch.1 + Ch.3 rewrite (axis framing, dependency tree) | `01_introduction.tex`, `03_workbench_architecture_and_axes.tex` | dependency-tree figure | introduction and workbench reconciled with V2 | low |
| 3 | Ch.8 ↔ Ch.9 reconciliation decision + evidence-note update | `08_global_surrogate.tex`, `09_best_model.tex`, `ch8_global_surrogate.md`, `ch9_best_model.md` | none | written decision recorded; one chapter's spec is rewritten to match | BLOCKER risk |
| 4 | Re-run cand01 with 3 seeds (if needed) on Condor | n/a | training | new MIA / cand01 run in CSV | requires Condor wall time |
| 5 | Ch.8 rewrite | `08_global_surrogate.tex` | new R-excluded marginal panel | clean Ch.8 with R-excluded panel, correct axis IDs | low |
| 6 | Ch.9 rewrite | `09_best_model.tex` | none (existing figures) | clean Ch.9 matching Ch.8 promotion | low |
| 7 | Central figure refresh + caption rewrites | all Ch.4–9 figures | plotting modules | uniform style; self-contained captions | medium |
| 8 | Ch.5 + Ch.6 + Ch.7 polish | three `.tex` files | none | chapter-number fossils fixed; anthropomorphisations softened | low |
| 9 | Ch.2 + Ch.10 polish; bibliography | `02_…`, `10_…`, `report.bib` | one figure for Ch.2 | bib resolved; `\todo` resolved; one Ch.2 figure added | low |
| 10 | Final compile, `\wandbaxis` URL population, PDF inspection, appendices | `report.tex`, appendices | none | final PDF | low |

In both plans, priorities are: (1) freeze storyline and claims, (2) freeze figure list, (3) rerun / replot only the necessary analyses, (4) rewrite the core results chapters (5, 6, 7, 8, 9), (5) polish intro / conclusion / captions / bibliography / appendices.

---

## 10. Final prioritised task list

### BLOCKER

- [ ] **B1.** Resolve "33 orthogonal axes" framing across `01_introduction.tex:217` and `10_conclusion_and_outlook.tex:6`. Use the V2 framing: ~90 configurable dimensions in a dependency tree with activation sets. **Assignee:** thesis-writer.
- [ ] **B2.** Reconcile Chapter 8 candidate promotion (`08_global_surrogate.tex:516–647`) with Chapter 9 model evaluation (`09_best_model.tex:22–30`). Either Ch.8 promotes the MIA spec or Ch.9 evaluates cand01. Decision required from human. **Assignee:** human decision → thesis-writer.
- [ ] **B3.** Remove or archive the unused `thesis_report/mainmatter/08_what_does_not_matter.tex` to prevent the next agent editing the wrong Chapter 8 file. **Assignee:** human / thesis-writer.
- [ ] **B4.** Fix broken cross-reference `\ref{ch:preenc}` in `10_conclusion_and_outlook.tex:39`. Replace with the correct chapter label or rewrite the sentence. **Assignee:** thesis-writer.
- [ ] **B5.** Correct the Chapter-10 claim that the significance reduction at Z=5 is "directly quantifiable" (`10_…:18–21`). Chapter 9 explicitly says it is not. **Assignee:** thesis-writer.

### HIGH

- [ ] **H1.** Fix chapter-number fossils:
      - `05_…:679` "Ch.\,8" → check intent and use correct chapter.
      - `05_…:734` "Ch.\,6 and Ch.\,8" → verify.
      - `03_…:273` "Ch.\,9" for H-axis variation → should be Ch.7.
      - `10_…:28` "Chapter~10" for significance → Chapter 9. **Assignee:** thesis-writer.
- [ ] **H2.** Update `01_introduction.tex:258` reader's guide to match actual `report.tex` structure (Part II = Chapters 4–7, Part III = Chapters 8–10). **Assignee:** thesis-writer.
- [ ] **H3.** Update `01_introduction.tex:265` axis-group enumeration to include all of G, D, T, E, P, A, F, B, C, H per V2. **Assignee:** thesis-writer.
- [ ] **H4.** In Chapter 3, add a dependency-tree diagram (rendered from V2 §0.2 mermaid graph). **Assignee:** plotting-report agent.
- [ ] **H5.** In Chapter 3, fix F-group listing (add F1-a1) and decide on full vs. abbreviated B-group enumeration. **Assignee:** thesis-writer.
- [ ] **H6.** In Chapter 8, add a second marginal-ranked panel excluding R-family axes, to expose architectural sensitivities. **Assignee:** plotting-report agent + analysis-rerun.
- [ ] **H7.** In Chapter 8, normalise Optuna axis names (`D1` → `D01`, `H1` → `H01`, `H2` → `H02`). **Assignee:** thesis-writer.
- [ ] **H8.** In Chapter 8, decide whether L-family (logging / interpretability) axes should be excluded from SHAP family aggregation; if yes, regenerate `surrogate_shap_family_bar.pdf`. **Assignee:** analysis-rerun + plotting-report.
- [ ] **H9.** In Chapter 9, cite the originating Hydra config for the `mia_od_followup_seeds` group and the explicit seed list. **Assignee:** thesis-writer.
- [ ] **H10.** Verify all chapter plotting modules import `apply_thesis_style()` at module level; replace any `plt.cm.tab10` / `tab20` for axis-group comparisons. **Assignee:** reviewer + plotting-report.
- [ ] **H11.** Confirm no Ch.4 or Ch.5 result depends on the pre-cleanup `03_analysis_ready.csv`. **Assignee:** reviewer.
- [ ] **H12.** Add citations for anti-kT, MLM merging, NNPDF3.1, Delphes ATLAS card in Chapter 2 background paragraph (currently only the dataset is cited). **Assignee:** thesis-writer.
- [ ] **H13.** In Chapter 3, soften RMSNorm "improves efficiency without degrading performance" — A2 was not directly ablated. **Assignee:** thesis-writer.
- [ ] **H14.** Verify the `\wandbaxis{}` placeholder URLs render acceptably in draft and are scheduled for population in the final-compile day. **Assignee:** thesis-writer.

### MEDIUM

- [ ] **M1.** Promote `scripts/one_off/ch8_*.py` to `src/thesis_ml/reports/analyses/ch8_*.py`, or lock the one-off scripts at a referenced commit. **Assignee:** analysis-rerun agent.
- [ ] **M2.** Add a single `ch5_*.py` analysis module to consolidate Ch.5 plotting code. **Assignee:** analysis-rerun + plotting-report.
- [ ] **M3.** Rename `thesis_report/mainmatter/06_(differential)_attention.tex` to `06_attention_mechanisms.tex` to remove parentheses from filename. **Assignee:** thesis-writer.
- [ ] **M4.** Resolve Ch.2 `\todo{ask supervisor: …}` items by either documenting an answer or moving them to an explicit "open questions" subsection. **Assignee:** human / thesis-writer.
- [ ] **M5.** Add at least one figure to Chapter 2 (token-count histogram or event-display sketch). **Assignee:** plotting-report.
- [ ] **M6.** Tighten captions for Ch.8 SHAP beeswarm and Cramér's V heatmap to make the encoding self-explanatory. **Assignee:** thesis-writer.
- [ ] **M7.** Caption fix for `ch6_A3a_auroc_seedspread_diff_bias_mode.pdf`: state the cell-size imbalance (n=3 vs n=21) explicitly. **Assignee:** thesis-writer.
- [ ] **M8.** Decide whether per-head per-layer attention grids in Ch.5 (`figure-attn_typepair_per_head_*.pdf`) belong in the main text or in an appendix. **Assignee:** thesis-writer + plotting-report.
- [ ] **M9.** Verify the on-figure annotation "Not a full profile-likelihood fit" is rendered in `ch9_mia_significance_vs_threshold.pdf`. **Assignee:** reviewer.
- [ ] **M10.** Confirm baseline group `4tbg_physics_baseline` matches the smallest preset (`d32_L3`?) cited as "smallest architecture preset" in `09_…:37`. **Assignee:** reviewer.
- [ ] **M11.** Replace anthropomorphic phrases ("the model understands", "the attention learns to focus", "moderate overtraining over 50 epochs") with descriptive equivalents. **Assignee:** thesis-writer.
- [ ] **M12.** Verify `Visive:2026mjd` bib entry for the T=18 token-length citation. **Assignee:** thesis-writer.
- [ ] **M13.** Across all chapters, switch from informal `R5`, `H1`, `D1` to V2 zero-padded forms `R05`, `H01`, `D01`. **Assignee:** thesis-writer.
- [ ] **M14.** Confirm heatmap colour conventions: `RdBu_r` for AUROC-difference diverging maps, `viridis` for non-negative magnitudes, `Blues` for confusion. **Assignee:** plotting-report.

### LOW

- [ ] **L1.** Resolve six `\todo{Figure: …}` placeholders in Chapter 1 (SM table, Feynman, LHC, ATLAS, supervisor 4-top diagram, transformer). **Assignee:** human / thesis-writer.
- [ ] **L2.** Across Part II chapters, replace one-sentence paragraphs (e.g. `04_…:368` "The narrative arc of this chapter is therefore: …") with full topic + support + conclusion paragraphs, per writing rules. **Assignee:** thesis-writer.
- [ ] **L3.** Remove contractions, em dashes used non-vitally, FANBOYS sentence starts; verify no rhetorical questions across all chapters. **Assignee:** thesis-writer.
- [ ] **L4.** Decide on a single chapter-reference form ("Ch.\,8" vs. "Chapter~8" vs. "\Cref{ch:...}") and apply consistently. **Assignee:** thesis-writer.
- [ ] **L5.** Verify all figures use `figure_size()` from `style.py` (no bare hardcoded tuples). **Assignee:** plotting-report.
- [ ] **L6.** Verify top / right spines are absent in all PDFs (set in `THESIS_RC` but worth a visual check). **Assignee:** reviewer.
- [ ] **L7.** Inspect appendices a, b, c for content suitability and decide whether to keep, merge, or delete. **Assignee:** thesis-writer.
- [ ] **L8.** Verify bibliography entries cited in the text are all present in `report.bib` (no `?` in compiled PDF). **Assignee:** thesis-writer.
- [ ] **L9.** Add a Z=2 / Z=5 explanation paragraph in Ch.9 §9.3 to make the working points understandable to a non-HEP reader. **Assignee:** thesis-writer.
- [ ] **L10.** Confirm `report.pdf` compiles cleanly with no unresolved `\ref` warnings after BLOCKER fixes. **Assignee:** thesis-writer.

---

**End of staging note.** The next phase is the chapter-by-chapter rewrite, which must start by resolving items B1–B5 (axis-count framing, Ch.8↔Ch.9 model identity, duplicate Ch.8 file, broken cross-reference, and the Z=5 claim in the conclusion).
