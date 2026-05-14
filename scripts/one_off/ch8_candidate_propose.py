"""ch8_candidate_propose.py — Phase D: propose top-10 novel candidate configs.

Standalone script (no Hydra, no W&B).  Activate environment before running:
  source /data/atlas/users/nterlind/venvs/miniconda3/etc/profile.d/conda.sh && conda activate /data/atlas/users/nterlind/venvs/thesis-ml
  python3 scripts/one_off/ch8_candidate_propose.py

Pipeline:
  1. Load surrogate (surrogate_xgb.json) and training-time column list (X_shap_sample.csv).
  2. Load primary cohort CSV (05_ch8_streamlined_primary.csv).
  3. Sample ~100k candidate configs from the observed hull (product-of-marginals).
  4. Filter via is_legal_config.
  5. Build feature matrices (same one-hot pipeline as training), aligned to training columns.
  6. Predict AUROC with the surrogate.
  7. Deduplicate against already-trained configs (fingerprint = sorted tuple of axis values
     excluding R5_Seed, matching make_groups logic).
  8. Pick top-10 novel candidates.
  9. Estimate uncertainty as uniform band = CV Spearman std (per-fold models not saved).
 10. Write outputs to .../candidates/:
       top10.json, top10_overrides.txt, top10_predicted.csv, top10_scatter.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_ml.reports.ch8_surrogate import build_feature_matrix, make_groups
from thesis_ml.reports.ch8_constraints import is_legal_config, sample_observed_hull
from thesis_ml.reports.plots.style import apply_thesis_style, figure_size, axis_color
from thesis_ml.monitoring.io_utils import save_figure

apply_thesis_style()

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------

SURROGATE_DIR = Path(
    "/data/atlas/users/nterlind/outputs/reports/"
    "report_ch8_patching_G1_2/surrogate"
)
CANDIDATES_DIR = Path(
    "/data/atlas/users/nterlind/outputs/reports/"
    "report_ch8_patching_G1_2/candidates"
)
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = (
    REPO_ROOT
    / "thesis_results/ch8_streamlined/05_ch8_streamlined_primary.csv"
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

N_SAMPLES = 100_000
RNG_SEED = 42
TOP_K = 10

# ---------------------------------------------------------------------------
# Step 1: Load surrogate and training-time column list
# ---------------------------------------------------------------------------

print("Loading surrogate model …")
booster = xgb.Booster()
booster.load_model(str(SURROGATE_DIR / "surrogate_xgb.json"))

print("Loading training-time column list from X_shap_sample.csv …")
shap_df = pd.read_csv(SURROGATE_DIR / "X_shap_sample.csv", nrows=0)
TRAIN_COLS: list[str] = list(shap_df.columns)
print(f"  Training feature columns: {len(TRAIN_COLS)}")

print("Loading CV metrics …")
with open(SURROGATE_DIR / "cv_metrics.json") as f:
    cv_metrics = json.load(f)
spearman_std = cv_metrics["spearman_std"]
spearman_mean = cv_metrics["spearman_mean"]
print(f"  Surrogate CV: Spearman = {spearman_mean:.3f} ± {spearman_std:.3f}")

# ---------------------------------------------------------------------------
# Step 2: Load primary cohort CSV
# ---------------------------------------------------------------------------

print(f"\nLoading primary cohort CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"  Rows: {len(df)}")

AXIS_COLS = [c for c in df.columns if c.startswith("config/axes/")]
print(f"  Axis columns: {len(AXIS_COLS)}")

# Seed column (used for fingerprint, excluded from dedup)
SEED_COL = next((c for c in AXIS_COLS if "R5_Seed" in c), None)

# ---------------------------------------------------------------------------
# Step 3: Sample ~100k candidate configs from observed hull
# ---------------------------------------------------------------------------

print(f"\nSampling {N_SAMPLES:,} candidates from observed hull …")
rng = np.random.default_rng(RNG_SEED)
raw_candidates = sample_observed_hull(df, AXIS_COLS, N_SAMPLES, rng)
print(f"  Sampled: {len(raw_candidates):,}")

# ---------------------------------------------------------------------------
# Step 4: Filter via is_legal_config
# ---------------------------------------------------------------------------

print("Filtering by is_legal_config …")
legal_candidates = [c for c in raw_candidates if is_legal_config(c)]
print(f"  Legal candidates: {len(legal_candidates):,} / {len(raw_candidates):,} "
      f"({100 * len(legal_candidates) / len(raw_candidates):.1f}%)")

if len(legal_candidates) == 0:
    print("ERROR: No legal candidates found.  Check constraint logic.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 5: Build feature matrices aligned to training columns
# ---------------------------------------------------------------------------

print("\nBuilding feature matrices for legal candidates …")
cand_df = pd.DataFrame(legal_candidates)

# Ensure all axis cols are present (fill missing with "inactive")
for col in AXIS_COLS:
    if col not in cand_df.columns:
        cand_df[col] = "inactive"

# Build one-hot features the same way as training
X_cand, _ = build_feature_matrix(cand_df, AXIS_COLS)

# Align to training-time columns:
#   - Columns present in training but missing in X_cand → add as 0 (unseen level)
#   - Columns present in X_cand but not in training → drop (novel level)
X_cand = X_cand.reindex(columns=TRAIN_COLS, fill_value=0)
print(f"  Feature matrix shape: {X_cand.shape}")

# ---------------------------------------------------------------------------
# Step 6: Predict AUROC
# ---------------------------------------------------------------------------

print("Predicting AUROC …")
dmatrix = xgb.DMatrix(X_cand.values, feature_names=TRAIN_COLS)
preds = booster.predict(dmatrix)
print(f"  Predicted AUROC: min={preds.min():.4f}, max={preds.max():.4f}, "
      f"mean={preds.mean():.4f}")

# ---------------------------------------------------------------------------
# Step 7: Deduplicate against already-trained configs
# ---------------------------------------------------------------------------

# Build fingerprint columns (all axis cols except seed)
fp_cols = [c for c in AXIS_COLS if c != SEED_COL]

# Build set of existing fingerprints from training data
def make_fingerprint(row: pd.Series | dict, cols: list[str]) -> tuple:
    """Build a fingerprint as a tuple of (col, value) pairs, sorted by col name.

    Using sorted col names (not sorted values) so the fingerprint is unique per
    config and not just per set-of-values.
    """
    return tuple((c, str(row[c])) for c in sorted(cols))

print("\nBuilding existing fingerprints from training data …")
existing_fps: set[tuple] = set()
for _, row in df[fp_cols].iterrows():
    existing_fps.add(make_fingerprint(row, fp_cols))
print(f"  Unique existing fingerprints: {len(existing_fps)}")

# Build fingerprints for candidates
print("Computing candidate fingerprints …")
cand_fps = []
for cand in legal_candidates:
    fp = make_fingerprint(cand, fp_cols)
    cand_fps.append(fp)

# Filter: keep only novel candidates
novel_mask = np.array([fp not in existing_fps for fp in cand_fps])
n_novel = novel_mask.sum()
print(f"  Novel candidates: {n_novel:,} / {len(legal_candidates):,}")

if n_novel == 0:
    print("WARNING: No novel candidates found.  All sampled configs match existing runs.")
    print("  Relaxing: reporting top-10 by predicted AUROC regardless of novelty …")
    novel_mask = np.ones(len(legal_candidates), dtype=bool)

novel_indices = np.where(novel_mask)[0]
novel_preds = preds[novel_mask]

# ---------------------------------------------------------------------------
# Step 8: Pick top-10 novel candidates
# ---------------------------------------------------------------------------

print(f"\nRanking top-{TOP_K} novel candidates …")
top_order = np.argsort(novel_preds)[::-1][:TOP_K]
top_indices = novel_indices[top_order]
top_preds = preds[top_indices]

top_candidates = [legal_candidates[i] for i in top_indices]
top_fps = [cand_fps[i] for i in top_indices]

print("Top-10 predicted AUROCs:")
for rank, (pred, fp) in enumerate(zip(top_preds, top_fps), 1):
    print(f"  Rank {rank}: {pred:.4f}")

# ---------------------------------------------------------------------------
# Step 9: Uncertainty estimate
# ---------------------------------------------------------------------------

# Per-fold models are not saved; we use the CV Spearman std as a uniform
# uncertainty band.  This is a conservative estimate: the true prediction
# interval is approximately ±spearman_std in rank-correlation units, which
# does not directly translate to ±AUROC.  We report it as-is with a note.
uncertainty = spearman_std
print(f"\nUncertainty estimate: CV Spearman std = {uncertainty:.4f} (uniform)")
print("  NOTE: This is expressed in Spearman rank-correlation units, not AUROC units.")
print("  It reflects how much the surrogate's rank-ordering agreement varies across folds.")

# ---------------------------------------------------------------------------
# Step 10: Write outputs
# ---------------------------------------------------------------------------

# ── Axis ID → Hydra override key mapping ────────────────────────────────────
# Derived from AXES_REFERENCE_V2.md (Table 6.*) + Hydra config inspection.
# Format: axis_short_label (without 'config/axes/') → Hydra config key.
# Keys not listed here are omitted from top10_overrides.txt.

AXIS_TO_HYDRA: dict[str, str] = {
    "A1_Normalization Policy":        "classifier.model.norm.policy",
    "A2_Normalization Type":          "classifier.model.norm.type",
    "A3_Attention Type":              "classifier.model.attention.type",
    "A3-a_Differential Attention Bias Mode": "classifier.model.attention.diff_bias_mode",
    "A4_Attention Internal Normalization": "classifier.model.attention.norm",
    "A5_Causal Masking":              "classifier.model.causal_attention",
    "B1_Bias Activation Set":         "classifier.model.attention_biases",
    "B1-G1_Global-Conditioned Mode":  "classifier.model.bias_config.global_conditioned.mode",
    "B1-L1_Lorentz Feature Set":      "classifier.model.bias_config.lorentz_scalar.features",
    "B1-L2_Lorentz MLP Type":         "classifier.model.bias_config.lorentz_scalar.mlp_type",
    "B1-L4_Lorentz Per-Head Mode":    "classifier.model.bias_config.lorentz_scalar.per_head",
    "B1-L5_Lorentz Sparse Gating":    "classifier.model.bias_config.lorentz_scalar.sparse_gating",
    "B1-S1_SM Interaction Mode":      "classifier.model.bias_config.sm_interaction.mode",
    "B1-T1_Type-Pair Initialization": "classifier.model.bias_config.typepair_kinematic.init_from_physics",
    "B1-T2_Type-Pair Freeze Table":   "classifier.model.bias_config.typepair_kinematic.freeze_table",
    "B1-T3_Type-Pair Kinematic Gate": "classifier.model.bias_config.typepair_kinematic.kinematic_gate",
    "C1_Head Realization":            "classifier.model.head.type",
    "C2_Pooling Strategy":            "classifier.model.head.pooling",
    "D1_Feature Set":                 "data.cont_features",
    "D2_MET Treatment":               "classifier.globals.include_met",
    "D3_Token Ordering":              "data.sort_tokens_by",
    "E1_PE Type":                     "classifier.model.positional",
    "E1-a_PE Space":                  "classifier.model.positional_space",
    "F1_FFN Type":                    "classifier.model.ffn.type",
    "F1-a_KAN FFN Variant":           "classifier.model.ffn.kan.variant",
    "F1-b_MoE Encoder Scope":         "classifier.model.moe.scope",
    "F1-moe_MoE Enabled":             "classifier.model.moe.enabled",
    "H1_Model Dimension":             "classifier.model.dim",
    "H2_Encoder Depth":               "classifier.model.depth",
    "H3_Attention Heads":             "classifier.model.heads",
    "H4_FFN Hidden Dimension":        "classifier.model.mlp_dim",
    "H5_Dropout":                     "classifier.model.dropout",
    "K1_KAN Grid Size":               "classifier.model.kan.grid_size",
    "K2_KAN Spline Order":            "classifier.model.kan.spline_order",
    "M1_MoE Number of Experts":       "classifier.model.moe.num_experts",
    "M2_MoE Top K":                   "classifier.model.moe.top_k",
    "M3_MoE Routing Level":           "classifier.model.moe.routing_level",
    "P1_Nodewise Mass Enabled":       "classifier.model.nodewise_mass.enabled",
    "P2_MIA Pre-Encoder Enabled":     "classifier.model.mia_blocks.enabled",
    "P2-a_MIA Placement":             "classifier.model.mia_blocks.placement",
    "R1_Epochs":                      "classifier.trainer.epochs",
    "R2_Learning Rate":               "classifier.trainer.lr",
    "R3_Weight Decay":                "classifier.trainer.weight_decay",
    "R4_Batch Size":                  "classifier.trainer.batch_size",
    "R5_Seed":                        "classifier.trainer.seed",
    "R6_Warmup Steps":                "classifier.trainer.warmup_steps",
    "R7_LR Schedule":                 "classifier.trainer.lr_schedule",
    "R8_Label Smoothing":             "classifier.trainer.label_smoothing",
    "T1_Tokenizer Family":            "classifier.model.tokenizer.name",
    "T1-a_PID Embedding Mode":        "classifier.model.tokenizer.pid_mode",
    "T1-b_PID Embedding Dimension":   "classifier.model.tokenizer.id_embed_dim",
}

# ── Build output structures ──────────────────────────────────────────────────

top10_records = []
override_lines = []

# Key axis columns to include in the CSV table (most informative for thesis)
KEY_AXES_SHORT = [
    "T1_Tokenizer Family",
    "T1-a_PID Embedding Mode",
    "T1-b_PID Embedding Dimension",
    "B1_Bias Activation Set",
    "B1-G1_Global-Conditioned Mode",
    "D2_MET Treatment",
    "F1_FFN Type",
    "F1-moe_MoE Enabled",
    "F1-eff_FFN Realization",
    "E1_PE Type",
    "A1_Normalization Policy",
    "A2_Normalization Type",
    "H1_Model Dimension",
    "H2_Encoder Depth",
    "H3_Attention Heads",
    "H4_FFN Hidden Dimension",
    "H5_Dropout",
    "R1_Epochs",
    "R2_Learning Rate",
    "R4_Batch Size",
    "P1_Nodewise Mass Enabled",
    "P2_MIA Pre-Encoder Enabled",
]

# Map to full column names
KEY_AXES_FULL = [f"config/axes/{s}" for s in KEY_AXES_SHORT]

for rank_0, (cand, pred, fp) in enumerate(zip(top_candidates, top_preds, top_fps)):
    rank = rank_0 + 1

    # Build record for JSON
    record = {
        "rank": rank,
        "predicted_auroc": float(pred),
        "uncertainty_spearman_std": float(uncertainty),
        "uncertainty_note": (
            "Uniform uncertainty estimate = CV Spearman std across 5 folds. "
            "Expressed in rank-correlation units, not AUROC units. "
            "Per-fold models were not saved; this is an approximation."
        ),
        "fingerprint": list(fp),
        "config": {k: v for k, v in cand.items()},
    }
    top10_records.append(record)

    # Build Hydra override string: only include axes with known mapping
    # and non-inactive values
    override_parts = []
    for axis_full, hydra_key in AXIS_TO_HYDRA.items():
        full_col = f"config/axes/{axis_full}"
        val = cand.get(full_col, "inactive")
        if str(val).lower() not in ("inactive", "nan", "none", ""):
            # Skip logging/interpretability axes and axes rarely changed
            if axis_full.startswith("L"):
                continue
            override_parts.append(f"{hydra_key}={val}")
    override_lines.append(f"# Rank {rank} (predicted_auroc={pred:.4f})\n" + " ".join(override_parts))

# ── top10.json ───────────────────────────────────────────────────────────────
out_json = CANDIDATES_DIR / "top10.json"
with open(out_json, "w") as f:
    json.dump(top10_records, f, indent=2)
print(f"\nWritten: {out_json}")

# ── top10_overrides.txt ───────────────────────────────────────────────────────
out_overrides = CANDIDATES_DIR / "top10_overrides.txt"
with open(out_overrides, "w") as f:
    f.write("# top10_overrides.txt — Ch8 Phase D candidate Hydra overrides\n")
    f.write("# Generated by ch8_candidate_propose.py\n")
    f.write("# Uncertainty = CV Spearman std (rank-corr units); see top10.json for details.\n\n")
    for line in override_lines:
        f.write(line + "\n\n")
print(f"Written: {out_overrides}")

# ── top10_predicted.csv ───────────────────────────────────────────────────────
rows = []
for rank_0, (cand, pred, fp) in enumerate(zip(top_candidates, top_preds, top_fps)):
    row = {"rank": rank_0 + 1, "predicted_auroc": round(float(pred), 4),
           "uncertainty_spearman_std": round(float(uncertainty), 4)}
    for key_short, key_full in zip(KEY_AXES_SHORT, KEY_AXES_FULL):
        row[key_short] = cand.get(key_full, "inactive")
    rows.append(row)

csv_df = pd.DataFrame(rows)
out_csv = CANDIDATES_DIR / "top10_predicted.csv"
csv_df.to_csv(out_csv, index=False)
print(f"Written: {out_csv}")

# ── top10_scatter.pdf ─────────────────────────────────────────────────────────

print("\nGenerating top10_scatter.pdf …")

cfg_logging = {"fig_format": "pdf", "dpi": 300}

fig, ax = plt.subplots(figsize=figure_size("full"))

ranks = np.arange(1, TOP_K + 1)
colors = [axis_color("recommended")] * TOP_K

ax.errorbar(
    ranks,
    top_preds,
    yerr=uncertainty * np.ones(TOP_K),
    fmt="o",
    color=axis_color("recommended"),
    ecolor="gray",
    elinewidth=0.8,
    capsize=3,
    markersize=6,
    zorder=3,
    label="Predicted AUROC",
)

# Reference line: best observed AUROC in training data
best_observed = float(df["eval_v2/test_auroc"].max())
ax.axhline(
    best_observed,
    color="gray",
    linestyle="--",
    linewidth=0.8,
    alpha=0.5,
    label=f"Best observed AUROC ({best_observed:.4f})",
)

ax.set_xlabel("Candidate rank")
ax.set_ylabel("Predicted AUROC (surrogate)")
ax.set_xticks(ranks)
ax.set_xlim(0.5, TOP_K + 0.5)
ax.legend()

out_fig = CANDIDATES_DIR / "top10_scatter"
save_figure(fig, CANDIDATES_DIR, "top10_scatter", cfg_logging)
plt.close(fig)
print(f"Written: {out_fig}.pdf")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Phase D candidate proposal complete.")
print(f"  Sampled:            {N_SAMPLES:,}")
print(f"  Legal:              {len(legal_candidates):,}")
print(f"  Novel:              {n_novel:,}")
print(f"  Output directory:   {CANDIDATES_DIR}")
print(f"  Uncertainty method: CV Spearman std = {uncertainty:.4f} (uniform band)")
print("=" * 60)
