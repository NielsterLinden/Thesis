"""ch8_candidate_propose_marginal.py — Marginal-greedy candidate proposal for Ch8.

Standalone script (no Hydra, no W&B).  Activate environment before running:
  source /data/atlas/users/nterlind/venvs/miniconda3/etc/profile.d/conda.sh && conda activate /data/atlas/users/nterlind/venvs/thesis-ml
  python3 scripts/one_off/ch8_candidate_propose_marginal.py

Strategy: for each axis column independently pick the value with the highest
mean test_auroc (per-value groupby mean), then combine those into a config.
Produces 3 candidates:
  cand_m1 — full greedy (best value on every axis)
  cand_m2 — m1 but second-best value for the #1 highest-impact axis
  cand_m3 — m1 but second-best value for the #2 highest-impact axis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_ml.reports.ch8_constraints import is_legal_config  # noqa: E402

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------

CANDIDATES_DIR = Path(
    "/data/atlas/users/nterlind/outputs/reports/"
    "report_ch8_patching_G1_2/candidates"
)
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = REPO_ROOT / "thesis_results/ch8_streamlined/05_ch8_streamlined_primary.csv"
SURROGATE_CANDIDATES = CANDIDATES_DIR / "top10.json"
YAML_OUT_DIR = (
    REPO_ROOT
    / "configs/classifier/experiment/thesis_experiments/ch8_candidates"
)
YAML_OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "eval_v2/test_auroc"

# ---------------------------------------------------------------------------
# Axis → Hydra key mapping (same as ch8_candidate_propose.py)
# ---------------------------------------------------------------------------

AXIS_TO_HYDRA: dict[str, str] = {
    "A1_Normalization Policy":             "classifier.model.norm.policy",
    "A2_Normalization Type":               "classifier.model.norm.type",
    "A3_Attention Type":                   "classifier.model.attention.type",
    "A3-a_Differential Attention Bias Mode": "classifier.model.attention.diff_bias_mode",
    "A4_Attention Internal Normalization": "classifier.model.attention.norm",
    "A5_Causal Masking":                   "classifier.model.causal_attention",
    "B1_Bias Activation Set":              "classifier.model.attention_biases",
    "B1-G1_Global-Conditioned Mode":       "classifier.model.bias_config.global_conditioned.mode",
    "B1-L1_Lorentz Feature Set":           "classifier.model.bias_config.lorentz_scalar.features",
    "B1-L2_Lorentz MLP Type":             "classifier.model.bias_config.lorentz_scalar.mlp_type",
    "B1-L4_Lorentz Per-Head Mode":        "classifier.model.bias_config.lorentz_scalar.per_head",
    "B1-L5_Lorentz Sparse Gating":        "classifier.model.bias_config.lorentz_scalar.sparse_gating",
    "B1-S1_SM Interaction Mode":          "classifier.model.bias_config.sm_interaction.mode",
    "B1-T1_Type-Pair Initialization":     "classifier.model.bias_config.typepair_kinematic.init_from_physics",
    "B1-T2_Type-Pair Freeze Table":       "classifier.model.bias_config.typepair_kinematic.freeze_table",
    "B1-T3_Type-Pair Kinematic Gate":     "classifier.model.bias_config.typepair_kinematic.kinematic_gate",
    "C1_Head Realization":                "classifier.model.head.type",
    "C2_Pooling Strategy":                "classifier.model.head.pooling",
    "D1_Feature Set":                     "data.cont_features",
    "D2_MET Treatment":                   "classifier.globals.include_met",
    "D3_Token Ordering":                  "data.sort_tokens_by",
    "E1_PE Type":                         "classifier.model.positional",
    "E1-a_PE Space":                      "classifier.model.positional_space",
    "F1_FFN Type":                        "classifier.model.ffn.type",
    "F1-a_KAN FFN Variant":               "classifier.model.ffn.kan.variant",
    "F1-b_MoE Encoder Scope":            "classifier.model.moe.scope",
    "F1-moe_MoE Enabled":                "classifier.model.moe.enabled",
    "H1_Model Dimension":                "classifier.model.dim",
    "H2_Encoder Depth":                  "classifier.model.depth",
    "H3_Attention Heads":                "classifier.model.heads",
    "H4_FFN Hidden Dimension":           "classifier.model.mlp_dim",
    "H5_Dropout":                        "classifier.model.dropout",
    "K1_KAN Grid Size":                  "classifier.model.kan.grid_size",
    "K2_KAN Spline Order":               "classifier.model.kan.spline_order",
    "M1_MoE Number of Experts":          "classifier.model.moe.num_experts",
    "M2_MoE Top K":                      "classifier.model.moe.top_k",
    "M3_MoE Routing Level":              "classifier.model.moe.routing_level",
    "P1_Nodewise Mass Enabled":          "classifier.model.nodewise_mass.enabled",
    "P2_MIA Pre-Encoder Enabled":        "classifier.model.mia_blocks.enabled",
    "P2-a_MIA Placement":               "classifier.model.mia_blocks.placement",
    "R1_Epochs":                         "classifier.trainer.epochs",
    "R2_Learning Rate":                  "classifier.trainer.lr",
    "R3_Weight Decay":                   "classifier.trainer.weight_decay",
    "R4_Batch Size":                     "classifier.trainer.batch_size",
    "R5_Seed":                           "classifier.trainer.seed",
    "R6_Warmup Steps":                   "classifier.trainer.warmup_steps",
    "R7_LR Schedule":                    "classifier.trainer.lr_schedule",
    "R8_Label Smoothing":               "classifier.trainer.label_smoothing",
    "T1_Tokenizer Family":              "classifier.model.tokenizer.name",
    "T1-a_PID Embedding Mode":          "classifier.model.tokenizer.pid_mode",
    "T1-b_PID Embedding Dimension":     "classifier.model.tokenizer.id_embed_dim",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fingerprint(cfg: dict, cols: list[str]) -> tuple:
    """Build sorted (col, value) fingerprint, excluding seed."""
    return tuple((c, str(cfg[c])) for c in sorted(cols))


def per_axis_rankings(df: pd.DataFrame, axis_cols: list[str]) -> dict[str, pd.Series]:
    """Return {col: Series of mean AUROC indexed by value, sorted descending}."""
    rankings: dict[str, pd.Series] = {}
    for col in axis_cols:
        means = df.groupby(col)[TARGET].mean().sort_values(ascending=False)
        rankings[col] = means
    return rankings


def greedy_config(rankings: dict[str, pd.Series], overrides: dict | None = None) -> dict:
    """Build config dict by picking argmax for each axis."""
    cfg: dict = {}
    for col, means in rankings.items():
        cfg[col] = means.index[0]
    if overrides:
        cfg.update(overrides)
    return cfg


def fix_constraints(
    cfg: dict, rankings: dict[str, pd.Series]
) -> tuple[dict, list[str]]:
    """Iteratively fall back to second-best for axes that violate is_legal_config.

    Returns updated config and list of substitution notes.
    """
    notes: list[str] = []
    # Try up to len(rankings) passes in case of cascading violations
    for _ in range(len(rankings)):
        if is_legal_config(cfg):
            break
        # Find the first axis with an available fallback and swap it
        for col, means in rankings.items():
            test = dict(cfg)
            # Try each ranked value until one yields a legal config
            for rank, val in enumerate(means.index):
                test[col] = val
                if is_legal_config(test):
                    if cfg[col] != val:
                        notes.append(
                            f"{col}: substituted rank-{rank+1} value "
                            f"'{val}' (was '{cfg[col]}')"
                        )
                        cfg[col] = val
                    break
    return cfg, notes


def marginal_auroc(df: pd.DataFrame, cfg: dict, axis_cols: list[str]) -> float:
    """Estimate AUROC by averaging per-axis best-value means."""
    means = [
        df.groupby(col)[TARGET].mean().get(cfg[col], float("nan"))
        for col in axis_cols
    ]
    return float(np.nanmean(means))


def build_override_str(cfg: dict) -> str:
    """Build Hydra override string from config dict (full axis col keys)."""
    parts = []
    for axis_short, hydra_key in AXIS_TO_HYDRA.items():
        full_col = f"config/axes/{axis_short}"
        val = cfg.get(full_col, "inactive")
        if str(val).lower() not in ("inactive", "nan", "none", ""):
            if axis_short.startswith("L"):
                continue
            parts.append(f"{hydra_key}={val}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# YAML generation helpers
# ---------------------------------------------------------------------------


def _cast(val: str):
    """Cast string value to Python bool/int/float/str as appropriate."""
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    # Try int
    try:
        return int(val)
    except (ValueError, TypeError):
        pass
    # Try float
    try:
        return float(val)
    except (ValueError, TypeError):
        pass
    # Try list (feature sets like "[0, 1, 2, 3]")
    if val.startswith("[") and val.endswith("]"):
        inner = val.strip("[]")
        try:
            return [int(x.strip()) for x in inner.split(",")]
        except ValueError:
            pass
    return val


def set_nested(d: dict, dotted_key: str, val):
    """Set a value in a nested dict using a dotted key path."""
    parts = dotted_key.split(".")
    node = d
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = val


def build_yaml_dict(cfg: dict, cand_name: str, auroc_est: float) -> dict:
    """Build the full nested YAML config dict for a candidate."""
    doc: dict = {}

    # meta
    doc["meta"] = {"goal": "classification"}

    # experiment
    doc["experiment"] = {"name": cand_name}

    # hydra
    doc["hydra"] = {
        "mode": "MULTIRUN",
        "job": {"chdir": True, "name": "${experiment.name}"},
        "run": {
            "dir": (
                "${env.output_root}/runs/"
                "run_${now:%Y%m%d-%H%M%S}_${experiment.name}_job${zpad:${hydra.job.num}}"
            )
        },
        "sweep": {
            "dir": "${env.output_root}/multiruns/exp_${now:%Y%m%d-%H%M%S}_${experiment.name}",
            "subdir": (
                "${env.output_root}/runs/"
                "run_${now:%Y%m%d-%H%M%S}_${experiment.name}_job${zpad:${hydra.job.num}}"
            ),
        },
        "sweeper": {"params": {"classifier.trainer.seed": "42,123,456"}},
    }

    # Build nested classifier + data sections from AXIS_TO_HYDRA mapping
    for axis_short, hydra_key in AXIS_TO_HYDRA.items():
        if axis_short == "R5_Seed":
            continue  # handled by sweeper
        full_col = f"config/axes/{axis_short}"
        val = cfg.get(full_col, "inactive")
        if str(val).lower() in ("inactive", "nan", "none", ""):
            continue
        set_nested(doc, hydra_key, _cast(str(val)))

    # Ensure required blocks exist with sensible defaults (following cand01.yaml)
    clf = doc.setdefault("classifier", {})
    model = clf.setdefault("model", {})
    model.setdefault("moe", {}).setdefault("enabled", False)
    model["moe"].setdefault("scope", "all_blocks")
    model["moe"].setdefault("routing_level", "token")
    model.setdefault("nodewise_mass", {}).setdefault("enabled", False)
    model.setdefault("mia_blocks", {}).setdefault("enabled", False)
    model.setdefault("kan", {}).update({"grid_size": 5, "spline_order": 3})
    # Override kan defaults if axes specified them
    kan_axes = {
        "config/axes/K1_KAN Grid Size": "grid_size",
        "config/axes/K2_KAN Spline Order": "spline_order",
    }
    for full_col, key in kan_axes.items():
        val = cfg.get(full_col, "inactive")
        if str(val).lower() not in ("inactive", "nan", "none", ""):
            model["kan"][key] = _cast(str(val))

    # data block
    data = doc.setdefault("data", {})
    data["_struct_"] = False
    data.setdefault("cont_features", [0, 1, 2, 3])
    data.setdefault("sort_tokens_by", "input_order")
    data.setdefault("classifier", {}).update(
        {
            "signal_vs_background": {"signal": 1, "background": [2, 3, 4, 5]},
            "selected_labels": None,
        }
    )

    return doc


def write_yaml(doc: dict, path: Path, cand_name: str, auroc_est: float, notes: str):
    """Write YAML with header comment, matching cand01.yaml style."""
    header = (
        f"# @package _global_\n"
        f"# Chapter 8 marginal-greedy candidate {cand_name} "
        f"(marginal_auroc_estimate={auroc_est:.4f}, 4t-vs-background).\n"
        f"# {notes}\n"
        f"# Sweep: 3 seeds.\n"
        f"#\n"
        f"# Local smoke:\n"
        f"# thesis-train env=local data.limit_samples=500 classifier.trainer.epochs=1 "
        f"logging.use_wandb=false "
        f"classifier/experiment=thesis_experiments/ch8_candidates/{cand_name} --multirun\n"
        f"#\n"
        f"# Stoomboot full:\n"
        f"# thesis-train env=stoomboot "
        f"classifier/experiment=thesis_experiments/ch8_candidates/{cand_name} --multirun\n\n"
    )
    body = yaml.dump(doc, default_flow_style=False, sort_keys=False, allow_unicode=True)
    path.write_text(header + body)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Rows: {len(df)}")

    AXIS_COLS = [c for c in df.columns if c.startswith("config/axes/")]
    SEED_COL = next((c for c in AXIS_COLS if "R5_Seed" in c), None)
    FP_COLS = [c for c in AXIS_COLS if c != SEED_COL]
    print(f"  Axis columns: {len(AXIS_COLS)}, fingerprint cols: {len(FP_COLS)}")

    # --- Per-axis rankings --------------------------------------------------
    print("\nComputing per-axis marginal rankings …")
    rankings = per_axis_rankings(df, AXIS_COLS)

    # Impact = range of mean AUROC across values (best − worst)
    impact: dict[str, float] = {
        col: float(means.iloc[0] - means.iloc[-1])
        for col, means in rankings.items()
        if len(means) > 1
    }
    sorted_impact = sorted(impact.items(), key=lambda x: x[1], reverse=True)
    top2_axes = [col for col, _ in sorted_impact[:2]]
    print(f"  Top-impact axis #1: {top2_axes[0]}  (Δ={impact[top2_axes[0]]:.4f})")
    print(f"  Top-impact axis #2: {top2_axes[1]}  (Δ={impact[top2_axes[1]]:.4f})")

    # --- Build cand_m1: full greedy -----------------------------------------
    print("\nBuilding cand_m1 (full greedy) …")
    cfg_m1 = greedy_config(rankings)
    cfg_m1, notes_m1 = fix_constraints(cfg_m1, rankings)
    if notes_m1:
        print(f"  Constraint fixes: {notes_m1}")
    else:
        print("  No constraint violations.")
    auroc_m1 = marginal_auroc(df, cfg_m1, AXIS_COLS)
    print(f"  marginal_auroc_estimate = {auroc_m1:.4f}")

    # --- Build cand_m2: swap #1 impact axis to second-best ------------------
    print("\nBuilding cand_m2 (second-best on top-impact axis) …")
    top1_col = top2_axes[0]
    top1_second_best = rankings[top1_col].index[1] if len(rankings[top1_col]) > 1 else rankings[top1_col].index[0]
    cfg_m2 = greedy_config(rankings, overrides={top1_col: top1_second_best})
    cfg_m2, notes_m2 = fix_constraints(cfg_m2, rankings)
    if notes_m2:
        print(f"  Constraint fixes: {notes_m2}")
    auroc_m2 = marginal_auroc(df, cfg_m2, AXIS_COLS)
    print(f"  marginal_auroc_estimate = {auroc_m2:.4f}")

    # --- Build cand_m3: swap #2 impact axis to second-best ------------------
    print("\nBuilding cand_m3 (second-best on 2nd-impact axis) …")
    top2_col = top2_axes[1]
    top2_second_best = rankings[top2_col].index[1] if len(rankings[top2_col]) > 1 else rankings[top2_col].index[0]
    cfg_m3 = greedy_config(rankings, overrides={top2_col: top2_second_best})
    cfg_m3, notes_m3 = fix_constraints(cfg_m3, rankings)
    if notes_m3:
        print(f"  Constraint fixes: {notes_m3}")
    auroc_m3 = marginal_auroc(df, cfg_m3, AXIS_COLS)
    print(f"  marginal_auroc_estimate = {auroc_m3:.4f}")

    candidates = [
        ("cand_m1", cfg_m1, auroc_m1, notes_m1,
         f"Full greedy (best value per axis independently)"),
        ("cand_m2", cfg_m2, auroc_m2, notes_m2,
         f"As m1 but second-best value for {top1_col} (Δ={impact[top1_col]:.4f})"),
        ("cand_m3", cfg_m3, auroc_m3, notes_m3,
         f"As m1 but second-best value for {top2_col} (Δ={impact[top2_col]:.4f})"),
    ]

    # --- Load surrogate candidates for overlap check ------------------------
    surrogate_fps: set[tuple] = set()
    surrogate_fp_map: dict[tuple, int] = {}
    if SURROGATE_CANDIDATES.exists():
        with open(SURROGATE_CANDIDATES) as f:
            top10 = json.load(f)
        for rec in top10:
            fp = tuple((k, v) for k, v in rec["fingerprint"])
            surrogate_fps.add(fp)
            surrogate_fp_map[fp] = rec["rank"]
        print(f"\nLoaded {len(surrogate_fps)} surrogate candidate fingerprints for overlap check.")
    else:
        print(f"\nWARNING: surrogate candidates file not found at {SURROGATE_CANDIDATES}")

    # --- Write outputs -------------------------------------------------------

    # top3_marginal.json
    json_records = []
    for cand_name, cfg, auroc, notes, desc in candidates:
        fp = make_fingerprint(cfg, FP_COLS)
        overlap_rank = surrogate_fp_map.get(fp)
        json_records.append({
            "candidate": cand_name,
            "description": desc,
            "marginal_auroc_estimate": round(auroc, 6),
            "note": (
                "marginal_auroc_estimate is the mean of per-axis best-value means, "
                "not a surrogate-model prediction."
            ),
            "constraint_fixes": notes,
            "surrogate_overlap_rank": overlap_rank,
            "fingerprint": list(fp),
            "config": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                           int(v) if isinstance(v, (np.integer,)) else
                           float(v) if isinstance(v, (np.floating,)) else
                           str(v))
                       for k, v in cfg.items()},
        })

    out_json = CANDIDATES_DIR / "top3_marginal.json"
    with open(out_json, "w") as f:
        json.dump(json_records, f, indent=2)
    print(f"\nWritten: {out_json}")

    # top3_marginal_overrides.txt
    out_overrides = CANDIDATES_DIR / "top3_marginal_overrides.txt"
    with open(out_overrides, "w") as f:
        f.write("# top3_marginal_overrides.txt — Ch8 marginal-greedy candidate Hydra overrides\n")
        f.write("# Generated by ch8_candidate_propose_marginal.py\n")
        f.write("# marginal_auroc_estimate = mean of per-axis groupby means (not surrogate).\n\n")
        for cand_name, cfg, auroc, notes, desc in candidates:
            f.write(f"# {cand_name} (marginal_auroc_estimate={auroc:.4f})\n")
            f.write(f"# {desc}\n")
            f.write(build_override_str(cfg) + "\n\n")
    print(f"Written: {out_overrides}")

    # top3_marginal.csv
    KEY_AXES = [
        "T1_Tokenizer Family", "B1_Bias Activation Set", "A1_Normalization Policy",
        "A2_Normalization Type", "E1_PE Type", "F1_FFN Type", "F1-moe_MoE Enabled",
        "H1_Model Dimension", "H2_Encoder Depth", "H3_Attention Heads",
        "H4_FFN Hidden Dimension", "H5_Dropout", "P1_Nodewise Mass Enabled",
        "P2_MIA Pre-Encoder Enabled", "D2_MET Treatment",
    ]
    csv_rows = []
    for rank_i, (cand_name, cfg, auroc, notes, desc) in enumerate(candidates, 1):
        row: dict = {"candidate": cand_name, "rank": rank_i,
                     "marginal_auroc_estimate": round(auroc, 4), "description": desc}
        for ax in KEY_AXES:
            row[ax] = cfg.get(f"config/axes/{ax}", "inactive")
        csv_rows.append(row)
    out_csv = CANDIDATES_DIR / "top3_marginal.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"Written: {out_csv}")

    # YAML experiment files
    print("\nWriting YAML experiment files …")
    for cand_name, cfg, auroc, notes, desc in candidates:
        doc = build_yaml_dict(cfg, cand_name, auroc)
        yaml_path = YAML_OUT_DIR / f"{cand_name}.yaml"
        write_yaml(doc, yaml_path, cand_name, auroc, desc)
        print(f"  Written: {yaml_path}")

    # --- Summary ------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Marginal-greedy candidate proposal complete.")
    print(f"  Output directory: {CANDIDATES_DIR}")
    print(f"  YAML directory:   {YAML_OUT_DIR}")
    print()
    for cand_name, cfg, auroc, notes, desc in candidates:
        fp = make_fingerprint(cfg, FP_COLS)
        overlap_rank = surrogate_fp_map.get(fp)
        overlap_str = f"  OVERLAPS surrogate rank {overlap_rank}" if overlap_rank else ""
        print(f"  {cand_name}: marginal_auroc={auroc:.4f}  {overlap_str}")
        print(f"    {desc}")
        if notes:
            for n in notes:
                print(f"    [CONSTRAINT FIX] {n}")
    print()
    print("  Top-impact axes (range of per-value mean AUROC):")
    for col, delta in sorted_impact[:5]:
        short = col.replace("config/axes/", "")
        best_val = rankings[col].index[0]
        print(f"    {short}: Δ={delta:.4f}  best='{best_val}'")
    print("=" * 65)
