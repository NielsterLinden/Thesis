"""Static allowlists and target values for G1/G3 axis repair (W&B + thesis_results CSV).

Source: thesis audit + cohort definitions (db_completeness 2026-05 vs Feb 2026 binning runs).
"""

from __future__ import annotations

# --- Cohort A1: G3 should be 5-way multiclass (29 runs) ---
G3_DISPLAY_5WAY = "4t | ttH | ttW | ttWW | ttZ"
G3_COMPACT_5WAY = "4t|ttH|ttW|ttWW|ttZ"

G3_COHORT_A1_IDS: frozenset[str] = frozenset(
    {
        "45avja8u",
        "hhuxnj8h",
        "a0k58fly",
        "p7aud2qc",
        "3q4vl8a5",
        "8wmjdinf",
        "ryy2459v",
        "kqxqv7m3",
        "qyc8wqlo",
        "gy88jq8k",
        "ndovg0af",
        "rp41kuru",
        "ft87v2ey",
        "6v6fupe2",
        "p68q6a2y",
        "7fr7osmv",
        "7l784i0y",
        "yqk10l2v",
        "94gfir5c",
        "eg4yfpr0",
        "xzw7q0zc",
        "q1i53x27",
        "u3enk82m",
        "rr34efdz",
        "l5arx4vk",
        "ar9h4kp8",
        "h62a9no8",
        "r7x3bdj4",
        "0v1ddkg6",
    }
)

# --- Cohort A2: G3 should be 4t vs ttH binary (20 runs) ---
G3_DISPLAY_TTH = "4t | ttH"
G3_COMPACT_TTH = "4t|ttH"

G3_COHORT_A2_IDS: frozenset[str] = frozenset(
    {
        "24ntjuny",
        "yns321da",
        "sa3iw694",
        "kpphuqxz",
        "l2sukdg9",
        "6u72aptz",
        "3vvcw43y",
        "7s83bda9",
        "a57cd5gf",
        "bhcm17x7",
        "uq2spbq5",
        "soihu2u6",
        "xjeyjuv7",
        "rxwgmd87",
        "d71fmtkc",
        "jkilb9i4",
        "e9ij4ib6",
        "kf0rutwf",
        "sstxf4lx",
        "2q9qtjya",
    }
)

# --- Cohort B: G1 transformer -> transformer_classifier (39 classification runs) ---
G1_FIX_IDS: frozenset[str] = frozenset(
    {
        "3uy57sf5",
        "x0jwyhwy",
        "mmitswir",
        "dn3q2gzv",
        "suv93ztg",
        "yf1bvzo7",
        "ktrbti6c",
        "otivnrjh",
        "lv3msgg7",
        "t1dvmoey",
        "yya0vf7o",
        "4dc5ecbm",
        "v5s4suqp",
        "3tx6tvms",
        "0iymkevd",
        "hgh2302e",
        "uh7yufnr",
        "8p392xoc",
        "b9th7d41",
        "sov5wt8e",
        "09xp30o9",
        "e7hnaaq7",
        "c2vgutim",
        "pglx8iu4",
        "zhkne2ej",
        "g4mbvs0b",
        "4sog9llh",
        "gv59fy9y",
        "sgj9wz4o",
        "wp1om1go",
        "zqrrksvg",
        "ofibdnox",
        "xluvne5b",
        "75ff0nbz",
        "d726c4cc",
        "j25pmsq5",
        "31831qff",
        "ec4mpegj",
        "23g52yi5",
    }
)

COL_G1 = "config/axes/G1_Task Type"
COL_G3 = "config/axes/G3_Classification Task"
COL_PG = "config/meta.process_groups_key"
COL_CD = "config/meta.class_def_str"
COL_ROW = "config/meta.row_key"

G1_TARGET = "transformer_classifier"


def csv_patch_for_run(run_id: str) -> dict[str, str]:
    """CSV column name -> new value. Only keys returned are written."""
    out: dict[str, str] = {}
    rid = run_id.strip()
    if rid in G1_FIX_IDS:
        out[COL_G1] = G1_TARGET
    if rid in G3_COHORT_A1_IDS:
        out[COL_G3] = G3_DISPLAY_5WAY
        out[COL_PG] = G3_COMPACT_5WAY
        out[COL_CD] = G3_DISPLAY_5WAY
    elif rid in G3_COHORT_A2_IDS:
        out[COL_G3] = G3_DISPLAY_TTH
        out[COL_PG] = G3_COMPACT_TTH
        out[COL_CD] = G3_DISPLAY_TTH
    return out


def wandb_patch_for_run(run_id: str) -> dict[str, object]:
    """Flat W&B config keys -> values."""
    out: dict[str, object] = {}
    rid = run_id.strip()
    if rid in G1_FIX_IDS:
        out["axes/G1_Task Type"] = G1_TARGET
    if rid in G3_COHORT_A1_IDS:
        out["axes/G3_Classification Task"] = G3_DISPLAY_5WAY
        out["meta.process_groups_key"] = G3_COMPACT_5WAY
        out["meta.class_def_str"] = G3_DISPLAY_5WAY
    elif rid in G3_COHORT_A2_IDS:
        out["axes/G3_Classification Task"] = G3_DISPLAY_TTH
        out["meta.process_groups_key"] = G3_COMPACT_TTH
        out["meta.class_def_str"] = G3_DISPLAY_TTH
    return out


def all_patched_run_ids() -> frozenset[str]:
    return G1_FIX_IDS | G3_COHORT_A1_IDS | G3_COHORT_A2_IDS
