"""End-to-end smoke test for TransformerClassifier with all new modules enabled."""

import sys

sys.path.insert(0, "src")

import torch
from omegaconf import OmegaConf

from thesis_ml.architectures.transformer_classifier.base import build_from_config

B, T = 4, 18

META = {
    "n_tokens": T,
    "token_feat_dim": 4,
    "has_globals": True,
    "n_classes": 2,
    "num_types": 8,
}


def make_cfg(attention_bias=None, nodewise=False, mia=False, include_met=False, legacy_pairwise=False):
    cfg_dict = {
        "classifier": {
            "globals": {"include_met": include_met},
            "model": {
                "dim": 64,
                "depth": 2,
                "heads": 8,
                "mlp_dim": 128,
                "dropout": 0.0,
                "norm": {"policy": "pre"},
                "positional": "sinusoidal",
                "positional_space": "model",
                "positional_dim_mask": None,
                "rotary": {"base": 10000.0},
                "causal_attention": False,
                "pooling": "cls",
                "attention_bias": attention_bias or {},
                "nodewise_mass": {"enabled": nodewise, "k_values": [4, 8], "hidden_dim": 32},
                "mia_blocks": {"enabled": mia, "num_blocks": 2, "interaction_dim": 32, "reduction_dim": 8, "dropout": 0.0},
                "attn_pairwise": {"enabled": legacy_pairwise, "features": ["m2", "deltaR"], "hidden_dim": 8, "per_head": False},
                "tokenizer": {"name": "identity", "id_embed_dim": 8, "pid_mode": "learned"},
            },
        }
    }
    return OmegaConf.create(cfg_dict)


def run_forward(model, include_met=False, label=""):
    tokens_cont = torch.randn(B, T, 4)
    tokens_id = torch.randint(1, 8, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 15:] = False
    globals_ = torch.randn(B, 2) if include_met else None

    out = model(tokens_cont, tokens_id, globals_) if include_met else model(tokens_cont, tokens_id)
    print(f"  {label}: output shape {out.shape}")  # expected [4, 2]
    assert out.shape == (B, 2), f"Expected [4, 2], got {out.shape}"
    # Check gradients flow
    loss = out.sum()
    loss.backward()
    print(f"  {label}: backward pass OK")


print("=== Test 1: Baseline (no bias, no nodewise, no MIA) ===")
model = build_from_config(make_cfg(), META)
run_forward(model, label="baseline")

print("\n=== Test 2: LorentzScalarBias only ===")
cfg = make_cfg(attention_bias={"lorentz_scalar": {"enabled": True, "features": ["m2", "deltaR", "log_kt"], "hidden_dim": 8, "per_head": True}})
model = build_from_config(cfg, META)
run_forward(model, label="lorentz_scalar")

print("\n=== Test 3: TypePairKinematicBias only ===")
cfg = make_cfg(attention_bias={"typepair_kinematic": {"enabled": True, "kinematic_gate": True, "kinematic_feature": "deltaR", "init_from_physics": "none", "mask_value": -5.0}})
model = build_from_config(cfg, META)
run_forward(model, label="typepair_kinematic")

print("\n=== Test 4: SMInteractionBias (binary) ===")
cfg = make_cfg(attention_bias={"sm_interaction": {"enabled": True, "mode": "binary", "mask_value": -100.0}})
model = build_from_config(cfg, META)
run_forward(model, label="sm_interaction/binary")

print("\n=== Test 5: SMInteractionBias (running_coupling) ===")
cfg = make_cfg(attention_bias={"sm_interaction": {"enabled": True, "mode": "running_coupling", "mask_value": -100.0}})
model = build_from_config(cfg, META)
run_forward(model, label="sm_interaction/running")

print("\n=== Test 6: GlobalConditionedBias with include_met ===")
cfg = make_cfg(
    attention_bias={"global_conditioned": {"enabled": True, "global_dim": 16, "mode": "met_direction"}},
    include_met=True,
)
model = build_from_config(cfg, META)
run_forward(model, include_met=True, label="global_conditioned/met_direction")

print("\n=== Test 7: NodewiseMassBias ===")
cfg = make_cfg(nodewise=True)
model = build_from_config(cfg, META)
run_forward(model, label="nodewise_mass")

print("\n=== Test 8: MIAEncoder ===")
cfg = make_cfg(mia=True)
model = build_from_config(cfg, META)
run_forward(model, label="mia_encoder")

print("\n=== Test 9: All modules combined ===")
cfg = make_cfg(
    attention_bias={
        "lorentz_scalar": {"enabled": True, "features": ["m2", "deltaR"], "hidden_dim": 8, "per_head": False},
        "typepair_kinematic": {"enabled": True, "kinematic_gate": False, "kinematic_feature": "deltaR", "init_from_physics": "binary", "mask_value": -5.0},
        "global_conditioned": {"enabled": True, "global_dim": 8, "mode": "global_scale"},
    },
    nodewise=True,
    mia=True,
    include_met=True,
)
model = build_from_config(cfg, META)
run_forward(model, include_met=True, label="all_modules")

print("\n=== Test 10: Legacy attn_pairwise backward-compat ===")
cfg = make_cfg(legacy_pairwise=True)
model = build_from_config(cfg, META)
run_forward(model, label="legacy_attn_pairwise")

print("\n=== Test 11: include_met + CLS (mask bug fix verification) ===")
cfg = make_cfg(
    attention_bias={"lorentz_scalar": {"enabled": True, "features": ["m2", "deltaR"], "hidden_dim": 8, "per_head": False}},
    include_met=True,
)
model = build_from_config(cfg, META)
run_forward(model, include_met=True, label="include_met+lorentz_scalar (bug fix)")

print("\n=== All end-to-end tests PASSED ===")
