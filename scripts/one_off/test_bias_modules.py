"""Quick smoke test for all new physics-informed bias modules."""

import sys

sys.path.insert(0, "src")

import torch
from omegaconf import OmegaConf

from thesis_ml.architectures.transformer_classifier.modules.biases import (
    BiasComposer,
    GlobalConditionedBias,
    LorentzScalarBias,
    MIAEncoder,
    NodewiseMassBias,
    SMInteractionBias,
    TypePairKinematicBias,
    build_bias_composer,
)

B, T = 4, 18
tokens_cont = torch.randn(B, T, 4)
mask = torch.ones(B, T, dtype=torch.bool)
mask[:, 15:] = False
tokens_id = torch.randint(0, 8, (B, T))
globals_ = torch.randn(B, 2)

print("--- LorentzScalarBias ---")
ls = LorentzScalarBias(["m2", "deltaR", "log_kt", "z"], cont_dim=4, hidden_dim=8, num_heads=8, per_head=True)
out = ls(tokens_cont, mask=mask)
print(f"  5-vec output shape: {out.shape}")  # expected [4, 8, 18, 18]
ls_3 = LorentzScalarBias(["m2", "deltaR"], cont_dim=3, hidden_dim=8, num_heads=8, per_head=False)
out_3 = ls_3(tokens_cont[:, :, 1:], mask=mask)
print(f"  4-vec (no E) output shape: {out_3.shape}")  # expected [4, 1, 18, 18] — m2 dropped, deltaR only

print("\n--- TypePairKinematicBias ---")
tpk = TypePairKinematicBias(num_heads=8, cont_dim=4, kinematic_gate=True, kinematic_feature="log_m2")
out2 = tpk(tokens_id=tokens_id, tokens_cont=tokens_cont, mask=mask)
print(f"  output shape: {out2.shape}")  # expected [4, 8, 18, 18]
tpk_nk = TypePairKinematicBias(num_heads=8, cont_dim=4, kinematic_gate=False)
out2_nk = tpk_nk(tokens_id=tokens_id, mask=mask)
print(f"  no kinematic gate: {out2_nk.shape}")

print("\n--- SMInteractionBias ---")
for mode in ("binary", "fixed_coupling", "running_coupling"):
    sm = SMInteractionBias(num_heads=8, cont_dim=4, mode=mode)
    out3 = sm(tokens_id=tokens_id, tokens_cont=tokens_cont, mask=mask)
    print(f"  mode={mode}: {out3.shape}")  # expected [4, 8, 18, 18]

print("\n--- NodewiseMassBias ---")
nm = NodewiseMassBias(model_dim=256, cont_dim=4, k_values=[4, 8, 16], hidden_dim=32)
out5 = nm(tokens_cont, mask=mask)
print(f"  output shape: {out5.shape}")  # expected [4, 18, 256]
nm3 = NodewiseMassBias(model_dim=256, cont_dim=3)  # no E, should return None
out5_3 = nm3(tokens_cont[:, :, 1:], mask=mask)
print(f"  no-E output (should be None): {out5_3}")

print("\n--- GlobalConditionedBias ---")
gc = GlobalConditionedBias(num_heads=8, cont_dim=4, mode="global_scale")
out6 = gc(globals_=globals_)
print(f"  global_scale output: {out6.shape}")  # expected [4, 8, 1, 1]
gc2 = GlobalConditionedBias(num_heads=8, cont_dim=4, mode="met_direction")
out7 = gc2(globals_=globals_, tokens_cont=tokens_cont, mask=mask)
print(f"  met_direction output: {out7.shape}")  # expected [4, 8, 18, 18]

print("\n--- MIAEncoder ---")
x = torch.randn(B, T, 256)
mia = MIAEncoder(model_dim=256, cont_dim=4, num_blocks=2, interaction_dim=64, reduction_dim=8)
x_up, U2 = mia(x, tokens_cont, mask=mask)
print(f"  x_updated: {x_up.shape}")  # expected [4, 18, 256]
print(f"  U2: {U2.shape}")  # expected [4, 8, 18, 18]

print("\n--- BiasComposer (all modules) ---")
composer = BiasComposer(
    bias_modules={
        "lorentz_scalar": LorentzScalarBias(["m2", "deltaR"], cont_dim=4, num_heads=8, per_head=True),
        "typepair": TypePairKinematicBias(num_heads=8, cont_dim=4),
    },
    use_cls=True,
    num_met_tokens=2,
)
bias = composer(tokens_cont=tokens_cont, tokens_id=tokens_id, mask=mask, globals_=None)
print(f"  BiasComposer output (with cls+met): {bias.shape}")  # expected [4, 8, 21, 21]

print("\n--- backward compat: build_bias_composer ---")
cfg = OmegaConf.create({"attn_pairwise": {"enabled": True, "features": ["m2", "deltaR"], "hidden_dim": 8, "per_head": False}})
bc = build_bias_composer(cfg, num_heads=8, model_dim=256, cont_dim=4, use_cls=True, num_met_tokens=0)
bias2 = bc(tokens_cont=tokens_cont, tokens_id=None, mask=mask)
print(f"  Legacy attn_pairwise → BiasComposer output: {bias2.shape}")  # expected [4, 1, 19, 19]

print("\n=== All tests PASSED ===")
