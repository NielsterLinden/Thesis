"""Unit tests for scripts/wandb/v2_axes.py.

Pure-python tests — no W&B import. Invokes ``derive_v2_axes`` on synthetic
run-config dicts and asserts the expected V2 key outputs. Run with:

.. code-block:: bash

    pytest tests/test_v2_axes.py -q

Target-key naming scheme under test: ``axes/<ID>_<Canonical Name>`` where the
ID uses no leading zeros (``G1`` not ``G01``, ``D2`` not ``D02``, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WANDB_DIR = Path(__file__).resolve().parents[1] / "scripts" / "wandb"
if str(_WANDB_DIR) not in sys.path:
    sys.path.insert(0, str(_WANDB_DIR))

import v2_axes as va  # noqa: E402


def _k(axis_id: str) -> str:
    """Short helper — resolve an axis id to its current target key."""
    return f"axes/{axis_id}_{va.HUMAN_NAMES[axis_id]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_identity_minimal() -> dict:
    """Minimal config for a raw/identity-tokenizer transformer run."""
    return {
        "raw/loop": "classifier",
        "raw/data/cont_features": "[0, 1, 2, 3]",
        "raw/classifier/globals/include_met": True,
        "data/token_order": "pt_sorted",
        "raw/classifier/model/tokenizer/name": "identity",
        "raw/classifier/model/tokenizer/pid_mode": "learned",
        "raw/classifier/model/tokenizer/id_embed_dim": 8,
        "raw/classifier/model/positional": "sinusoidal",
        "raw/classifier/model/positional_space": "model",
        "raw/classifier/model/norm/policy": "pre",
        "raw/classifier/model/norm/type": "layernorm",
        "raw/classifier/model/attention/type": "standard",
        "raw/classifier/model/attention/norm": "softmax",
        "raw/classifier/model/causal_attention": False,
        "raw/classifier/model/ffn/type": "standard",
        "raw/classifier/model/moe/enabled": False,
        "raw/classifier/model/head/pooling": "cls",
        "raw/classifier/model/head/type": "standard",
        "raw/classifier/model/attention_biases": "none",
        "raw/classifier/model/dim": 128,
        "raw/classifier/model/depth": 4,
        "raw/classifier/model/heads": 4,
        "raw/classifier/trainer/epochs": 20,
        "raw/classifier/trainer/lr": 1e-4,
    }


# ---------------------------------------------------------------------------
# Basic shape invariants
# ---------------------------------------------------------------------------


def test_derive_returns_one_string_per_registered_target_key():
    out = va.derive_v2_axes({})
    assert set(out.keys()) == {a.target_key for a in va.V2_AXES}
    assert all(isinstance(v, str) for v in out.values())
    assert len(out) == len(va.V2_AXES)


def test_target_keys_follow_id_name_convention():
    for a in va.V2_AXES:
        assert a.target_key.startswith("axes/")
        assert a.id in a.target_key
        assert va.HUMAN_NAMES[a.id] in a.target_key


def test_no_axis_id_has_leading_zero():
    for a in va.V2_AXES:
        parts = a.id.split("-")
        # head must not have leading zero (e.g. "G1" OK, "G01" banned).
        head = parts[0]
        if len(head) >= 2 and head[1:].isdigit():
            assert not head[1:].startswith("0"), f"Axis {a.id} has leading-zero head"


def test_empty_config_all_strings():
    out = va.derive_v2_axes({})
    for v in out.values():
        assert isinstance(v, str)


def test_v2_axes_graph_is_acyclic():
    sorted_axes = va.topological_sort(va.V2_AXES)
    assert len(sorted_axes) == len(va.V2_AXES)


# ---------------------------------------------------------------------------
# T1 / T1-a / T1-b — tokenizer + one_hot override
# ---------------------------------------------------------------------------


def test_t1b_empty_when_t1_binned():
    cfg = {
        "raw/classifier/model/tokenizer/name": "binned",
        "raw/classifier/model/tokenizer/id_embed_dim": 16,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("T1-b")] == ""
    assert out[_k("T1-a")] == ""


def test_t1b_num_types_override_when_onehot():
    cfg = {
        "raw/classifier/model/tokenizer/name": "identity",
        "raw/classifier/model/tokenizer/pid_mode": "one_hot",
        "raw/classifier/model/tokenizer/id_embed_dim": 42,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("T1")] == "identity"
    assert out[_k("T1-a")] == "one_hot"
    assert out[_k("T1-b")] == "num_types"


def test_t1b_integer_when_learned():
    cfg = {
        "raw/classifier/model/tokenizer/name": "identity",
        "raw/classifier/model/tokenizer/pid_mode": "learned",
        "raw/classifier/model/tokenizer/id_embed_dim": 8,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("T1-b")] == "8"


def test_t1c_pretrained_default_vq_when_missing():
    cfg = {"raw/classifier/model/tokenizer/name": "pretrained"}
    out = va.derive_v2_axes(cfg)
    assert out[_k("T1")] == "pretrained"
    assert out[_k("T1-c")] == "vq"


def test_t1c_empty_when_not_pretrained():
    cfg = {"raw/classifier/model/tokenizer/name": "identity"}
    out = va.derive_v2_axes(cfg)
    assert out[_k("T1-c")] == ""


# ---------------------------------------------------------------------------
# E1-a1 dim_mask null semantics
# ---------------------------------------------------------------------------


def test_e1a1_null_when_prereq_met_and_no_config():
    cfg = {
        "raw/classifier/model/positional": "sinusoidal",
        "raw/classifier/model/positional_space": "token",
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("E1-a")] == "token"
    assert out[_k("E1-a1")] == "null"


def test_e1a1_list_when_configured():
    cfg = {
        "raw/classifier/model/positional": "sinusoidal",
        "raw/classifier/model/positional_space": "token",
        "raw/classifier/model/positional_dim_mask": [0, 1, 2],
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("E1-a1")] == "[0,1,2]"


def test_e1a1_empty_when_prereq_fails():
    cfg = {"raw/classifier/model/positional": "rotary"}
    out = va.derive_v2_axes(cfg)
    assert out[_k("E1-a1")] == ""


# ---------------------------------------------------------------------------
# D1 / P1 — cont_features parse & energy gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cont_features",
    ["[0, 1, 2, 3]", "[0,1,2,3]", [0, 1, 2, 3], "[0,1,2]"],
)
def test_p1_energy_gate_passes_when_0_in_cont_features(cont_features):
    cfg = {
        "raw/data/cont_features": cont_features,
        "raw/classifier/model/tokenizer/name": "identity",
        "raw/classifier/model/nodewise_mass/enabled": True,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("P1")] == "true"


def test_p1_gate_fails_when_energy_missing():
    cfg = {
        "raw/data/cont_features": "[1, 2, 3]",
        "raw/classifier/model/tokenizer/name": "identity",
        "raw/classifier/model/nodewise_mass/enabled": True,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("P1")] == ""


def test_p1_gate_fails_when_t1_binned():
    cfg = {
        "raw/data/cont_features": "[0, 1, 2, 3]",
        "raw/classifier/model/tokenizer/name": "binned",
        "raw/classifier/model/nodewise_mass/enabled": True,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("P1")] == ""


# ---------------------------------------------------------------------------
# F1 — effective realization priority (MoE > KAN > standard)
# ---------------------------------------------------------------------------


def test_f1_moe_wins_over_kan():
    cfg = {
        "raw/classifier/model/ffn/type": "kan",
        "raw/classifier/model/moe/enabled": True,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("F1")] == "kan"
    assert out[_k("F1-moe")] == "true"
    assert out[_k("F1-eff")] == "moe"


def test_f1_kan_when_moe_disabled_and_ffn_kan():
    cfg = {
        "raw/classifier/model/ffn/type": "kan",
        "raw/classifier/model/moe/enabled": False,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("F1-eff")] == "kan"


def test_f1_standard_when_neither_moe_nor_kan():
    cfg = {
        "raw/classifier/model/ffn/type": "standard",
        "raw/classifier/model/moe/enabled": False,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("F1-eff")] == "standard"


def test_f1a_kan_variant_only_when_realization_kan():
    cfg = {
        "raw/classifier/model/ffn/type": "kan",
        "raw/classifier/model/moe/enabled": False,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("F1-a")] == "hybrid"


def test_f1a_empty_when_moe_wins():
    cfg = {
        "raw/classifier/model/ffn/type": "kan",
        "raw/classifier/model/moe/enabled": True,
        "raw/classifier/model/ffn/kan/variant": "full",
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("F1-a")] == ""


# ---------------------------------------------------------------------------
# B1 — pure read (no legacy attn_pairwise remap)
# ---------------------------------------------------------------------------


def test_b1_reads_attention_biases_directly():
    cfg = {
        "raw/classifier/model/tokenizer/name": "identity",
        "raw/classifier/model/attention_biases": "lorentz_scalar+typepair",
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("B1")] == "lorentz_scalar+typepair"


def test_b1_empty_when_pre_bias_era_run():
    cfg = {"raw/classifier/model/tokenizer/name": "identity"}
    out = va.derive_v2_axes(cfg)
    assert out[_k("B1")] == ""


def test_b1_none_emitted_literally_when_stored():
    cfg = {
        "raw/classifier/model/tokenizer/name": "identity",
        "raw/classifier/model/attention_biases": "none",
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("B1")] == "none"


def test_b1_ignores_attn_pairwise_legacy():
    cfg = {
        "raw/classifier/model/tokenizer/name": "identity",
        "raw/classifier/model/attention_biases": "none",
        "raw/classifier/model/attn_pairwise/enabled": True,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("B1")] == "none"
    assert out[_k("B1-L1")] == ""


def test_b1_lorentz_features_default_when_family_active():
    cfg = {
        "raw/classifier/model/tokenizer/name": "identity",
        "raw/classifier/model/attention_biases": "lorentz_scalar",
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("B1-L1")] == "[m2,deltaR]"


# ---------------------------------------------------------------------------
# H10 / model size
# ---------------------------------------------------------------------------


def test_h10_formats_integer_cast():
    cfg = {
        "raw/classifier/model/dim": 128,
        "raw/classifier/model/depth": 4,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("H1")] == "128"
    assert out[_k("H2")] == "4"
    assert out[_k("H10")] == "d128_L4"


def test_h10_floats_become_integers():
    cfg = {
        "raw/classifier/model/dim": 128.0,
        "raw/classifier/model/depth": 4.0,
    }
    out = va.derive_v2_axes(cfg)
    assert out[_k("H10")] == "d128_L4"


# ---------------------------------------------------------------------------
# D3 token_order
# ---------------------------------------------------------------------------


def test_d3_reads_stored_token_order():
    out = va.derive_v2_axes({"data/token_order": "pt_sorted"})
    assert out[_k("D3")] == "pt_sorted"


def test_d3_inferred_shuffled_when_shuffle_true():
    out = va.derive_v2_axes({"raw/data/shuffle_tokens": True})
    assert out[_k("D3")] == "shuffled"


def test_d3_inferred_pt_sorted_when_sort_by_pt():
    out = va.derive_v2_axes({"raw/data/sort_tokens_by": "pt"})
    assert out[_k("D3")] == "pt_sorted"


# ---------------------------------------------------------------------------
# G2 — model family inference from G1
# ---------------------------------------------------------------------------


def test_g2_explicit_model_type_wins():
    cfg = {"raw/loop": "classifier", "raw/model/type": "transformer"}
    out = va.derive_v2_axes(cfg)
    assert out[_k("G1")] == "classifier"
    assert out[_k("G2")] == "transformer"


def test_g2_inferred_from_loop_substring():
    out = va.derive_v2_axes({"raw/loop": "train_mlp_classifier"})
    assert out[_k("G2")] == "mlp"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotency_derive_of_derive():
    cfg = _raw_identity_minimal()
    v1 = va.derive_v2_axes(cfg)
    cfg_after = dict(cfg)
    cfg_after.update(v1)
    v2 = va.derive_v2_axes(cfg_after)
    assert v1 == v2


def test_idempotency_across_reruns_unchanged_cfg():
    cfg = _raw_identity_minimal()
    assert va.derive_v2_axes(cfg) == va.derive_v2_axes(cfg)


# ---------------------------------------------------------------------------
# Flag bucket wiring
# ---------------------------------------------------------------------------


def test_flag_bucket_records_prereq_failures():
    cfg = {"raw/classifier/model/tokenizer/name": "binned"}
    flags: dict = {}
    va.derive_v2_axes(cfg, flag_bucket=flags)
    assert "T1-a" in flags["keys_left_empty_by_prereq"]
    assert "T1-b" in flags["keys_left_empty_by_prereq"]
    assert "T1-c" in flags["keys_left_empty_by_prereq"]
    assert flags["unresolved_flags"] == []


def test_flag_bucket_records_missing_config():
    cfg = {"raw/classifier/model/tokenizer/name": "identity", "raw/classifier/model/tokenizer/pid_mode": "learned"}
    flags: dict = {}
    va.derive_v2_axes(cfg, flag_bucket=flags)
    assert "T1-b" in flags["keys_empty_missing_config"]


def test_target_key_contains_space_readable_name():
    """Spot-check: G1's target key is 'axes/G1_Task Type'."""
    axes_by_id = {a.id: a for a in va.V2_AXES}
    assert axes_by_id["G1"].target_key == "axes/G1_Task Type"
    assert axes_by_id["T1-b"].target_key == "axes/T1-b_PID Embedding Dimension"
    assert axes_by_id["D2"].target_key == "axes/D2_MET Treatment"
