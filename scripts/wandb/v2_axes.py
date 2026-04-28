"""V2 axis registry + pure derivation function for the thesis-ml W&B backfill.

This module is pure Python, no W&B imports. It is consumed by
``scripts/wandb/backfill_labels.py --mode v2`` to compute V2 ``axes/*`` keys
from each run's stored ``run.config`` dict.

See ``docs/backfill_plan/02_mapping_spec.md`` for the per-axis derivation
rules, ``docs/backfill_plan/00_doc_issues.md`` for the discharged ambiguities,
and ``docs/backfill_plan/01_audit.md`` §2.1 for the config-source lookup table.

Naming scheme (from user direction, 2026-04-22)
------------------------------------------------
- Axis IDs strip leading zeros: ``G1`` not ``G01``, ``D2`` not ``D02``,
  ``H1`` not ``H01``, ``R1`` not ``R01``. Two-digit IDs (``H10``, ``R10``...``R17``)
  are unchanged.
- W&B target keys follow the form ``axes/<ID>_<Canonical Name>`` where the
  canonical name comes from the V2 reference doc's section headings (Title Case,
  with acronyms preserved like ``PID``, ``MET``, ``MIA``, ``MoE``, ``KAN``,
  ``SM``, ``LR``, ``PE``, ``FFN``). Spaces inside the name are kept — W&B
  config keys allow spaces and this makes the Runs-table columns self-labelling.

Design invariants
-----------------
- Every value returned by ``derive_v2_axes`` is a ``str`` (possibly empty).
- Empty string ``""`` is the canonical "not applicable / missing" marker.
- Prerequisite gates are evaluated on already-derived parent values,
  topologically. Source config is not read when the gate fails.
- Config-source priority is RAW-FIRST: ``raw/classifier/.../<leaf>`` beats
  the namespaced slice, which beats legacy ``axes/<key>``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Human names (section-heading titles from docs/AXES_REFERENCE_V2.md)
# ---------------------------------------------------------------------------

HUMAN_NAMES: dict[str, str] = {
    # G — Study framing
    "G1": "Task Type",
    "G2": "Model Family",
    "G3": "Classification Task",
    # D — Data treatment
    "D1": "Feature Set",
    "D2": "MET Treatment",
    "D3": "Token Ordering",
    # T — Tokenizer
    "T1": "Tokenizer Family",
    "T1-a": "PID Embedding Mode",
    "T1-b": "PID Embedding Dimension",
    "T1-c": "Pretrained Model Type",
    # E — Positional encoding
    "E1": "PE Type",
    "E1-a": "PE Space",
    "E1-a1": "PE Dimension Mask",
    "E1-b": "Rotary Base Frequency",
    # A — Attention & encoder block
    "A1": "Normalization Policy",
    "A2": "Normalization Type",
    "A3": "Attention Type",
    "A3-a": "Differential Attention Bias Mode",
    "A4": "Attention Internal Normalization",
    "A5": "Causal Masking",
    # F — Encoder FFN (F1 raw type, F1-moe raw moe_enabled, F1-eff derived realization)
    "F1": "FFN Type",
    "F1-moe": "MoE Enabled",
    "F1-eff": "FFN Realization",
    "F1-a": "KAN FFN Variant",
    "F1-a1": "KAN FFN Bottleneck Dimension",
    "F1-b": "MoE Encoder Scope",
    # C — Classifier head
    "C2": "Pooling Strategy",
    "C1": "Head Realization",
    # P — Pre-encoder modules
    "P1": "Nodewise Mass Enabled",
    "P1-a": "Nodewise Mass Neighbourhood Sizes",
    "P1-b": "Nodewise Mass Hidden Dimension",
    "P2": "MIA Pre-Encoder Enabled",
    "P2-a": "MIA Placement",
    "P2-b": "MIA Number of Blocks",
    "P2-c": "MIA Interaction Dimension",
    "P2-d": "MIA Reduction Dimension",
    "P2-e": "MIA Dropout",
    # B — Physics biases
    "B1": "Bias Activation Set",
    "B1-L1": "Lorentz Feature Set",
    "B1-L2": "Lorentz MLP Type",
    "B1-L3": "Lorentz Hidden Dimension",
    "B1-L4": "Lorentz Per-Head Mode",
    "B1-L5": "Lorentz Sparse Gating",
    "B1-T1": "Type-Pair Initialization",
    "B1-T2": "Type-Pair Freeze Table",
    "B1-T3": "Type-Pair Kinematic Gate",
    "B1-T4": "Type-Pair Kinematic Feature",
    "B1-T5": "Type-Pair Mask Value",
    "B1-S1": "SM Interaction Mode",
    "B1-S2": "SM Mask Value",
    "B1-G1": "Global-Conditioned Mode",
    "B1-G2": "Global-Conditioned MLP Type",
    "B1-G3": "Global-Conditioned Global Dimension",
    # K — KAN shared
    "K1": "KAN Grid Size",
    "K2": "KAN Spline Order",
    "K3": "KAN Grid Range",
    "K4": "KAN Spline Regularization Weight",
    "K5": "KAN Grid Update Frequency",
    # M — MoE shared
    "M1": "MoE Number of Experts",
    "M2": "MoE Top K",
    "M3": "MoE Routing Level",
    "M4": "MoE Load Balance Weight",
    "M5": "MoE Noisy Gating",
    # S — Shared backbone
    "S1": "Shared Backbone Enabled",
    "S2": "Shared Backbone Features",
    # H — Model-size
    "H1": "Model Dimension",
    "H2": "Encoder Depth",
    "H3": "Attention Heads",
    "H4": "FFN Hidden Dimension",
    "H5": "Dropout",
    "H10": "Model Size Label",
    # R — Training protocol
    "R1": "Epochs",
    "R2": "Learning Rate",
    "R3": "Weight Decay",
    "R4": "Batch Size",
    "R5": "Seed",
    "R6": "Warmup Steps",
    "R7": "LR Schedule",
    "R8": "Label Smoothing",
    "R9": "Gradient Clipping",
    "R10": "Early Stop Enabled",
    "R11": "Early Stop Patience",
    "R12": "Early Stop Min Delta",
    "R13": "Restore Best Weights",
    "R14": "PID Schedule Mode",
    "R15": "PID Transition Epoch",
    "R16": "PID Reinit Mode",
    "R17": "PID Separate LR",
    # L — Logging & interpretability
    "L1": "Log PID Embeddings",
    "L2": "Interpretability Enabled",
    "L3": "Save Attention Maps",
    "L4": "Save KAN Splines",
    "L5": "Save MoE Routing",
    "L6": "Save Gradient Norms",
    "L7": "Checkpoint Epochs",
}


def _target_key(axis_id: str) -> str:
    """Build ``axes/<ID>_<Canonical Name>``. Raises KeyError on unknown ID."""
    return f"axes/{axis_id}_{HUMAN_NAMES[axis_id]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return bool(isinstance(v, list | tuple | dict) and len(v) == 0)


def _read_chain(cfg: dict, keys: list[str]) -> Any:
    """Return the value of the first key in *keys* whose value is non-empty."""
    for k in keys:
        if k in cfg:
            v = cfg[k]
            if not _is_empty(v):
                return v
    return None


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return s


def _as_lower(v: Any) -> str:
    return _as_str(v).lower()


def _as_bool_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    s = _as_str(v).lower()
    if s in ("true", "1", "yes", "y"):
        return "true"
    if s in ("false", "0", "no", "n"):
        return "false"
    return ""


def _as_int_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return ""


def _as_float_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        return str(float(v))
    except (ValueError, TypeError):
        return ""


def _as_list_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list | tuple):
        inner = ",".join(str(x).strip() for x in v)
        return f"[{inner}]"
    s = _as_str(v)
    if s.startswith("[") and s.endswith("]"):
        return s.replace(" ", "")
    return s


def _parse_cont_features(v: Any) -> set[int] | None:
    if v is None:
        return None
    if isinstance(v, list | tuple):
        try:
            return {int(x) for x in v}
        except (ValueError, TypeError):
            return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        s_json = s.replace(" ", "")
        try:
            parsed = json.loads(s_json)
            if isinstance(parsed, list):
                return {int(x) for x in parsed}
        except (ValueError, TypeError):
            pass
    return None


def _bias_list(b1_value: str) -> list[str]:
    if not b1_value or b1_value.strip().lower() in ("", "none"):
        return []
    tokens = [t.strip().lower() for t in b1_value.split("+")]
    alias = {"typepair": "typepair_kinematic", "sm": "sm_interaction", "global": "global_conditioned"}
    result = []
    for t in tokens:
        if t in ("", "none"):
            continue
        t = alias.get(t, t)
        if t not in result:
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# Axis dataclass
# ---------------------------------------------------------------------------


@dataclass
class V2Axis:
    id: str
    target_key: str
    parents: tuple[str, ...] = ()
    prereq: Callable[[dict], bool] | None = None  # derived dict -> bool
    emit: Callable[[dict, dict, dict], str] | None = None  # (cfg, derived, flags) -> str
    notes: str = ""


def _simple_emit(sources: list[str], cast: str = "str", default: str = "") -> Callable[[dict, dict, dict], str]:
    caster = {
        "str": _as_str,
        "lower": _as_lower,
        "int": _as_int_str,
        "float": _as_float_str,
        "bool": _as_bool_str,
        "list": _as_list_str,
    }[cast]

    def _emit(cfg: dict, derived: dict, flags: dict) -> str:
        v = _read_chain(cfg, sources)
        if v is None:
            return default
        return caster(v)

    return _emit


# ---------------------------------------------------------------------------
# Emit closures for the non-trivial axes
# ---------------------------------------------------------------------------


def _emit_G1(cfg: dict, derived: dict, flags: dict) -> str:
    return _as_lower(_read_chain(cfg, ["raw/loop", "model/loop", "meta.model_family"]))


def _emit_G2(cfg: dict, derived: dict, flags: dict) -> str:
    v = _read_chain(cfg, ["raw/model/type", "model/type"])
    if v is not None:
        return _as_lower(v)
    g1 = derived.get("G1", "")
    if not g1:
        return ""
    for keyword, family in (
        ("transformer", "transformer"),
        ("mlp", "mlp"),
        ("bdt", "bdt"),
        ("autoencoder", "autoencoder"),
        ("ae", "autoencoder"),
        ("gan", "autoencoder"),
        ("diffusion", "autoencoder"),
    ):
        if keyword in g1:
            return family
    return g1


def _emit_G3(cfg: dict, derived: dict, flags: dict) -> str:
    return _as_str(_read_chain(cfg, ["meta.class_def_str", "meta.process_groups_key", "meta.row_key"]))


def _emit_D1(cfg: dict, derived: dict, flags: dict) -> str:
    v = _read_chain(
        cfg,
        ["raw/data/cont_features", "raw/classifier/data/cont_features", "data/cont_features", "axes/cont_features"],
    )
    if v is None:
        return "[0,1,2,3]"
    return _as_list_str(v)


def _emit_D2(cfg: dict, derived: dict, flags: dict) -> str:
    return _as_bool_str(
        _read_chain(
            cfg,
            ["raw/classifier/globals/include_met", "globals/include_met", "axes/include_met"],
        )
    )


def _emit_D3(cfg: dict, derived: dict, flags: dict) -> str:
    v = _read_chain(cfg, ["data/token_order", "axes/token_order"])
    if v is not None:
        return _as_lower(v)
    shuffle = _read_chain(cfg, ["raw/data/shuffle_tokens", "data/shuffle_tokens"])
    if shuffle is not None and _as_bool_str(shuffle) == "true":
        return "shuffled"
    sort_by = _read_chain(cfg, ["raw/data/sort_tokens_by", "data/sort_tokens_by"])
    if sort_by is not None and _as_lower(sort_by) == "pt":
        return "pt_sorted"
    return "input_order" if (shuffle is not None or sort_by is not None) else ""


def _emit_T1(cfg: dict, derived: dict, flags: dict) -> str:
    v = _read_chain(
        cfg,
        ["raw/classifier/model/tokenizer/name", "tokenizer/type", "axes/tokenizer_name"],
    )
    if v is None:
        return "identity"
    return _as_lower(v)


def _emit_T1a(cfg: dict, derived: dict, flags: dict) -> str:
    return _as_lower(
        _read_chain(
            cfg,
            ["raw/classifier/model/tokenizer/pid_mode", "tokenizer/pid_mode", "axes/pid_mode"],
        )
    )


def _emit_T1b(cfg: dict, derived: dict, flags: dict) -> str:
    if derived.get("T1-a") == "one_hot":
        return "num_types"
    return _as_int_str(
        _read_chain(
            cfg,
            ["raw/classifier/model/tokenizer/id_embed_dim", "tokenizer/id_embed_dim", "axes/id_embed_dim"],
        )
    )


def _emit_T1c(cfg: dict, derived: dict, flags: dict) -> str:
    v = _read_chain(cfg, ["raw/classifier/model/tokenizer/model_type", "tokenizer/model_type"])
    if v is None:
        return "vq"
    return _as_lower(v)


def _emit_E1a1(cfg: dict, derived: dict, flags: dict) -> str:
    v = _read_chain(
        cfg,
        ["raw/classifier/model/positional_dim_mask", "pos_enc/dim_mask", "axes/positional_dim_mask"],
    )
    if v is None:
        return "null"
    if isinstance(v, str) and v.strip().lower() in ("null", "none"):
        return "null"
    return _as_list_str(v)


def _emit_P1(cfg: dict, derived: dict, flags: dict) -> str:
    return _as_bool_str(
        _read_chain(
            cfg,
            ["raw/classifier/model/nodewise_mass/enabled", "nodewise_mass/enabled", "axes/nodewise_mass_enabled"],
        )
    )


def _emit_P2a(cfg: dict, derived: dict, flags: dict) -> str:
    v = _read_chain(cfg, ["raw/classifier/model/mia_blocks/placement", "mia/placement", "axes/mia_placement"])
    s = _as_lower(v)
    if s == "interleave":
        return "prepend"
    return s


def _emit_F1_ffn_type(cfg: dict, derived: dict, flags: dict) -> str:
    return _as_lower(
        _read_chain(
            cfg,
            ["raw/classifier/model/ffn/type", "ffn/type", "axes/ffn_type"],
        )
    )


def _emit_F1_moe_enabled(cfg: dict, derived: dict, flags: dict) -> str:
    return _as_bool_str(
        _read_chain(
            cfg,
            ["raw/classifier/model/moe/enabled", "moe/enabled", "axes/moe_enabled"],
        )
    )


def _emit_F1_realization(cfg: dict, derived: dict, flags: dict) -> str:
    moe = derived.get("F1-moe", "")
    ffn = derived.get("F1", "")
    if moe == "true":
        return "moe"
    if ffn == "kan":
        return "kan"
    if moe == "" and ffn == "":
        return ""
    return "standard"


def _emit_B1(cfg: dict, derived: dict, flags: dict) -> str:
    v = _read_chain(
        cfg,
        ["raw/classifier/model/attention_biases", "bias/selector", "axes/attention_biases"],
    )
    if v is None:
        return "none"
    return _as_lower(v)


def _emit_H10(cfg: dict, derived: dict, flags: dict) -> str:
    dim = derived.get("H1", "")
    depth = derived.get("H2", "")
    if not dim or not depth:
        return ""
    try:
        d = int(float(dim))
        L = int(float(depth))
        return f"d{d}_L{L}"
    except (ValueError, TypeError):
        flags["unresolved_flags"].append("H10: cast failure")
        return ""


# ---------------------------------------------------------------------------
# Gate predicates
# ---------------------------------------------------------------------------


def _gate_T1_identity(d: dict) -> bool:
    return d.get("T1") == "identity"


def _gate_T1_pretrained(d: dict) -> bool:
    return d.get("T1") == "pretrained"


def _gate_E1_sincos_learned(d: dict) -> bool:
    return d.get("E1") in ("sinusoidal", "learned")


def _gate_E1a_token(d: dict) -> bool:
    return d.get("E1-a") == "token"


def _gate_E1_rotary(d: dict) -> bool:
    return d.get("E1") == "rotary"


def _gate_T1_raw_identity(d: dict) -> bool:
    return d.get("T1") in ("raw", "identity")


def _gate_P2_true(d: dict) -> bool:
    return d.get("P2") == "true"


def _gate_P1_true(d: dict) -> bool:
    return d.get("P1") == "true"


def _gate_A3_differential(d: dict) -> bool:
    return d.get("A3") == "differential"


def _gate_ffn_kan(d: dict) -> bool:
    return d.get("F1-eff") == "kan"


def _gate_ffn_moe(d: dict) -> bool:
    return d.get("F1-eff") == "moe"


def _gate_F1a_bottleneck(d: dict) -> bool:
    return d.get("F1-a") == "bottleneck"


def _gate_B1_lorentz(d: dict) -> bool:
    return "lorentz_scalar" in _bias_list(d.get("B1", ""))


def _gate_B1_typepair(d: dict) -> bool:
    return "typepair_kinematic" in _bias_list(d.get("B1", ""))


def _gate_B1_sm(d: dict) -> bool:
    return "sm_interaction" in _bias_list(d.get("B1", ""))


def _gate_B1_global(d: dict) -> bool:
    return "global_conditioned" in _bias_list(d.get("B1", ""))


def _gate_kan_consumer(d: dict) -> bool:
    return d.get("F1-a") not in ("", None) or d.get("C1") == "kan" or d.get("B1-L2") == "kan" or d.get("B1-G2") == "kan"


def _gate_moe_consumer(d: dict) -> bool:
    return d.get("F1-eff") == "moe" or d.get("C1") == "moe"


def _gate_backbone_consumer(d: dict) -> bool:
    biases = _bias_list(d.get("B1", ""))
    return len(biases) > 0 or d.get("P2") == "true"


def _gate_pid_learned(d: dict) -> bool:
    return d.get("T1") == "identity" and d.get("T1-a") == "learned"


def _gate_L2(d: dict) -> bool:
    return d.get("L2") == "true"


def _gate_R10(d: dict) -> bool:
    return d.get("R10") == "true"


def _gate_P1(d: dict) -> bool:
    if d.get("T1") not in ("raw", "identity"):
        return False
    parsed = d.get("_D1_parsed")
    return bool(parsed is not None and 0 in parsed)


# ---------------------------------------------------------------------------
# Build the V2_AXES list (topological order)
# ---------------------------------------------------------------------------


def _A(axis_id: str, **kwargs) -> V2Axis:
    """Shorthand: auto-fill target_key from HUMAN_NAMES[axis_id]."""
    return V2Axis(id=axis_id, target_key=_target_key(axis_id), **kwargs)


def _build_registry() -> list[V2Axis]:
    A: list[V2Axis] = []

    # §1 G
    A.append(_A("G1", emit=_emit_G1))
    A.append(_A("G2", parents=("G1",), emit=_emit_G2))
    A.append(_A("G3", emit=_emit_G3))

    # §2 D
    A.append(_A("D1", emit=_emit_D1))
    A.append(_A("D2", emit=_emit_D2))
    A.append(_A("D3", emit=_emit_D3))

    # §3 T
    A.append(_A("T1", emit=_emit_T1))
    A.append(_A("T1-a", parents=("T1",), prereq=_gate_T1_identity, emit=_emit_T1a))
    A.append(_A("T1-b", parents=("T1", "T1-a"), prereq=_gate_T1_identity, emit=_emit_T1b))
    A.append(_A("T1-c", parents=("T1",), prereq=_gate_T1_pretrained, emit=_emit_T1c))

    # §4 E
    A.append(
        _A(
            "E1",
            emit=_simple_emit(
                ["raw/classifier/model/positional", "pos_enc/type", "axes/positional"],
                cast="lower",
            ),
        )
    )
    A.append(
        _A(
            "E1-a",
            parents=("E1",),
            prereq=_gate_E1_sincos_learned,
            emit=_simple_emit(
                ["raw/classifier/model/positional_space", "pos_enc/space", "axes/positional_space"],
                cast="lower",
                default="model",
            ),
        )
    )
    A.append(_A("E1-a1", parents=("E1-a",), prereq=_gate_E1a_token, emit=_emit_E1a1))
    A.append(
        _A(
            "E1-b",
            parents=("E1",),
            prereq=_gate_E1_rotary,
            emit=_simple_emit(
                ["raw/classifier/model/rotary/base", "pos_enc/rotary_base"],
                cast="float",
                default="10000.0",
            ),
        )
    )

    # §6 A
    A.append(
        _A(
            "A1",
            emit=_simple_emit(
                ["raw/classifier/model/norm/policy", "norm/policy", "axes/norm_policy"],
                cast="lower",
                default="pre",
            ),
        )
    )
    A.append(
        _A(
            "A2",
            emit=_simple_emit(
                ["raw/classifier/model/norm/type", "norm/type", "axes/norm_type"],
                cast="lower",
                default="layernorm",
            ),
        )
    )
    A.append(
        _A(
            "A3",
            emit=_simple_emit(
                ["raw/classifier/model/attention/type", "attention/type", "axes/attention_type"],
                cast="lower",
                default="standard",
            ),
        )
    )
    A.append(
        _A(
            "A3-a",
            parents=("A3",),
            prereq=_gate_A3_differential,
            emit=_simple_emit(
                ["raw/classifier/model/attention/diff_bias_mode", "attention/diff_bias_mode", "axes/diff_bias_mode"],
                cast="lower",
                default="shared",
            ),
        )
    )
    A.append(
        _A(
            "A4",
            emit=_simple_emit(
                ["raw/classifier/model/attention/norm", "attention/norm", "axes/attention_norm"],
                cast="lower",
                default="none",
            ),
        )
    )
    A.append(
        _A(
            "A5",
            emit=_simple_emit(
                ["raw/classifier/model/causal_attention", "model/causal_attention", "axes/causal_attention"],
                cast="bool",
                default="false",
            ),
        )
    )

    # §7 F (three entries) + §9 C
    A.append(_A("F1", emit=_emit_F1_ffn_type))
    A.append(_A("F1-moe", emit=_emit_F1_moe_enabled))
    A.append(_A("F1-eff", parents=("F1", "F1-moe"), emit=_emit_F1_realization))
    A.append(
        _A(
            "F1-a",
            parents=("F1-eff",),
            prereq=_gate_ffn_kan,
            emit=_simple_emit(
                ["raw/classifier/model/ffn/kan/variant", "ffn/kan_variant", "axes/kan_ffn_variant"],
                cast="lower",
                default="hybrid",
            ),
        )
    )
    A.append(
        _A(
            "F1-a1",
            parents=("F1-a",),
            prereq=_gate_F1a_bottleneck,
            emit=_simple_emit(
                ["raw/classifier/model/ffn/kan/bottleneck_dim"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "F1-b",
            parents=("F1-eff",),
            prereq=_gate_ffn_moe,
            emit=_simple_emit(
                ["raw/classifier/model/moe/scope", "moe/scope", "axes/moe_scope"],
                cast="lower",
                default="all_blocks",
            ),
        )
    )
    A.append(
        _A(
            "C2",
            emit=_simple_emit(
                ["raw/classifier/model/head/pooling", "raw/classifier/model/pooling", "pooling/type", "axes/pooling"],
                cast="lower",
            ),
        )
    )
    A.append(
        _A(
            "C1",
            emit=_simple_emit(
                ["raw/classifier/model/head/type", "head/type", "axes/head_type"],
                cast="lower",
            ),
        )
    )

    # §5 P
    A.append(_A("P1", parents=("T1", "D1"), prereq=_gate_P1, emit=_emit_P1))
    A.append(
        _A(
            "P1-a",
            parents=("P1",),
            prereq=_gate_P1_true,
            emit=_simple_emit(
                ["raw/classifier/model/nodewise_mass/k_values", "nodewise_mass/k_values"],
                cast="list",
                default="[2,4,8]",
            ),
        )
    )
    A.append(
        _A(
            "P1-b",
            parents=("P1",),
            prereq=_gate_P1_true,
            emit=_simple_emit(
                ["raw/classifier/model/nodewise_mass/hidden_dim"],
                cast="int",
                default="64",
            ),
        )
    )
    A.append(
        _A(
            "P2",
            parents=("T1",),
            prereq=_gate_T1_raw_identity,
            emit=_simple_emit(
                ["raw/classifier/model/mia_blocks/enabled", "mia/enabled", "axes/mia_enabled"],
                cast="bool",
            ),
        )
    )
    A.append(_A("P2-a", parents=("P2",), prereq=_gate_P2_true, emit=_emit_P2a))
    A.append(
        _A(
            "P2-b",
            parents=("P2",),
            prereq=_gate_P2_true,
            emit=_simple_emit(
                ["raw/classifier/model/mia_blocks/num_blocks", "mia/num_blocks"],
                cast="int",
                default="5",
            ),
        )
    )
    A.append(
        _A(
            "P2-c",
            parents=("P2",),
            prereq=_gate_P2_true,
            emit=_simple_emit(
                ["raw/classifier/model/mia_blocks/interaction_dim"],
                cast="int",
                default="64",
            ),
        )
    )
    A.append(
        _A(
            "P2-d",
            parents=("P2",),
            prereq=_gate_P2_true,
            emit=_simple_emit(
                ["raw/classifier/model/mia_blocks/reduction_dim"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "P2-e",
            parents=("P2",),
            prereq=_gate_P2_true,
            emit=_simple_emit(
                ["raw/classifier/model/mia_blocks/dropout"],
                cast="float",
                default="0.0",
            ),
        )
    )

    # §8 B
    A.append(_A("B1", parents=("T1",), prereq=_gate_T1_raw_identity, emit=_emit_B1))
    A.append(
        _A(
            "B1-L1",
            parents=("B1",),
            prereq=_gate_B1_lorentz,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/lorentz_scalar/features", "bias/lorentz_features", "axes/lorentz_features"],
                cast="list",
                default="[m2,deltaR]",
            ),
        )
    )
    A.append(
        _A(
            "B1-L2",
            parents=("B1",),
            prereq=_gate_B1_lorentz,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/lorentz_scalar/mlp_type", "kan/bias_lorentz_mlp_type", "axes/lorentz_mlp_type"],
                cast="lower",
                default="standard",
            ),
        )
    )
    A.append(
        _A(
            "B1-L3",
            parents=("B1",),
            prereq=_gate_B1_lorentz,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/lorentz_scalar/hidden_dim"],
                cast="int",
                default="8",
            ),
        )
    )
    A.append(
        _A(
            "B1-L4",
            parents=("B1",),
            prereq=_gate_B1_lorentz,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/lorentz_scalar/per_head", "bias/lorentz_per_head", "axes/lorentz_per_head"],
                cast="bool",
                default="false",
            ),
        )
    )
    A.append(
        _A(
            "B1-L5",
            parents=("B1",),
            prereq=_gate_B1_lorentz,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/lorentz_scalar/sparse_gating", "bias/lorentz_sparse_gating", "axes/lorentz_sparse_gating"],
                cast="bool",
                default="false",
            ),
        )
    )
    A.append(
        _A(
            "B1-T1",
            parents=("B1",),
            prereq=_gate_B1_typepair,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/typepair_kinematic/init_from_physics", "bias/typepair_init", "axes/typepair_init"],
                cast="lower",
                default="none",
            ),
        )
    )
    A.append(
        _A(
            "B1-T2",
            parents=("B1",),
            prereq=_gate_B1_typepair,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/typepair_kinematic/freeze_table", "bias/typepair_freeze", "axes/typepair_freeze"],
                cast="bool",
                default="false",
            ),
        )
    )
    A.append(
        _A(
            "B1-T3",
            parents=("B1",),
            prereq=_gate_B1_typepair,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/typepair_kinematic/kinematic_gate", "bias/typepair_kinematic_gate", "axes/typepair_kinematic_gate"],
                cast="bool",
                default="true",
            ),
        )
    )
    A.append(
        _A(
            "B1-T4",
            parents=("B1",),
            prereq=_gate_B1_typepair,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/typepair_kinematic/kinematic_feature"],
                cast="lower",
                default="log_m2",
            ),
        )
    )
    A.append(
        _A(
            "B1-T5",
            parents=("B1",),
            prereq=_gate_B1_typepair,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/typepair_kinematic/mask_value"],
                cast="float",
                default="-5.0",
            ),
        )
    )
    A.append(
        _A(
            "B1-S1",
            parents=("B1",),
            prereq=_gate_B1_sm,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/sm_interaction/mode", "bias/sm_mode", "axes/sm_mode"],
                cast="lower",
                default="binary",
            ),
        )
    )
    A.append(
        _A(
            "B1-S2",
            parents=("B1",),
            prereq=_gate_B1_sm,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/sm_interaction/mask_value"],
                cast="float",
                default="-100.0",
            ),
        )
    )
    A.append(
        _A(
            "B1-G1",
            parents=("B1", "D2"),
            prereq=_gate_B1_global,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/global_conditioned/mode", "bias/global_mode", "axes/global_conditioned_mode"],
                cast="lower",
                default="global_scale",
            ),
        )
    )
    A.append(
        _A(
            "B1-G2",
            parents=("B1",),
            prereq=_gate_B1_global,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/global_conditioned/mlp_type", "kan/bias_global_mlp_type"],
                cast="lower",
                default="standard",
            ),
        )
    )
    A.append(
        _A(
            "B1-G3",
            parents=("B1",),
            prereq=_gate_B1_global,
            emit=_simple_emit(
                ["raw/classifier/model/bias_config/global_conditioned/global_dim"],
                cast="int",
                default="16",
            ),
        )
    )

    # §11 §K
    A.append(
        _A(
            "K1",
            prereq=_gate_kan_consumer,
            emit=_simple_emit(
                ["raw/classifier/model/kan/grid_size", "kan/grid_size", "axes/kan_grid_size"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "K2",
            prereq=_gate_kan_consumer,
            emit=_simple_emit(
                ["raw/classifier/model/kan/spline_order", "kan/spline_order", "axes/kan_spline_order"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "K3",
            prereq=_gate_kan_consumer,
            emit=_simple_emit(
                ["raw/classifier/model/kan/grid_range", "kan/grid_range"],
                cast="list",
            ),
        )
    )
    A.append(
        _A(
            "K4",
            prereq=_gate_kan_consumer,
            emit=_simple_emit(
                ["raw/classifier/model/kan/spline_regularization_weight", "kan/spline_reg_weight"],
                cast="float",
            ),
        )
    )
    A.append(
        _A(
            "K5",
            prereq=_gate_kan_consumer,
            emit=_simple_emit(
                ["raw/classifier/model/kan/grid_update_freq", "kan/grid_update_freq"],
                cast="int",
            ),
        )
    )

    # §12 §M
    A.append(
        _A(
            "M1",
            prereq=_gate_moe_consumer,
            emit=_simple_emit(
                ["raw/classifier/model/moe/num_experts", "moe/num_experts"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "M2",
            prereq=_gate_moe_consumer,
            emit=_simple_emit(
                ["raw/classifier/model/moe/top_k", "moe/top_k", "axes/moe_top_k"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "M3",
            prereq=_gate_moe_consumer,
            emit=_simple_emit(
                ["raw/classifier/model/moe/routing_level", "moe/routing_level", "axes/moe_routing_level"],
                cast="lower",
            ),
        )
    )
    A.append(
        _A(
            "M4",
            prereq=_gate_moe_consumer,
            emit=_simple_emit(
                ["raw/classifier/model/moe/load_balance_loss_weight", "moe/lb_weight"],
                cast="float",
            ),
        )
    )
    A.append(
        _A(
            "M5",
            prereq=_gate_moe_consumer,
            emit=_simple_emit(
                ["raw/classifier/model/moe/noisy_gating", "moe/noisy_gating"],
                cast="bool",
            ),
        )
    )

    # §13 §S
    A.append(
        _A(
            "S1",
            prereq=_gate_backbone_consumer,
            emit=_simple_emit(
                ["raw/classifier/model/shared_backbone/enabled"],
                cast="bool",
                default="false",
            ),
        )
    )
    A.append(
        _A(
            "S2",
            prereq=_gate_backbone_consumer,
            emit=_simple_emit(
                ["raw/classifier/model/shared_backbone/features"],
                cast="list",
            ),
        )
    )

    # §10 H
    A.append(
        _A(
            "H1",
            emit=_simple_emit(
                ["raw/classifier/model/dim", "model/dim", "axes/dim"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "H2",
            emit=_simple_emit(
                ["raw/classifier/model/depth", "model/depth", "axes/depth"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "H3",
            emit=_simple_emit(
                ["raw/classifier/model/heads", "model/heads", "axes/heads"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "H4",
            emit=_simple_emit(
                ["raw/classifier/model/mlp_dim", "model/mlp_dim", "axes/mlp_dim"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "H5",
            emit=_simple_emit(
                ["raw/classifier/model/dropout", "model/dropout", "axes/dropout"],
                cast="float",
            ),
        )
    )
    A.append(_A("H10", parents=("H1", "H2"), emit=_emit_H10))

    # §14 §R
    A.append(
        _A(
            "R1",
            emit=_simple_emit(
                ["raw/classifier/trainer/epochs", "training/epochs", "axes/epochs"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "R2",
            emit=_simple_emit(
                ["raw/classifier/trainer/lr", "training/lr", "axes/lr"],
                cast="float",
            ),
        )
    )
    A.append(
        _A(
            "R3",
            emit=_simple_emit(
                ["raw/classifier/trainer/weight_decay", "training/weight_decay", "axes/weight_decay"],
                cast="float",
            ),
        )
    )
    A.append(
        _A(
            "R4",
            emit=_simple_emit(
                ["raw/classifier/trainer/batch_size", "training/batch_size", "axes/batch_size"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "R5",
            emit=_simple_emit(
                ["raw/classifier/trainer/seed", "training/seed", "axes/seed"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "R6",
            emit=_simple_emit(
                ["raw/classifier/trainer/warmup_steps", "training/warmup_steps"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "R7",
            emit=_simple_emit(
                ["raw/classifier/trainer/lr_schedule", "training/lr_schedule"],
                cast="lower",
            ),
        )
    )
    A.append(
        _A(
            "R8",
            emit=_simple_emit(
                ["raw/classifier/trainer/label_smoothing", "training/label_smoothing"],
                cast="float",
            ),
        )
    )
    A.append(
        _A(
            "R9",
            emit=_simple_emit(
                ["raw/classifier/trainer/grad_clip", "training/grad_clip"],
                cast="float",
            ),
        )
    )
    A.append(
        _A(
            "R10",
            emit=_simple_emit(
                ["raw/classifier/trainer/early_stopping/enabled", "early_stop/enabled"],
                cast="bool",
            ),
        )
    )
    A.append(
        _A(
            "R11",
            parents=("R10",),
            prereq=_gate_R10,
            emit=_simple_emit(
                ["raw/classifier/trainer/early_stopping/patience", "early_stop/patience"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "R12",
            parents=("R10",),
            prereq=_gate_R10,
            emit=_simple_emit(
                ["raw/classifier/trainer/early_stopping/min_delta"],
                cast="float",
            ),
        )
    )
    A.append(
        _A(
            "R13",
            parents=("R10",),
            prereq=_gate_R10,
            emit=_simple_emit(
                ["raw/classifier/trainer/early_stopping/restore_best_weights"],
                cast="bool",
            ),
        )
    )
    A.append(
        _A(
            "R14",
            parents=("T1", "T1-a"),
            prereq=_gate_pid_learned,
            emit=_simple_emit(
                ["raw/classifier/trainer/pid_schedule/mode", "pid/schedule_mode"],
                cast="lower",
            ),
        )
    )
    A.append(
        _A(
            "R15",
            parents=("T1", "T1-a"),
            prereq=_gate_pid_learned,
            emit=_simple_emit(
                ["raw/classifier/trainer/pid_schedule/transition_epoch", "pid/transition_epoch"],
                cast="int",
            ),
        )
    )
    A.append(
        _A(
            "R16",
            parents=("T1", "T1-a"),
            prereq=_gate_pid_learned,
            emit=_simple_emit(
                ["raw/classifier/trainer/pid_schedule/reinit_mode"],
                cast="lower",
            ),
        )
    )
    A.append(
        _A(
            "R17",
            parents=("T1", "T1-a"),
            prereq=_gate_pid_learned,
            emit=_simple_emit(
                ["raw/classifier/trainer/pid_schedule/pid_lr"],
                cast="float",
            ),
        )
    )

    # §15 §L
    A.append(
        _A(
            "L1",
            emit=_simple_emit(
                ["raw/classifier/trainer/log_pid_embeddings"],
                cast="bool",
            ),
        )
    )
    A.append(
        _A(
            "L2",
            emit=_simple_emit(
                ["raw/classifier/trainer/interpretability/enabled"],
                cast="bool",
            ),
        )
    )
    A.append(
        _A(
            "L3",
            parents=("L2",),
            prereq=_gate_L2,
            emit=_simple_emit(
                ["raw/classifier/trainer/interpretability/save_attention_maps"],
                cast="bool",
            ),
        )
    )
    A.append(
        _A(
            "L4",
            parents=("L2",),
            prereq=_gate_L2,
            emit=_simple_emit(
                ["raw/classifier/trainer/interpretability/save_kan_splines"],
                cast="bool",
            ),
        )
    )
    A.append(
        _A(
            "L5",
            parents=("L2",),
            prereq=_gate_L2,
            emit=_simple_emit(
                ["raw/classifier/trainer/interpretability/save_moe_routing"],
                cast="bool",
            ),
        )
    )
    A.append(
        _A(
            "L6",
            parents=("L2",),
            prereq=_gate_L2,
            emit=_simple_emit(
                ["raw/classifier/trainer/interpretability/save_gradient_norms"],
                cast="bool",
            ),
        )
    )
    A.append(
        _A(
            "L7",
            parents=("L2",),
            prereq=_gate_L2,
            emit=_simple_emit(
                ["raw/classifier/trainer/interpretability/checkpoint_epochs"],
                cast="list",
            ),
        )
    )

    return A


V2_AXES: list[V2Axis] = _build_registry()
V2_TARGET_KEYS: list[str] = [a.target_key for a in V2_AXES]


# ---------------------------------------------------------------------------
# Topological sort (sanity check; V2_AXES is already in topo order)
# ---------------------------------------------------------------------------


def topological_sort(axes: list[V2Axis]) -> list[V2Axis]:
    by_id = {a.id: a for a in axes}
    in_deg: dict[str, int] = {a.id: 0 for a in axes}
    children: dict[str, list[str]] = {a.id: [] for a in axes}
    for a in axes:
        for p in a.parents:
            if p not in by_id:
                continue
            in_deg[a.id] += 1
            children[p].append(a.id)
    ready = [aid for aid, d in in_deg.items() if d == 0]
    order: list[str] = []
    while ready:
        aid = ready.pop(0)
        order.append(aid)
        for c in children[aid]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                ready.append(c)
    if len(order) != len(axes):
        cycle = [aid for aid, d in in_deg.items() if d > 0]
        raise ValueError(f"V2 axis graph has a cycle involving: {cycle}")
    return [by_id[aid] for aid in order]


# ---------------------------------------------------------------------------
# Main derivation
# ---------------------------------------------------------------------------


def derive_v2_axes(
    run_config: dict,
    *,
    flag_bucket: dict | None = None,
) -> dict[str, str]:
    """Pure function: run config -> {target_key: string_value}.

    Returns ``{target_key: str_value}`` for every registered V2 axis. All
    values are strings; empty strings mean "not applicable" or "missing".
    """
    if flag_bucket is None:
        flag_bucket = {}
    flag_bucket.setdefault("keys_left_empty_by_prereq", [])
    flag_bucket.setdefault("keys_empty_missing_config", [])
    flag_bucket.setdefault("unresolved_flags", [])

    derived: dict[str, Any] = {}

    # Pre-compute parsed cont_features for the P1 energy gate.
    derived["_D1_parsed"] = _parse_cont_features(
        _read_chain(
            run_config,
            [
                "raw/data/cont_features",
                "raw/classifier/data/cont_features",
                "data/cont_features",
                "axes/cont_features",
            ],
        )
    )

    for axis in V2_AXES:
        if axis.prereq is not None:
            try:
                gate_ok = bool(axis.prereq(derived))
            except Exception as e:
                flag_bucket["unresolved_flags"].append(f"{axis.id}: prereq error {e}")
                gate_ok = False
            if not gate_ok:
                derived[axis.id] = ""
                flag_bucket["keys_left_empty_by_prereq"].append(axis.id)
                continue

        try:
            value = axis.emit(run_config, derived, flag_bucket) if axis.emit else ""
        except Exception as e:
            flag_bucket["unresolved_flags"].append(f"{axis.id}: emit error {e}")
            value = ""

        if value is None:
            value = ""
        if not isinstance(value, str):
            value = str(value)

        derived[axis.id] = value
        if value == "" and axis.prereq is not None:
            flag_bucket["keys_empty_missing_config"].append(axis.id)

    out: dict[str, str] = {}
    for axis in V2_AXES:
        out[axis.target_key] = derived.get(axis.id, "")
    return out
