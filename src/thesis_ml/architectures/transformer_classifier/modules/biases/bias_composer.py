"""BiasComposer: orchestrates physics-informed attention bias modules.

Config selector pattern
-----------------------
Instead of per-module ``enabled: true/false``, a single string
``classifier.model.attention_biases`` lists the active modules,
separated by ``+``:

    attention_biases: "none"                     # baseline (no physics bias)
    attention_biases: "lorentz_scalar"            # single module
    attention_biases: "lorentz_scalar+typepair"   # combination

Valid tokens:
    none, lorentz_scalar, typepair_kinematic, sm_interaction,
    global_conditioned

MET/global conditioning is automatically included when ``global_conditioned``
is in the list AND ``globals_`` is available.

``nodewise_mass`` and ``mia_blocks`` are NOT part of this string — they
modify ``x`` rather than the attention logits and have their own config keys.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

# ---------------------------------------------------------------------------
# Special-token padding helper
# ---------------------------------------------------------------------------


def _pad_bias_for_special_tokens(
    bias: torch.Tensor,
    use_cls: bool,
    num_met_tokens: int,
) -> torch.Tensor:
    """Pad a physics bias tensor to the full encoder sequence length.

    Accepts ``[B, T, T]`` (shared) or ``[B, H, T, T]`` (per-head).
    CLS and MET rows/columns are filled with zeros.

    Parameters
    ----------
    bias : torch.Tensor
        ``[B, T, T]`` or ``[B, H, T, T]`` — physical-token block only.
    use_cls : bool
        Whether a CLS token is prepended.
    num_met_tokens : int
        Number of MET tokens appended (0 or 2).

    Returns
    -------
    torch.Tensor
        ``[B, T_full, T_full]`` or ``[B, H, T_full, T_full]``.
    """
    cls_offset = int(use_cls)
    if bias.dim() == 3:
        B, T, _ = bias.shape
        T_full = T + cls_offset + num_met_tokens
        full = bias.new_zeros(B, T_full, T_full)
        full[:, cls_offset : cls_offset + T, cls_offset : cls_offset + T] = bias
    else:
        B, H, T, _ = bias.shape
        T_full = T + cls_offset + num_met_tokens
        full = bias.new_zeros(B, H, T_full, T_full)
        full[:, :, cls_offset : cls_offset + T, cls_offset : cls_offset + T] = bias
    return full


# ---------------------------------------------------------------------------
# GlobalConditionedBias
# ---------------------------------------------------------------------------


class GlobalConditionedBias(nn.Module):
    """MET-conditioned additive attention bias.

    Two modes:
      ``global_scale``   — MLP(globals_) → [B, H] broadcast to [B, H, 1, 1]
      ``met_direction``  — per-token Δφ relative to METphi → [B, H, T, T]

    Source: Novel, inspired by Option B in planning session.
    Per-module gate (init=0).
    """

    def __init__(
        self,
        num_heads: int,
        cont_dim: int,
        global_dim: int = 16,
        mode: str = "global_scale",
        mlp_type: str = "standard",
        kan_cfg: dict | None = None,
    ):
        super().__init__()
        if mode not in ("global_scale", "met_direction"):
            raise ValueError(f"mode must be 'global_scale' or 'met_direction'; got {mode!r}")
        self.mode = mode
        self.num_heads = num_heads
        self.cont_dim = cont_dim

        from thesis_ml.architectures.transformer_classifier.modules.kan.utils import (
            build_bias_mlp,
        )

        in_dim = 2 if mode == "global_scale" else 5
        self.mlp = build_bias_mlp(in_dim, global_dim, num_heads, mlp_type=mlp_type, kan_cfg=kan_cfg)

        if mlp_type == "standard":
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        globals_: torch.Tensor,
        tokens_cont: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor | None:
        if globals_ is None:
            return None

        if self.mode == "global_scale":
            head_scales = self.mlp(globals_)  # [B, H]
            bias = head_scales.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

        else:  # met_direction
            if tokens_cont is None or self.cont_dim < 3:
                return None
            phi_col = 3 if self.cont_dim >= 4 else 2
            phi = tokens_cont[..., phi_col]  # [B, T]
            met_phi = globals_[:, 1:2]
            met_mag = globals_[:, 0:1]
            dphi = phi - met_phi
            B, T = phi.shape

            sin_i = torch.sin(dphi).unsqueeze(2).expand(B, T, T)
            cos_i = torch.cos(dphi).unsqueeze(2).expand(B, T, T)
            sin_j = torch.sin(dphi).unsqueeze(1).expand(B, T, T)
            cos_j = torch.cos(dphi).unsqueeze(1).expand(B, T, T)
            met_exp = met_mag.unsqueeze(-1).expand(B, T, T)

            pair_feat = torch.stack([sin_i, cos_i, sin_j, cos_j, met_exp], dim=-1)
            bias = self.mlp(pair_feat).permute(0, 3, 1, 2)  # [B, H, T, T]

            if mask is not None:
                if mask.size(1) != T:
                    mask = mask[:, :T]
                valid = mask.unsqueeze(2) & mask.unsqueeze(1)
                bias = bias * valid.unsqueeze(1).to(bias.dtype)

        return torch.tanh(self.gate) * bias


# ---------------------------------------------------------------------------
# BiasComposer
# ---------------------------------------------------------------------------


class BiasComposer(nn.Module):
    """Sums all enabled physics bias modules and pads for CLS/MET tokens.

    All sub-modules receive the *physical* T tokens only.  The output is
    padded to the full encoder sequence length.
    """

    def __init__(
        self,
        bias_modules: dict[str, nn.Module],
        use_cls: bool,
        num_met_tokens: int,
        global_conditioner: GlobalConditionedBias | None = None,
    ):
        super().__init__()
        self.bias_modules = nn.ModuleDict(bias_modules)
        self.use_cls = use_cls
        self.num_met_tokens = num_met_tokens
        if global_conditioner is not None:
            self.add_module("global_conditioner", global_conditioner)
        self.global_conditioner = global_conditioner

    def forward(
        self,
        tokens_cont: torch.Tensor,
        tokens_id: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        globals_: torch.Tensor | None = None,
        F_ij: torch.Tensor | None = None,
        feature_to_idx: dict[str, int] | None = None,
    ) -> torch.Tensor | None:
        total: torch.Tensor | None = None

        for module in self.bias_modules.values():
            bias = module(
                tokens_cont=tokens_cont,
                tokens_id=tokens_id,
                mask=mask,
                globals_=globals_,
                F_ij=F_ij,
                feature_to_idx=feature_to_idx,
            )
            if bias is None:
                continue
            if bias.ndim == 3:
                bias = bias.unsqueeze(1)
            total = bias if total is None else total + bias

        if self.global_conditioner is not None and globals_ is not None:
            g_bias = self.global_conditioner(globals_=globals_, tokens_cont=tokens_cont, mask=mask)
            if g_bias is not None:
                if g_bias.ndim == 3:
                    g_bias = g_bias.unsqueeze(1)
                total = g_bias if total is None else total + g_bias

        if total is None:
            return None

        T = tokens_cont.size(1)
        if total.size(-1) != T or total.size(-2) != T:
            total = total.expand(tokens_cont.size(0), -1, T, T).contiguous()

        # Normalise single-head [B, 1, T, T] → [B, T, T] for broadcast
        if total.dim() == 4 and total.size(1) == 1:
            total = total.squeeze(1)

        return _pad_bias_for_special_tokens(total, self.use_cls, self.num_met_tokens)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Mapping from selector token to the module name it enables
_SELECTOR_MAP: dict[str, str] = {
    "lorentz_scalar": "lorentz_scalar",
    "typepair_kinematic": "typepair_kinematic",
    "typepair": "typepair_kinematic",  # short alias
    "sm_interaction": "sm_interaction",
    "sm": "sm_interaction",  # short alias
    "global_conditioned": "global_conditioned",
    "global": "global_conditioned",  # short alias
}


def parse_attention_biases(selector: str) -> list[str]:
    """Parse ``attention_biases`` string to a list of canonical module names.

    Examples::

        parse_attention_biases("none")                       → []
        parse_attention_biases("lorentz_scalar")             → ["lorentz_scalar"]
        parse_attention_biases("lorentz_scalar+typepair")    → ["lorentz_scalar", "typepair_kinematic"]
    """
    if not selector or selector.strip().lower() in ("none", ""):
        return []
    tokens = [t.strip().lower() for t in selector.split("+")]
    result: list[str] = []
    for tok in tokens:
        if tok in ("none", ""):
            continue
        canonical = _SELECTOR_MAP.get(tok, tok)
        if canonical not in result:
            result.append(canonical)
    return result


def build_bias_composer(
    cfg: DictConfig,
    num_heads: int,
    model_dim: int,
    cont_dim: int,
    use_cls: bool,
    num_met_tokens: int,
    kan_cfg: dict | None = None,
) -> BiasComposer | None:
    """Build a BiasComposer from Hydra config.

    Reads ``classifier.model.attention_biases`` (string selector) and
    ``classifier.model.bias_config.*`` (per-module hyperparameters).

    Backward compatibility: if ``attn_pairwise.enabled=true`` is set and
    ``attention_biases`` is not configured, silently maps to
    ``lorentz_scalar`` with the old config values.

    Parameters
    ----------
    cfg : DictConfig
        The ``classifier.model`` sub-config.
    num_heads : int
        Number of attention heads.
    model_dim : int
        Transformer model dimension (unused currently; reserved for future).
    cont_dim : int
        Number of continuous features per token (3 or 4).
    use_cls : bool
        Whether a CLS token is prepended.
    num_met_tokens : int
        Number of MET tokens appended (0 or 2).
    kan_cfg : dict | None
        Global KAN hyperparameters.  Passed to bias modules when their
        ``mlp_type`` is ``"kan"``.

    Returns
    -------
    BiasComposer | None
    """
    from .lorentz_scalar import LorentzScalarBias
    from .type_pair_bias import SMInteractionBias, TypePairKinematicBias

    # --- resolve selector string ---
    selector_raw = str(cfg.get("attention_biases", "none"))

    # Backward-compat: old attn_pairwise.enabled  →  lorentz_scalar
    attn_pairwise_cfg = cfg.get("attn_pairwise", {})
    if selector_raw.strip().lower() in ("none", "") and attn_pairwise_cfg.get("enabled", False):
        selector_raw = "lorentz_scalar"

    # Backward-compat: old attention_bias dict with enabled keys
    if selector_raw.strip().lower() in ("none", ""):
        old_bias = cfg.get("attention_bias", {})
        if isinstance(old_bias, dict):
            enabled_from_dict = [k for k, v in old_bias.items() if isinstance(v, dict) and v.get("enabled", False)]
            if enabled_from_dict:
                selector_raw = "+".join(enabled_from_dict)

    enabled = parse_attention_biases(selector_raw)
    bias_cfg = dict(cfg.get("bias_config", {}))
    # Backward-compat: merge old attention_bias dict into bias_config
    old_bias = cfg.get("attention_bias", {})
    if isinstance(old_bias, dict):
        for k, v in old_bias.items():
            if isinstance(v, dict) and k not in bias_cfg:
                bias_cfg[k] = v

    modules: dict[str, nn.Module] = {}

    # LorentzScalarBias
    if "lorentz_scalar" in enabled:
        c = bias_cfg.get("lorentz_scalar", {})
        # also check attn_pairwise for backward compat values
        ap = attn_pairwise_cfg
        features = list(c.get("features", ap.get("features", ["m2", "deltaR"])))
        hidden_dim = int(c.get("hidden_dim", ap.get("hidden_dim", 8)))
        per_head = bool(c.get("per_head", ap.get("per_head", False)))
        sparse_gating = bool(c.get("sparse_gating", False))
        ls_mlp_type = str(c.get("mlp_type", "standard"))
        modules["lorentz_scalar"] = LorentzScalarBias(
            features=features,
            cont_dim=cont_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            per_head=per_head,
            sparse_gating=sparse_gating,
            mlp_type=ls_mlp_type,
            kan_cfg=kan_cfg,
        )

    # TypePairKinematicBias
    if "typepair_kinematic" in enabled:
        c = bias_cfg.get("typepair_kinematic", {})
        modules["typepair_kinematic"] = TypePairKinematicBias(
            num_heads=num_heads,
            cont_dim=cont_dim,
            kinematic_gate=bool(c.get("kinematic_gate", True)),
            kinematic_feature=str(c.get("kinematic_feature", "log_m2")),
            init_from_physics=str(c.get("init_from_physics", "none")),
            mask_value=float(c.get("mask_value", -5.0)),
            freeze_table=bool(c.get("freeze_table", False)),
        )

    # SMInteractionBias
    if "sm_interaction" in enabled:
        c = bias_cfg.get("sm_interaction", {})
        modules["sm_interaction"] = SMInteractionBias(
            num_heads=num_heads,
            cont_dim=cont_dim,
            mode=str(c.get("mode", "binary")),
            mask_value=float(c.get("mask_value", -100.0)),
        )

    # GlobalConditionedBias
    global_conditioner: GlobalConditionedBias | None = None
    if "global_conditioned" in enabled:
        c = bias_cfg.get("global_conditioned", {})
        gc_mlp_type = str(c.get("mlp_type", "standard"))
        global_conditioner = GlobalConditionedBias(
            num_heads=num_heads,
            cont_dim=cont_dim,
            global_dim=int(c.get("global_dim", 16)),
            mode=str(c.get("mode", "global_scale")),
            mlp_type=gc_mlp_type,
            kan_cfg=kan_cfg,
        )

    if not modules and global_conditioner is None:
        return None

    return BiasComposer(
        bias_modules=modules,
        use_cls=use_cls,
        num_met_tokens=num_met_tokens,
        global_conditioner=global_conditioner,
    )
