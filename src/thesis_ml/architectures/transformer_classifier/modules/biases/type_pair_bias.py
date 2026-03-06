"""Type-pair and SM interaction attention bias modules.

Particle type IDs:
  0=padding  1=jet  2=b-jet  3=e+  4=e-  5=mu+  6=mu-  7=photon

Sources:
  TypePairKinematicBias: Novel, inspired by Builtjes et al. (2025)
  SMInteractionBias:     Builtjes et al. (2025), Section 4.2
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ._features import _FEATURES_NEED_E, compute_pairwise_feature_set

# ---------------------------------------------------------------------------
# SM interaction table constants (Builtjes et al. 2025)
# ---------------------------------------------------------------------------

NUM_TYPES: int = 8

_GS: float = 1.22
_GE: float = 0.31
_GZ: float = 0.758

_ITYPE_NONE: int = 0
_ITYPE_QCD: int = 1
_ITYPE_QED_JET: int = 2
_ITYPE_QED_BJET: int = 3
_ITYPE_QED_LEP: int = 4
_ITYPE_EW: int = 5


def _build_sm_tables(mask_value: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    binary = torch.full((NUM_TYPES, NUM_TYPES), mask_value)
    coupling = torch.full((NUM_TYPES, NUM_TYPES), mask_value)
    itype = torch.zeros(NUM_TYPES, NUM_TYPES, dtype=torch.long)

    def _set(i: int, j: int, val_b: float, val_c: float, it: int) -> None:
        binary[i, j] = val_b
        binary[j, i] = val_b
        coupling[i, j] = val_c
        coupling[j, i] = val_c
        itype[i, j] = it
        itype[j, i] = it

    _set(1, 1, 1.0, _GS, _ITYPE_QCD)
    _set(1, 2, 1.0, _GS, _ITYPE_QCD)
    _set(2, 2, 1.0, _GS, _ITYPE_QCD)
    _set(1, 7, 1.0, _GE * 0.5, _ITYPE_QED_JET)
    _set(2, 7, 1.0, _GE / 3.0, _ITYPE_QED_BJET)
    _set(3, 7, 1.0, _GE, _ITYPE_QED_LEP)
    _set(4, 7, 1.0, _GE, _ITYPE_QED_LEP)
    _set(5, 7, 1.0, _GE, _ITYPE_QED_LEP)
    _set(6, 7, 1.0, _GE, _ITYPE_QED_LEP)
    _set(3, 4, 1.0, _GZ, _ITYPE_EW)
    _set(5, 6, 1.0, _GZ, _ITYPE_EW)

    binary[0, :] = 0.0
    binary[:, 0] = 0.0
    coupling[0, :] = 0.0
    coupling[:, 0] = 0.0

    return binary, coupling, itype


# ---------------------------------------------------------------------------
# TypePairKinematicBias
# ---------------------------------------------------------------------------


class TypePairKinematicBias(nn.Module):
    """Learnable symmetric 8×8 type-pair table with optional kinematic gate.

    Initialised at zero (neutral baseline); learns which particle-type pairs
    matter.  Optionally multiplied by an MLP over a pairwise kinematic feature
    to condition on kinematics (e.g. closer pairs → stronger effect).

    Source: Novel, inspired by Builtjes et al. (2025).
    Per-module gate (init=0): all biases zero at init.
    """

    def __init__(
        self,
        num_heads: int,
        cont_dim: int,
        kinematic_gate: bool = True,
        kinematic_feature: str = "log_m2",
        init_from_physics: str = "none",
        mask_value: float = -5.0,
        num_types: int = NUM_TYPES,
        freeze_table: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_types = num_types
        self.mask_value = mask_value

        table_init = torch.zeros(num_types, num_types)
        if init_from_physics in ("binary", "fixed_coupling"):
            binary_tbl, coupling_tbl, _ = _build_sm_tables(mask_value)
            table_init = binary_tbl if init_from_physics == "binary" else coupling_tbl
        elif init_from_physics != "none":
            raise ValueError(f"init_from_physics must be 'none', 'binary', or 'fixed_coupling'; " f"got {init_from_physics!r}")

        self.table_raw = nn.Parameter(table_init.clone())
        if freeze_table:
            self.table_raw.requires_grad_(False)
        pad_mask = torch.ones(num_types, num_types)
        pad_mask[0, :] = 0.0
        pad_mask[:, 0] = 0.0
        self.register_buffer("pad_mask", pad_mask)

        self.head_proj = nn.Linear(1, num_heads, bias=True)
        nn.init.zeros_(self.head_proj.weight)
        nn.init.zeros_(self.head_proj.bias)

        has_E = cont_dim >= 4
        self.use_kinematic_gate = kinematic_gate and (has_E or kinematic_feature not in _FEATURES_NEED_E)
        self.kinematic_feature = kinematic_feature
        if self.use_kinematic_gate:
            self.kinematic_mlp = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 1))
            nn.init.zeros_(self.kinematic_mlp[-1].weight)
            nn.init.zeros_(self.kinematic_mlp[-1].bias)

        self.gate = nn.Parameter(torch.zeros(1))

    def _symmetric_table(self) -> torch.Tensor:
        return 0.5 * (self.table_raw + self.table_raw.T) * self.pad_mask

    def forward(
        self,
        tokens_id: torch.Tensor,
        tokens_cont: torch.Tensor | None = None,
        F_ij: torch.Tensor | None = None,
        feature_to_idx: dict[str, int] | None = None,
        mask: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor | None:
        if tokens_id is None:
            return None
        B, T = tokens_id.shape
        ids = tokens_id.clamp(0, self.num_types - 1)
        tbl = self._symmetric_table()
        ids_i = ids.unsqueeze(2).expand(B, T, T)
        ids_j = ids.unsqueeze(1).expand(B, T, T)
        base = tbl[ids_i, ids_j]

        feat = None
        if self.use_kinematic_gate:
            if F_ij is not None and feature_to_idx is not None and self.kinematic_feature in feature_to_idx:
                idx = feature_to_idx[self.kinematic_feature]
                feat = F_ij[..., idx : idx + 1]
            elif tokens_cont is not None:
                f, active = compute_pairwise_feature_set(tokens_cont, [self.kinematic_feature], mask=mask)
                if f is not None and active:
                    feat = f
        if feat is not None and feat.numel() > 0:
            gate_val = self.kinematic_mlp(feat)
            base = base * gate_val.squeeze(-1)

        bias = self.head_proj(base.unsqueeze(-1)).permute(0, 3, 1, 2)

        if mask is not None:
            if mask.size(1) != T:
                mask = mask[:, :T]
            valid = mask.unsqueeze(2) & mask.unsqueeze(1)
            bias = bias * valid.unsqueeze(1).to(bias.dtype)

        return torch.tanh(self.gate) * bias


# ---------------------------------------------------------------------------
# SMInteractionBias
# ---------------------------------------------------------------------------


class SMInteractionBias(nn.Module):
    """Standard Model interaction strength as a fixed prior attention bias.

    Three modes (Builtjes et al. 2025, Section 4.2):
      binary          — 0/1 interaction flags
      fixed_coupling  — fixed SM coupling constants (gs, ge, gz)
      running_coupling — one-loop RGE at scale Q²=p̄_T²

    Per-module gate (init=0).
    Source: Builtjes et al. (2025).
    """

    def __init__(
        self,
        num_heads: int,
        cont_dim: int,
        mode: str = "binary",
        mask_value: float = -100.0,
        num_types: int = NUM_TYPES,
    ):
        super().__init__()
        if mode not in ("binary", "fixed_coupling", "running_coupling"):
            raise ValueError(f"mode must be 'binary', 'fixed_coupling', or 'running_coupling'; got {mode!r}")
        self.mode = mode
        self.mask_value = mask_value
        self.num_types = num_types
        self.cont_dim = cont_dim

        binary_tbl, coupling_tbl, itype_tbl = _build_sm_tables(mask_value)
        self.register_buffer("binary_table", binary_tbl)
        self.register_buffer("coupling_table", coupling_tbl)
        self.register_buffer("itype_table", itype_tbl)

        self.head_proj = nn.Linear(1, num_heads, bias=True)
        nn.init.zeros_(self.head_proj.weight)
        nn.init.zeros_(self.head_proj.bias)

        self.gate = nn.Parameter(torch.zeros(1))

        if mode == "running_coupling":
            self._mu0_sq = 91.1876**2
            self._alpha_s0 = 0.118
            self._alpha_em0 = 1.0 / 127.5
            self._nf = 6
            self._n_lep = 3

    def _running_gs(self, Q2: torch.Tensor) -> torch.Tensor:
        beta0 = (33 - 2 * self._nf) / (12 * math.pi)
        log_ratio = torch.log((Q2 / self._mu0_sq).clamp(min=1e-6))
        denom = (1.0 + self._alpha_s0 * beta0 * log_ratio).clamp(min=0.01)
        return torch.sqrt((4 * math.pi * self._alpha_s0 / denom).clamp(min=0.0, max=25.0))

    def _running_ge(self, Q2: torch.Tensor) -> torch.Tensor:
        b_em = (self._nf + self._n_lep) / (3 * math.pi)
        log_ratio = torch.log((Q2 / self._mu0_sq).clamp(min=1e-6))
        denom = (1.0 - self._alpha_em0 * b_em * log_ratio).clamp(min=0.01)
        return torch.sqrt((4 * math.pi * self._alpha_em0 / denom).clamp(min=0.0, max=25.0))

    def forward(
        self,
        tokens_id: torch.Tensor,
        tokens_cont: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor | None:
        if tokens_id is None:
            return None
        B, T = tokens_id.shape
        ids = tokens_id.clamp(0, self.num_types - 1)
        ids_i = ids.unsqueeze(2).expand(B, T, T)
        ids_j = ids.unsqueeze(1).expand(B, T, T)

        if self.mode == "binary":
            vals = self.binary_table[ids_i, ids_j]
        elif self.mode == "fixed_coupling":
            vals = self.coupling_table[ids_i, ids_j]
        else:
            vals = self._running_coupling_vals(tokens_cont, ids_i, ids_j, B, T)

        bias = self.head_proj(vals.unsqueeze(-1)).permute(0, 3, 1, 2)

        if mask is not None:
            if mask.size(1) != T:
                mask = mask[:, :T]
            valid = mask.unsqueeze(2) & mask.unsqueeze(1)
            bias = bias * valid.unsqueeze(1).to(bias.dtype)

        return torch.tanh(self.gate) * bias

    def _running_coupling_vals(
        self,
        tokens_cont: torch.Tensor | None,
        ids_i: torch.Tensor,
        ids_j: torch.Tensor,
        B: int,
        T: int,
    ) -> torch.Tensor:
        itype = self.itype_table[ids_i, ids_j]
        if tokens_cont is not None:
            pt_col = 1 if self.cont_dim >= 4 else 0
            Pt = tokens_cont[..., pt_col].float()
            Q2 = (0.5 * (Pt.unsqueeze(2) + Pt.unsqueeze(1))).pow(2).clamp(min=1e-4)
            gs = self._running_gs(Q2)
            ge = self._running_ge(Q2)
        else:
            gs = torch.full((B, T, T), _GS, device=ids_i.device, dtype=torch.float32)
            ge = torch.full((B, T, T), _GE, device=ids_i.device, dtype=torch.float32)

        vals = torch.full((B, T, T), self.mask_value, device=ids_i.device, dtype=torch.float32)
        vals = torch.where(itype == _ITYPE_QCD, gs, vals)
        vals = torch.where(itype == _ITYPE_QED_JET, ge * 0.5, vals)
        vals = torch.where(itype == _ITYPE_QED_BJET, ge / 3.0, vals)
        vals = torch.where(itype == _ITYPE_QED_LEP, ge, vals)
        vals = torch.where(itype == _ITYPE_EW, torch.full_like(vals, _GZ), vals)
        vals = vals.clamp(min=self.mask_value, max=5.0)

        pad = (ids_i == 0) | (ids_j == 0)
        vals = vals.masked_fill(pad, 0.0)
        return vals
