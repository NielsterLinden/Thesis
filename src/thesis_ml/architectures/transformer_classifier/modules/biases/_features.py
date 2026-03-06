"""Shared pairwise kinematic feature computation for all bias modules.

tokens_cont layout:
  C=4: (E=0, Pt=1, eta=2, phi=3)   "5-vec" (cont_features=[0,1,2,3])
  C=3: (Pt=0, eta=1, phi=2)         "4-vec" (cont_features=[1,2,3], no E)

All bias modules import from here to avoid duplicating feature computation.
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

#: Features that require E (column 0 of 5-vec tokens_cont)
_FEATURES_NEED_E: frozenset[str] = frozenset({"m2", "dot", "log_m2"})

#: All valid feature names for pairwise bias modules
VALID_FEATURES: frozenset[str] = frozenset({"m2", "dot", "log_m2", "log_kt", "z", "deltaR", "deltaR_ptw"})


# ---------------------------------------------------------------------------
# Kinematics extraction
# ---------------------------------------------------------------------------


def _extract_kinematics(
    tokens_cont: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (E, Pt, eta, phi) from tokens_cont, E=None when C < 4."""
    C = tokens_cont.size(-1)
    if C >= 4:
        return tokens_cont[..., 0], tokens_cont[..., 1], tokens_cont[..., 2], tokens_cont[..., 3]
    return None, tokens_cont[..., 0], tokens_cont[..., 1], tokens_cont[..., 2]


# ---------------------------------------------------------------------------
# Core pairwise feature computation
# ---------------------------------------------------------------------------


def compute_pairwise_feature_set(
    tokens_cont: torch.Tensor,
    features: list[str],
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, list[str]]:
    """Compute a stacked pairwise feature tensor.

    Parameters
    ----------
    tokens_cont : torch.Tensor
        [B, T, C] — C=4 (E,Pt,eta,phi) or C=3 (Pt,eta,phi).
    features : list[str]
        Feature names from VALID_FEATURES.  E-dependent features are silently
        dropped when C < 4.
    mask : torch.Tensor, optional
        [B, T] True=valid.  Padding pairs are zeroed in the output.

    Returns
    -------
    tuple[torch.Tensor | None, list[str]]
        (feat [B, T, T, F_actual], active_names).
        (None, []) if no features can be computed.
    """
    if tokens_cont.dim() != 3 or tokens_cont.size(-1) < 3:
        return None, []

    E, Pt, eta, phi = _extract_kinematics(tokens_cont)
    has_E = E is not None

    active = [f for f in features if f in VALID_FEATURES and (has_E or f not in _FEATURES_NEED_E)]
    if not active:
        return None, []

    T = tokens_cont.size(1)

    # Angular quantities (always available for C >= 3)
    deta = eta.unsqueeze(2) - eta.unsqueeze(1)  # [B, T, T]
    dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
    dphi = (dphi + math.pi) % (2 * math.pi) - math.pi  # wrap to [-π, π]
    deltaR = torch.sqrt(deta**2 + dphi**2 + 1e-8)  # [B, T, T]

    Pt_i = Pt.unsqueeze(2)  # [B, T, 1]
    Pt_j = Pt.unsqueeze(1)  # [B, 1, T]
    Pt_min = torch.minimum(Pt_i, Pt_j)  # [B, T, T]
    Pt_sum = Pt_i + Pt_j  # [B, T, T]

    if has_E:
        px = Pt * torch.cos(phi)
        py = Pt * torch.sin(phi)
        pz = Pt * torch.sinh(eta.clamp(-20.0, 20.0))
        E_i = E.unsqueeze(2)
        E_j = E.unsqueeze(1)
        px_i = px.unsqueeze(2)
        px_j = px.unsqueeze(1)
        py_i = py.unsqueeze(2)
        py_j = py.unsqueeze(1)
        pz_i = pz.unsqueeze(2)
        pz_j = pz.unsqueeze(1)
        m2 = ((E_i + E_j) ** 2 - (px_i + px_j) ** 2 - (py_i + py_j) ** 2 - (pz_i + pz_j) ** 2).clamp(min=0.0)
        dot = E_i * E_j - px_i * px_j - py_i * py_j - pz_i * pz_j

    feat_list: list[torch.Tensor] = []
    for name in active:
        if name == "m2":
            feat_list.append(m2)
        elif name == "dot":
            feat_list.append(dot)
        elif name == "log_m2":
            feat_list.append(torch.log(m2.clamp(min=1e-6)))
        elif name == "log_kt":
            feat_list.append(torch.log((Pt_min * deltaR).clamp(min=1e-6)))
        elif name == "z":
            feat_list.append(Pt_min / Pt_sum.clamp(min=1e-6))
        elif name == "deltaR":
            feat_list.append(deltaR)
        elif name == "deltaR_ptw":
            feat_list.append(deltaR * Pt_sum)

    out = torch.stack(feat_list, dim=-1)  # [B, T, T, F_actual]

    if mask is not None:
        if mask.size(1) != T:
            mask = mask[:, :T]
        valid = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1)
        out = out * valid.to(out.dtype)

    return out, active
