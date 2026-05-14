"""ch8_constraints.py — Constraint predicate and hull sampler for Ch8 Phase D.

Implemented constraints (conservative subset from AXES_REFERENCE_V2.md §0.3):

  RULE 1  (Raw-token path): B1 biases, P1, P2 require T1 ∈ {raw, identity}.
          If T1 = identity or T1 = raw, fine. Any active bias/P1/P2 with
          T1 = binned (or anything else) is illegal.

  RULE 2  (MET requirement): B1-G1 = met_direction requires D2_MET Treatment = True.

  RULE 3  (FFN mutual exclusivity via effective realization):
          If F1-eff_FFN Realization = moe, then F1_FFN Type must NOT be kan.
          (moe takes priority; kan + moe.enabled=true is technically allowed but
          records as moe, so the combination is fine — we just reject kan
          without moe when moe_enabled is True and vice versa.)
          More precisely: if F1-moe_MoE Enabled = True, F1_FFN Type should be
          standard (the effective realization is moe).  If both ffn_type=kan AND
          moe_enabled=True, that's ok (code resolves to moe), but if moe is
          inactive and F1-eff_FFN Realization says moe, that's a contradiction.

  RULE 4  (MoE scope / head overlap): If F1-b_MoE Encoder Scope = head then
          F1-eff_FFN Realization should be moe (you can't set scope=head without
          moe enabled). Conversely scope=head with moe_enabled=False is illegal.

  RULE 5  (PID dim override): If T1-a_PID Embedding Mode = one_hot, then
          T1-b_PID Embedding Dimension must be num_types (not a numeric dim).

  RULE 6  (T1-a / T1-b only meaningful for identity tokenizer):
          If T1_Tokenizer Family = raw, T1-a and T1-b should be inactive.

  RULE 7  (Energy requirement for P1): If P1_Nodewise Mass Enabled = True,
          D1_Feature Set must include index 0 (energy). We check that the string
          representation of D1 contains '0'.

APPROXIMATIONS / NOT IMPLEMENTED:
  - KAN active → KAN hyperparams (K1-K5) present: not enforced — the observed
    hull always includes valid KAN params when KAN is active, so this will
    rarely trigger. Noted as gap.
  - MoE shared coupling (M1-M5): not enforced — same reason.
  - §S shared backbone: not enforced as explicit constraint; it is implied by
    the observed hull.
  - A3-a (diff_bias_mode) active only when A3 = differential: not enforced
    (sparse in data, unlikely to create bad candidates).
  - Combinations of B1 activation set strings (e.g. multi-family strings):
    not individually validated beyond the raw-token check.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column name helper
# ---------------------------------------------------------------------------

_AXES_PREFIX = "config/axes/"


def _v(cfg: dict, short_key: str) -> Any:
    """Get axis value by short key (without 'config/axes/' prefix)."""
    full_key = _AXES_PREFIX + short_key
    return cfg.get(full_key, cfg.get(short_key, None))


def _str(val: Any) -> str:
    """Normalise to lowercase string; treat nan/None as 'inactive'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "inactive"
    return str(val).strip().lower()


def _is_inactive(val: Any) -> bool:
    return _str(val) in ("inactive", "nan", "none", "")


def _bias_active(cfg: dict) -> bool:
    """Return True if any attention bias family is active."""
    b1 = _str(_v(cfg, "B1_Bias Activation Set"))
    return b1 != "none" and not _is_inactive(b1)


def _lorentz_active(cfg: dict) -> bool:
    b1 = _str(_v(cfg, "B1_Bias Activation Set"))
    return "lorentz_scalar" in b1


def _global_active(cfg: dict) -> bool:
    b1 = _str(_v(cfg, "B1_Bias Activation Set"))
    return "global_conditioned" in b1


# ---------------------------------------------------------------------------
# Constraint predicate
# ---------------------------------------------------------------------------


def is_legal_config(cfg: dict) -> bool:
    """Return True iff the config dict passes all implemented constraints.

    Parameters
    ----------
    cfg : dict
        Flat config dict.  Keys may include the 'config/axes/' prefix or not —
        the helper ``_v`` handles both.

    Returns
    -------
    bool
    """

    # RULE 1: B1 biases, P1, P2 require T1 ∈ {raw, identity}
    t1 = _str(_v(cfg, "T1_Tokenizer Family"))
    raw_token_ok = t1 in ("raw", "identity")

    if _bias_active(cfg) and not raw_token_ok:
        return False  # biases require raw-token tokenizer

    p1_enabled = _str(_v(cfg, "P1_Nodewise Mass Enabled"))
    if p1_enabled == "true" and not raw_token_ok:
        return False  # nodewise mass requires raw/identity

    p2_enabled = _str(_v(cfg, "P2_MIA Pre-Encoder Enabled"))
    if p2_enabled == "true" and not raw_token_ok:
        return False  # MIA requires raw/identity

    # RULE 2: B1-G1 = met_direction requires D2 = True
    if _global_active(cfg):
        b1g1 = _str(_v(cfg, "B1-G1_Global-Conditioned Mode"))
        if b1g1 == "met_direction":
            d2 = _str(_v(cfg, "D2_MET Treatment"))
            if d2 not in ("true", "1"):
                return False

    # RULE 3 + 4: MoE/KAN/standard effective realization consistency
    moe_enabled = _str(_v(cfg, "F1-moe_MoE Enabled"))
    ffn_type = _str(_v(cfg, "F1_FFN Type"))
    ffn_eff = _str(_v(cfg, "F1-eff_FFN Realization"))
    moe_scope = _str(_v(cfg, "F1-b_MoE Encoder Scope"))

    # If effective realization is moe, moe_enabled should be True
    if ffn_eff == "moe" and moe_enabled not in ("true", "1"):
        return False
    # If moe scope is head, moe must be enabled
    if moe_scope == "head" and moe_enabled not in ("true", "1"):
        return False
    # If moe scope = head, effective realization should be moe
    if moe_scope == "head" and ffn_eff != "moe" and not _is_inactive(moe_scope):
        return False

    # RULE 5: T1-a = one_hot → T1-b = num_types
    t1a = _str(_v(cfg, "T1-a_PID Embedding Mode"))
    t1b = _str(_v(cfg, "T1-b_PID Embedding Dimension"))
    if t1a == "one_hot":
        if t1b not in ("num_types", "inactive"):
            return False

    # RULE 6: T1 = raw → T1-a and T1-b should be inactive
    if t1 == "raw":
        if not _is_inactive(t1a) and t1a != "inactive":
            return False
        if not _is_inactive(t1b) and t1b not in ("inactive",):
            return False

    # RULE 7: P1 requires energy feature (D1 must include 0)
    if p1_enabled == "true":
        d1 = _str(_v(cfg, "D1_Feature Set"))
        # The feature set is stored as a string like '[0, 1, 2, 3]' or '[0,1,2,3]'
        # Check whether '0' appears as an index (not just substring of e.g. '10')
        nums = re.findall(r"\d+", d1)
        if "0" not in nums:
            return False

    return True


# ---------------------------------------------------------------------------
# Observed-hull sampler
# ---------------------------------------------------------------------------


def sample_observed_hull(
    df: pd.DataFrame,
    axis_cols: list[str],
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Randomly sample combinations of values observed in df.

    For each axis column, the sampled value is drawn independently from the
    empirical marginal distribution (with replacement, weighted by frequency).
    This is a product-of-marginals approximation — it samples from a
    slightly wider space than the observed joint distribution, but stays
    within the observed marginals for each axis.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    axis_cols : list[str]
        Columns to sample from.
    n_samples : int
        Number of candidate dicts to return.
    rng : np.random.Generator
        Seeded RNG.

    Returns
    -------
    list of dict
        Each dict maps axis_col → sampled value.
    """
    # Pre-compute marginal distributions
    marginals: dict[str, tuple[list, np.ndarray]] = {}
    for col in axis_cols:
        counts = df[col].astype(str).value_counts(dropna=False)
        vals = counts.index.tolist()
        probs = counts.values / counts.values.sum()
        marginals[col] = (vals, probs)

    results: list[dict] = []
    for _ in range(n_samples):
        candidate: dict = {}
        for col in axis_cols:
            vals, probs = marginals[col]
            chosen = rng.choice(len(vals), p=probs)
            candidate[col] = vals[chosen]
        results.append(candidate)

    return results
