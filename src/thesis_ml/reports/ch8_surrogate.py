"""ch8_surrogate.py — Shared library for Ch8 Phase C/D: XGBoost surrogate + SHAP.

Provides reusable building blocks:
  - build_feature_matrix   : one-hot encode axis columns
  - make_groups            : GroupKFold fingerprint excluding R5_Seed
  - fit_surrogate          : cross-validated XGBRegressor + full refit
  - compute_shap           : TreeExplainer SHAP values
  - aggregate_shap_to_families : roll up one-hot SHAP to axis family letter
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    axis_cols: list[str],
    extra_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode axis columns and optionally include numeric extras.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (rows = runs).
    axis_cols : list[str]
        Categorical axis columns to one-hot encode.  May contain the string
        ``"inactive"`` as a level — that is treated as a regular category.
    extra_cols : list[str] | None
        Numeric columns to include as-is after ``fillna(median)``.

    Returns
    -------
    (X, feature_names)
        X              : pd.DataFrame with encoded features
        feature_names  : list of column names in X
    """
    parts: list[pd.DataFrame] = []

    if axis_cols:
        cat_df = df[axis_cols].astype(str)
        dummies = pd.get_dummies(cat_df, prefix_sep="__LEVEL__", drop_first=False)
        # XGBoost does not allow [ ] < > in feature names — sanitize
        dummies.columns = [
            c.replace("[", "(").replace("]", ")").replace("<", "lt_").replace(">", "gt_")
            for c in dummies.columns
        ]
        parts.append(dummies)

    if extra_cols:
        num_df = df[extra_cols].copy()
        for col in extra_cols:
            med = num_df[col].median()
            num_df[col] = num_df[col].fillna(med)
        parts.append(num_df)

    if not parts:
        raise ValueError("No columns to encode — provide axis_cols or extra_cols.")

    X = pd.concat(parts, axis=1).reset_index(drop=True)
    feature_names = list(X.columns)
    return X, feature_names


# ---------------------------------------------------------------------------
# Group fingerprint for GroupKFold
# ---------------------------------------------------------------------------


def make_groups(df: pd.DataFrame, axis_cols: list[str]) -> np.ndarray:
    """Build a group fingerprint for GroupKFold, excluding R5_Seed.

    All runs that differ only in seed belong to the same group, preventing
    seed-duplicated information leaking from train to validation fold.

    Parameters
    ----------
    df : pd.DataFrame
    axis_cols : list[str]

    Returns
    -------
    np.ndarray of int64
        Group label per row (hash of the non-seed axis values).
    """
    seed_col = next(
        (c for c in axis_cols if "R5_Seed" in c),
        None,
    )
    fp_cols = [c for c in axis_cols if c != seed_col] if seed_col else axis_cols
    groups = (
        df[fp_cols]
        .astype(str)
        .apply(lambda r: hash(tuple(r)), axis=1)
        .values.astype(np.int64)
    )
    return groups


# ---------------------------------------------------------------------------
# Surrogate fitting
# ---------------------------------------------------------------------------


def fit_surrogate(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[Any, np.ndarray, dict]:
    """Fit XGBRegressor with GroupKFold CV, then refit on all data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (one-hot encoded).
    y : np.ndarray
        Target values (eval_v2/test_auroc).
    groups : np.ndarray
        Group labels from ``make_groups``.
    n_splits : int
        Number of GroupKFold folds.
    random_state : int

    Returns
    -------
    (final_model, oof_preds, cv_metrics)
        final_model : fitted XGBRegressor on all data
        oof_preds   : out-of-fold predictions (same length as y)
        cv_metrics  : dict with keys r2_mean, r2_std, spearman_mean,
                      spearman_std, per_fold
    """
    import xgboost as xgb
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from scipy.stats import spearmanr

    model_params = dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )

    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.full(len(y), np.nan)
    per_fold: list[dict] = []

    X_arr = X.values  # numpy array for XGB

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X_arr, y, groups)):
        m = xgb.XGBRegressor(**model_params)
        m.fit(X_arr[train_idx], y[train_idx])
        preds = m.predict(X_arr[val_idx])
        oof_preds[val_idx] = preds
        r2 = r2_score(y[val_idx], preds)
        sp = spearmanr(y[val_idx], preds).statistic
        per_fold.append({"fold": fold_i, "r2": float(r2), "spearman": float(sp)})

    r2_vals = [f["r2"] for f in per_fold]
    sp_vals = [f["spearman"] for f in per_fold]

    cv_metrics = {
        "r2_mean": float(np.mean(r2_vals)),
        "r2_std": float(np.std(r2_vals)),
        "spearman_mean": float(np.mean(sp_vals)),
        "spearman_std": float(np.std(sp_vals)),
        "per_fold": per_fold,
    }

    # Refit on all data
    final_model = xgb.XGBRegressor(**model_params)
    final_model.fit(X_arr, y)

    return final_model, oof_preds, cv_metrics


# ---------------------------------------------------------------------------
# SHAP values
# ---------------------------------------------------------------------------


def compute_shap(
    model: Any,
    X: pd.DataFrame,
    max_rows: int = 800,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model : XGBRegressor
        Fitted surrogate model.
    X : pd.DataFrame
        Full feature matrix.
    max_rows : int
        Maximum rows to pass to SHAP (subsampled if needed for speed).

    Returns
    -------
    (shap_values, X_sample)
    """
    import shap

    if len(X) > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        idx.sort()
        X_sample = X.iloc[idx].reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    return sv, X_sample


# ---------------------------------------------------------------------------
# Aggregate SHAP to families
# ---------------------------------------------------------------------------


def _axis_family_from_feature(feature_name: str) -> str:
    """Extract axis family letter from a one-hot feature name.

    One-hot column naming convention from build_feature_matrix:
        ``config/axes/B1-L3_Lorentz Hidden Dimension__LEVEL__32``
    Strip the level suffix first, then strip ``config/axes/``, then take the
    uppercase letter prefix (everything before the first digit or hyphen).
    """
    # Strip __LEVEL__<value> suffix
    base = re.sub(r"__LEVEL__.*$", "", feature_name)
    # Strip config/axes/ prefix
    short = base.replace("config/axes/", "")
    # Extract leading uppercase letters (family)
    m = re.match(r"^([A-Z]+)", short)
    return m.group(1) if m else "?"


def _axis_col_from_feature(feature_name: str) -> str:
    """Recover original axis column name by stripping __LEVEL__<value>."""
    return re.sub(r"__LEVEL__.*$", "", feature_name)


def aggregate_shap_to_families(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Aggregate mean |SHAP| per one-hot feature to axis family letters.

    Parameters
    ----------
    shap_values : np.ndarray, shape (n_samples, n_features)
    feature_names : list[str]
        Column names of the SHAP sample dataframe (one-hot features).

    Returns
    -------
    dict mapping family letter → normalised mean |SHAP| (sums to 1.0).
    """
    mean_abs = np.abs(shap_values).mean(axis=0)  # shape (n_features,)

    family_sums: dict[str, float] = {}
    for feat, val in zip(feature_names, mean_abs):
        fam = _axis_family_from_feature(feat)
        family_sums[fam] = family_sums.get(fam, 0.0) + float(val)

    total = sum(family_sums.values())
    if total > 0:
        return {k: v / total for k, v in family_sums.items()}
    return family_sums
