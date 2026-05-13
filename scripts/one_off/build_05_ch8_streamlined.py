"""Build streamlined ``05_ch8_streamlined_*.csv`` files from the frozen 04 analysis CSV.

Phase A of the Ch8 plan: row-filter (needs_review/state/auroc), dedupe on
``(checkpoint_sha256, G3, R5_Seed)``, drop zero-variance / all-NaN
``config/axes/*`` columns, encode branch-conditional NaN as the string
``"inactive"``, and split into three task-specific CSVs by
``config/axes/G3_Classification Task``.

Encoding decision: a ``config/axes/*`` column with both null and non-null
entries reflects a branch-conditional axis (e.g. KAN hyperparams only exist
when KAN is enabled). For all such columns we fill NaN with the literal
string ``"inactive"`` and cast to ``str`` for downstream one-hot encoding.
Axis columns that are entirely non-null are left untouched in dtype but cast
to ``str`` for uniformity. Residual numeric NaN counts (should be zero after
the rule above) are reported in the summary JSON as a sanity check.

Inputs (read-only):
    thesis_results/04_cleaned_backfilled_analysis_ready.csv

Outputs:
    thesis_results/05_ch8_streamlined_primary.csv
    thesis_results/05_ch8_streamlined_g3_4t_tth.csv
    thesis_results/05_ch8_streamlined_g3_5class.csv
    thesis_results/05_ch8_streamlined_build_summary.json
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from thesis_ml.reports.ch8_analysis_ready import DEDUPE_COLUMNS, dedupe_key

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path("/project/atlas/users/nterlind/Thesis-Code")
INPUT_CSV = REPO_ROOT / "thesis_results" / "04_cleaned_backfilled_analysis_ready.csv"
OUT_DIR = REPO_ROOT / "thesis_results"

PRIMARY_G3 = "ttH+ttW+ttWW+ttZ | 4t"
APPENDIX_G3_4T_TTH = "4t | ttH"
APPENDIX_G3_5CLASS = "4t | ttH | ttW | ttWW | ttZ"

G3_COL = "config/axes/G3_Classification Task"
AUROC_COL = "eval_v2/test_auroc"
NEEDS_REVIEW_COL = "config/meta.needs_review"
STATE_COL = "meta_run/state"


def load_input(path: Path) -> pd.DataFrame:
    """Read 04 CSV with conservative dtypes."""
    logger.info("Reading %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded shape: %s", df.shape)
    return df


def apply_row_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Drop ``needs_review`` rows and non-finished states; assert AUROC present."""
    n0 = len(df)
    state_dist = df[STATE_COL].value_counts(dropna=False).to_dict()
    # Stringify keys so the summary JSON is valid (NaN -> "nan").
    state_dist_json = {str(k): int(v) for k, v in state_dist.items()}

    mask_review = df[NEEDS_REVIEW_COL] == True  # noqa: E712  explicit bool match
    n_review = int(mask_review.sum())
    df = df.loc[~mask_review].copy()
    logger.info("Dropped %d rows with needs_review=True", n_review)

    mask_not_finished = df[STATE_COL] != "finished"
    n_not_finished = int(mask_not_finished.sum())
    frac = n_not_finished / max(n0, 1)
    if frac > 0.05:
        logger.warning(
            "More than 5%% (%.1f%%) of rows are non-finished. State distribution: %s",
            100 * frac,
            state_dist_json,
        )
    df = df.loc[~mask_not_finished].copy()
    logger.info("Dropped %d rows with meta_run/state != 'finished'", n_not_finished)

    n_auroc_null = int(df[AUROC_COL].isna().sum())
    assert n_auroc_null == 0, f"Expected eval_v2/test_auroc all non-null, found {n_auroc_null}"

    info: dict[str, object] = {
        "source_rows": n0,
        "state_distribution": state_dist_json,
        "row_filters": {
            "needs_review_dropped": n_review,
            "not_finished_dropped": n_not_finished,
            "auroc_null_dropped": 0,
        },
        "rows_after_filters": len(df),
    }
    return df, info


def apply_dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Dedupe on ``(checkpoint_sha256, G3, R5_Seed)`` keeping first by ``meta_run/id``.

    Assumes ``dedupe_key`` from :mod:`thesis_ml.reports.ch8_analysis_ready`.
    """
    for col in DEDUPE_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"Missing dedupe column in input: {col}")

    df = df.copy()
    df["_dedupe_key"] = df.apply(lambda row: dedupe_key(row.to_dict()), axis=1)
    df = df.sort_values("meta_run/id", kind="mergesort")
    before = len(df)
    df = df.drop_duplicates(subset="_dedupe_key", keep="first")
    after = len(df)
    df = df.drop(columns="_dedupe_key")
    dropped = before - after
    logger.info("Dedupe dropped %d duplicate rows (%d -> %d)", dropped, before, after)
    return df, dropped


def find_drop_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (all_nan_cols, zero_variance_cols) among ``config/axes/*`` only."""
    axis_cols = [c for c in df.columns if c.startswith("config/axes/")]
    all_nan: list[str] = []
    zero_var: list[str] = []
    for c in axis_cols:
        s = df[c]
        if s.isna().all():
            all_nan.append(c)
        elif s.nunique(dropna=False) <= 1:
            zero_var.append(c)
    return all_nan, zero_var


def encode_branch_conditional_nans(
    df: pd.DataFrame, axis_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill NaN in mixed null/non-null axis columns with ``"inactive"`` and cast to str.

    Returns the modified frame and a per-column residual-NaN count (post-cast).
    """
    df = df.copy()
    residuals: dict[str, int] = {}
    for c in axis_cols:
        s = df[c]
        n_null = int(s.isna().sum())
        n_total = len(s)
        if 0 < n_null < n_total:
            # Mixed null/non-null -> branch-conditional. Fill and cast to str.
            df[c] = s.where(s.notna(), "inactive").astype(str)
        else:
            # All non-null (zero-variance already dropped). Cast for uniformity.
            df[c] = s.astype(str)
        # Sanity: residual literal-NaN should be zero (pandas will render NaN -> 'nan').
        residual = int((df[c] == "nan").sum())
        if residual > 0:
            residuals[c] = residual
    return df, residuals


def split_by_g3(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split frame into the three G3 cohorts; error if any expected string is absent."""
    wanted = {
        "primary": PRIMARY_G3,
        "g3_4t_tth": APPENDIX_G3_4T_TTH,
        "g3_5class": APPENDIX_G3_5CLASS,
    }
    seen = set(df[G3_COL].dropna().unique().tolist())
    missing = [v for v in wanted.values() if v not in seen]
    if missing:
        raise RuntimeError(
            f"Expected G3 strings not present in deduped frame: {missing}. "
            f"Observed unique G3 values: {sorted(seen)}"
        )
    out: dict[str, pd.DataFrame] = {}
    for key, g3_value in wanted.items():
        out[key] = df.loc[df[G3_COL] == g3_value].copy()
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_input(INPUT_CSV)
    df, filter_info = apply_row_filters(df)

    df, n_dedupe_dropped = apply_dedupe(df)

    all_nan, zero_var = find_drop_columns(df)
    to_drop = all_nan + zero_var
    logger.info("Dropping %d axis columns (all-NaN: %d, zero-variance: %d)",
                len(to_drop), len(all_nan), len(zero_var))
    for c in to_drop:
        logger.info("  drop: %s", c)
    df = df.drop(columns=to_drop)

    axis_cols_kept = [c for c in df.columns if c.startswith("config/axes/")]
    df, residuals = encode_branch_conditional_nans(df, axis_cols_kept)

    splits = split_by_g3(df)
    split_paths = {
        "primary": OUT_DIR / "05_ch8_streamlined_primary.csv",
        "g3_4t_tth": OUT_DIR / "05_ch8_streamlined_g3_4t_tth.csv",
        "g3_5class": OUT_DIR / "05_ch8_streamlined_g3_5class.csv",
    }
    task_split_counts: dict[str, int] = {}
    for key, out_path in split_paths.items():
        sub = splits[key]
        sub.to_csv(out_path, index=False)
        task_split_counts[key] = int(len(sub))
        logger.info("Wrote %s (%d rows, %d cols)", out_path, len(sub), sub.shape[1])

    summary: dict[str, object] = {
        "source": str(INPUT_CSV.relative_to(REPO_ROOT)),
        "source_rows": filter_info["source_rows"],
        "state_distribution": filter_info["state_distribution"],
        "row_filters": filter_info["row_filters"],
        "rows_after_row_filters": filter_info["rows_after_filters"],
        "dedupe_dropped": int(n_dedupe_dropped),
        "rows_after_dedupe": int(len(df)),
        "columns_dropped_zero_variance_or_all_nan": sorted(to_drop),
        "columns_dropped_all_nan": sorted(all_nan),
        "columns_dropped_zero_variance": sorted(zero_var),
        "axes_columns_kept": int(len(axis_cols_kept)),
        "task_split": task_split_counts,
        "encoding": (
            "branch-conditional NaN in config/axes/* (mixed null+non-null) filled "
            "with 'inactive'; all axis columns cast to str"
        ),
        "axis_numeric_nan_residuals": residuals,
        "outputs": {k: str(p.relative_to(REPO_ROOT)) for k, p in split_paths.items()},
    }
    summary_path = OUT_DIR / "05_ch8_streamlined_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info("Wrote %s", summary_path)

    # Drop accounting sanity check (must sum to 1320 source rows).
    src = int(filter_info["source_rows"])
    accounted = (
        int(filter_info["row_filters"]["needs_review_dropped"])  # type: ignore[index]
        + int(filter_info["row_filters"]["not_finished_dropped"])  # type: ignore[index]
        + int(n_dedupe_dropped)
        + sum(task_split_counts.values())
    )
    drop_counts = Counter(
        {
            "needs_review_dropped": filter_info["row_filters"]["needs_review_dropped"],  # type: ignore[index]
            "not_finished_dropped": filter_info["row_filters"]["not_finished_dropped"],  # type: ignore[index]
            "dedupe_dropped": n_dedupe_dropped,
            **task_split_counts,
        }
    )
    logger.info("Drop accounting: %s", dict(drop_counts))
    if accounted != src:
        # Difference = rows that survived row filters + dedupe but did not match any
        # of the three G3 strings. Should be zero per the split assertion.
        raise RuntimeError(
            f"Drop accounting mismatch: accounted={accounted} != source_rows={src}"
        )
    logger.info("Drop accounting matches source rows (%d).", src)


if __name__ == "__main__":
    main()
