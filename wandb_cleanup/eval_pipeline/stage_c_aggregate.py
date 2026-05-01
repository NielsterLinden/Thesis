#!/usr/bin/env python3
"""Stage C: merge eval results into raw export, validate schema, emit anomalies."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


def _is_na(val: str, na_lit: bool) -> bool:
    s = str(val).strip()
    if s == "<not_applicable>":
        return na_lit
    return s == "" or s.lower() == "nan"


def _dedupe_results(res: pd.DataFrame) -> pd.DataFrame:
    if res.empty:
        return res
    if "eval_v2/timestamp" in res.columns:
        return res.sort_values("eval_v2/timestamp").drop_duplicates("run_id", keep="last")
    return res.drop_duplicates("run_id", keep="last")


def _merge_raw_and_results(raw: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """Left-join raw export to eval rows on ``meta_run/id`` == ``run_id``, with optional name fallback."""
    res = _dedupe_results(res.copy())
    drop_join = {"run_id"}
    eval_cols = [c for c in res.columns if c not in drop_join]

    merged = raw.merge(
        res,
        left_on="meta_run/id",
        right_on="run_id",
        how="left",
        suffixes=("", "_eval"),
    )
    if "run_id" in merged.columns:
        merged = merged.drop(columns=["run_id"])

    marker = "eval_v2/test_auroc"
    if "run_name" in res.columns and marker in merged.columns:
        res_name = res.copy()
        if "eval_v2/timestamp" in res_name.columns:
            res_name = res_name.sort_values("eval_v2/timestamp")
        res_name = res_name.drop_duplicates("run_name", keep="last")
        name_idx = res_name.set_index("run_name")
        need = merged[marker].isna() & merged["meta_run/name"].notna()
        if need.any():
            for idx in merged.index[need]:
                nm = str(merged.at[idx, "meta_run/name"]).strip()
                if not nm or nm not in name_idx.index:
                    continue
                for col in eval_cols:
                    if col == "run_name":
                        continue
                    if col not in merged.columns or col not in name_idx.columns:
                        continue
                    cur = merged.at[idx, col]
                    if pd.isna(cur) or (isinstance(cur, str) and cur.strip() == ""):
                        val = name_idx.at[nm, col]
                        if pd.notna(val):
                            merged.at[idx, col] = val

    return merged


def _validate_row(row: dict, rules: dict) -> list[str]:
    errs: list[str] = []
    for col, spec in rules.items():
        if col not in row:
            continue
        val = row.get(col, "")
        typ = spec.get("type", "str")
        na_ok = bool(spec.get("na_literal", False))
        if _is_na(val, na_ok):
            continue
        if typ == "float":
            try:
                float(val)
            except ValueError:
                errs.append(f"{col}: not float ({val!r})")
        elif typ == "int":
            try:
                int(float(val))
            except ValueError:
                errs.append(f"{col}: not int ({val!r})")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-csv", type=Path, required=True)
    ap.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Merged eval CSV (e.g. 01_eval_outcomes.csv from merge_stage_b_shards.py)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--schema", type=Path, default=Path(__file__).parent / "config" / "schema.yaml")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.raw_csv, low_memory=False)
    res = pd.read_csv(args.results, low_memory=False)
    merged = _merge_raw_and_results(raw, res)

    merged.to_csv(args.out_dir / "02_eval_combined.csv", index=False)

    schema = OmegaConf.load(args.schema)
    rules = OmegaConf.to_container(schema.columns, resolve=True) or {}
    assert isinstance(rules, dict)

    val_lines: list[str] = []
    an_rows: list[dict[str, str]] = []
    for i, row in merged.iterrows():
        rd = row.to_dict()
        errs = _validate_row({str(k): "" if pd.isna(v) else str(v) for k, v in rd.items()}, rules)
        if errs:
            val_lines.append(f"row {i}: " + "; ".join(errs))
        rid = str(rd.get("meta_run/id", ""))
        try:
            auroc = float(rd.get("eval_v2/test_auroc", "nan"))
        except (TypeError, ValueError):
            auroc = float("nan")
        try:
            lat = float(rd.get("eval_v2/inference_latency_ms_b1_mean", "nan"))
        except (TypeError, ValueError):
            lat = float("nan")
        try:
            pm = float(rd.get("eval_v2/peak_memory_mib_inference_b512", "nan"))
        except (TypeError, ValueError):
            pm = float("nan")
        reasons: list[str] = []
        if auroc == auroc and auroc < 0.5:
            reasons.append("auroc_lt_0.5")
        if auroc == 1.0:
            reasons.append("auroc_eq_1")
        if lat == lat and lat <= 0:
            reasons.append("latency_zero")
        if pm == pm and pm > 81920:
            reasons.append("peak_memory_gt_80gb")
        if reasons:
            an_rows.append({"run_id": rid, "reasons": ",".join(reasons)})

    (args.out_dir / "schema_validation_report.txt").write_text("\n".join(val_lines) or "OK\n", encoding="utf-8")
    pd.DataFrame(an_rows).to_csv(args.out_dir / "anomalies_report.csv", index=False)
    print(f"Wrote {args.out_dir / '02_eval_combined.csv'}")


if __name__ == "__main__":
    main()
