#!/usr/bin/env python3
"""Stage A: build 00_eval_manifest.csv from raw W&B export + HPC run layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from omegaconf import OmegaConf


def _resolve_classifier_weights_path(run_dir: Path) -> tuple[Path, str]:
    """Match thesis_ml.reports.utils.inference.resolve_classifier_weights_path (no torch import)."""
    ordered = [
        (run_dir / "best_val.pt", "best_val"),
        (run_dir / "last.pt", "last"),
        (run_dir / "model.pt", "model"),
    ]
    for path, kind in ordered:
        if path.is_file():
            return path, kind
    best_ep, best_p = -1, None
    for path in run_dir.glob("epoch_*.pt"):
        m = re.match(r"^epoch_(\d+)\.pt$", path.name)
        if m and int(m.group(1)) > best_ep:
            best_ep, best_p = int(m.group(1)), path
    if best_p is not None:
        return best_p, "epoch"
    raise FileNotFoundError(f"No weights in {run_dir}")

FIELDS = [
    "run_id",
    "run_name",
    "source_created_at",
    "run_dir",
    "checkpoint_path",
    "checkpoint_kind",
    "checkpoint_size_mb",
    "checkpoint_sha256",
    "task_canonical",
    "task_status",
    "model_family",
    "selected_labels",
    "n_test_events_expected",
]
COL = {
    "id": "meta_run/id",
    "name": "meta_run/name",
    "created": "meta_run/created_at",
    "src_dir": "config/source/run_dir",
    "g2": "config/axes/G2_Model Family",
    "g3": "config/axes/G3_Classification Task",
    "sel": "config/raw/classifier/selected_labels",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_run_dir(row: dict[str, str], runs_root: Path) -> Path | None:
    raw = (row.get(COL["src_dir"]) or "").strip()
    if raw:
        return Path(raw)
    name = (row.get(COL["name"]) or "").strip()
    return runs_root / name if name else None


def _task_from_g3(g3: str, g3_map: dict[str, str]) -> tuple[str, str]:
    key = (g3 or "").strip()
    if not key:
        return "", "skip_unsupported_task"
    canon = g3_map.get(key, "skip_unsupported_task")
    if canon == "skip_unsupported_task":
        return "", "skip_unsupported_task"
    return str(canon), "evaluable"


def _emit(w: csv.DictWriter, **kw: str) -> None:
    w.writerow({f: kw.get(f, "") for f in FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A: eval manifest")
    ap.add_argument("--raw-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, default=Path("/data/atlas/users/nterlind/outputs/runs"))
    ap.add_argument("--eval-spec", type=Path, default=Path(__file__).parent / "config" / "eval_spec.yaml")
    ap.add_argument("--test-splits", type=Path, default=Path(__file__).parent / "config" / "test_splits.yaml")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N rows of raw CSV")
    args = ap.parse_args()

    g3_map: dict[str, str] = OmegaConf.to_container(OmegaConf.load(args.eval_spec).g3_to_canonical, resolve=True)  # type: ignore[assignment]
    tasks: dict = OmegaConf.to_container(OmegaConf.load(args.test_splits).tasks, resolve=True)  # type: ignore[assignment]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "00_eval_manifest.csv"
    with args.raw_csv.open(newline="", encoding="utf-8") as fin, out_path.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        writer.writeheader()
        for i, row in enumerate(reader):
            if args.limit and i >= args.limit:
                break
            rid = (row.get(COL["id"]) or "").strip()
            rname = (row.get(COL["name"]) or "").strip()
            created = (row.get(COL["created"]) or "").strip()
            g2 = (row.get(COL["g2"]) or "").strip().lower()
            g3 = (row.get(COL["g3"]) or "").strip()
            sel = (row.get(COL["sel"]) or "").strip()
            base = {"run_id": rid, "run_name": rname, "source_created_at": created, "model_family": g2, "selected_labels": sel}
            rd = _resolve_run_dir(row, args.runs_root)
            if rd is None or not rd.is_dir():
                _emit(writer, **base, task_status="skip_missing_run_dir", run_dir=str(rd) if rd else "")
                continue
            base["run_dir"] = str(rd.resolve())
            if g2 != "transformer":
                _emit(writer, **base, task_status="skipped_non_transformer")
                continue
            tcanon, tstat = _task_from_g3(g3, g3_map)
            if tstat != "evaluable":
                _emit(writer, **base, task_canonical=tcanon, task_status=tstat)
                continue
            try:
                ckpt_path, ckpt_kind = _resolve_classifier_weights_path(rd)
            except FileNotFoundError:
                _emit(writer, **base, task_canonical=tcanon, task_status="skip_no_checkpoint")
                continue
            tdef = tasks.get(tcanon, {})
            nexp = str(tdef["approx_n_test_events"]) if isinstance(tdef, dict) and "approx_n_test_events" in tdef else ""
            if not sel and isinstance(tdef, dict):
                if "selected_labels" in tdef:
                    sel = json.dumps(tdef["selected_labels"])
                elif "signal_vs_background" in tdef:
                    sel = json.dumps({"signal_vs_background": tdef["signal_vs_background"]})
            base["selected_labels"] = sel
            sz = ckpt_path.stat().st_size / (1024 * 1024)
            _emit(
                writer,
                **base,
                checkpoint_path=str(ckpt_path.resolve()),
                checkpoint_kind=ckpt_kind,
                checkpoint_size_mb=f"{sz:.4f}",
                checkpoint_sha256=_sha256_file(ckpt_path),
                task_canonical=tcanon,
                task_status="evaluable",
                n_test_events_expected=nexp,
            )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
