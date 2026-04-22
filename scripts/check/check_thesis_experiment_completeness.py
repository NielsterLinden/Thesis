#!/usr/bin/env python3
"""Check completeness of thesis experiment sweeps (Ch 4/5/6/8).

This script evaluates completeness at the experiment-sweep level by:
1) discovering experiment configs in thesis_experiments,
2) finding corresponding multirun directories,
3) resolving run directories via timestamp + experiment name,
4) reusing run-level checks from check_run_completeness.py.

It prints a copy/paste-friendly summary block so results can be shared in chat.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIGS_DIR = Path("configs/classifier/experiment/thesis_experiments")
DEFAULT_MULTIRUNS_DIR = Path("/data/atlas/users/nterlind/outputs/multiruns")
DEFAULT_RUNS_DIR = Path("/data/atlas/users/nterlind/outputs/runs")
DEFAULT_CHAPTERS = (4, 5, 6, 8)

SUMMARY_START = "===THESIS_COMPLETENESS_SUMMARY_START==="
SUMMARY_END = "===THESIS_COMPLETENESS_SUMMARY_END==="
CANONICAL_START = "===THESIS_CANONICAL_MAPPING_START==="
CANONICAL_END = "===THESIS_CANONICAL_MAPPING_END==="


def load_run_checker() -> Any:
    """Load check_run_completeness from sibling script via file path."""
    current = Path(__file__).resolve()
    target = current.parent / "check_run_completeness.py"
    spec = importlib.util.spec_from_file_location("check_run_completeness_module", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load checker module from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "check_run_completeness"):
        raise RuntimeError("check_run_completeness function not found in check_run_completeness.py")
    return module.check_run_completeness


CHECK_RUN_COMPLETENESS = load_run_checker()


@dataclass
class SweepCandidateResult:
    """One multirun candidate evaluated for one experiment config."""

    multirun_dir: Path
    timestamp: str
    run_dirs: list[Path]
    expected_runs: int
    complete_runs: int
    incomplete_runs: int

    @property
    def is_complete(self) -> bool:
        return self.expected_runs > 0 and len(self.run_dirs) == self.expected_runs and self.incomplete_runs == 0 and self.complete_runs == self.expected_runs


@dataclass
class ExperimentResult:
    """Resolved completeness status for one experiment config."""

    chapter: int
    config_path: Path
    experiment_name: str
    expected_runs: int
    status: str  # COMPLETE | INCOMPLETE | MISSING
    selected: SweepCandidateResult | None
    all_candidates: list[SweepCandidateResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check thesis experiment completeness")
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=DEFAULT_CONFIGS_DIR,
        help=f"Directory with thesis experiment YAMLs (default: {DEFAULT_CONFIGS_DIR})",
    )
    parser.add_argument(
        "--multiruns-dir",
        type=Path,
        default=DEFAULT_MULTIRUNS_DIR,
        help=f"Multiruns root directory (default: {DEFAULT_MULTIRUNS_DIR})",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Runs root directory (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--chapters",
        type=str,
        default="4,5,6,8",
        help="Comma-separated chapter numbers to include (default: 4,5,6,8)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Optional comma-separated experiment config stems to include",
    )
    parser.add_argument(
        "--verbose-runs",
        action="store_true",
        help="Enable verbose run-level checks (very noisy)",
    )
    return parser.parse_args()


def split_csv_top_level(value: str) -> list[str]:
    """Split comma-separated sweeper values, respecting [] and quotes."""
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    quote: str | None = None

    for ch in value:
        if quote is not None:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue

        if ch in {"'", '"'}:
            quote = ch
            cur.append(ch)
            continue

        if ch == "[":
            depth += 1
            cur.append(ch)
            continue

        if ch == "]":
            depth = max(0, depth - 1)
            cur.append(ch)
            continue

        if ch == "," and depth == 0:
            part = "".join(cur).strip()
            if part:
                parts.append(part)
            cur = []
            continue

        cur.append(ch)

    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def compute_expected_runs(config_path: Path) -> int:
    """Compute expected Cartesian sweep size from hydra.sweeper.params."""
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    params: dict[str, Any] = cfg.get("hydra", {}).get("sweeper", {}).get("params", {}) or {}
    if not params:
        return 1

    cardinalities: list[int] = []
    for value in params.values():
        if isinstance(value, str):
            cardinalities.append(max(1, len(split_csv_top_level(value))))
        elif isinstance(value, list):
            cardinalities.append(max(1, len(value)))
        else:
            cardinalities.append(1)

    return reduce(mul, cardinalities, 1)


def chapter_from_name(stem: str) -> int | None:
    m = re.match(r"ch(\d+)_", stem)
    return int(m.group(1)) if m else None


def parse_multirun_name(multirun_name: str) -> tuple[str, str] | None:
    m = re.match(r"exp_(\d{8}-\d{6})_(.+)$", multirun_name)
    if not m:
        return None
    return m.group(1), m.group(2)


def evaluate_candidate(
    multirun_dir: Path,
    runs_dir: Path,
    expected_runs: int,
    verbose_runs: bool,
) -> SweepCandidateResult:
    parsed = parse_multirun_name(multirun_dir.name)
    if parsed is None:
        return SweepCandidateResult(
            multirun_dir=multirun_dir,
            timestamp="UNKNOWN",
            run_dirs=[],
            expected_runs=expected_runs,
            complete_runs=0,
            incomplete_runs=0,
        )

    timestamp, experiment_name = parsed
    run_dirs = sorted(runs_dir.glob(f"run_{timestamp}_{experiment_name}_job*"))
    complete = 0
    incomplete = 0

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        status = CHECK_RUN_COMPLETENESS(run_dir, verbose=verbose_runs)
        if status.is_complete:
            complete += 1
        else:
            incomplete += 1

    return SweepCandidateResult(
        multirun_dir=multirun_dir,
        timestamp=timestamp,
        run_dirs=[p for p in run_dirs if p.is_dir()],
        expected_runs=expected_runs,
        complete_runs=complete,
        incomplete_runs=incomplete,
    )


def choose_candidate(candidates: list[SweepCandidateResult]) -> SweepCandidateResult | None:
    if not candidates:
        return None

    complete_candidates = [c for c in candidates if c.is_complete]
    if complete_candidates:
        return sorted(complete_candidates, key=lambda c: c.timestamp)[-1]

    return sorted(candidates, key=lambda c: c.timestamp)[-1]


def discover_experiment_configs(
    configs_dir: Path,
    allowed_chapters: set[int],
    experiment_filter: set[str] | None,
) -> list[Path]:
    config_paths = sorted(configs_dir.glob("*.yaml"))
    selected: list[Path] = []
    for path in config_paths:
        chapter = chapter_from_name(path.stem)
        if chapter is None or chapter not in allowed_chapters:
            continue
        if experiment_filter is not None and path.stem not in experiment_filter:
            continue
        selected.append(path)
    return selected


def evaluate_experiments(args: argparse.Namespace) -> list[ExperimentResult]:
    chapters = {int(part.strip()) for part in args.chapters.split(",") if part.strip()}
    experiment_filter = None
    if args.experiments:
        experiment_filter = {part.strip() for part in args.experiments.split(",") if part.strip()}

    configs = discover_experiment_configs(args.configs_dir, chapters, experiment_filter)
    results: list[ExperimentResult] = []

    for config_path in configs:
        stem = config_path.stem
        chapter = chapter_from_name(stem) or -1
        expected_runs = compute_expected_runs(config_path)

        candidates = sorted(
            [d for d in args.multiruns_dir.glob(f"exp_*_{stem}") if d.is_dir()],
            key=lambda p: p.name,
        )
        evaluated = [
            evaluate_candidate(
                multirun_dir=c,
                runs_dir=args.runs_dir,
                expected_runs=expected_runs,
                verbose_runs=args.verbose_runs,
            )
            for c in candidates
        ]
        selected = choose_candidate(evaluated)

        if selected is None:
            status = "MISSING"
        elif selected.is_complete:
            status = "COMPLETE"
        else:
            status = "INCOMPLETE"

        results.append(
            ExperimentResult(
                chapter=chapter,
                config_path=config_path,
                experiment_name=stem,
                expected_runs=expected_runs,
                status=status,
                selected=selected,
                all_candidates=evaluated,
            )
        )
    return results


def print_summary(results: list[ExperimentResult]) -> None:
    total = len(results)
    complete = sum(1 for r in results if r.status == "COMPLETE")
    incomplete = sum(1 for r in results if r.status == "INCOMPLETE")
    missing = sum(1 for r in results if r.status == "MISSING")

    print(f"Checked experiments: {total}")
    print(f"COMPLETE: {complete} | INCOMPLETE: {incomplete} | MISSING: {missing}")
    print("")

    print(SUMMARY_START)
    print("CH|EXPERIMENT|STATUS|EXPECTED_RUNS|FOUND_RUNS|COMPLETE_RUNS|MULTIRUN_DIR|RUN_GLOB")
    for r in results:
        if r.selected is None:
            found_runs = 0
            complete_runs = 0
            multirun = "-"
            run_glob = "-"
        else:
            found_runs = len(r.selected.run_dirs)
            complete_runs = r.selected.complete_runs
            multirun = str(r.selected.multirun_dir)
            run_glob = f"{r.selected.multirun_dir.parent.parent / 'runs'}/" f"run_{r.selected.timestamp}_{r.experiment_name}_job*"

        print(f"{r.chapter}|{r.experiment_name}|{r.status}|{r.expected_runs}|" f"{found_runs}|{complete_runs}|{multirun}|{run_glob}")
    print(SUMMARY_END)
    print("")

    print(CANONICAL_START)
    for r in results:
        if r.status != "COMPLETE" or r.selected is None:
            continue
        run_glob = f"{r.selected.multirun_dir.parent.parent / 'runs'}/" f"run_{r.selected.timestamp}_{r.experiment_name}_job***"
        print(f"- config: {r.config_path.name}\n" f"  multirun_dir: {r.selected.multirun_dir}\n" f"  run_dir_pattern: {run_glob}\n" f"  expected_runs: {r.expected_runs}")
    print(CANONICAL_END)


def main() -> int:
    args = parse_args()
    results = evaluate_experiments(args)
    print_summary(results)
    return 0 if all(r.status == "COMPLETE" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
