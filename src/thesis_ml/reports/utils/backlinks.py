"""Utilities for managing report backlinks in runs."""

from __future__ import annotations

from pathlib import Path


def append_report_pointer(run_dir: Path | str, report_id: str) -> None:
    """Atomically append report_id to report_pointer.txt in run directory.

    This function is idempotent and uses atomic writes (temp file + rename)
    to handle concurrency safely on Stoomboot.

    Parameters
    ----------
    run_dir : Path | str
        Path to run directory
    report_id : str
        Report ID to append (e.g., "report_20251024-190000_compare_tokenizers")
    """
    run_dir = Path(run_dir)
    pointer_file = run_dir / "report_pointer.txt"
    temp_file = pointer_file.with_suffix(".tmp")

    # Read existing entries (if any)
    existing = []
    if pointer_file.exists():
        existing = [line.strip() for line in pointer_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Idempotent: don't add if already present
    if report_id not in existing:
        existing.append(report_id)
        # Write to temp file first
        temp_file.write_text("\n".join(existing) + "\n", encoding="utf-8")
        # Atomic rename (works on Windows and Unix)
        temp_file.replace(pointer_file)


def read_report_pointers(run_dir: Path | str) -> list[str]:
    """Read all report IDs from report_pointer.txt.

    Parameters
    ----------
    run_dir : Path | str
        Path to run directory

    Returns
    -------
    list[str]
        List of report IDs (one per line)
    """
    run_dir = Path(run_dir)
    pointer_file = run_dir / "report_pointer.txt"
    if not pointer_file.exists():
        return []
    return [line.strip() for line in pointer_file.read_text(encoding="utf-8").splitlines() if line.strip()]
