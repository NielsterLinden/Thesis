"""Backlinks from runs to reports for bidirectional navigation."""

from __future__ import annotations

import os
from pathlib import Path


def append_report_pointer(run_dir: Path, report_id: str) -> None:
    """Append report ID to report_pointer.txt in run directory (append-only, atomic).

    Creates a bidirectional link: runs know which reports reference them.
    This is useful for navigation and discovering all reports that used a given run.

    Parameters
    ----------
    run_dir : Path
        Path to run directory
    report_id : str
        Report ID to append (e.g., "report_20251024-190000_compare_tokenizers")
    """
    run_dir = Path(run_dir)
    pointer_file = run_dir / "report_pointer.txt"

    # Read existing content to avoid duplicates
    existing_ids = set()
    if pointer_file.exists():
        try:
            content = pointer_file.read_text(encoding="utf-8")
            existing_ids = {line.strip() for line in content.splitlines() if line.strip()}
        except Exception:
            # If read fails, we'll append anyway (safer than skipping)
            pass

    # Append if not already present (idempotent)
    if report_id not in existing_ids:
        # Use append mode with newline for atomic append
        with pointer_file.open("a", encoding="utf-8") as f:
            f.write(f"{report_id}\n")
            # Force flush to ensure atomic write
            f.flush()
            os.fsync(f.fileno())
