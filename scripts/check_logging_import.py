#!/usr/bin/env python3
"""Check that files using logger = logging.getLogger(__name__) have import logging."""

import re
import sys
from pathlib import Path


def check_file(filepath: Path) -> bool:
    """Check if file has proper logging import."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return True  # Skip files we can't read

    # Check if file uses logger = logging.getLogger(__name__)
    has_logger_usage = "logger = logging.getLogger(__name__)" in content

    if not has_logger_usage:
        return True  # File doesn't use logger, skip

    # Check if file has import logging
    has_import = bool(re.search(r"^import logging", content, re.MULTILINE))

    if not has_import:
        print(f"{filepath}: Uses logger but missing 'import logging'")
        return False

    return True


def main():
    """Check all Python files passed as arguments."""
    if len(sys.argv) < 2:
        return 0

    files = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]
    all_ok = all(check_file(f) for f in files)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
