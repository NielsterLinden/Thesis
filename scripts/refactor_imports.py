"""Import rewrite script for codebase refactoring.

Automatically updates imports from old structure to new structure.

Usage:
    python scripts/refactor_imports.py
"""

import re
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parents[1]

# Import mappings: old pattern → new replacement
IMPORT_MAPPINGS = {
    # Phase1 to architectures/autoencoder
    r"from\s+thesis_ml\.phase1\.autoenc": "from thesis_ml.architectures.autoencoder",
    r"import\s+thesis_ml\.phase1\.autoenc": "import thesis_ml.architectures.autoencoder",
    # General to architectures/simple
    r"from\s+thesis_ml\.general\.models": "from thesis_ml.architectures.simple",
    r"import\s+thesis_ml\.general\.models": "import thesis_ml.architectures.simple",
    # Phase1 train to training_loops
    r"from\s+thesis_ml\.phase1\.train": "from thesis_ml.training_loops",
    r"import\s+thesis_ml\.phase1\.train": "import thesis_ml.training_loops",
    # General train to training_loops
    r"from\s+thesis_ml\.general\.train": "from thesis_ml.training_loops",
    r"import\s+thesis_ml\.general\.train": "import thesis_ml.training_loops",
    # Plots to monitoring
    r"from\s+thesis_ml\.plots": "from thesis_ml.monitoring",
    r"import\s+thesis_ml\.plots": "import thesis_ml.monitoring",
    # Reports experiments to analyses
    r"from\s+thesis_ml\.reports\.experiments": "from thesis_ml.reports.analyses",
    r"import\s+thesis_ml\.reports\.experiments": "import thesis_ml.reports.analyses",
    # Facts system consolidation
    r"from\s+thesis_ml\.utils\.facts_builder": "from thesis_ml.facts.builders",
    r"import\s+thesis_ml\.utils\.facts_builder": "import thesis_ml.facts.builders",
    r"from\s+thesis_ml\.plots\.io_utils\s+import\s+(append_jsonl_event|append_scalars_csv|ensure_facts_dir)": r"from thesis_ml.facts.writers import \1",
    r"from\s+thesis_ml\.reports\.utils\.read_facts": "from thesis_ml.facts.readers",
    r"import\s+thesis_ml\.reports\.utils\.read_facts": "import thesis_ml.facts.readers",
    # CLI reorganization
    r"from\s+thesis_ml\.train\s+import\s+DISPATCH": "from thesis_ml.cli.train import DISPATCH",
}


def rewrite_file(file_path: Path) -> bool:
    """Rewrite imports in a single file.

    Returns:
        True if file was modified, False otherwise
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        original = content

        # Apply all mappings
        for pattern, replacement in IMPORT_MAPPINGS.items():
            content = re.sub(pattern, replacement, content)

        # Check if anything changed
        if content != original:
            file_path.write_text(content, encoding="utf-8")
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Run import rewrite on all Python files in the repository."""
    # Find all Python files, excluding venv/cache
    python_files = [p for p in ROOT.rglob("*.py") if not any(part in str(p) for part in ["venv", "__pycache__", ".venv", "node_modules", ".git"])]

    print(f"Found {len(python_files)} Python files to process...")

    modified_count = 0
    for file_path in python_files:
        if rewrite_file(file_path):
            print(f"[OK] Updated: {file_path.relative_to(ROOT)}")
            modified_count += 1

    print(f"\nComplete! Modified {modified_count} files.")


if __name__ == "__main__":
    main()
