"""Experiments of the paper. Run the scripts as modules from the repository root, e.g.
    python -m experiments.exp1_dataset_size.evaluate
The fpnode package is used from src/ if it is not installed (pip install -e .)."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import fpnode  # noqa: F401
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def rel(path) -> str:
    """Path relative to the repository root when inside it (for printed messages)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return os.path.relpath(path)
