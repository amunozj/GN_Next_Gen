"""Repository-relative paths.

Resolved from this file's location, so they are valid for the editable install created by the
conda environment files (``pip install -e .``).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
MANIFEST = DATA_RAW / "MANIFEST.csv"

CONFIGS = REPO_ROOT / "configs"
OUTPUTS = REPO_ROOT / "outputs"
LEGACY = REPO_ROOT / "legacy"
DOCS = REPO_ROOT / "docs"
