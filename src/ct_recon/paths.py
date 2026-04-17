from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample_1"
OUTPUTS_DIR = REPO_ROOT / "outputs"
EXPORTS_DIR = REPO_ROOT / "exports"
DOCS_DIR = REPO_ROOT / "docs"
REFERENCES_DIR = REPO_ROOT / "references"


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate
