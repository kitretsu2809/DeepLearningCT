from __future__ import annotations

from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
EXPORTS_DIR = REPO_ROOT / "exports"
DOCS_DIR = REPO_ROOT / "docs"
REFERENCES_DIR = REPO_ROOT / "references"

SAMPLE_DIRS = {
    "sample_1": DATA_DIR / "sample_1",
    "sample_2": DATA_DIR / "sample_2",
}

SAMPLE_DIR = SAMPLE_DIRS["sample_1"]


def set_sample(sample_name: Literal["sample_1", "sample_2"]) -> None:
    global SAMPLE_DIR
    SAMPLE_DIR = SAMPLE_DIRS[sample_name]


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate
