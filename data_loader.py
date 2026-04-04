from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


@dataclass
class CTScanData:
    sample_dir: Path
    settings_path: Path
    projections_dir: Path
    settings: dict[str, dict[str, Any]]
    projections: np.ndarray
    projection_files: list[Path]


class CasePreservingConfigParser(ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    upper = value.upper()

    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False

    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_cto_settings(settings_path: str | Path) -> dict[str, dict[str, Any]]:
    settings_path = Path(settings_path)
    parser = CasePreservingConfigParser()

    with settings_path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)

    parsed: dict[str, dict[str, Any]] = {}
    for section in parser.sections():
        parsed[section] = {
            key: _parse_value(value)
            for key, value in parser.items(section)
        }
    return parsed


def list_projection_files(projections_dir: str | Path) -> list[Path]:
    projections_dir = Path(projections_dir)
    files = sorted(projections_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No TIFF projections found in {projections_dir}")
    return files


def load_projection_stack(projections_dir: str | Path, dtype=np.float32) -> tuple[np.ndarray, list[Path]]:
    files = list_projection_files(projections_dir)
    stack = np.stack(
        [tifffile.imread(str(path)).astype(dtype, copy=False) for path in files],
        axis=0,
    )
    return stack, files


def load_sample1(sample_dir: str | Path = "sample 1", dtype=np.float32) -> CTScanData:
    sample_dir = Path(sample_dir)
    settings_path = sample_dir / "settings.cto"
    projections_dir = sample_dir / "projections"

    if not settings_path.exists():
        raise FileNotFoundError(f"Missing settings file: {settings_path}")
    if not projections_dir.exists():
        raise FileNotFoundError(f"Missing projections directory: {projections_dir}")

    settings = load_cto_settings(settings_path)
    projections, projection_files = load_projection_stack(projections_dir, dtype=dtype)

    return CTScanData(
        sample_dir=sample_dir,
        settings_path=settings_path,
        projections_dir=projections_dir,
        settings=settings,
        projections=projections,
        projection_files=projection_files,
    )


if __name__ == "__main__":
    data = load_sample1()
    print("Loaded sample:", data.sample_dir)
    print("Projection stack shape:", data.projections.shape)
    print("Projection dtype:", data.projections.dtype)
    print("Number of projection files:", len(data.projection_files))
    print("Sections:", list(data.settings))
