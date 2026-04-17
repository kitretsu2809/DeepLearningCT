from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_recon.reconstruct_fdk_astra import run_fdk_reconstruction


if __name__ == "__main__":
    outputs = run_fdk_reconstruction()
    for key, value in outputs.items():
        print(f"{key}: {value}")
