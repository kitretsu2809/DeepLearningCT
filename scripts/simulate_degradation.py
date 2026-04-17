from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_recon.simulate_degradation import build_and_save_default_degradation_sets


if __name__ == "__main__":
    outputs = build_and_save_default_degradation_sets()
    for item in outputs:
        print(item["dataset_dir"])
