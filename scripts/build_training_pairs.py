from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_recon.build_training_pairs import build_default_training_pairs


if __name__ == "__main__":
    outputs = build_default_training_pairs()
    for item in outputs:
        print(item["pair_dir"])
