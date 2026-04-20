from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_recon.simulate_degradation import build_and_save_default_degradation_sets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build default degraded projection sets.")
    parser.add_argument("--sample-dir", default=str(REPO_ROOT / "data" / "sample_1"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "degradation_sets"))
    args = parser.parse_args()

    outputs = build_and_save_default_degradation_sets(
        sample_dir=args.sample_dir,
        output_dir=args.output_dir,
    )
    for item in outputs:
        print(item["dataset_dir"])
