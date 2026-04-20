from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_recon.build_training_pairs import build_default_training_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build default U-Net training pairs for one sample.")
    parser.add_argument("--sample-dir", default=str(REPO_ROOT / "data" / "sample_1"))
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "training_pairs"))
    args = parser.parse_args()

    outputs = build_default_training_pairs(
        sample_dir=args.sample_dir,
        downsample_factor=args.downsample_factor,
        output_dir=args.output_dir,
    )
    for item in outputs:
        print(item["pair_dir"])
