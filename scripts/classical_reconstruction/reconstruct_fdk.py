from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_recon.reconstruct_fdk_astra import run_fdk_reconstruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FDK reconstruction for sample_1.")
    parser.add_argument("--sample-dir", default=str(REPO_ROOT / "data" / "sample_1"))
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "fdk_full_ds2"))
    args = parser.parse_args()

    outputs = run_fdk_reconstruction(
        sample_dir=args.sample_dir,
        downsample_factor=args.downsample,
        output_dir=args.output_dir,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
