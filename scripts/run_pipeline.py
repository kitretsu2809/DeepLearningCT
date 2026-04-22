#!/usr/bin/env python3
"""
Master pipeline script for CT reconstruction workflows.

Usage:
    # Sinogram reconstruction (direct: sinogram -> image)
    python scripts/run_pipeline.py sinogram --sample sample_1 --epochs 50
    
    # U-Net enhancement (post-processing: degraded FBP -> enhanced)
    python scripts/run_pipeline.py unet --sample sample_1 --epochs 50
    
    # Classical FDK reconstruction only
    python scripts/run_pipeline.py classical --sample sample_1
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Get repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "common"))

from sample_config import get_sample_paths, get_sample_config


def run_command(cmd: list[str], description: str, check: bool = True) -> int:
    """Run a command with error handling."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"Error: {description} failed with code {result.returncode}")
        return result.returncode
    return 0


def run_classical(sample_name: str, skip_if_exists: bool = True, no_downsample: bool = False, downsample_factor_override: int | None = None):
    """Run classical FDK reconstruction."""
    paths = get_sample_paths(sample_name)
    config = get_sample_config(sample_name)
    
    fdk_volume = paths["fdk_volume"]
    
    if skip_if_exists and fdk_volume.exists():
        print(f"Classical reconstruction already exists: {fdk_volume}")
        return 0
    
    fdk_volume.parent.mkdir(parents=True, exist_ok=True)
    
    if no_downsample:
        ds = 1
        print("Running FULL RESOLUTION reconstruction (no downsampling)")
    elif downsample_factor_override:
        ds = downsample_factor_override
    else:
        ds = config["downsample_factor"]
    
    if sample_name == "sample_1":
        if ds == 1:
            output_name = "fdk_full"
        else:
            output_name = f"fdk_full_ds{ds}"
        result = run_command(
            [sys.executable, str(SCRIPTS_DIR / "classical_reconstruction" / "reconstruct_fdk.py"),
             "--downsample", str(ds),
             "--output-dir", str(REPO_ROOT / "outputs" / output_name)],
            "Running FDK reconstruction for sample_1"
        )
        if result != 0:
            return result
        # Copy to standard location
        source = REPO_ROOT / "outputs" / output_name / "fdk_volume.tif"
        if source.exists():
            shutil.copy(source, fdk_volume)
    else:
        if ds == 1:
            output_name = "sample_2_fdk_full"
        else:
            output_name = f"sample_2_fdk_ds{ds}"
        result = run_command(
            [sys.executable, str(SCRIPTS_DIR / "classical_reconstruction" / "reconstruct_fdk_sample2.py"),
             "--downsample", str(ds)],
            "Running FDK reconstruction for sample_2"
        )
        if result != 0:
            return result
        source = REPO_ROOT / "outputs" / output_name / "fdk_volume.tif"
        if source.exists():
            shutil.copy(source, fdk_volume)
        preview_source = REPO_ROOT / "outputs" / output_name / "preview.png"
        preview_dest = fdk_volume.parent / "fdk_preview.png"
        if preview_source.exists():
            shutil.copy(preview_source, preview_dest)
    
    print(f"FDK volume saved to: {fdk_volume}")
    return 0


def run_sinogram_pipeline(sample_name: str, epochs: int = 50):
    """Run direct sinogram reconstruction pipeline."""
    paths = get_sample_paths(sample_name)
    config = get_sample_config(sample_name)
    
    print(f"\n{'='*60}")
    print(f"Sinogram Reconstruction Pipeline for {sample_name}")
    print(f"{'='*60}")
    
    # Step 1: Classical reconstruction for reference
    result = run_classical(sample_name)
    if result != 0:
        return result
    
    # Step 2: Build dataset
    print("\n[Step 2/4] Building sparse sinogram dataset...")
    paths["sinogram_dataset"].parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [sys.executable, str(SCRIPTS_DIR / "sinogram_reconstruction" / "01_build_dataset.py"),
         "--sample-dir", str(paths["sample_dir"]),
         "--target-volume", str(paths["fdk_volume"]),
         "--output-path", str(paths["sinogram_dataset"]),
         "--downsample-factor", str(config["downsample_factor"]),
         "--sparse-step", "4",
         "--detector-count", "256",
         "--image-size", "256"],
        "Building sparse sinogram dataset"
    )
    if result != 0:
        return result
    
    # Step 3: Train model
    print("\n[Step 3/4] Training model...")
    paths["sinogram_checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [sys.executable, str(SCRIPTS_DIR / "sinogram_reconstruction" / "02_train_model.py"),
         "--dataset-path", str(paths["sinogram_dataset"]),
         "--output-dir", str(paths["sinogram_checkpoint"].parent),
         "--epochs", str(epochs),
         "--batch-size", "4",
         "--learning-rate", "1e-3"],
        "Training sinogram reconstruction model"
    )
    if result != 0:
        return result
    
    # Step 4: Run inference
    print("\n[Step 4/4] Running inference...")
    paths["sinogram_inference"].mkdir(parents=True, exist_ok=True)
    result = run_command(
        [sys.executable, str(SCRIPTS_DIR / "sinogram_reconstruction" / "03_run_inference.py"),
         "--checkpoint", str(paths["sinogram_checkpoint"]),
         "--sample-dir", str(paths["sample_dir"]),
         "--output-dir", str(paths["sinogram_inference"]),
         "--sparse-step", "4"],
        "Running inference"
    )
    
    print(f"\n{'='*60}")
    print("Sinogram reconstruction pipeline complete!")
    print(f"Output: {paths['sinogram_inference']}")
    print(f"{'='*60}")
    return 0


def run_unet_pipeline(sample_name: str, epochs: int = 50):
    """Run U-Net enhancement pipeline."""
    paths = get_sample_paths(sample_name)
    config = get_sample_config(sample_name)
    
    print(f"\n{'='*60}")
    print(f"U-Net Enhancement Pipeline for {sample_name}")
    print(f"{'='*60}")
    
    # Step 1: Classical reference (used for pair generation)
    result = run_classical(sample_name)
    if result != 0:
        return result

    print("\n[Step 2/4] Building training pairs (degraded vs reference)...")
    paths["unet_training_pairs"].mkdir(parents=True, exist_ok=True)
    result = run_command(
        [
            sys.executable,
            str(SCRIPTS_DIR / "unet_enhancement" / "01_build_training_pairs.py"),
            "--sample-dir",
            str(paths["sample_dir"]),
            "--downsample-factor",
            str(config["downsample_factor"]),
            "--output-dir",
            str(paths["unet_training_pairs"]),
        ],
        "Building U-Net training pairs",
    )
    if result != 0:
        return result

    print("\n[Step 3/4] Training U-Net...")
    paths["unet_checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [
            sys.executable,
            str(SCRIPTS_DIR / "unet_enhancement" / "02_train_model.py"),
            "--pairs-root",
            str(paths["unet_training_pairs"]),
            "--output-dir",
            str(paths["unet_checkpoint"].parent),
            "--epochs",
            str(epochs),
            "--batch-size",
            "8",
            "--learning-rate",
            "1e-3",
        ],
        "Training U-Net enhancement model",
    )
    if result != 0:
        return result

    print("\n[Step 4/4] Preparing inference output directory...")
    paths["unet_inference"].mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("U-Net enhancement pipeline complete (training stage).")
    print(f"Training pairs: {paths['unet_training_pairs']}")
    print(f"Model checkpoint: {paths['unet_checkpoint']}")
    print(f"Inference output dir: {paths['unet_inference']}")
    print("Note: inference script is not yet implemented in this repository.")
    print(f"{'='*60}")
    return 0


def run_enhance_pipeline(sample_name: str, epochs: int = 50):
    """Run enhance pipeline: full sinogram input -> FDK target."""
    paths = get_sample_paths(sample_name)
    config = get_sample_config(sample_name)

    print(f"\n{'='*60}")
    print(f"Enhance Pipeline for {sample_name}")
    print(f"(Full sinogram -> FDK reconstruction)")
    print(f"{'='*60}")

    result = run_classical(sample_name)
    if result != 0:
        return result

    print("\n[Step 2/4] Building enhance dataset...")
    paths["enhance_dataset"].parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [sys.executable, str(SCRIPTS_DIR / "enhance_reconstruction" / "01_build_dataset.py"),
         "--sample-dir", str(paths["sample_dir"]),
         "--target-volume", str(paths["fdk_volume"]),
         "--output-path", str(paths["enhance_dataset"]),
         "--downsample-factor", str(config["downsample_factor"]),
         "--detector-count", "256",
         "--image-size", "256"],
        "Building enhance dataset (full sinogram -> FDK)"
    )
    if result != 0:
        return result

    print("\n[Step 3/4] Training enhance model...")
    paths["enhance_checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [sys.executable, str(SCRIPTS_DIR / "enhance_reconstruction" / "02_train_model.py"),
         "--dataset-path", str(paths["enhance_dataset"]),
         "--output-dir", str(paths["enhance_checkpoint"].parent),
         "--epochs", str(epochs),
         "--batch-size", "4",
         "--learning-rate", "1e-3"],
        "Training enhance model"
    )
    if result != 0:
        return result

    print("\n[Step 4/4] Running inference...")
    paths["enhance_inference"].mkdir(parents=True, exist_ok=True)
    result = run_command(
        [sys.executable, str(SCRIPTS_DIR / "enhance_reconstruction" / "03_run_inference.py"),
         "--checkpoint", str(paths["enhance_checkpoint"]),
         "--sample-dir", str(paths["sample_dir"]),
         "--output-dir", str(paths["enhance_inference"])],
        "Running inference"
    )

    print(f"\n{'='*60}")
    print("Enhance pipeline complete!")
    print(f"Output: {paths['enhance_inference']}")
    print(f"{'='*60}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="CT Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sinogram reconstruction on sample_1
  python scripts/run_pipeline.py sinogram --sample sample_1 --epochs 50
  
  # Run classical reconstruction only
  python scripts/run_pipeline.py classical --sample sample_2
  
  # Run U-Net enhancement
  python scripts/run_pipeline.py unet --sample sample_1 --epochs 50
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Pipeline type")
    
    # Sinogram reconstruction command
    sinogram_parser = subparsers.add_parser("sinogram", help="Direct sinogram reconstruction")
    sinogram_parser.add_argument("--sample", required=True, choices=["sample_1", "sample_2"],
                                 help="Sample to process")
    sinogram_parser.add_argument("--epochs", type=int, default=50,
                                 help="Training epochs (default: 50)")
    
    # U-Net enhancement command
    unet_parser = subparsers.add_parser("unet", help="U-Net enhancement (degraded FBP -> enhanced)")
    unet_parser.add_argument("--sample", required=True, choices=["sample_1", "sample_2"],
                            help="Sample to process")
    unet_parser.add_argument("--epochs", type=int, default=50,
                            help="Training epochs (default: 50)")
    
    # Classical command
    classical_parser = subparsers.add_parser("classical", help="Classical FDK reconstruction only")
    classical_parser.add_argument("--sample", required=True, choices=["sample_1", "sample_2"],
                                   help="Sample to process")
    classical_parser.add_argument("--no-downsample", action="store_true",
                                   help="Skip downsampling for better quality (requires more GPU memory)")
    classical_parser.add_argument("--downsample-factor", type=int, default=None,
                                   help="Override downsample factor")

    # Enhance pipeline (full input -> full target)
    enhance_parser = subparsers.add_parser("enhance", help="Enhance FDK with full sinogram input")
    enhance_parser.add_argument("--sample", required=True, choices=["sample_1", "sample_2"],
                            help="Sample to process")
    enhance_parser.add_argument("--epochs", type=int, default=50,
                            help="Training epochs (default: 50)")

    args = parser.parse_args()

    if args.command == "sinogram":
        return run_sinogram_pipeline(args.sample, args.epochs)
    elif args.command == "unet":
        return run_unet_pipeline(args.sample, args.epochs)
    elif args.command == "enhance":
        return run_enhance_pipeline(args.sample, args.epochs)
    elif args.command == "classical":
        return run_classical(
            args.sample,
            skip_if_exists=False,
            no_downsample=args.no_downsample,
            downsample_factor_override=args.downsample_factor
        )
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
