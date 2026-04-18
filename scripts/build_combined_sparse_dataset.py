#!/usr/bin/env python3
"""Build sparse sinogram datasets for multiple samples and combine them."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_recon.data_loader import load_sample1, load_sample2
from ct_recon.geometry import parse_geometry
from ct_recon.paths import OUTPUTS_DIR, set_sample, resolve_repo_path
from ct_recon.reconstruct_fdk_astra import convert_to_attenuation, downsample_projection_stack
from ct_recon.sparse_ct_reconstruction import SparseSinogramDatasetMetadata, resize_2d_array


def build_single_dataset(
    sample_name: str,
    sample_dir: Path,
    target_volume_path: Path | None,
    downsample_factor: int,
    sparse_step: int,
    detector_count: int,
    image_size: int,
    slice_stride: int = 1,
):
    """Build dataset for a single sample."""
    print(f"\n=== Building dataset for {sample_name} ===")
    
    if sample_name == "sample_1":
        data = load_sample1()
    else:
        set_sample(sample_name)
        data = load_sample2()
    
    geometry = parse_geometry(sample_dir / "settings.cto")
    
    if target_volume_path is None:
        target_volume_path = OUTPUTS_DIR / f"{sample_name}_fdk_ds{downsample_factor}" / "fdk_volume.tif"
    
    if not target_volume_path.exists():
        raise FileNotFoundError(f"Target volume not found: {target_volume_path}")
    
    target_volume = tifffile.imread(target_volume_path).astype(np.float32)
    print(f"Target volume shape: {target_volume.shape}")
    
    projections_ds = downsample_projection_stack(data.projections, downsample_factor)
    attenuation = convert_to_attenuation(projections_ds)
    print(f"Attenuation shape: {attenuation.shape}")
    
    dense_angle_count = int(attenuation.shape[0])
    sparse_indices = np.arange(0, dense_angle_count, sparse_step, dtype=np.int32)
    
    row_start = 0
    row_stop = min(attenuation.shape[1] - 1, target_volume.shape[0] - 1)
    
    image_min = float(target_volume.min())
    image_max = float(target_volume.max())
    image_scale = max(image_max - image_min, 1e-6)
    sinogram_scale = float(np.percentile(attenuation, 99.5))
    sinogram_scale = max(sinogram_scale, 1e-6)
    
    selected_slice_indices = np.arange(0, target_volume.shape[0], slice_stride, dtype=np.int32)
    
    input_sinograms = []
    target_sinograms = []
    target_images = []
    
    for slice_idx in selected_slice_indices:
        if slice_idx >= target_volume.shape[0]:
            break
        
        dense_sinogram = attenuation[:, slice_idx, :] if slice_idx < attenuation.shape[1] else np.zeros((dense_angle_count, attenuation.shape[2]))
        sparse_sinogram = dense_sinogram[sparse_indices]
        
        dense_sinogram_resized = resize_2d_array(dense_sinogram, (dense_angle_count, detector_count))
        sparse_sinogram_resized = resize_2d_array(sparse_sinogram, (len(sparse_indices), detector_count))
        target_image = resize_2d_array(target_volume[slice_idx], (image_size, image_size))
        
        input_sinograms.append(np.clip(sparse_sinogram_resized / sinogram_scale, 0.0, None).astype(np.float32))
        target_sinograms.append(np.clip(dense_sinogram_resized / sinogram_scale, 0.0, None).astype(np.float32))
        target_images.append(np.clip((target_image - image_min) / image_scale, 0.0, 1.0).astype(np.float32))
    
    metadata = SparseSinogramDatasetMetadata(
        sparse_step=sparse_step,
        dense_angle_count=dense_angle_count,
        sparse_angle_count=len(sparse_indices),
        detector_count=detector_count,
        image_size=image_size,
        downsample_factor=downsample_factor,
        row_start=row_start,
        row_stop=row_stop,
        slice_count=len(selected_slice_indices),
        sinogram_scale=sinogram_scale,
        image_min=image_min,
        image_max=image_max,
        target_volume_path=str(target_volume_path),
        sample_name=sample_name,
    )
    
    return (
        np.stack(input_sinograms, axis=0).astype(np.float32),
        np.stack(target_sinograms, axis=0).astype(np.float32),
        np.stack(target_images, axis=0).astype(np.float32),
        selected_slice_indices,
        sparse_indices,
        metadata,
    )


def main():
    parser = argparse.ArgumentParser(description="Build combined sparse sinogram dataset from multiple samples.")
    
    # Sample 1 options
    parser.add_argument("--sample1-target", default=None, help="Target volume path for sample_1")
    parser.add_argument("--skip-sample1", action="store_true", help="Skip sample_1")
    
    # Sample 2 options
    parser.add_argument("--sample2-target", default=None, help="Target volume path for sample_2")
    parser.add_argument("--skip-sample2", action="store_true", help="Skip sample_2")
    
    # Common options
    parser.add_argument("--output-path", default=str(OUTPUTS_DIR / "combined_sparse_dataset.npz"))
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--sparse-step", type=int, default=4)
    parser.add_argument("--detector-count", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--slice-stride", type=int, default=1)
    
    args = parser.parse_args()
    
    all_input_sinograms = []
    all_target_sinograms = []
    all_target_images = []
    samples_metadata = []
    
    # Build sample 1 dataset
    if not args.skip_sample1:
        from ct_recon.paths import SAMPLE_DIRS
        inp, tgt_sin, tgt_img, sel_idx, sparse_idx, meta = build_single_dataset(
            "sample_1",
            SAMPLE_DIRS["sample_1"],
            resolve_repo_path(args.sample1_target) if args.sample1_target else None,
            args.downsample_factor,
            args.sparse_step,
            args.detector_count,
            args.image_size,
            args.slice_stride,
        )
        all_input_sinograms.append(inp)
        all_target_sinograms.append(tgt_sin)
        all_target_images.append(tgt_img)
        samples_metadata.append({"sample": "sample_1", "slices": len(inp), "metadata": meta.__dict__})
        print(f"Sample 1: added {len(inp)} slices")
    
    # Build sample 2 dataset
    if not args.skip_sample2:
        from ct_recon.paths import SAMPLE_DIRS
        inp, tgt_sin, tgt_img, sel_idx, sparse_idx, meta = build_single_dataset(
            "sample_2",
            SAMPLE_DIRS["sample_2"],
            resolve_repo_path(args.sample2_target) if args.sample2_target else None,
            args.downsample_factor,
            args.sparse_step,
            args.detector_count,
            args.image_size,
            args.slice_stride,
        )
        all_input_sinograms.append(inp)
        all_target_sinograms.append(tgt_sin)
        all_target_images.append(tgt_img)
        samples_metadata.append({"sample": "sample_2", "slices": len(inp), "metadata": meta.__dict__})
        print(f"Sample 2: added {len(inp)} slices")
    
    if not all_input_sinograms:
        raise ValueError("No samples selected. Use --skip-sample1 or --skip-sample2 to skip.")
    
    # Concatenate all data
    combined_input = np.concatenate(all_input_sinograms, axis=0)
    combined_target_sin = np.concatenate(all_target_sinograms, axis=0)
    combined_target_img = np.concatenate(all_target_images, axis=0)
    
    print(f"\n=== Combined dataset ===")
    print(f"Total slices: {len(combined_input)}")
    print(f"Input sinograms shape: {combined_input.shape}")
    print(f"Target images shape: {combined_target_img.shape}")
    
    # Use metadata from first sample (they should be compatible)
    combined_metadata = SparseSinogramDatasetMetadata(
        sparse_step=args.sparse_step,
        dense_angle_count=all_target_sinograms[0].shape[1],
        sparse_angle_count=all_input_sinograms[0].shape[1],
        detector_count=args.detector_count,
        image_size=args.image_size,
        downsample_factor=args.downsample_factor,
        row_start=0,
        row_stop=combined_input.shape[0] - 1,
        slice_count=combined_input.shape[0],
        sinogram_scale=1.0,
        image_min=0.0,
        image_max=1.0,
        target_volume_path="combined",
        sample_name="combined",
    )
    
    output_path = resolve_repo_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        input_sinograms=combined_input,
        target_sinograms=combined_target_sin,
        target_images=combined_target_img,
        metadata_json=json.dumps(combined_metadata.__dict__),
        samples_json=json.dumps(samples_metadata),
    )
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
