#!/usr/bin/env python3
"""Build dataset for enhance pipeline: full sinogram -> FDK reconstruction.

This pipeline trains a network to enhance FDK reconstruction using full projection data.
Input: Full sinogram (all projections)
Target: FDK reconstruction from the same projections
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_recon.data_loader import load_sample
from ct_recon.geometry import parse_geometry
from ct_recon.paths import OUTPUTS_DIR, SAMPLE_DIR, resolve_repo_path
from ct_recon.reconstruct_fdk_astra import convert_to_attenuation, downsample_projection_stack
from ct_recon.sparse_ct_reconstruction import resize_2d_array


class EnhanceDatasetMetadata:
    def __init__(self, dense_angle_count: int, detector_count: int, image_size: int,
                 downsample_factor: int, row_start: int, row_stop: int, slice_count: int,
                 sinogram_scale: float, image_min: float, image_max: float,
                 target_volume_path: str):
        self.dense_angle_count = dense_angle_count
        self.detector_count = detector_count
        self.image_size = image_size
        self.downsample_factor = downsample_factor
        self.row_start = row_start
        self.row_stop = row_stop
        self.slice_count = slice_count
        self.sinogram_scale = sinogram_scale
        self.image_min = image_min
        self.image_max = image_max
        self.target_volume_path = target_volume_path


def main():
    parser = argparse.ArgumentParser(description="Build full-sinogram to FDK training data for enhancement.")
    parser.add_argument("--sample-dir", default=str(SAMPLE_DIR))
    parser.add_argument("--target-volume", default=None, help="Path to FDK target volume.")
    parser.add_argument("--output-path", default=str(OUTPUTS_DIR / "enhance_dataset.npz"))
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--detector-count", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--slice-stride", type=int, default=1, help="Keep every Nth slice.")
    args = parser.parse_args()

    sample_dir = resolve_repo_path(args.sample_dir)
    sample = load_sample(sample_dir)
    geometry = parse_geometry(sample_dir / "settings.cto")

    if args.target_volume:
        target_volume_path = resolve_repo_path(args.target_volume)
    else:
        target_volume_path = None
        for candidate in OUTPUTS_DIR.glob("**/fdk_volume.tif"):
            if candidate.exists():
                target_volume_path = candidate
                break
    if not target_volume_path or not target_volume_path.exists():
        raise FileNotFoundError("No target volume. Run classical reconstruction first or pass --target-volume.")

    target_volume = tifffile.imread(target_volume_path).astype(np.float32)

    projections_ds = downsample_projection_stack(sample.projections, args.downsample_factor)
    attenuation = convert_to_attenuation(projections_ds)

    dense_angle_count = int(attenuation.shape[0])
    row_start = max(0, int(geometry.zmin // args.downsample_factor))
    row_stop = min(attenuation.shape[1] - 1, int(geometry.zmax // args.downsample_factor))

    actual_slices = target_volume.shape[0]
    expected_slices = row_stop - row_start + 1
    if actual_slices != expected_slices:
        usable_slices = min(actual_slices, expected_slices)
        row_stop = row_start + usable_slices - 1
        print(f"Adjusted to {usable_slices} slices")

    image_min = float(target_volume.min())
    image_max = float(target_volume.max())
    image_scale = max(image_max - image_min, 1e-6)
    sinogram_scale = float(np.percentile(attenuation, 99.5))
    sinogram_scale = max(sinogram_scale, 1e-6)

    selected_slice_indices = np.arange(0, target_volume.shape[0], args.slice_stride, dtype=np.int32)
    input_sinograms = []
    target_sinograms = []
    target_images = []

    for slice_idx in selected_slice_indices:
        detector_row = row_start + int(slice_idx)
        full_sinogram = attenuation[:, detector_row, :]

        full_sinogram_resized = resize_2d_array(full_sinogram, (dense_angle_count, args.detector_count))
        target_image = resize_2d_array(target_volume[slice_idx], (args.image_size, args.image_size))

        input_sinograms.append(np.clip(full_sinogram_resized / sinogram_scale, 0.0, None).astype(np.float32))
        target_sinograms.append(np.clip(full_sinogram_resized / sinogram_scale, 0.0, None).astype(np.float32))
        target_images.append(np.clip((target_image - image_min) / image_scale, 0.0, 1.0).astype(np.float32))

    output_path = resolve_repo_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = EnhanceDatasetMetadata(
        dense_angle_count=dense_angle_count,
        detector_count=args.detector_count,
        image_size=args.image_size,
        downsample_factor=args.downsample_factor,
        row_start=row_start,
        row_stop=row_stop,
        slice_count=len(selected_slice_indices),
        sinogram_scale=sinogram_scale,
        image_min=image_min,
        image_max=image_max,
        target_volume_path=str(target_volume_path),
    )

    np.savez_compressed(
        output_path,
        input_sinograms=np.stack(input_sinograms, axis=0).astype(np.float32),
        target_sinograms=np.stack(target_sinograms, axis=0).astype(np.float32),
        target_images=np.stack(target_images, axis=0).astype(np.float32),
        selected_slice_indices=selected_slice_indices.astype(np.int32),
        metadata_json=json.dumps(metadata.__dict__),
    )
    print(output_path)


if __name__ == "__main__":
    main()