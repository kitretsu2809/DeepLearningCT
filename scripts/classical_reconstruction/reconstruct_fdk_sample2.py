#!/usr/bin/env python3
"""Reconstruct sample_2 with downsampling for practical processing."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_recon.data_loader import load_sample2
from ct_recon.geometry import CTGeometry, parse_geometry
from ct_recon.paths import set_sample, OUTPUTS_DIR
from ct_recon.reconstruct_fdk_astra import (
    build_cone_geometry,
    build_volume_geometry,
    convert_to_attenuation,
    downsample_projection_stack,
    normalize_image,
    save_preview,
    run_fdk_reconstruction,
)
import astra
import numpy as np
import tifffile


def run_sample2_reconstruction(downsample_factor: int = 4, max_projections: int | None = None):
    """Run FDK reconstruction for sample_2."""
    set_sample("sample_2")
    
    print("Loading sample_2 data...")
    data = load_sample2()
    print(f"Original projections shape: {data.projections.shape}")
    
    # Parse geometry
    geometry = parse_geometry(data.settings_path)
    print(f"Original detector size: {geometry.detector_rows}x{geometry.detector_cols}")
    
    # Downsample projections for practical processing
    ds = downsample_factor
    projections = downsample_projection_stack(data.projections, ds)
    print(f"Downsampled projections shape: {projections.shape}")
    
    # Convert to attenuation
    projections = convert_to_attenuation(projections)
    
    # Optionally limit projections for sparse-view reconstruction
    if max_projections and max_projections < projections.shape[0]:
        step = projections.shape[0] // max_projections
        projections = projections[::step]
        print(f"Using {projections.shape[0]} sparse-view projections")
    
    # Build ASTRA geometry
    angles_rad = np.linspace(0, 2*np.pi, projections.shape[0], endpoint=False).astype(np.float32)
    detector_rows = projections.shape[1]
    detector_cols = projections.shape[2]
    detector_pixel_mm = geometry.detector_pixel_size_mm * ds
    
    proj_geom = astra.create_proj_geom(
        "cone",
        detector_pixel_mm,
        detector_pixel_mm,
        detector_rows,
        detector_cols,
        angles_rad,
        geometry.source_to_object_mm,
        geometry.source_to_detector_mm - geometry.source_to_object_mm,
    )
    
    vol_geom = build_volume_geometry(detector_rows, detector_cols, detector_pixel_mm)
    
    # Create sinogram
    # ASTRA expects sinogram in (detector_rows, angles, detector_cols) order
    projections = np.transpose(projections, (1, 0, 2))
    print("Creating sinogram...")
    sinogram_id = astra.data3d.create("-sino", proj_geom, projections)
    
    # Reconstruct
    print("Running FDK reconstruction...")
    rec_id = astra.data3d.create("-vol", vol_geom)
    
    cfg = astra.astra_dict("FDK_CUDA")
    cfg["ProjectionDataId"] = sinogram_id
    cfg["ReconstructionDataId"] = rec_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    volume = astra.data3d.get(rec_id)
    
    # Cleanup
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sinogram_id)
    
    # Normalize and save
    volume_norm = normalize_image(volume)
    
    output_dir = OUTPUTS_DIR / f"sample_2_fdk_ds{ds}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    volume_path = output_dir / "fdk_volume.tif"
    tifffile.imwrite(volume_path, volume_norm.astype(np.float32))
    print(f"Saved volume to {volume_path}")
    
    preview_path = save_preview(volume_norm, detector_pixel_mm, output_dir / "preview.png")
    print(f"Saved preview to {preview_path}")
    
    return {"volume_path": volume_path, "volume_shape": volume.shape}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reconstruct sample_2")
    parser.add_argument("--downsample", "-d", type=int, default=4, help="Downsample factor (default: 4)")
    parser.add_argument("--max-projections", "-p", type=int, default=None, help="Max projections for sparse view")
    
    args = parser.parse_args()
    result = run_sample2_reconstruction(args.downsample, args.max_projections)
    print(result)
