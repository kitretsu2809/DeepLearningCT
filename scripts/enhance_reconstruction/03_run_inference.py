#!/usr/bin/env python3
"""Run inference with enhancement model: full sinogram -> enhanced FDK."""
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
from ct_recon.sparse_ct_reconstruction import _import_torch_or_exit, resize_2d_array


def save_preview_with_windowing(volume: np.ndarray, output_path: Path,
                         window_center: float = None, window_width: float = None) -> Path:
    """Save preview with CT windowing."""
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2

    if window_center is None:
        window_center = float(volume.mean())
    if window_width is None:
        window_width = float(volume.std() * 4)

    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2

    axial = np.clip(volume[z_mid], window_min, window_max)
    coronal = np.clip(volume[:, y_mid, :], window_min, window_max)
    sagittal = np.clip(volume[:, :, x_mid], window_min, window_max)

    axial = (axial - window_min) / window_width
    coronal = (coronal - window_min) / window_width
    sagittal = (sagittal - window_min) / window_width

    axial = (axial * 255).clip(0, 255).astype(np.uint8)
    coronal = (coronal * 255).clip(0, 255).astype(np.uint8)
    sagittal = (sagittal * 255).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(axial, cmap="gray")
    axes[0].set_title(f"Axial z={z_mid}")
    axes[0].axis("off")
    axes[1].imshow(coronal, cmap="gray")
    axes[1].set_title(f"Coronal y={y_mid}")
    axes[1].axis("off")
    axes[2].imshow(sagittal, cmap="gray")
    axes[2].set_title(f"Sagittal x={x_mid}")
    axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    torch, _, _ = _import_torch_or_exit()
    from ct_recon.sparse_ct_reconstruction import SparseCTReconstructionModel

    parser = argparse.ArgumentParser(description="Enhance FDK with full sinogram input.")
    parser.add_argument("--checkpoint", default=str(OUTPUTS_DIR / "enhance_training" / "best_model.pt"))
    parser.add_argument("--sample-dir", default=str(SAMPLE_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "enhance_inference"))
    args = parser.parse_args()

    checkpoint = torch.load(resolve_repo_path(args.checkpoint), map_location="cpu")
    metadata = checkpoint["metadata"]

    sample_dir = resolve_repo_path(args.sample_dir)
    sample = load_sample(sample_dir)
    geometry = parse_geometry(sample_dir / "settings.cto")
    projections_ds = downsample_projection_stack(sample.projections, int(metadata["downsample_factor"]))
    attenuation = convert_to_attenuation(projections_ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseCTReconstructionModel(
        sparse_angle_count=int(metadata["dense_angle_count"]),
        dense_angle_count=int(metadata["dense_angle_count"]),
        detector_count=int(metadata["detector_count"]),
        image_size=int(metadata["image_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downsample_factor = int(metadata["downsample_factor"])
    row_start = max(0, int(geometry.zmin // downsample_factor))
    row_stop = min(attenuation.shape[1] - 1, int(geometry.zmax // downsample_factor))
    print(f"Processing rows {row_start} to {row_stop}")

    sinogram_scale = float(metadata.get("sinogram_scale", 1.0))
    if sinogram_scale == 1.0:
        sinogram_scale = max(float(np.percentile(attenuation, 99.5)), 1e-6)

    recon_slices = []
    with torch.no_grad():
        for detector_row in range(row_start, row_stop + 1):
            full_sinogram = attenuation[:, detector_row, :]
            full_sinogram_resized = resize_2d_array(
                full_sinogram,
                (int(metadata["dense_angle_count"]), int(metadata["detector_count"])),
            )
            sparse_input = torch.from_numpy(
                np.clip(full_sinogram_resized / sinogram_scale, 0.0, None).astype(np.float32)
            )[None, None, :, :].to(device)
            prediction = model(sparse_input)["reconstruction"][0, 0].cpu().numpy()
            recon_slices.append(prediction)

    recon_volume = np.stack(recon_slices, axis=0).astype(np.float32)

    image_min = float(metadata.get("image_min", 0.0))
    image_max = float(metadata.get("image_max", 1.0))
    if image_max - image_min > 0.9:
        actual_min = float(recon_volume.min())
        actual_max = float(recon_volume.max())
        if actual_max > actual_min:
            physical_volume = (recon_volume - actual_min) / (actual_max - actual_min) * 0.2 - 0.05
        else:
            physical_volume = recon_volume
    else:
        physical_volume = recon_volume * (image_max - image_min) + image_min

    volume_path = output_dir / "enhance_volume.tif"
    preview_path = output_dir / "enhance_preview.png"
    metadata_path = output_dir / "enhance_info.json"

    tifffile.imwrite(volume_path, physical_volume)
    save_preview_with_windowing(physical_volume, preview_path)
    metadata_path.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "sample_dir": str(args.sample_dir),
        "output_volume_shape": list(physical_volume.shape),
        "training_metadata": metadata,
    }, indent=2))

    print(volume_path)


if __name__ == "__main__":
    main()