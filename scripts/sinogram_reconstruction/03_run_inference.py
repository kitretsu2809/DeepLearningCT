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


def save_preview_with_windowing(volume: np.ndarray, output_path: str | Path, window_center: float = 0.14, window_width: float = 0.1) -> Path:
    """Save preview with CT windowing to show contrast."""
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2

    # CT windowing: clip to [center - width/2, center + width/2] then normalize
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    axial = np.clip(volume[z_mid], window_min, window_max)
    coronal = np.clip(volume[:, y_mid, :], window_min, window_max)
    sagittal = np.clip(volume[:, :, x_mid], window_min, window_max)
    
    # Normalize to [0, 1]
    axial = (axial - window_min) / window_width
    coronal = (coronal - window_min) / window_width
    sagittal = (sagittal - window_min) / window_width
    
    # Convert to 8-bit
    axial = (axial * 255).clip(0, 255).astype(np.uint8)
    coronal = (coronal * 255).clip(0, 255).astype(np.uint8)
    sagittal = (sagittal * 255).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(axial, cmap="gray")
    axes[0].set_title(f"Axial z={z_mid} (window: {window_center} ± {window_width/2})")
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

    parser = argparse.ArgumentParser(description="Reconstruct a slice volume directly from sparse sinograms with a trained DL model.")
    parser.add_argument("--checkpoint", default=str(OUTPUTS_DIR / "sparse_recon_training" / "best_model.pt"))
    parser.add_argument("--sample-dir", default=str(SAMPLE_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "sparse_recon_inference"))
    parser.add_argument("--sparse-step", type=int, default=None, help="Override sparse step from the checkpoint metadata.")
    args = parser.parse_args()

    checkpoint = torch.load(resolve_repo_path(args.checkpoint), map_location="cpu")
    metadata = checkpoint["metadata"]
    sparse_step = int(args.sparse_step or metadata["sparse_step"])

    sample_dir = resolve_repo_path(args.sample_dir)
    sample = load_sample(sample_dir)
    geometry = parse_geometry(sample_dir / "settings.cto")
    projections_ds = downsample_projection_stack(sample.projections, int(metadata["downsample_factor"]))
    attenuation = convert_to_attenuation(projections_ds)

    sparse_indices = np.arange(0, attenuation.shape[0], sparse_step, dtype=np.int32)
    
    # Compute valid row range from geometry (zmin/zmax) based on current sample
    # Not from training metadata which may be from combined dataset
    downsample_factor = int(metadata["downsample_factor"])
    row_start = max(0, int(geometry.zmin // downsample_factor))
    row_stop = min(attenuation.shape[1] - 1, int(geometry.zmax // downsample_factor))
    print(f"Processing rows {row_start} to {row_stop} (attenuation shape: {attenuation.shape})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseCTReconstructionModel(
        sparse_angle_count=len(sparse_indices),
        dense_angle_count=attenuation.shape[0],
        detector_count=int(metadata["detector_count"]),
        image_size=int(metadata["image_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recon_slices = []
    with torch.no_grad():
        for detector_row in range(row_start, row_stop + 1):
            dense_sinogram = attenuation[:, detector_row, :]
            sparse_sinogram = dense_sinogram[sparse_indices]
            sparse_sinogram_resized = resize_2d_array(
                sparse_sinogram,
                (len(sparse_indices), int(metadata["detector_count"])),
            )
            # Compute sinogram scale from data if metadata has placeholder value
            sinogram_scale = float(metadata.get("sinogram_scale", 1.0))
            if sinogram_scale == 1.0:
                # Compute from percentiles of the actual sinogram
                sinogram_scale = max(float(np.percentile(sparse_sinogram_resized, 99.5)), 1e-6)
            sparse_input = torch.from_numpy(
                np.clip(sparse_sinogram_resized / sinogram_scale, 0.0, None).astype(np.float32)
            )[None, None, :, :].to(device)
            prediction = model(sparse_input)["reconstruction"][0, 0].detach().cpu().numpy()
            recon_slices.append(prediction)

    recon_volume = np.stack(recon_slices, axis=0).astype(np.float32)
    
    # Denormalize: model outputs [0,1] from sigmoid
    # Map to physical range using metadata or auto-scale
    image_min = float(metadata.get("image_min", 0.0))
    image_max = float(metadata.get("image_max", 1.0))
    if image_max - image_min > 0.9:  # Placeholder range [0,1]
        # Auto-scale based on output statistics
        actual_min = float(recon_volume.min())
        actual_max = float(recon_volume.max())
        if actual_max > actual_min:
            # Stretch to typical CT attenuation range
            physical_volume = (recon_volume - actual_min) / (actual_max - actual_min) * 0.2 - 0.05
        else:
            physical_volume = recon_volume
    else:
        physical_volume = recon_volume * (image_max - image_min) + image_min

    volume_path = output_dir / "dl_sparse_reconstruction_volume.tif"
    preview_path = output_dir / "dl_sparse_reconstruction_preview.png"
    metadata_path = output_dir / "dl_sparse_reconstruction_info.json"
    tifffile.imwrite(volume_path, physical_volume)

    voxel_size_mm = geometry.detector_pixel_size_mm * int(metadata["downsample_factor"]) * (
        geometry.source_to_object_mm / geometry.source_to_detector_mm
    )
    save_preview_with_windowing(physical_volume, output_path=preview_path, window_center=float(physical_volume.mean()), window_width=float(physical_volume.std() * 4))
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "sample_dir": str(args.sample_dir),
                "sparse_step": sparse_step,
                "output_volume_shape": list(physical_volume.shape),
                "training_metadata": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(volume_path)


if __name__ == "__main__":
    main()
