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

from ct_recon.data_loader import load_sample
from ct_recon.geometry import parse_geometry
from ct_recon.paths import OUTPUTS_DIR, SAMPLE_DIR, resolve_repo_path
from ct_recon.reconstruct_fdk_astra import convert_to_attenuation, downsample_projection_stack, save_preview
from ct_recon.sparse_ct_reconstruction import _import_torch_or_exit, resize_2d_array


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
    row_start = int(metadata["row_start"])
    row_stop = int(metadata["row_stop"])

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
            sparse_input = torch.from_numpy(
                np.clip(sparse_sinogram_resized / float(metadata["sinogram_scale"]), 0.0, None).astype(np.float32)
            )[None, None, :, :].to(device)
            prediction = model(sparse_input)["reconstruction"][0, 0].detach().cpu().numpy()
            recon_slices.append(prediction)

    recon_volume = np.stack(recon_slices, axis=0).astype(np.float32)
    physical_volume = recon_volume * float(metadata["image_max"] - metadata["image_min"]) + float(metadata["image_min"])

    volume_path = output_dir / "dl_sparse_reconstruction_volume.tif"
    preview_path = output_dir / "dl_sparse_reconstruction_preview.png"
    metadata_path = output_dir / "dl_sparse_reconstruction_info.json"
    tifffile.imwrite(volume_path, physical_volume)

    voxel_size_mm = geometry.detector_pixel_size_mm * int(metadata["downsample_factor"]) * (
        geometry.source_to_object_mm / geometry.source_to_detector_mm
    )
    save_preview(physical_volume, voxel_size_mm=voxel_size_mm, output_path=preview_path)
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
