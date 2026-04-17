from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from .data_loader import CTScanData, load_sample1
from .geometry import CTGeometry, geometry_for_projection_count, parse_geometry
from .paths import OUTPUTS_DIR, SAMPLE_DIR, resolve_repo_path


def construct_sinogram(projections: np.ndarray, detector_row: int) -> np.ndarray:
    num_rows = projections.shape[1]
    if detector_row < 0 or detector_row >= num_rows:
        raise IndexError(f"detector_row {detector_row} is out of range for shape {projections.shape}")
    return projections[:, detector_row, :].T


def build_detector_positions(num_detectors: int, pixel_size_mm: float, center_of_rotation_px: float) -> np.ndarray:
    detector_center = (num_detectors - 1) / 2.0
    offset_px = center_of_rotation_px - detector_center
    indices = np.arange(num_detectors, dtype=np.float32)
    return (indices - detector_center - offset_px) * pixel_size_mm


def ramp_filter_sinogram(sinogram: np.ndarray, detector_spacing_mm: float) -> np.ndarray:
    num_detectors = sinogram.shape[0]
    frequencies = np.fft.fftfreq(num_detectors, d=detector_spacing_mm).astype(np.float32)
    ramp = np.abs(frequencies)
    spectrum = np.fft.fft(sinogram, axis=0)
    filtered = np.fft.ifft(spectrum * ramp[:, None], axis=0).real
    return filtered.astype(np.float32)


def fbp_reconstruct_slice(
    sinogram: np.ndarray,
    angles_rad: np.ndarray,
    detector_positions_mm: np.ndarray,
    image_size: int = 512,
) -> np.ndarray:
    if sinogram.shape[1] != len(angles_rad):
        raise ValueError("sinogram angle dimension does not match number of angles")

    filtered = ramp_filter_sinogram(
        sinogram=sinogram,
        detector_spacing_mm=float(np.abs(detector_positions_mm[1] - detector_positions_mm[0])),
    )

    extent_mm = float(np.max(np.abs(detector_positions_mm)))
    coords = np.linspace(-extent_mm, extent_mm, image_size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")

    reconstruction = np.zeros((image_size, image_size), dtype=np.float32)

    for angle_index, theta in enumerate(angles_rad):
        detector_coordinate = xx * np.cos(theta) + yy * np.sin(theta)
        sampled = np.interp(
            detector_coordinate.ravel(),
            detector_positions_mm,
            filtered[:, angle_index],
            left=0.0,
            right=0.0,
        )
        reconstruction += sampled.reshape(image_size, image_size)

    reconstruction *= np.pi / max(len(angles_rad), 1)
    return reconstruction


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(image.min())
    max_value = float(image.max())
    if np.isclose(max_value, min_value):
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def reconstruct_rows(
    data: CTScanData,
    geometry: CTGeometry,
    detector_rows: Iterable[int],
    image_size: int = 512,
) -> tuple[np.ndarray, list[int]]:
    detector_positions_mm = build_detector_positions(
        num_detectors=geometry.detector_cols,
        pixel_size_mm=geometry.detector_pixel_size_mm,
        center_of_rotation_px=geometry.center_of_rotation_px,
    )

    reconstructed_rows = []
    used_rows: list[int] = []

    for row in detector_rows:
        sinogram = construct_sinogram(data.projections, row)
        image = fbp_reconstruct_slice(
            sinogram=sinogram,
            angles_rad=geometry.angles_rad,
            detector_positions_mm=detector_positions_mm,
            image_size=image_size,
        )
        reconstructed_rows.append(image)
        used_rows.append(row)

    volume = np.stack(reconstructed_rows, axis=0)
    return volume, used_rows


def save_volume_as_tiff_stack(volume: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), volume.astype(np.float32))
    return output_path


def save_preview(volume: np.ndarray, output_path: str | Path) -> Path:
    raise NotImplementedError("Use save_preview_with_geometry instead.")


def detector_row_to_z_mm(row: int, geometry: CTGeometry) -> float:
    return (row - geometry.vertical_center_px) * geometry.detector_pixel_size_mm


def save_preview_with_geometry(
    volume: np.ndarray,
    used_rows: list[int],
    geometry: CTGeometry,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not used_rows:
        raise ValueError("used_rows must not be empty")

    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2

    axial = normalize_image(volume[z_mid])
    coronal = normalize_image(volume[:, y_mid, :])
    sagittal = normalize_image(volume[:, :, x_mid])

    xy_extent_mm = (volume.shape[1] / 2.0) * geometry.detector_pixel_size_mm
    z_positions_mm = np.array([detector_row_to_z_mm(row, geometry) for row in used_rows], dtype=np.float32)
    z_min_mm = float(z_positions_mm.min() - geometry.detector_pixel_size_mm / 2.0)
    z_max_mm = float(z_positions_mm.max() + geometry.detector_pixel_size_mm / 2.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        axial,
        cmap="gray",
        extent=[-xy_extent_mm, xy_extent_mm, xy_extent_mm, -xy_extent_mm],
        aspect="equal",
    )
    axes[0].set_title(f"Axial z-slice index={z_mid}")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")

    axes[1].imshow(
        coronal,
        cmap="gray",
        extent=[-xy_extent_mm, xy_extent_mm, z_max_mm, z_min_mm],
        aspect="equal",
    )
    axes[1].set_title(f"Coronal y={y_mid}")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("z (mm)")

    axes[2].imshow(
        sagittal,
        cmap="gray",
        extent=[-xy_extent_mm, xy_extent_mm, z_max_mm, z_min_mm],
        aspect="equal",
    )
    axes[2].set_title(f"Sagittal x={x_mid}")
    axes[2].set_xlabel("y (mm)")
    axes[2].set_ylabel("z (mm)")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def default_row_subset(geometry: CTGeometry, count: int = 64) -> list[int]:
    start = max(geometry.zmin, 0)
    stop = min(geometry.zmax, geometry.detector_rows - 1)
    available = stop - start + 1
    if count >= available:
        return list(range(start, stop + 1))

    center = int(round((start + stop) / 2))
    half = count // 2
    subset_start = center - half
    subset_stop = subset_start + count - 1

    if subset_start < start:
        subset_start = start
        subset_stop = start + count - 1
    if subset_stop > stop:
        subset_stop = stop
        subset_start = stop - count + 1

    return list(range(subset_start, subset_stop + 1))


def run_baseline_reconstruction(
    sample_dir: str | Path = SAMPLE_DIR,
    image_size: int = 512,
    num_rows: int = 64,
    output_dir: str | Path = OUTPUTS_DIR / "fbp_baseline",
) -> dict[str, Path]:
    sample_dir = resolve_repo_path(sample_dir)
    output_dir = resolve_repo_path(output_dir)

    data = load_sample1(sample_dir)
    geometry = parse_geometry(sample_dir / "settings.cto")
    geometry = geometry_for_projection_count(geometry, data.projections.shape[0])
    row_subset = default_row_subset(geometry, count=num_rows)

    volume, used_rows = reconstruct_rows(
        data=data,
        geometry=geometry,
        detector_rows=row_subset,
        image_size=image_size,
    )

    volume_path = save_volume_as_tiff_stack(volume, output_dir / "fbp_volume.tif")
    preview_path = save_preview_with_geometry(volume, used_rows, geometry, output_dir / "fbp_preview.png")
    rows_path = output_dir / "reconstructed_rows.txt"
    rows_path.write_text("\n".join(str(row) for row in used_rows), encoding="utf-8")

    return {
        "volume_path": volume_path,
        "preview_path": preview_path,
        "rows_path": rows_path,
    }


if __name__ == "__main__":
    outputs = run_baseline_reconstruction()
    for key, value in outputs.items():
        print(f"{key}: {value}")
