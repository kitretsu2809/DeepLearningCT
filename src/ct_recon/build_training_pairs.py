from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from .paths import OUTPUTS_DIR, SAMPLE_DIR, resolve_repo_path
from .reconstruct_fdk_astra import reconstruct_volume_from_projection_dataset, save_reconstruction_outputs
from .simulate_degradation import (
    DegradedProjectionData,
    build_default_degradation_sets,
    make_full_projection_dataset,
)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    min_value = float(volume.min())
    max_value = float(volume.max())
    if np.isclose(min_value, max_value):
        return np.zeros_like(volume, dtype=np.float32)
    return (volume - min_value) / (max_value - min_value)


def center_crop_to_match(source: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    z, y, x = source.shape
    tz, ty, tx = target_shape
    if tz > z or ty > y or tx > x:
        raise ValueError(f"target_shape {target_shape} cannot exceed source shape {source.shape}")

    z0 = (z - tz) // 2
    y0 = (y - ty) // 2
    x0 = (x - tx) // 2
    return source[z0 : z0 + tz, y0 : y0 + ty, x0 : x0 + tx]


def align_pair_shapes(input_volume: np.ndarray, target_volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shared_shape = tuple(min(a, b) for a, b in zip(input_volume.shape, target_volume.shape))
    return (
        center_crop_to_match(input_volume, shared_shape),
        center_crop_to_match(target_volume, shared_shape),
    )


def save_pair_outputs(
    name: str,
    input_volume: np.ndarray,
    target_volume: np.ndarray,
    output_dir: str | Path,
    metadata: dict,
) -> dict[str, Path]:
    output_dir = resolve_repo_path(output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_dir / "input_volume.tif"
    target_path = output_dir / "target_volume.tif"
    slices_path = output_dir / "axial_slices.npz"
    metadata_path = output_dir / "pair_metadata.json"

    tifffile.imwrite(input_path, input_volume.astype(np.float32))
    tifffile.imwrite(target_path, target_volume.astype(np.float32))
    np.savez_compressed(
        slices_path,
        input_slices=normalize_volume(input_volume).astype(np.float32),
        target_slices=normalize_volume(target_volume).astype(np.float32),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "pair_dir": output_dir,
        "input_path": input_path,
        "target_path": target_path,
        "slices_path": slices_path,
        "metadata_path": metadata_path,
    }


def build_pair_for_dataset(
    degraded_dataset: DegradedProjectionData,
    sample_dir: str | Path = SAMPLE_DIR,
    downsample_factor: int = 2,
    output_dir: str | Path = OUTPUTS_DIR / "training_pairs",
    target_volume: np.ndarray | None = None,
    target_info: dict | None = None,
) -> dict[str, Path]:
    if target_volume is None or target_info is None:
        full_dataset = make_full_projection_dataset(sample_dir)
        target_volume, target_info = reconstruct_volume_from_projection_dataset(
            dataset=full_dataset,
            sample_dir=sample_dir,
            downsample_factor=downsample_factor,
        )

    input_volume, input_info = reconstruct_volume_from_projection_dataset(
        dataset=degraded_dataset,
        sample_dir=sample_dir,
        downsample_factor=downsample_factor,
    )
    input_volume, target_volume_aligned = align_pair_shapes(input_volume, target_volume)

    pair_metadata = {
        "name": degraded_dataset.name,
        "downsample_factor": int(downsample_factor),
        "degraded_metadata": degraded_dataset.metadata,
        "input_info": input_info,
        "target_info": target_info,
        "aligned_shape": list(input_volume.shape),
    }

    outputs = save_pair_outputs(
        name=degraded_dataset.name,
        input_volume=input_volume,
        target_volume=target_volume_aligned,
        output_dir=output_dir,
        metadata=pair_metadata,
    )

    save_reconstruction_outputs(
        input_volume,
        input_info | {"output_volume_shape": tuple(int(v) for v in input_volume.shape)},
        output_dir=outputs["pair_dir"],
        prefix="input_fdk",
    )
    save_reconstruction_outputs(
        target_volume_aligned,
        target_info | {"output_volume_shape": tuple(int(v) for v in target_volume_aligned.shape)},
        output_dir=outputs["pair_dir"],
        prefix="target_fdk",
    )
    return outputs


def build_default_training_pairs(
    sample_dir: str | Path = SAMPLE_DIR,
    downsample_factor: int = 2,
    output_dir: str | Path = OUTPUTS_DIR / "training_pairs",
) -> list[dict[str, Path]]:
    full_dataset = make_full_projection_dataset(sample_dir)
    target_volume, target_info = reconstruct_volume_from_projection_dataset(
        dataset=full_dataset,
        sample_dir=sample_dir,
        downsample_factor=downsample_factor,
    )

    outputs = []
    for dataset in build_default_degradation_sets(sample_dir=sample_dir):
        outputs.append(
            build_pair_for_dataset(
                degraded_dataset=dataset,
                sample_dir=sample_dir,
                downsample_factor=downsample_factor,
                output_dir=output_dir,
                target_volume=target_volume,
                target_info=target_info,
            )
        )
    return outputs


if __name__ == "__main__":
    outputs = build_default_training_pairs()
    for item in outputs:
        print(item["pair_dir"])
