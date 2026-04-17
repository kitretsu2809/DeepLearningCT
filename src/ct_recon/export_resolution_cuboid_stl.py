from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .paths import OUTPUTS_DIR, resolve_repo_path


def mm_to_voxels(length_mm: float, voxel_mm: float, minimum: int = 1) -> int:
    return max(minimum, int(round(length_mm / voxel_mm)))


def carve_internal_resolution_block(
    occupancy: np.ndarray,
    voxel_mm: float,
    feature_sizes_mm: list[float],
    block_origin_mm: tuple[float, float, float],
    block_size_mm: tuple[float, float, float],
    orientation: str = "x",
) -> list[dict]:
    """
    Carve fully enclosed alternating void lamellae inside a chamber.
    The chamber itself remains surrounded by solid shell on all sides.
    """
    z0_mm, y0_mm, x0_mm = block_origin_mm
    z_size_mm, y_size_mm, x_size_mm = block_size_mm

    z0 = mm_to_voxels(z0_mm, voxel_mm)
    y0 = mm_to_voxels(y0_mm, voxel_mm)
    x0 = mm_to_voxels(x0_mm, voxel_mm)
    z1 = min(occupancy.shape[0] - 1, z0 + mm_to_voxels(z_size_mm, voxel_mm))
    y1 = min(occupancy.shape[1] - 1, y0 + mm_to_voxels(y_size_mm, voxel_mm))
    x1 = min(occupancy.shape[2] - 1, x0 + mm_to_voxels(x_size_mm, voxel_mm))

    shell = mm_to_voxels(1.0, voxel_mm)
    iz0, iz1 = z0 + shell, z1 - shell
    iy0, iy1 = y0 + shell, y1 - shell
    ix0, ix1 = x0 + shell, x1 - shell
    if iz1 <= iz0 or iy1 <= iy0 or ix1 <= ix0:
        return []

    cursor = ix0 if orientation == "x" else iy0
    end = ix1 if orientation == "x" else iy1
    feature_meta: list[dict] = []

    for idx, size_mm in enumerate(feature_sizes_mm):
        lamella = mm_to_voxels(size_mm, voxel_mm)
        period = 2 * lamella
        if cursor + period > end:
            break

        if orientation == "x":
            occupancy[iz0:iz1, iy0:iy1, cursor : cursor + lamella] = False
            feature_meta.append(
                {
                    "feature_index": idx,
                    "size_mm": size_mm,
                    "orientation": orientation,
                    "void_start_mm": cursor * voxel_mm,
                    "void_end_mm": (cursor + lamella) * voxel_mm,
                }
            )
        else:
            occupancy[iz0:iz1, cursor : cursor + lamella, ix0:ix1] = False
            feature_meta.append(
                {
                    "feature_index": idx,
                    "size_mm": size_mm,
                    "orientation": orientation,
                    "void_start_mm": cursor * voxel_mm,
                    "void_end_mm": (cursor + lamella) * voxel_mm,
                }
            )
        cursor += period

    return feature_meta


def generate_resolution_cuboid_voxels(
    size_x_mm: float = 80.0,
    size_y_mm: float = 60.0,
    size_z_mm: float = 24.0,
    voxel_mm: float = 0.25,
) -> tuple[np.ndarray, dict]:
    x_count = int(round(size_x_mm / voxel_mm))
    y_count = int(round(size_y_mm / voxel_mm))
    z_count = int(round(size_z_mm / voxel_mm))

    occupancy = np.ones((z_count, y_count, x_count), dtype=bool)

    # Fully enclosed internal feature chambers with alternating bar-space patterns.
    sizes_x = [3.0, 2.5, 2.0, 1.6, 1.2, 1.0, 0.8, 0.6]
    sizes_y = [2.8, 2.3, 1.9, 1.5, 1.1, 0.9, 0.7, 0.5]

    chamber_1 = carve_internal_resolution_block(
        occupancy=occupancy,
        voxel_mm=voxel_mm,
        feature_sizes_mm=sizes_x,
        block_origin_mm=(4.0, 6.0, 8.0),
        block_size_mm=(14.0, 14.0, 64.0),
        orientation="x",
    )
    chamber_2 = carve_internal_resolution_block(
        occupancy=occupancy,
        voxel_mm=voxel_mm,
        feature_sizes_mm=sizes_y,
        block_origin_mm=(4.0, 26.0, 8.0),
        block_size_mm=(14.0, 28.0, 64.0),
        orientation="y",
    )

    meta = {
        "model_type": "resolution_cuboid_with_enclosed_internal_bar_space_patterns",
        "base_size_mm": {"x": size_x_mm, "y": size_y_mm, "z": size_z_mm},
        "voxel_mm": voxel_mm,
        "feature_sizes_x_mm": sizes_x,
        "feature_sizes_y_mm": sizes_y,
        "outer_shell_visible": False,
        "internal_chambers": [
            {
                "name": "chamber_x",
                "origin_mm": {"z": 4.0, "y": 6.0, "x": 8.0},
                "size_mm": {"z": 14.0, "y": 14.0, "x": 64.0},
                "orientation": "x",
                "features": chamber_1,
            },
            {
                "name": "chamber_y",
                "origin_mm": {"z": 4.0, "y": 26.0, "x": 8.0},
                "size_mm": {"z": 14.0, "y": 28.0, "x": 64.0},
                "orientation": "y",
                "features": chamber_2,
            },
        ],
    }
    return occupancy, meta


def append_face(tris: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]], a, b, c, d):
    tris.append((a, b, c))
    tris.append((a, c, d))


def voxel_surface_triangles(occupancy: np.ndarray, voxel_mm: float):
    z_count, y_count, x_count = occupancy.shape
    tris: list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = []

    occ = occupancy

    # +X faces
    mask = occ & ~np.pad(occ[:, :, 1:], ((0, 0), (0, 0), (0, 1)), constant_values=False)
    for z, y, x in np.argwhere(mask):
        xw = (x + 1) * voxel_mm
        y0 = y * voxel_mm
        y1 = (y + 1) * voxel_mm
        z0 = z * voxel_mm
        z1 = (z + 1) * voxel_mm
        append_face(tris, (xw, y0, z0), (xw, y1, z0), (xw, y1, z1), (xw, y0, z1))

    # -X faces
    mask = occ & ~np.pad(occ[:, :, :-1], ((0, 0), (0, 0), (1, 0)), constant_values=False)
    for z, y, x in np.argwhere(mask):
        xw = x * voxel_mm
        y0 = y * voxel_mm
        y1 = (y + 1) * voxel_mm
        z0 = z * voxel_mm
        z1 = (z + 1) * voxel_mm
        append_face(tris, (xw, y0, z0), (xw, y0, z1), (xw, y1, z1), (xw, y1, z0))

    # +Y faces
    mask = occ & ~np.pad(occ[:, 1:, :], ((0, 0), (0, 1), (0, 0)), constant_values=False)
    for z, y, x in np.argwhere(mask):
        yw = (y + 1) * voxel_mm
        x0 = x * voxel_mm
        x1 = (x + 1) * voxel_mm
        z0 = z * voxel_mm
        z1 = (z + 1) * voxel_mm
        append_face(tris, (x0, yw, z0), (x1, yw, z0), (x1, yw, z1), (x0, yw, z1))

    # -Y faces
    mask = occ & ~np.pad(occ[:, :-1, :], ((0, 0), (1, 0), (0, 0)), constant_values=False)
    for z, y, x in np.argwhere(mask):
        yw = y * voxel_mm
        x0 = x * voxel_mm
        x1 = (x + 1) * voxel_mm
        z0 = z * voxel_mm
        z1 = (z + 1) * voxel_mm
        append_face(tris, (x0, yw, z0), (x0, yw, z1), (x1, yw, z1), (x1, yw, z0))

    # +Z faces
    mask = occ & ~np.pad(occ[1:, :, :], ((0, 1), (0, 0), (0, 0)), constant_values=False)
    for z, y, x in np.argwhere(mask):
        zw = (z + 1) * voxel_mm
        x0 = x * voxel_mm
        x1 = (x + 1) * voxel_mm
        y0 = y * voxel_mm
        y1 = (y + 1) * voxel_mm
        append_face(tris, (x0, y0, zw), (x1, y0, zw), (x1, y1, zw), (x0, y1, zw))

    # -Z faces
    mask = occ & ~np.pad(occ[:-1, :, :], ((1, 0), (0, 0), (0, 0)), constant_values=False)
    for z, y, x in np.argwhere(mask):
        zw = z * voxel_mm
        x0 = x * voxel_mm
        x1 = (x + 1) * voxel_mm
        y0 = y * voxel_mm
        y1 = (y + 1) * voxel_mm
        append_face(tris, (x0, y0, zw), (x0, y1, zw), (x1, y1, zw), (x1, y0, zw))

    return tris


def triangle_normal(a, b, c):
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    norm = (nx * nx + ny * ny + nz * nz) ** 0.5
    if norm == 0:
        return 0.0, 0.0, 0.0
    return nx / norm, ny / norm, nz / norm


def write_ascii_stl(triangles, output_path: Path, solid_name: str, scale: float = 1.0):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"solid {solid_name}\n")
        for a, b, c in triangles:
            a_s = (a[0] * scale, a[1] * scale, a[2] * scale)
            b_s = (b[0] * scale, b[1] * scale, b[2] * scale)
            c_s = (c[0] * scale, c[1] * scale, c[2] * scale)
            n = triangle_normal(a_s, b_s, c_s)
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {a_s[0]:.6e} {a_s[1]:.6e} {a_s[2]:.6e}\n")
            f.write(f"      vertex {b_s[0]:.6e} {b_s[1]:.6e} {b_s[2]:.6e}\n")
            f.write(f"      vertex {c_s[0]:.6e} {c_s[1]:.6e} {c_s[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def export_resolution_cuboid_stls(
    output_dir: str | Path = OUTPUTS_DIR / "usaf_phantom_stl",
    printer_max_mm: tuple[float, float, float] = (220.0, 220.0, 250.0),
) -> dict[str, str]:
    output_dir = resolve_repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    occupancy, meta = generate_resolution_cuboid_voxels()
    triangles = voxel_surface_triangles(occupancy, meta["voxel_mm"])

    sx, sy, sz = meta["base_size_mm"]["x"], meta["base_size_mm"]["y"], meta["base_size_mm"]["z"]
    longest = max(sx, sy, sz)
    scale_100 = 100.0 / longest
    scale_max = min(printer_max_mm[0] / sx, printer_max_mm[1] / sy, printer_max_mm[2] / sz)

    base_path = output_dir / "resolution_cuboid_base.stl"
    s100_path = output_dir / "resolution_cuboid_100mm.stl"
    smax_path = output_dir / "resolution_cuboid_max_printer.stl"
    meta_path = output_dir / "resolution_cuboid_metadata.json"

    write_ascii_stl(triangles, base_path, "resolution_cuboid_base", scale=1.0)
    write_ascii_stl(triangles, s100_path, "resolution_cuboid_100mm", scale=scale_100)
    write_ascii_stl(triangles, smax_path, "resolution_cuboid_max_printer", scale=scale_max)

    meta.update(
        {
            "printer_max_mm_assumed": {"x": printer_max_mm[0], "y": printer_max_mm[1], "z": printer_max_mm[2]},
            "scale_for_100mm": scale_100,
            "scale_for_max_printer": scale_max,
            "scaled_100mm_size_mm": {"x": sx * scale_100, "y": sy * scale_100, "z": sz * scale_100},
            "scaled_max_size_mm": {"x": sx * scale_max, "y": sy * scale_max, "z": sz * scale_max},
            "triangle_count": len(triangles),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "base_stl": str(base_path),
        "scaled_100mm_stl": str(s100_path),
        "scaled_max_printer_stl": str(smax_path),
        "metadata": str(meta_path),
    }


if __name__ == "__main__":
    outputs = export_resolution_cuboid_stls()
    for k, v in outputs.items():
        print(f"{k}: {v}")
