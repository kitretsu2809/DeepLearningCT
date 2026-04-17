from __future__ import annotations

import argparse
from pathlib import Path


def scale_ascii_stl(input_path: str | Path, output_path: str | Path, scale: float) -> Path:
    if scale <= 0:
        raise ValueError("scale must be positive")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines_out: list[str] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("vertex "):
                parts = stripped.split()
                x = float(parts[1]) * scale
                y = float(parts[2]) * scale
                z = float(parts[3]) * scale
                indent = line[: line.index("v")] if "v" in line else "      "
                lines_out.append(f"{indent}vertex {x:.6e} {y:.6e} {z:.6e}\n")
            else:
                lines_out.append(line)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.writelines(lines_out)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniformly scale an ASCII STL file.")
    parser.add_argument("--input", required=True, help="Input ASCII STL path")
    parser.add_argument("--output", required=True, help="Output ASCII STL path")
    parser.add_argument("--scale", required=True, type=float, help="Uniform scale factor")
    args = parser.parse_args()

    output = scale_ascii_stl(args.input, args.output, args.scale)
    print(f"scaled_stl: {output}")


if __name__ == "__main__":
    main()
