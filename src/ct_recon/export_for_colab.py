from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .paths import EXPORTS_DIR, OUTPUTS_DIR, resolve_repo_path


def copy_pair_folder(pair_dir: Path, export_root: Path) -> Path:
    destination = export_root / pair_dir.name
    destination.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "axial_slices.npz",
        "pair_metadata.json",
    ]

    for filename in files_to_copy:
        source = pair_dir / filename
        if source.exists():
            shutil.copy2(source, destination / filename)

    return destination


def export_training_pairs(
    pairs_root: str | Path = OUTPUTS_DIR / "training_pairs",
    export_root: str | Path = EXPORTS_DIR / "colab_training_pairs",
    zip_output: bool = True,
) -> dict[str, Path]:
    pairs_root = resolve_repo_path(pairs_root)
    export_root = resolve_repo_path(export_root)

    if not pairs_root.exists():
        raise FileNotFoundError(f"Pairs root does not exist: {pairs_root}")

    if export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    exported_dirs = []
    for pair_dir in sorted(path for path in pairs_root.iterdir() if path.is_dir()):
        if (pair_dir / "axial_slices.npz").exists():
            exported_dirs.append(copy_pair_folder(pair_dir, export_root))

    if not exported_dirs:
        raise FileNotFoundError(f"No pair folders with axial_slices.npz found under {pairs_root}")

    manifest_path = export_root / "manifest.txt"
    manifest_path.write_text(
        "\n".join(path.name for path in exported_dirs),
        encoding="utf-8",
    )

    outputs = {
        "export_root": export_root,
        "manifest_path": manifest_path,
    }

    if zip_output:
        archive_base = export_root.parent / export_root.name
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=export_root.parent, base_dir=export_root.name)
        outputs["zip_path"] = Path(archive_path)

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Export compact training-pair files for Colab.")
    parser.add_argument("--pairs-root", default=str(OUTPUTS_DIR / "training_pairs"), help="Directory containing pair subfolders")
    parser.add_argument("--export-root", default=str(EXPORTS_DIR / "colab_training_pairs"), help="Output directory for compact Colab export")
    parser.add_argument("--no-zip", action="store_true", help="Do not create a zip archive")
    args = parser.parse_args()

    outputs = export_training_pairs(
        pairs_root=args.pairs_root,
        export_root=args.export_root,
        zip_output=not args.no_zip,
    )

    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
