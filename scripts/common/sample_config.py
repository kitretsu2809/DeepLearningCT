"""Universal sample configuration."""
from pathlib import Path

# Sample configurations with their specific parameters
SAMPLE_CONFIG = {
    "sample_1": {
        "downsample_factor": 2,
        "detector_size": 1000,  # Original projection size
        "z_range": (180, 1060),  # zmin, zmax from settings
    },
    "sample_2": {
        "downsample_factor": 4,
        "detector_size": 2850,
        "z_range": (0, 2849),
    },
}


def get_sample_config(sample_name: str) -> dict:
    """Get configuration for a sample."""
    if sample_name not in SAMPLE_CONFIG:
        raise ValueError(f"Unknown sample: {sample_name}. Available: {list(SAMPLE_CONFIG.keys())}")
    return SAMPLE_CONFIG[sample_name]


def get_sample_paths(sample_name: str, repo_root: Path | None = None) -> dict:
    """Get all paths for a sample."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    
    sample_dir = repo_root / "data" / sample_name
    output_base = repo_root / "outputs" / f"{sample_name}_pipeline"
    config = get_sample_config(sample_name)
    
    return {
        "repo_root": repo_root,
        "sample_dir": sample_dir,
        "sample_name": sample_name,
        "downsample_factor": config["downsample_factor"],
        # Classical reconstruction outputs
        "fdk_volume": output_base / "classical" / "fdk_volume.tif",
        # Sinogram reconstruction pipeline
        "sinogram_dataset": output_base / "sinogram_recon" / "dataset.npz",
        "sinogram_checkpoint": output_base / "sinogram_recon" / "training" / "best_model.pt",
        "sinogram_inference": output_base / "sinogram_recon" / "inference",
        # U-Net enhancement pipeline  
        "unet_training_pairs": output_base / "unet_enhance" / "training_pairs",
        "unet_checkpoint": output_base / "unet_enhance" / "training" / "best_model.pt",
        "unet_inference": output_base / "unet_enhance" / "inference",
    }
