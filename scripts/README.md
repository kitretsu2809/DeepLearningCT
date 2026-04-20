# CT Reconstruction Scripts - Organized Structure

## Quick Start

```bash
# Run complete sinogram reconstruction pipeline on sample_1
python scripts/run_pipeline.py sinogram --sample sample_1 --epochs 50

# Run classical FDK reconstruction only
python scripts/run_pipeline.py classical --sample sample_2
```

## Directory Structure

### `scripts/sinogram_reconstruction/` - Direct Sinogram → Image
End-to-end deep learning reconstruction from sparse sinograms.

| Script | Purpose |
|--------|---------|
| `01_build_dataset.py` | Create sparse sinogram → target slice dataset |
| `02_train_model.py` | Train SinogramToImageDecoder model |
| `03_run_inference.py` | Reconstruct new volumes using trained model |

**Workflow:**
1. Classical FDK reconstruction (for reference targets)
2. Build dataset: Extract detector rows → sparse sinograms
3. Train model: Learn mapping `sparse_sinogram (90,256) → slice (256,256)`
4. Inference: Process new sparse-view data

### `scripts/unet_enhancement/` - FBP Enhancement
Post-processing enhancement of degraded FBP reconstructions.

| Script | Purpose |
|--------|---------|
| `01_build_training_pairs.py` | Create degraded ↔ reference slice pairs |
| `02_train_model.py` | Train U-Net for image enhancement |

**Workflow:**
1. Generate degraded reconstructions (sparse/limited/noisy)
2. Build pairs: degraded FBP slice → reference FBP slice
3. Train U-Net: Learn `degraded_slice → enhanced_slice`

### `scripts/classical_reconstruction/` - Traditional Methods
Classical CT reconstruction without deep learning.

| Script | Purpose |
|--------|---------|
| `reconstruct_fdk.py` | FDK reconstruction for sample_1 |
| `reconstruct_fdk_sample2.py` | FDK reconstruction for sample_2 |
| `simulate_degradations.py` | Simulate sparse/limited/noisy views |

### `scripts/common/` - Shared Resources
- `sample_config.py` - Universal sample configuration
- `paths.sh` - Bash path utilities
- `export_for_colab.py` - Export utilities

## Universal Output Structure

All outputs follow this pattern:
```
outputs/{sample_name}_pipeline/
├── classical/
│   └── fdk_volume.tif              # Reference FDK reconstruction
├── sinogram_recon/
│   ├── dataset.npz                 # Training dataset
│   ├── training/
│   │   └── best_model.pt         # Trained model
│   └── inference/
│       ├── dl_sparse_reconstruction_volume.tif
│       └── dl_sparse_reconstruction_preview.png
└── unet_enhance/
    ├── training_pairs/           # Degraded vs reference
    ├── training/
    │   └── best_model.pt
    └── inference/
```

## Sample Configuration

Samples have different properties:

| Sample | Downsample | Detector Size | Z Range |
|--------|------------|---------------|---------|
| sample_1 | 2 | 1000×1000 | 180-1060 |
| sample_2 | 4 | 2850×2850 | 0-2849 |

Configured in `scripts/common/sample_config.py`.
