# Deep Learning CT Reconstruction

This project implements three CT reconstruction pipeline modes:

1. **Classical**: Traditional FDK (Feldkamp-Davis-Kress) reconstruction
2. **Sinogram**: Direct neural network reconstruction from sparse sinograms
3. **U-Net**: Enhancement of degraded FDK reconstructions using U-Net

## Pipeline Modes

### 1. Classical (`classical`)
Traditional filtered back-projection (FDK) reconstruction using ASTRA Toolbox.

**What it does:**
- Runs FDK reconstruction on raw projection data
- Outputs a 3D volume reconstruction (.tif) and preview (.png)

**Command:**
```bash
# Default (downsampled)
python scripts/run_pipeline.py classical --sample sample_1
python scripts/run_pipeline.py classical --sample sample_2

# Full resolution (no downsampling - requires more GPU memory)
python scripts/run_pipeline.py classical --sample sample_2 --no-downsample

# Custom downsample factor
python scripts/run_pipeline.py classical --sample sample_2 --downsample-factor 2
```

**Options:**
- `--no-downsample` - Skip downsampling for better quality (requires more GPU memory)
- `--downsample-factor N` - Override default downsample factor

---

### 2. Sinogram (`sinogram`)
End-to-end deep learning: directly reconstructs images from sparse sinograms (fewer X-ray projections).

**What it does:**
1. Runs classical FDK reconstruction for reference
2. Builds sparse sinogram dataset (subsampled projections)
3. Trains a neural network to reconstruct from sparse sinograms
4. Runs inference on test data

**Command:**
```bash
python scripts/run_pipeline.py sinogram --sample sample_1 --epochs 50
python scripts/run_pipeline.py sinogram --sample sample_2 --epochs 50
```

**Options:**
- `--epochs N` - Number of training epochs (default: 50)

---

### 3. U-Net (`unet`)
Post-processing enhancement: improves degraded FDK reconstructions using a U-Net.

**What it does:**
1. Runs classical FDK reconstruction for reference
2. Creates training pairs (degraded FDK → high-quality reference)
3. Trains a U-Net to enhance degraded reconstructions
4. Outputs trained model for inference

**Command:**
```bash
python scripts/run_pipeline.py unet --sample sample_1 --epochs 50
python scripts/run_pipeline.py unet --sample sample_2 --epochs 50
```

**Options:**
- `--epochs N` - Number of training epochs (default: 50)

---

## Structure

- `src/ct_recon/`: shared library code
- `scripts/`: runnable pipelines and entrypoints
- `data/sample_1`, `data/sample_2`: raw datasets
- `outputs/sample_X_pipeline/`: pipeline outputs (classical, sinogram_recon, unet_enhance)

## Data Samples

| Sample | Projections | Original Size | Downsample | Notes |
|--------|------------|--------------|-----------|-------|
| sample_1 | 359 | 1000×1000 | 2 | Full volume |
| sample_2 | 361 | 2850×2850 | 4 | Limited z-range (slices 280-440 contain structure) |

**Note:** Sample_2 has empty slices at top and bottom. The z-range is automatically limited to only the active region.

## Notes

- Combined multi-sample training scripts were removed to keep workflows strictly separated by sample and task.
- Universal sample/output paths are managed via `scripts/common/sample_config.py`.
