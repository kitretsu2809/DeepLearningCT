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

# Full resolution (no downsampling)
python scripts/run_pipeline.py classical --sample sample_2 --no-downsample

# Custom downsample factor
python scripts/run_pipeline.py classical --sample sample_2 --downsample-factor 2
```

**Options:**
- `--no-downsample` - Skip downsampling for better quality
- `--downsample-factor N` - Override default downsample factor

---

### 2. Sinogram (`sinogram`)
End-to-end deep learning: directly reconstructs from sparse sinograms.

**What it does:**
1. Runs classical FDK for reference
2. Builds sparse sinogram dataset
3. Trains neural network
4. Runs inference

**Command:**
```bash
python scripts/run_pipeline.py sinogram --sample sample_1 --epochs 50
python scripts/run_pipeline.py sinogram --sample sample_2 --epochs 50
```

**Options:**
- `--epochs N` - Training epochs (default: 50)

---

### 3. U-Net (`unet`)
Post-processing enhancement of degraded FDK reconstructions.

**Command:**
```bash
python scripts/run_pipeline.py unet --sample sample_1 --epochs 50
python scripts/run_pipeline.py unet --sample sample_2 --epochs 50
```

**Options:**
- `--epochs N` - Training epochs (default: 50)

---

## Data Samples

| Sample | Projections | Size | Downsample |
|--------|------------|------|-----------|
| sample_1 | 359 | 1000×1000 | 2 |
| sample_2 | 361 | 2850×2850 | 4 |

---

## Notes

- Sample_2 z-range limited to slices 280-440 (only active region)
- Paths managed via `scripts/common/sample_config.py`

