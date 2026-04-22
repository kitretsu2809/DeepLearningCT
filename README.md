# Deep Learning CT Reconstruction

This project implements four CT reconstruction pipeline modes:

1. **Classical**: Traditional FDK (Feldkamp-Davis-Kress) reconstruction
2. **Sinogram**: Deep learning from sparse sinograms (fewer X-ray views)
3. **Enhance**: Deep learning from full sinograms (all X-ray views) - for improving FDK
4. **U-Net**: Post-processing enhancement of degraded FDK

## Pipeline Modes

### 1. Classical (`classical`)
Traditional filtered back-projection (FDK) reconstruction using ASTRA Toolbox.

**What it does:**
- Runs FDK on raw projection data
- Outputs 3D volume (.tif) and preview (.png)

**Command:**
```bash
# Default (downsampled)
python scripts/run_pipeline.py classical --sample sample_1

# Full resolution (no downsampling)
python scripts/run_pipeline.py classical --sample sample_2 --no-downsample
```

**Options:** `--no-downsample`, `--downsample-factor N`

---

### 2. Sinogram (`sinogram`)
Deep learning from **sparse** sinograms (subset of projections).

- Input: Every Nth projection (sparse)
- Target: FDK from all projections
- Use case: Reconstruct from fewer X-ray views

**Command:**
```bash
python scripts/run_pipeline.py sinogram --sample sample_1 --epochs 50
```

**Options:** `--epochs N`

---

### 3. Enhance (`enhance`)
Deep learning from **full** sinograms (all projections) to enhance FDK reconstruction.

- Input: Full projection set
- Target: FDK reconstruction (same projections)
- Use case: Improve reconstruction quality beyond classical FDK

**Command:**
```bash
python scripts/run_pipeline.py enhance --sample sample_1 --epochs 50
python scripts/run_pipeline.py enhance --sample sample_2 --epochs 50
```

**Options:** `--epochs N`

---

### 4. U-Net (`unet`)
Post-processing enhancement of degraded FDK using U-Net.

- Input: Degraded (downsampled) FDK
- Target: High-quality FDK
- Use case: Polish/reduce artifacts in reconstruction

**Command:**
```bash
python scripts/run_pipeline.py unet --sample sample_1 --epochs 50
```

**Options:** `--epochs N`

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
