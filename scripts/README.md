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
- Outputs a 3D volume reconstruction

**Command:**
```bash
python scripts/run_pipeline.py classical --sample sample_1
python scripts/run_pipeline.py classical --sample sample_2
```

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
- `outputs/`: generated reconstructions, datasets, and checkpoints

## Notes

- Combined multi-sample training scripts were removed to keep workflows strictly separated by sample and task.
- Universal sample/output paths are managed via `scripts/common/sample_config.py` and `scripts/common/paths.sh`.

