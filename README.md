# Deep Learning CT Reconstruction

This repository contains a CT reconstruction workflow built around two stages:

1. Classical reconstruction from projection images
2. Deep-learning preparation and training on degraded-vs-reference reconstructions

## Layout

- `src/ct_recon/`: reusable reconstruction and dataset code
- `scripts/`: runnable entry points
- `data/sample_1/`: raw CT scan sample and `settings.cto`
- `outputs/`: generated reconstructions, datasets, checkpoints, previews
- `exports/`: compact exported artifacts
- `docs/`: reports
- `references/`: papers and external reference repository

## Recommended workflow

1. Reconstruct the full reference volume with `scripts/reconstruct_fdk_astra.py`
2. Generate degraded projection datasets with `scripts/simulate_degradation.py`
3. Build aligned input-target reconstruction pairs with `scripts/build_training_pairs.py`
4. Train the first model with `scripts/train_unet.py`

## Direct sparse sinogram reconstruction

The repository now includes a direct sparse-view reconstruction path:

1. Sparse sinogram input
2. Learned sinogram-to-image generator
3. Direct supervision against high-quality target slices

This is implemented as a 2D slice model. It uses the real projection stack, extracts one detector-row sparse sinogram per reconstructed slice, and learns to predict the target slice directly. It does not take an FBP image as input and does not apply a post-FBP enhancement stage.

### Build the training dataset

```bash
./.venv/bin/python scripts/build_sparse_sinogram_dataset.py \
  --target-volume outputs/fdk_full_ds2/fdk_volume.tif \
  --output-path outputs/sparse_sinogram_dataset.npz \
  --sparse-step 4 \
  --detector-count 256 \
  --image-size 256
```

### Train the model

```bash
./.venv/bin/python scripts/train_sparse_sinogram_reconstructor.py \
  --dataset-path outputs/sparse_sinogram_dataset.npz \
  --output-dir outputs/sparse_recon_training \
  --epochs 20 \
  --batch-size 4
```

### Reconstruct a sparse-view volume

```bash
./.venv/bin/python scripts/reconstruct_sparse_volume_dl.py \
  --checkpoint outputs/sparse_recon_training/best_model.pt \
  --output-dir outputs/sparse_recon_inference
```

### Important scope note

This direct model is still slice-wise: one sparse detector-row sinogram maps to one reconstructed slice. That is a practical baseline for sparse-view DL reconstruction in the current repository. A later upgrade would be a true 3D cone-beam sinogram-to-volume model.

## Notes

- Package imports now resolve from `src/ct_recon`, and default sample data lives in `data/sample_1`.
- Raw datasets, generated outputs, checkpoints, previews, and exported training archives are excluded from Git.
- The main 3D reconstruction path uses ASTRA with CUDA.

## Training on Combined Dataset (Sample 1 + Sample 2)

To train the sparse sinogram reconstructor on data from both samples:

### Step 1: Reconstruct target volumes for each sample

```bash
# Sample 1 (smaller, faster)
python scripts/reconstruct_fdk_astra.py --downsample 2

# Sample 2 (larger, use downsampling)
python scripts/reconstruct_sample2.py --downsample 4
```

### Step 2: Build combined dataset

```bash
python scripts/build_combined_sparse_dataset.py \
  --downsample-factor 2 \
  --sparse-step 4 \
  --detector-count 256 \
  --image-size 256
```

Or with custom target volumes:
```bash
python scripts/build_combined_sparse_dataset.py \
  --sample1-target outputs/sample_1_fdk_ds2/fdk_volume.tif \
  --sample2-target outputs/sample_2_fdk_ds4/fdk_volume.tif \
  --sparse-step 4
```

### Step 3: Train on combined data

```bash
python scripts/train_combined_sparse_reconstructor.py \
  --dataset-path outputs/combined_sparse_dataset.npz \
  --output-dir outputs/combined_recon_training \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 1e-3
```

### Step 4: Inference with combined model

```bash
python scripts/reconstruct_sparse_volume_dl.py \
  --checkpoint outputs/combined_recon_training/best_model.pt \
  --output-dir outputs/combined_recon_inference
```
