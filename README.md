# Deep Learning CT Reconstruction

This repository contains a CT reconstruction workflow built around two stages:

1. Classical reconstruction from projection images
2. Deep-learning preparation and training on degraded-vs-reference reconstructions

## Current pipeline

- `data_loader.py`: load `sample 1` projections and scan settings
- `geometry.py`: parse CT geometry from `settings.cto`
- `reconstruct_fbp.py`: early slice-wise baseline
- `reconstruct_fdk_astra.py`: main cone-beam FDK reconstruction using ASTRA
- `simulate_degradation.py`: create sparse-view, limited-angle, and noisy projection datasets
- `build_training_pairs.py`: build degraded/input and reference/target reconstruction pairs
- `train_unet.py`: train a 2D U-Net on axial slice pairs
- `export_for_colab.py`: export compact training data for Colab

## Recommended workflow

1. Reconstruct the full reference volume with `reconstruct_fdk_astra.py`
2. Generate degraded projection datasets with `simulate_degradation.py`
3. Build aligned input-target reconstruction pairs with `build_training_pairs.py`
4. Train the first model with `train_unet.py`

## Notes

- Raw datasets, generated outputs, checkpoints, previews, and exported training archives are excluded from Git.
- The main 3D reconstruction path uses ASTRA with CUDA.
