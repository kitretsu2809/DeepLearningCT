# Deep Learning CT Reconstruction

This repository is organized into two separate deep-learning workflows:

1. Direct sinogram reconstruction: sparse sinogram -> reconstructed slice/volume
2. U-Net enhancement: degraded FDK reconstruction -> enhanced reconstruction

## Structure

- `src/ct_recon/`: shared library code
- `scripts/`: runnable pipelines and entrypoints
- `data/sample_1`, `data/sample_2`: raw datasets
- `outputs/`: generated reconstructions, datasets, and checkpoints

Details and per-script usage are documented in `scripts/README.md`.

## Quick Commands

```bash
# Direct sparse sinogram reconstruction pipeline
python scripts/run_pipeline.py sinogram --sample sample_1 --epochs 50

# U-Net enhancement training pipeline
python scripts/run_pipeline.py unet --sample sample_1 --epochs 50

# Classical FDK reconstruction only
python scripts/run_pipeline.py classical --sample sample_2
```

## Notes

- Combined multi-sample training scripts were removed to keep workflows strictly separated by sample and task.
- Universal sample/output paths are managed via `scripts/common/sample_config.py` and `scripts/common/paths.sh`.

