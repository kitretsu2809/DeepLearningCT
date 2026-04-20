from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_recon.paths import OUTPUTS_DIR, resolve_repo_path
from ct_recon.sparse_ct_reconstruction import _import_torch_or_exit, load_sparse_dataset, psnr_np, save_history


def split_indices(count: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(count * val_fraction))
    val_indices = set(indices[:val_count])
    train_indices = [idx for idx in indices if idx not in val_indices]
    valid_indices = [idx for idx in indices if idx in val_indices]
    return train_indices, valid_indices


def main():
    torch, nn, _ = _import_torch_or_exit()
    from torch.utils.data import DataLoader, Dataset
    from ct_recon.sparse_ct_reconstruction import SparseCTReconstructionModel

    class SparseSliceDataset(Dataset):
        def __init__(self, input_sinograms, target_sinograms, target_images, indices):
            self.input_sinograms = input_sinograms
            self.target_images = target_images
            self.indices = list(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            sample_idx = self.indices[index]
            sparse = torch.from_numpy(self.input_sinograms[sample_idx][None, :, :])
            target_image = torch.from_numpy(self.target_images[sample_idx][None, :, :])
            return sparse, target_image

    parser = argparse.ArgumentParser(description="Train a direct sparse sinogram to image reconstructor.")
    parser.add_argument("--dataset-path", default=str(OUTPUTS_DIR / "sparse_sinogram_dataset.npz"))
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "sparse_recon_training"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_sinograms, _target_sinograms, target_images, metadata = load_sparse_dataset(resolve_repo_path(args.dataset_path))
    train_indices, val_indices = split_indices(len(input_sinograms), args.val_fraction, args.seed)

    train_dataset = SparseSliceDataset(input_sinograms, _target_sinograms, target_images, train_indices)
    val_dataset = SparseSliceDataset(input_sinograms, _target_sinograms, target_images, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseCTReconstructionModel(
        sparse_angle_count=metadata.sparse_angle_count,
        dense_angle_count=metadata.dense_angle_count,
        detector_count=metadata.detector_count,
        image_size=metadata.image_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    l1_loss = nn.L1Loss()

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_model.pt"

    history = {
        "dataset_path": str(args.dataset_path),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "device": str(device),
        "metadata": metadata.__dict__,
        "epoch_logs": [],
    }

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for sparse, target_image in train_loader:
            sparse = sparse.to(device)
            target_image = target_image.to(device)

            outputs = model(sparse)
            loss = l1_loss(outputs["reconstruction"], target_image)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        val_psnr_scores = []
        with torch.no_grad():
            for sparse, target_image in val_loader:
                sparse = sparse.to(device)
                target_image = target_image.to(device)

                outputs = model(sparse)
                loss = l1_loss(outputs["reconstruction"], target_image)
                val_losses.append(float(loss.item()))

                predictions_np = outputs["reconstruction"].detach().cpu().numpy()
                targets_np = target_image.detach().cpu().numpy()
                for pred, target in zip(predictions_np, targets_np):
                    val_psnr_scores.append(psnr_np(np.clip(pred, 0.0, 1.0), target))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_psnr = float(np.mean(val_psnr_scores)) if val_psnr_scores else float("nan")
        history["epoch_logs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
            }
        )
        save_history(history, output_dir)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "metadata": metadata.__dict__,
                "epoch": epoch,
                "history": history,
            },
            last_checkpoint,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": metadata.__dict__,
                    "epoch": epoch,
                    "history": history,
                },
                best_checkpoint,
            )

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_psnr={val_psnr:.3f}dB"
        )


if __name__ == "__main__":
    main()
