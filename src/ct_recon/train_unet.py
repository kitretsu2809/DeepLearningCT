from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import OUTPUTS_DIR, resolve_repo_path


def _import_torch_or_exit():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError:
        raise SystemExit(
            "Missing PyTorch in this environment.\n"
            "Install it with:\n"
            "  ./.venv/bin/pip install torch torchvision"
        )
    return torch, nn, Dataset, DataLoader


@dataclass
class SliceRecord:
    pair_name: str
    slice_index: int
    input_slice: np.ndarray
    target_slice: np.ndarray


def load_slice_records(pairs_root: str | Path) -> list[SliceRecord]:
    pairs_root = Path(pairs_root)
    pair_dirs = sorted(path for path in pairs_root.iterdir() if path.is_dir())
    records: list[SliceRecord] = []

    for pair_dir in pair_dirs:
        slices_path = pair_dir / "axial_slices.npz"
        if not slices_path.exists():
            continue
        payload = np.load(slices_path)
        input_slices = payload["input_slices"].astype(np.float32)
        target_slices = payload["target_slices"].astype(np.float32)
        if input_slices.shape != target_slices.shape:
            raise ValueError(f"Slice shape mismatch in {slices_path}: {input_slices.shape} vs {target_slices.shape}")

        for index in range(input_slices.shape[0]):
            records.append(
                SliceRecord(
                    pair_name=pair_dir.name,
                    slice_index=index,
                    input_slice=input_slices[index],
                    target_slice=target_slices[index],
                )
            )

    if not records:
        raise FileNotFoundError(f"No axial_slices.npz files found under {pairs_root}")
    return records


def split_records(records: list[SliceRecord], val_fraction: float, seed: int) -> tuple[list[SliceRecord], list[SliceRecord]]:
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_count = max(1, int(len(indices) * val_fraction))
    val_indices = set(indices[:val_count])

    train_records = [record for idx, record in enumerate(records) if idx not in val_indices]
    val_records = [record for idx, record in enumerate(records) if idx in val_indices]
    return train_records, val_records


def build_transforms(slice_array: np.ndarray) -> np.ndarray:
    return np.expand_dims(slice_array.astype(np.float32, copy=False), axis=0)


def psnr(prediction, target, eps: float = 1e-8) -> float:
    mse = float(((prediction - target) ** 2).mean())
    if mse <= eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def save_history(history: dict, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_history.json"
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return path


def main():
    torch, nn, Dataset, DataLoader = _import_torch_or_exit()

    class SlicePairDataset(Dataset):
        def __init__(self, records: list[SliceRecord]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int):
            record = self.records[index]
            x = torch.from_numpy(build_transforms(record.input_slice))
            y = torch.from_numpy(build_transforms(record.target_slice))
            return x, y

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet2D(nn.Module):
        def __init__(self, in_channels: int = 1, out_channels: int = 1, features: tuple[int, ...] = (32, 64, 128, 256)):
            super().__init__()
            self.down_blocks = nn.ModuleList()
            self.up_transpose = nn.ModuleList()
            self.up_blocks = nn.ModuleList()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            channels = in_channels
            for feature in features:
                self.down_blocks.append(DoubleConv(channels, feature))
                channels = feature

            self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

            reversed_features = list(reversed(features))
            up_channels = features[-1] * 2
            for feature in reversed_features:
                self.up_transpose.append(nn.ConvTranspose2d(up_channels, feature, kernel_size=2, stride=2))
                self.up_blocks.append(DoubleConv(feature * 2, feature))
                up_channels = feature

            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        def forward(self, x):
            skips = []
            for down in self.down_blocks:
                x = down(x)
                skips.append(x)
                x = self.pool(x)

            x = self.bottleneck(x)
            skips = skips[::-1]

            for idx in range(len(self.up_transpose)):
                x = self.up_transpose[idx](x)
                skip = skips[idx]

                if x.shape[-2:] != skip.shape[-2:]:
                    x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

                x = torch.cat([skip, x], dim=1)
                x = self.up_blocks[idx](x)

            return self.final_conv(x)

    parser = argparse.ArgumentParser(description="Train a 2D U-Net on axial CT reconstruction slices.")
    parser.add_argument("--pairs-root", default=str(OUTPUTS_DIR / "training_pairs"), help="Directory containing pair subfolders with axial_slices.npz")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "unet_training"), help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_slice_records(resolve_repo_path(args.pairs_root))
    train_records, val_records = split_records(records, val_fraction=args.val_fraction, seed=args.seed)

    train_dataset = SlicePairDataset(train_records)
    val_dataset = SlicePairDataset(val_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.L1Loss()

    history = {
        "pairs_root": str(args.pairs_root),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "train_slices": len(train_records),
        "val_slices": len(val_records),
        "epoch_logs": [],
    }

    best_val_loss = float("inf")
    best_checkpoint = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        val_psnr_scores = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                val_losses.append(float(loss.item()))

                predictions_np = predictions.detach().cpu().numpy()
                targets_np = targets.detach().cpu().numpy()
                for pred, target in zip(predictions_np, targets_np):
                    val_psnr_scores.append(psnr(np.clip(pred, 0.0, 1.0), target))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_psnr_mean = float(np.mean(val_psnr_scores)) if val_psnr_scores else float("nan")

        epoch_log = {
            "epoch": epoch,
            "train_l1": train_loss,
            "val_l1": val_loss,
            "val_psnr": val_psnr_mean,
        }
        history["epoch_logs"].append(epoch_log)
        print(epoch_log)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            last_checkpoint,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                best_checkpoint,
            )

    save_history(history, output_dir)
    print({"best_checkpoint": str(best_checkpoint), "last_checkpoint": str(last_checkpoint)})


if __name__ == "__main__":
    main()
