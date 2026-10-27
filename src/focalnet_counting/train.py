from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from focalnet_counting.dataset import FSC147Dataset
from focalnet_counting.model import CountingModel, ReconstructionModel


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrain(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    checkpoint_dir: Path,
) -> Path:
    device = _device()
    dataset = FSC147Dataset(data_dir, split="train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = ReconstructionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for batch in tqdm(loader, desc=f"pretrain {epoch}/{epochs}"):
            image = batch["image"].to(device)
            reconstructed = model(image)
            loss = F.l1_loss(reconstructed, image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        print(f"epoch {epoch}: recon_l1={running / max(len(loader), 1):.4f}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "pretrain.pt"
    torch.save(model.encoder.state_dict(), path)
    print(f"Saved encoder -> {path}")
    return path


def train_counting(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    checkpoint_dir: Path,
    encoder_checkpoint: Path | None,
) -> Path:
    device = _device()
    dataset = FSC147Dataset(data_dir, split="train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = CountingModel().to(device)
    if encoder_checkpoint and encoder_checkpoint.is_file():
        model.load_encoder(torch.load(encoder_checkpoint, map_location=device, weights_only=True))
        print(f"Loaded encoder from {encoder_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for batch in tqdm(loader, desc=f"count {epoch}/{epochs}"):
            image = batch["image"].to(device)
            density = batch["density"].to(device)
            pred = model(image)
            loss = F.mse_loss(pred, density)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        print(f"epoch {epoch}: density_mse={running / max(len(loader), 1):.6f}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "counting.pt"
    torch.save(model.state_dict(), path)
    print(f"Saved counting model -> {path}")
    return path


@torch.inference_mode()
def evaluate(data_dir: Path, split: str, checkpoint: Path) -> tuple[float, float]:
    device = _device()
    dataset = FSC147Dataset(data_dir, split=split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = CountingModel().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    abs_err = 0.0
    sq_err = 0.0
    n = 0
    for batch in loader:
        pred = float(model.predicted_count(batch["image"].to(device)).item())
        true = float(batch["count"].item())
        abs_err += abs(pred - true)
        sq_err += (pred - true) ** 2
        n += 1
    mae = abs_err / max(n, 1)
    rmse = (sq_err / max(n, 1)) ** 0.5
    print(f"{split}: n={n} MAE={mae:.3f} RMSE={rmse:.3f}")
    return mae, rmse
