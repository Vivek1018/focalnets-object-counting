"""Minimal FSC-147 loader. Expects the official dataset layout."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FSC147Dataset(Dataset):
    def __init__(self, data_dir: Path, split: str = "train", image_size: int = 384) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        split_file = self.data_dir / "Train_Test_Val_FSC_147.json"
        annotation_file = self.data_dir / "annotation_FSC147_384.json"
        if not split_file.is_file() or not annotation_file.is_file():
            raise FileNotFoundError(
                f"FSC-147 files not found in {data_dir}. "
                "Download the dataset from "
                "https://github.com/cvlab-stonybrook/LearningToCountEverything"
            )
        splits = json.loads(split_file.read_text(encoding="utf-8"))
        self.annotations = json.loads(annotation_file.read_text(encoding="utf-8"))
        key = {"train": "train", "val": "val", "test": "test"}[split]
        self.names = splits[key]
        self.image_dir = self._first_existing(
            [self.data_dir / "images_384_VarV2", self.data_dir / "images"]
        )
        self.density_dir = self._first_existing(
            [self.data_dir / "gt_density_map_adaptive_384_VarV2", self.data_dir / "gt_density_maps"]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.image_size = image_size

    @staticmethod
    def _first_existing(candidates: list[Path]) -> Path:
        for path in candidates:
            if path.is_dir():
                return path
        return candidates[0]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name = self.names[index]
        image = Image.open(self.image_dir / name).convert("RGB")
        tensor = self.transform(image)
        density_path = (self.density_dir / name).with_suffix(".npy")
        if density_path.is_file():
            import numpy as np

            density = torch.from_numpy(np.load(density_path)).float()
            density = density.unsqueeze(0)
            density = torch.nn.functional.interpolate(
                density.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            points = self.annotations[name].get("points", [])
            density = torch.zeros(1, self.image_size, self.image_size)
            count = float(len(points))
            return {"image": tensor, "density": density, "count": torch.tensor(count), "name": name}

        count = torch.tensor(float(density.sum()))
        return {"image": tensor, "density": density, "count": count, "name": name}
