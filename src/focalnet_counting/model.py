"""Focal encoder, reverse patch-embedding, and a density counting head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from focalnet_counting.modulation import FocalBlock


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, dim: int = 256) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        tokens = self.proj(image)
        _, _, height, width = tokens.shape
        return tokens.flatten(2).transpose(1, 2), height, width


class ReversePatchEmbed(nn.Module):
    """Map tokens back to RGB patches for stage-1 reconstruction."""

    def __init__(self, patch_size: int = 16, dim: int = 256, out_chans: int = 3) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(dim, out_chans * patch_size * patch_size)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, _ = tokens.shape
        patches = self.proj(tokens)
        patches = patches.transpose(1, 2).reshape(
            batch, 3, self.patch_size, self.patch_size, height, width
        )
        patches = patches.permute(0, 1, 4, 2, 5, 3)
        return patches.reshape(
            batch, 3, height * self.patch_size, width * self.patch_size
        )


class FocalEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        patch_size: int = 16,
        focal_level: int = 3,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, dim=dim)
        self.blocks = nn.ModuleList(
            [FocalBlock(dim, focal_level=focal_level) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        tokens, height, width = self.patch_embed(image)
        for block in self.blocks:
            tokens = block(tokens, height, width)
        return self.norm(tokens), height, width


class ReconstructionModel(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 6, patch_size: int = 16) -> None:
        super().__init__()
        self.encoder = FocalEncoder(dim=dim, depth=depth, patch_size=patch_size)
        self.reverse = ReversePatchEmbed(patch_size=patch_size, dim=dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens, height, width = self.encoder(image)
        reconstructed = self.reverse(tokens, height, width)
        return F.interpolate(
            reconstructed, size=image.shape[-2:], mode="bilinear", align_corners=False
        )


class CountingModel(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 6, patch_size: int = 16) -> None:
        super().__init__()
        self.encoder = FocalEncoder(dim=dim, depth=depth, patch_size=patch_size)
        self.density_head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def load_encoder(self, state: dict[str, torch.Tensor]) -> None:
        self.encoder.load_state_dict(state, strict=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens, height, width = self.encoder(image)
        batch, _, dim = tokens.shape
        feature_map = tokens.transpose(1, 2).reshape(batch, dim, height, width)
        density = self.density_head(feature_map)
        return F.interpolate(
            density, size=image.shape[-2:], mode="bilinear", align_corners=False
        )

    def predicted_count(self, image: torch.Tensor) -> torch.Tensor:
        density = self.forward(image)
        return density.sum(dim=(1, 2, 3))
