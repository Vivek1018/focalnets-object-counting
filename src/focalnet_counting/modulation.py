"""Focal modulation block.

Replaces token self-attention with a gated multi-scale context signal
(Yang et al., Focal Modulation Networks, NeurIPS 2022). This file is an
original implementation of that idea, not a copy of the Microsoft repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalModulation(nn.Module):
    def __init__(
        self,
        dim: int,
        focal_window: int = 3,
        focal_level: int = 3,
        focal_factor: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.focal_level = focal_level
        self.input_proj = nn.Linear(dim, 2 * dim + (focal_level + 1))
        self.context_convs = nn.ModuleList()
        kernel = focal_window
        for _ in range(focal_level):
            self.context_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=kernel,
                        stride=1,
                        padding=kernel // 2,
                        groups=dim,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )
            kernel = kernel * focal_factor + (focal_factor % 2)
        self.global_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.GELU())
        self.mixer = nn.Conv2d(dim, dim, kernel_size=1)
        self.output_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, length, dim = tokens.shape
        projected = self.input_proj(tokens)
        query, context, gates = torch.split(
            projected, [dim, dim, self.focal_level + 1], dim=-1
        )
        context = context.transpose(1, 2).reshape(batch, dim, height, width)
        gates = gates.transpose(1, 2).reshape(
            batch, self.focal_level + 1, height, width
        )

        aggregator = context * gates[:, 0:1]
        for level, conv in enumerate(self.context_convs, start=1):
            context = conv(context)
            aggregator = aggregator + context * gates[:, level : level + 1]
        aggregator = aggregator + self.global_gate(context) * gates[:, -1:]
        modulator = self.mixer(aggregator)
        modulator = modulator.flatten(2).transpose(1, 2)
        mixed = query * modulator
        return self.norm(tokens + self.output_proj(mixed))


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.net(tokens)


class FocalBlock(nn.Module):
    def __init__(self, dim: int, focal_level: int = 3) -> None:
        super().__init__()
        self.modulation = FocalModulation(dim, focal_level=focal_level)
        self.ffn = FeedForward(dim)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        tokens = self.modulation(tokens, height, width)
        return self.ffn(tokens)
