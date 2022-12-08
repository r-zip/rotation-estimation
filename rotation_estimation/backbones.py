from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(
        self,
        layer_sizes: Tuple[int, int, int, int, int] = (64, 128, 1024, 512, 256),
        output_size: Tuple[int, int] = (3, 3),
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.mlp1 = nn.Sequential(
            nn.Linear(3, layer_sizes[0]),
            nn.ReLU(),
            nn.LayerNorm(layer_sizes[0]),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.LayerNorm(layer_sizes[1]),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.LayerNorm(layer_sizes[2]),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(layer_sizes[3], layer_sizes[4]),
            nn.ReLU(),
            nn.LayerNorm(layer_sizes[4]),
            nn.Linear(layer_sizes[4], output_size[0] * output_size[1]),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        out1 = self.mlp1(points)
        pooled = F.adaptive_max_pool2d(out1, output_size=(512, 1)).squeeze()
        return self.mlp2(pooled).reshape((points.shape[0], *self.output_size))


class PointNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.TNet1 = TNet()
