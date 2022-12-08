from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    Represents entire input/feature transform block.

    Reference: https://arxiv.org/pdf/1612.00593.pdf
    """

    def __init__(
        self,
        layer_sizes: Tuple[int, int, int, int, int] = (64, 128, 1024, 512, 256),
        input_features: int = 3,
        output_size: Tuple[int, int] = (3, 3),
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.mlp1 = nn.Sequential(
            nn.Linear(input_features, layer_sizes[0]),
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
        # TODO: WHY DOES THIS WORK (dimensionally)?
        pooled = F.adaptive_max_pool2d(out1, output_size=(1, 512)).squeeze()
        matrix = self.mlp2(pooled).reshape((points.shape[0], *self.output_size))
        return torch.einsum("b p d, b d a -> b p a", points, matrix)


class PointNet(nn.Module):
    """
    PointNet.

    Reference: https://arxiv.org/pdf/1612.00593.pdf
    """

    def __init__(self, output_dimension: int = 512) -> None:
        super().__init__()
        self.network = nn.Sequential(
            TNet(),
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.LayerNorm(64),  # maybe remove?
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),  # maybe remove?
            TNet(input_features=64, output_size=(64, 64)),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),  # maybe remove?
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),  # maybe remove?
            nn.AdaptiveMaxPool2d(output_size=(1, 1024)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
