from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import build_mlp


class TNet(nn.Module):
    """
    Represents entire input/feature transform block.

    Reference: https://arxiv.org/pdf/1612.00593.pdf
    """

    def __init__(
        self,
        layer_sizes: Tuple[int, int, int, int, int] = (64, 128, 1024, 512, 256),
        layer_norm: bool = True,
        input_features: int = 3,
        output_size: Tuple[int, int] = (3, 3),
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.mlp1 = build_mlp(input_features, layer_sizes[3], layer_sizes[:3], layer_norm=layer_norm)
        self.mlp2 = build_mlp(layer_sizes[3], output_size[0] * output_size[1], layer_sizes[4:])

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        out1 = self.mlp1(points)
        # TODO: WHY DOES THIS WORK (dimensionally)?
        pooled = F.adaptive_max_pool2d(out1, output_size=(1, 512)).squeeze()
        matrix = self.mlp2(pooled).reshape((points.shape[0], *self.output_size))
        return torch.einsum("b p d, b d a -> b p a", points, matrix)


def build_point_net(output_dimension: int = 512, final_activation: Optional[nn.Module] = None) -> nn.Module:
    return nn.Sequential(
        TNet(),
        build_mlp(3, 64, [64], layer_norm=True, final_activation=nn.ReLU()),
        TNet(input_features=64, output_size=(64, 64)),
        build_mlp(64, 1024, [64, 128], layer_norm=True, final_activation=nn.ReLU()),
        nn.AdaptiveMaxPool2d(output_size=(1, 1024)),
        nn.Flatten(),
        build_mlp(1024, output_dimension, [512, 256], layer_norm=True, final_activation=final_activation),
    )
