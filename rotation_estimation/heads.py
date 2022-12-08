from typing import List

import torch
import torch.nn as nn


def svd_projection(R: torch.Tensor) -> torch.Tensor:
    """
    Project R onto SO(3) using the SVD.

    Reference: https://proceedings.neurips.cc/paper/2020/file/fec3392b0dc073244d38eba1feb8e6b7-Paper.pdf
    """
    U, _, Vh = torch.linalg.svd(R)
    S_prime = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, torch.linalg.det(U @ Vh)]])
    return U @ S_prime @ Vh


class SVDHead(nn.Module):
    def __init__(self, input_dimension: int, layer_sizes: List[int], layer_norm: bool = False) -> None:
        super().__init__()

        input_sizes = [input_dimension, *layer_sizes[:-1]]
        output_sizes = [*layer_sizes[1:], 9]

        layers = []
        for k, (input_size, output_size) in enumerate(zip(input_sizes, output_sizes)):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            if layer_norm and k < min(len(input_sizes), len(output_sizes)):
                layers.append(nn.LayerNorm(output_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = self.mlp(x).reshape((*x.shape[:-1], 3, 3))
        return svd_projection(R)
