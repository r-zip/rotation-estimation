from typing import List

import torch
import torch.nn as nn

from .blocks import build_mlp


def svd_projection(R: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """
    Project R onto SO(3) using the SVD.

    Reference: https://proceedings.neurips.cc/paper/2020/file/fec3392b0dc073244d38eba1feb8e6b7-Paper.pdf
    """
    # TODO: test
    U, _, Vh = torch.linalg.svd(R + torch.stack([eps * torch.eye(3) for _ in range(R.shape[0])]))
    # det = torch.linalg.det(torch.matmul(U, Vh))
    # if len(R.shape) > 2:
    #     S_prime = torch.stack([torch.eye(3) for _ in range(R.shape[0])])
    #     S_prime[:, 2, 2] = det
    # elif len(R.shape) == 2:
    #     S_prime = torch.eye(3)
    #     S_prime[2, 2] = det
    # else:
    #     raise ValueError(f"Input tensor R (shape={R.shape}) has too few  dimensions.")

    # return torch.matmul(torch.matmul(U, S_prime), Vh)
    return torch.matmul(U, Vh)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.linalg.norm(x, dim=-1)[:, None]


def gram_schmidt(x: torch.Tensor) -> torch.Tensor:
    """Normalize each row of the rotation matrix"""
    # references:
    # https://en.wikipedia.org/wiki/Gram–Schmidt_process#The_Gram–Schmidt_process
    # https://arxiv.org/pdf/1812.07035.pdf
    y = torch.zeros((x.shape[0], 3, 3))
    y[:, 0, :] = _normalize(x[:, 0, :])
    proj = (y[:, 0, :].clone() * x[:, 1, :]).sum(dim=-1)
    y[:, 1, :] = _normalize(x[:, 1, :] - proj[:, None] * y[:, 0, :].clone())
    y[:, 2, :] = torch.linalg.cross(y[:, 0, :].clone(), y[:, 1, :].clone())
    return y


class SVDHead(nn.Module):
    def __init__(self, input_dimension: int, layer_sizes: List[int], layer_norm: bool = False) -> None:
        super().__init__()
        self.mlp = build_mlp(
            input_dimension=input_dimension, output_dimension=9, hidden_layer_sizes=layer_sizes, layer_norm=layer_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = self.mlp(x).reshape((*x.shape[:-1], 3, 3))
        return svd_projection(R)
