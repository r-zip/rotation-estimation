from typing import List

import torch
import torch.nn as nn

from .backbones import build_point_net
from .blocks import build_mlp
from .heads import svd_projection


class PointNetSVD(nn.Module):
    def __init__(self, point_net_embedding_dim: int, head_hidden_layer_sizes: List[int]) -> None:
        super().__init__()
        self.point_net = build_point_net(output_dimension=point_net_embedding_dim, final_activation=nn.ReLU())
        self.head_mlp = build_mlp(
            input_dimension=point_net_embedding_dim,
            output_dimension=9,
            hidden_layer_sizes=head_hidden_layer_sizes,
            final_activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        point_net_embedding = self.point_net(x)
        raw_matrix = self.head_mlp(point_net_embedding).reshape((*x.shape[:-2], 3, 3))
        return svd_projection(raw_matrix)
