from typing import List

import torch
import torch.nn as nn

from .backbones import build_point_net
from .blocks import build_mlp
from .heads import gram_schmidt, svd_projection


class PointNetRotationRegression(nn.Module):
    def __init__(
        self,
        point_net_embedding_dim: int = 32,
        head_hidden_layer_sizes: List[int] = None,
        layer_norm: bool = False,
        point_net: str = "simplified",
        svd_projection: bool = False,
        six_d: bool = False,
    ) -> None:
        super().__init__()
        self.svd_projection = svd_projection
        self.six_d = six_d

        if head_hidden_layer_sizes is None:
            head_hidden_layer_sizes = [128, 64]

        if point_net == "original":
            self.point_net = build_point_net(
                output_dimension=point_net_embedding_dim, final_activation=nn.ReLU(), layer_norm=layer_norm
            )

        elif point_net == "simplified":
            self.point_net = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d(output_size=(1, 64)),
                nn.Linear(64, point_net_embedding_dim),
                nn.ReLU(),
            )
        if self.six_d:
            self.head_mlp = build_mlp(
                input_dimension=point_net_embedding_dim,
                output_dimension=6,
                hidden_layer_sizes=head_hidden_layer_sizes,
                final_activation=None,
                layer_norm=layer_norm,
            )
        else:
            self.head_mlp = build_mlp(
                input_dimension=point_net_embedding_dim,
                output_dimension=9,
                hidden_layer_sizes=head_hidden_layer_sizes,
                final_activation=None,
                layer_norm=layer_norm,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        point_net_embedding = self.point_net(x)
        if self.six_d:
            raw_matrix = self.head_mlp(point_net_embedding).reshape((point_net_embedding.shape[0], 2, 3))
        else:
            raw_matrix = self.head_mlp(point_net_embedding).reshape((point_net_embedding.shape[0], 3, 3))

        if (not self.six_d) and self.svd_projection:
            return svd_projection(raw_matrix)
        elif self.six_d:
            return gram_schmidt(raw_matrix)

        return raw_matrix
