from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .backbones import build_point_net
from .blocks import build_mlp
from .heads import gram_schmidt, svd_projection
from .utils import identity


class PointNetRotationRegression(nn.Module):
    def __init__(
        self,
        point_net_embedding_dim: int = 32,
        head_hidden_layer_sizes: List[int] = None,
        layer_norm: bool = False,
        point_net: str = "simplified",
        svd: bool = False,
        six_d: bool = False,
        siamese: bool = True,
    ) -> None:
        super().__init__()
        self.svd_projection = svd
        self.six_d = six_d
        self.siamese = siamese

        head_hidden_layer_sizes = head_hidden_layer_sizes or [128, 64]

        self.point_net = build_point_net(
            output_dimension=point_net_embedding_dim,
            final_activation=nn.ReLU(),
            layer_norm=layer_norm,
            kind=point_net,
        )

        self.head_mlp = build_mlp(
            input_dimension=2 * point_net_embedding_dim if siamese else point_net_embedding_dim,
            output_dimension=6 if six_d else 9,
            hidden_layer_sizes=head_hidden_layer_sizes,
            final_activation=None,
            layer_norm=layer_norm,
        )

        self.output_shape = (2, 3) if six_d else (3, 3)

        if six_d:
            self.projection = gram_schmidt
        elif not six_d and svd:
            self.projection = svd_projection
        else:
            self.projection = identity

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.siamese and z is None:
            raise TypeError(
                f"{self.__class__.__name__} instance is siamese, but only x was provided. Please supply two arguments."
            )

        batch_size = x.shape[0]
        x_emb = self.point_net(x)
        if self.siamese:
            z_emb = self.point_net(z)
            raw_matrix = self.head_mlp(torch.concat([x_emb, z_emb], dim=1)).reshape(batch_size, *self.output_shape)
        else:
            raw_matrix = self.head_mlp(x_emb).reshape((batch_size, *self.output_shape))
        return raw_matrix, self.projection(raw_matrix)
