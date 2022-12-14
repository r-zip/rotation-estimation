from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .backbones import build_point_net
from .blocks import build_mlp
from .projections import gram_schmidt, svd_projection
from .utils import identity


class MultiHead(nn.Module):
    def __init__(
        self,
        point_net_embedding_dim: int = 32,
        head_hidden_layer_sizes: List[int] = None,
        layer_norm: bool = False,
        siamese: bool = True,
    ) -> None:
        super().__init__()
        head_hidden_layer_sizes = head_hidden_layer_sizes or [128, 64]
        self.projection = svd_projection
        self.input_dimension = 2 * point_net_embedding_dim if siamese else point_net_embedding_dim
        self.mlps = [
            build_mlp(
                input_dimension=self.input_dimension,
                output_dimension=6,
                hidden_layer_sizes=head_hidden_layer_sizes,
                final_activation=None,
                layer_norm=layer_norm,
            )
            for _ in range(6)
        ]

    def _multi_head_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts bxd as input, produces bx3x3 as output
        """
        six_d_outputs = torch.stack([gram_schmidt(mlp(x).reshape(x.shape[0], 2, 3)) for mlp in self.mlps])

        # init outputs
        output = torch.zeros((x.shape[0], 6, 3, 3))

        # head 1 (first two rows)
        output[:, 0, :3, :3] = six_d_outputs[0].clone()

        # head 2 (first and third row), multiply by -1 to preserve det(R) == 1
        output[:, 1, 0, :3] = -six_d_outputs[1, :, 0, :].clone()
        output[:, 1, 2, :3] = -six_d_outputs[1, :, 1, :].clone()
        output[:, 1, 1, :3] = -six_d_outputs[1, :, 2, :].clone()

        # head 3 (last two rows)
        output[:, 2, 1:, :3] = six_d_outputs[2, :, :2, :3].clone()
        output[:, 2, 0, :3] = six_d_outputs[2, :, 2, :3].clone()

        # head 4 (first two columns)
        output[:, 3, :3, :3] = six_d_outputs[3].clone().transpose(2, 1)

        # head 5 (first and last column), multiply by -1 to preserve det(R) == 1
        output[:, 4, :3, 0] = -six_d_outputs[4, :, 0, :].clone()
        output[:, 4, :3, 2] = -six_d_outputs[4, :, 1, :].clone()
        output[:, 4, :3, 1] = -six_d_outputs[4, :, 2, :].clone()

        # head 6 (last two columns)
        output[:, 5, :3, 1:] = six_d_outputs[5, :, :2, :].clone().transpose(2, 1)
        output[:, 5, :3, 0] = six_d_outputs[5, :, 2, :].clone()

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self._multi_head_embedding(x).mean(axis=1).squeeze()
        return svd_projection(emb)


class PointNetRotationRegression(nn.Module):
    def __init__(
        self,
        point_net_embedding_dim: int = 32,
        head_hidden_layer_sizes: List[int] = None,
        layer_norm: bool = False,
        point_net: str = "simplified",
        multi_head: bool = False,
        svd: bool = False,
        six_d: bool = False,
        siamese: bool = True,
    ) -> None:
        super().__init__()
        self.svd_projection = svd
        self.six_d = six_d
        self.siamese = siamese
        self.multi_head = multi_head

        head_hidden_layer_sizes = head_hidden_layer_sizes or [128, 64]

        self.point_net = build_point_net(
            output_dimension=point_net_embedding_dim,
            final_activation=nn.ReLU(),
            layer_norm=layer_norm,
            kind=point_net,
        )

        if self.multi_head:
            self.head = MultiHead(
                point_net_embedding_dim=32,
                head_hidden_layer_sizes=head_hidden_layer_sizes,
                layer_norm=layer_norm,
                siamese=siamese,
            )
        else:
            self.head = build_mlp(
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
            raw_matrix = self.head(torch.concat([x_emb, z_emb], dim=1)).reshape(batch_size, *self.output_shape)
        else:
            raw_matrix = self.head(x_emb).reshape((batch_size, *self.output_shape))
        return raw_matrix, self.projection(raw_matrix)
