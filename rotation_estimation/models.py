import torch

from .backbones import build_point_net
from .heads import svd_projection


# TODO: do we need to use a head with PointNet? (maybe for consistency)
class PointNetSVD:
    def __init__(self) -> None:
        self.point_net = build_point_net(output_dimension=9, final_activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_matrix = self.point_net(x)
        return svd_projection(raw_matrix)
