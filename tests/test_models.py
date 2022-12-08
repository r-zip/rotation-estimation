import torch

from rotation_estimation.models import PointNetSVD


def test_point_net_svd():
    point_net_svd = PointNetSVD(512, [128, 64])

    # batched
    x = torch.randn(10, 500, 3, 3)
    y = point_net_svd(x)
    breakpoint()
