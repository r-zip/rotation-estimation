import torch

from rotation_estimation.models import PointNetRotationRegression


def test_point_net_svd():
    point_net_svd = PointNetRotationRegression(512, [128, 64])

    # batched
    x = torch.randn(10, 500, 3)
    y = point_net_svd(x)
    assert y.shape == (10, 3, 3)
