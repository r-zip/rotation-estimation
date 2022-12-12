import torch

from rotation_estimation.models import PointNetRotationRegression


def test_point_net_svd():
    point_net_svd = PointNetRotationRegression(512, [128, 64])

    # batched
    x = torch.randn(10, 500, 3)
    z = torch.randn(10, 500, 3)
    P, Q = point_net_svd(x, z)
    assert P.shape == (10, 3, 3)
    assert Q.shape == (10, 3, 3)
