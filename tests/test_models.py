import torch

from rotation_estimation.models import MultiHead, PointNetRotationRegression


def test_multi_head():
    multi_head = MultiHead()
    x = torch.randn((10, multi_head.input_dimension))
    y = multi_head._multi_head_embedding(x)

    for batch in y:
        for R in batch:
            assert torch.allclose(R.T @ R, torch.eye(3), rtol=1e-6, atol=1e-6)
            assert torch.allclose(torch.linalg.det(R), torch.tensor(1.0))

    z = multi_head(x)
    assert z.shape == (10, 3, 3)
    for R in z:
        assert torch.allclose(R.T @ R, torch.eye(3), rtol=1e-6, atol=1e-6)
        assert torch.allclose(torch.linalg.det(R), torch.tensor(1.0))


def test_point_net_svd():
    point_net_svd = PointNetRotationRegression(512, [128, 64])

    # batched
    x = torch.randn(10, 500, 3)
    z = torch.randn(10, 500, 3)
    P, Q = point_net_svd(x, z)
    assert P.shape == (10, 3, 3)
    assert Q.shape == (10, 3, 3)
