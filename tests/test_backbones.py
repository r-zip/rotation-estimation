import pytest
import torch

from rotation_estimation.backbones import TNet


@pytest.mark.parametrize("batch_size,points,dim", [(100, 10, 3), (1000, 50, 3)])
def test_tnet(batch_size: int, points: int, dim: int):
    batch_size = 100
    points = 10
    dim = 3
    x = torch.randn((batch_size, points, dim))
    tnet = TNet()
    y = tnet(x)
    assert y.size() == (batch_size, points, dim)
