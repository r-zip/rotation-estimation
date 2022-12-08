import pytest
import torch

from rotation_estimation.backbones import PointNet, TNet


@pytest.mark.parametrize("batch_size,points,dim", [(100, 10, 3), (1000, 50, 3)])
def test_tnet(batch_size: int, points: int, dim: int):
    x = torch.randn((batch_size, points, dim))
    tnet = TNet()
    y = tnet(x)
    assert y.size() == (batch_size, points, dim)


@pytest.mark.parametrize("batch_size,points,dim,output_dimension", [(100, 10, 3, 256), (1000, 50, 3, 512)])
def test_tnet(batch_size: int, points: int, dim: int, output_dimension: int):
    x = torch.randn((batch_size, points, dim))
    pointnet = PointNet(output_dimension=output_dimension)
    y = pointnet(x)
    assert y.size() == (batch_size, output_dimension)
