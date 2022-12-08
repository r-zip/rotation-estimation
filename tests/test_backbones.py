import torch

from rotation_estimation.backbones import TNet


def test_tnet():
    batch_size = 100
    points = 10
    dim = 3
    x = torch.randn((batch_size, points, dim))
    tnet = TNet()
    y = tnet(x)
    assert y.size() == (100, 3, 3)
