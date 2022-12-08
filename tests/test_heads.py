import numpy as np
import torch

from rotation_estimation.heads import svd_projection
from rotation_estimation.utils import random_rotation


def test_svd_projection():
    R = random_rotation(seed=1234)
    torch.manual_seed(1234)
    R_prime = R + 1e-4 * torch.randn_like(R)
    R_hat = svd_projection(R_prime)

    assert np.abs(torch.linalg.det(R_hat).item() - 1) < 1e-4
    assert torch.allclose(R_hat.T @ R_hat, torch.eye(3), rtol=1e-6, atol=1e-6)
    assert torch.linalg.norm(R_hat - R, ord="fro") < 0.001
