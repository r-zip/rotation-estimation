import numpy as np
import torch

from rotation_estimation.projections import gram_schmidt, svd_projection
from rotation_estimation.utils import random_rotation


def test_svd_projection():
    R = torch.stack([random_rotation(seed=1234), random_rotation(seed=1235)])
    torch.manual_seed(1234)
    R_prime = R + 1e-4 * torch.randn_like(R)
    R_hat = svd_projection(R_prime)

    for k in range(R.shape[0]):
        assert np.abs(torch.linalg.det(R_hat[k]).item() - 1) < 1e-4
        assert torch.allclose(R_hat[k].T @ R_hat[k], torch.eye(3), rtol=1e-6, atol=1e-6)
        assert torch.linalg.norm(R_hat[k] - R[k], ord="fro").item() < 0.01


def test_gram_schmidt():
    R = torch.randn(10, 2, 3)
    G = gram_schmidt(R)
    for k in range(G.shape[0]):
        assert torch.allclose(G[k].T @ G[k], torch.eye(3), rtol=1e-6, atol=1e-6)
        assert np.abs(torch.linalg.det(G[k]).item() - 1) < 1e-4
