import numpy as np
import torch

from rotation_estimation.utils import random_rotations


def test_random_rotations():
    R = random_rotations(5, seed=1234)
    eye = torch.eye(3)
    for k in range(R.shape[0]):
        # ensure rotations are orthogonal
        assert torch.allclose(R[k].T @ R[k], eye, rtol=1e-4, atol=1e-4)

        # ensure rotations belong to SO(3), instead of O(3)
        assert np.abs(np.linalg.det(R[k].numpy()) - 1) < 1e-6
