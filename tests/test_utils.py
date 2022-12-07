import numpy as np

from rotation_estimation.utils import random_rotations


def test_random_rotations():
    R = random_rotations(5)
    eye = np.eye(3)
    for k in range(R.shape[0]):
        # ensure rotations are orthogonal
        assert np.allclose(R[k].T @ R[k], eye)

        # ensure rotations belong to SO(3), instead of O(3)
        assert np.abs(np.linalg.det(R[k]) - 1) < 1e-4
