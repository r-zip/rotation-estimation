import torch

from rotation_estimation.metrics import so3_distance
from rotation_estimation.utils import random_rotations


def test_so3_distance():
    P = random_rotations(10)
    Q = random_rotations(10)
    d = so3_distance(P, Q, unit="degrees")
    breakpoint()
