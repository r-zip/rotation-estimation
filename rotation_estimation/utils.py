from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure


def random_rotations(n: int, seed: Optional[int] = None) -> torch.Tensor:
    """
    Generate a random rotation matrix according to the Haar measure on SO(3).

    References:
        https://www.sciencedirect.com/science/article/pii/B9780080507552500348
        https://www.blopig.com/blog/2021/08/uniformly-sampled-3d-rotation-matrices

    Args:
        seed: The seed for the random generation.
    """
    # TODO: vectorize
    matrices = []
    for _ in range(n):
        # generate random rotation
        rng = np.random.default_rng(seed)
        theta, phi, z = rng.random(3)
        theta *= 2 * np.pi
        phi *= 2 * np.pi

        v = np.array(
            [
                np.cos(phi) * np.sqrt(z),
                np.sin(phi) * np.sqrt(z),
                np.sqrt(1 - z),
            ]
        ).reshape(-1, 1)

        # negative Householder matrix from the paper
        H_neg = 2 * v @ v.T - np.eye(3)
        R = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        matrices.append(H_neg @ R)

    return torch.tensor(np.array(matrices))


def visualize_rotations(show: bool = False) -> Figure:
    """
    Used for sanity checking rotation matrix distribution.

    References:
        https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    R = random_rotations(500).numpy()
    for axis in range(3):
        xs = np.array([R[k, :, axis] for k in range(R.shape[0])])
        ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], alpha=0.3)

    if show:
        plt.show()

    return fig
