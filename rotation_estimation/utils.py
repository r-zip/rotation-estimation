from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure


def synset_and_model_to_path(synset_id: str, model_id: str) -> Path:
    return Path(__file__).parents[1] / f"data/ShapeNetCore.v2/{synset_id}/{model_id}"


def random_rotation(seed: Optional[int] = None) -> torch.Tensor:
    """
    Generate a random rotation matrix according to the Haar measure on SO(3).

    References:
        https://www.sciencedirect.com/science/article/pii/B9780080507552500348
        https://www.blopig.com/blog/2021/08/uniformly-sampled-3d-rotation-matrices

    Args:
        seed: The seed for the random generation.

    Returns:
        A 3 x 3 torch.Tensor representing a rotation in 3D.
    """
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
    return torch.Tensor(H_neg @ R)


def random_rotations(n: int, seed: Optional[int] = None) -> torch.Tensor:
    """
    Generate n random rotation matrices.

    Args:
        n: The number of rotations to generate.
        seed: The seed to the RNG.

    Returns:
        An (n x 3 x 3) torch.Tensor, representing n 3x3 rotation matrices.
    """
    # TODO: vectorize
    matrices = []
    for _ in range(n):
        matrices.append(random_rotation(seed))

    return torch.stack(matrices)


def visualize_rotations(show: bool = False) -> Figure:
    """
    Used for sanity checking rotation matrix distribution.

    References:
        https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html

    Args:
        show: Whether to show the figure interactively.

    Returns:
        A matplotlib Figure with of the scatterplot.
    """
    # reference: https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    R = random_rotations(500).numpy()
    for axis in range(3):
        xs = np.array([R[k, :, axis] for k in range(R.shape[0])])
        ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], alpha=0.3)

    if show:
        plt.show()

    return fig
