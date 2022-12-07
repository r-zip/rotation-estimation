from pathlib import Path
from typing import Dict, List, Tuple

import torch
from pytorch3d.datasets.shapenet.shapenet_core import ShapeNetCore
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader

from .constants import DEFAULT_BATCH_SIZE, DEFAULT_POINTS_PER_SAMPLE
from .utils import random_rotation, synset_and_model_to_path


class ShapeNetRotation(ShapeNetCore):
    """Dataset that generates random rotations for each sample on-the-fly."""

    def __init__(
        self,
        data_dir,
        save_and_load_rotations: bool = False,
        synsets=None,
        version: int = 1,
        load_textures: bool = True,
        texture_resolution: int = 4,
    ) -> None:
        super().__init__(data_dir, synsets, version, load_textures, texture_resolution)
        self.save_and_load_rotations = save_and_load_rotations

    def __getitem__(self, idx: int) -> Dict:
        model = super().__getitem__(idx)

        if self.save_and_load_rotations:
            path = synset_and_model_to_path(model["synset_id"], model["model_id"]) / "rotation.pt"
            if path.exists():
                rotation = torch.load(path)
            else:
                rotation = random_rotation()
                torch.save(rotation, path)
        else:
            rotation = random_rotation()

        model["rotation"] = rotation
        return model


class PointCloudCollator:
    def __init__(self, points_per_sample: int = DEFAULT_POINTS_PER_SAMPLE) -> None:
        self.points_per_sample = points_per_sample

    def __call__(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            rotation_matrices = torch.stack([x["rotation"] for x in batch])
            meshes = Meshes(
                verts=[x["verts"] for x in batch],
                faces=[x["faces"] for x in batch],
            )

            # generate point clouds from vertices and faces
            point_clouds = sample_points_from_meshes(meshes, num_samples=self.points_per_sample)  # type: ignore

            # rotate point cloud
            output_point_clouds = torch.zeros_like(point_clouds)
            for k, (point_cloud, R) in enumerate(zip(point_clouds, rotation_matrices)):
                output_point_clouds[k] = (R @ point_cloud.T).T

            return output_point_clouds, rotation_matrices


class ImageCollator:
    def __init__(self, points_per_sample: int = DEFAULT_POINTS_PER_SAMPLE) -> None:
        self.points_per_sample = points_per_sample

    def __call__(self, batch: List[Dict]) -> torch.Tensor:
        pass


def get_point_cloud_data_loader(
    path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    points_per_sample: int = DEFAULT_POINTS_PER_SAMPLE,
    save_and_load_rotations: bool = False,
) -> DataLoader:
    """
    Return a point cloud data loader that internally handles building rotation matrices (and optionally saves these
    matrices to disk).

    Args:
        path: The path to the ShapeNetCore.v2 folder.
        batch_size: The number of samples to return per batch.
        points_per_sample: The number of point cloud points to return per sample.
        save_and_load_rotations: Whether to save/load rotations to disk to avoid re-generating. This is useful for
            consistency over epochs, or for hold-out set generation.

    Returns:
        PyTorch DataLoader object for randomly rotated point clouds. This provides an iterator over the dataset
        that produces point cloud/rotation matrix pairs.
    """
    return DataLoader(
        ShapeNetRotation(path, version=2, save_and_load_rotations=save_and_load_rotations),
        batch_size=batch_size,
        collate_fn=PointCloudCollator(points_per_sample),
    )
