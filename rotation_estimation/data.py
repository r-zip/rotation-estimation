from pathlib import Path
from typing import Dict, List

import torch
from pytorch3d.datasets.shapenet.shapenet_core import ShapeNetCore
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.data import DataLoader

from .constants import DEFAULT_BATCH_SIZE, DEFAULT_POINTS_PER_SAMPLE


class PointCloudCollator:
    def __init__(self, points_per_sample: int = DEFAULT_POINTS_PER_SAMPLE) -> None:
        self.points_per_sample = points_per_sample

    def __call__(self, batch: List[Dict]) -> torch.Tensor:
        meshes = Meshes(verts=[x["verts"] for x in batch], faces=[x["faces"] for x in batch])
        return sample_points_from_meshes(meshes, num_samples=self.points_per_sample)  # type: ignore


def get_point_cloud_data_loader(
    path: Path, batch_size: int = DEFAULT_BATCH_SIZE, points_per_sample: int = DEFAULT_POINTS_PER_SAMPLE
) -> DataLoader:
    return DataLoader(
        ShapeNetCore(path, version=2), batch_size=batch_size, collate_fn=PointCloudCollator(points_per_sample)
    )
