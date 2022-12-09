import torch
from pytorch3d.io import IO
from pytorch3d.ops import sample_points_from_meshes
from torch import Dataset

from rotation_estimation.utils import random_rotations


class RotationData(Dataset):
    def __init__(
        self, mesh_file_path: str = "../models/airplane.obj", num_points: int = 100, device: str = "cpu"
    ) -> None:
        mesh = IO().load_mesh(mesh_file_path, device=device)
        self.point_cloud, _, _ = sample_points_from_meshes(
            mesh, num_samples=num_points, return_normals=False, return_textures=False
        )
        self.num_points = num_points
        self.rotation_matrices = random_rotations(num_points)

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        return torch.matmul(self.point_cloud, self.rotation_matrices[idx])
