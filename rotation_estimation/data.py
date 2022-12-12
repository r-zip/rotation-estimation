from pathlib import Path
from typing import Tuple, Union

import torch
from torch.utils.data import Dataset

try:
    from pytorch3d.io import IO
    from pytorch3d.ops import sample_points_from_meshes

    from .constants import OBJ_FILE_PATH
    from .utils import random_rotations

    class RotationData(Dataset):
        def __init__(
            self,
            mesh_file_path: str = OBJ_FILE_PATH,
            num_points: int = 100,
            device: str = "cpu",
            dataset_size: int = 2400,
            seed: int = 1234,
        ) -> None:
            mesh = IO().load_mesh(mesh_file_path, device=device)
            self.point_cloud = sample_points_from_meshes(
                mesh, num_samples=num_points, return_normals=False, return_textures=False
            ).squeeze()
            self.dataset_size = dataset_size
            self.rotation_matrices = random_rotations(self.dataset_size, seed=seed)

        def __len__(self) -> int:
            return self.dataset_size

        @torch.no_grad()
        def __getitem__(self, idx: int) -> torch.Tensor:
            return torch.matmul(self.point_cloud, self.rotation_matrices[idx]), self.rotation_matrices[idx]

except ImportError:
    pass


class ProcessedDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        processed_data_dir: Path = Path(__file__).parents[1] / "data/processed",
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.split = split
        self.root_dir = processed_data_dir
        self.split_dir = processed_data_dir / split
        self.device = device
        self._num_files = None
        self._file_list = sorted(list((processed_data_dir / split).glob("*.pt")))
        self._samples = []

    def __len__(self) -> int:
        if self._num_files is None:
            self._num_files = len([f for f in self.split_dir.iterdir() if f.is_file()])
        return self._num_files

    def _load(self) -> None:
        self._samples.clear()
        for f in sorted(self._file_list):
            sample = torch.load(f)
            self._samples.append((sample["point_cloud"].squeeze().to(self.device), sample["rotation"].to(self.device)))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._samples:
            self._load()

        return self._samples[index]
