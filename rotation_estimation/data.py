from pathlib import Path
from typing import Tuple, Union

import torch
from torch.utils.data import Dataset

if torch.__version__.startswith("1.12"):
    from pytorch3d.io import IO
    from pytorch3d.ops import sample_points_from_meshes

    from .constants import SPLITS_PATH
    from .utils import random_rotation

    class RotationData(Dataset):
        def __init__(
            self,
            split: str,
            num_points: int = 256,
            device: str = "cpu",
            dataset_size: int = 10_000,
        ) -> None:
            self.num_points = num_points
            self.split_dir = SPLITS_PATH / split
            assert self.split_dir.exists(), "Split directory does not exist!"

            meshes = []
            models = []
            for folder in self.split_dir.iterdir():
                models.append(folder.name)
                obj_file = folder / "models" / "model_normalized.obj"
                meshes.append(IO().load_mesh(obj_file, device=device))

            self.meshes = meshes
            self.models = models
            self.num_models = len(models)
            self.dataset_size = dataset_size

        def __len__(self) -> int:
            return self.dataset_size

        @torch.no_grad()
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            point_cloud = sample_points_from_meshes(
                self.meshes[idx % self.num_models],
                num_samples=self.num_points,
                return_normals=False,
                return_textures=False,
            ).squeeze()
            rotation_matrix = random_rotation()
            return (
                self.models[idx % self.num_models],
                point_cloud,
                torch.matmul(point_cloud, rotation_matrix),
                rotation_matrix,
            )


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
            self._samples.append(
                (
                    sample["original_point_cloud"].squeeze().to(self.device),
                    sample["rotated_point_cloud"].squeeze().to(self.device),
                    sample["rotation"].to(self.device),
                )
            )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._samples:
            self._load()

        return self._samples[index]
