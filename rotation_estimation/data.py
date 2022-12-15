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
            pre_rotate: bool = False,
            re_sample: bool = True,
            center: bool = False,
        ) -> None:
            self.num_points = num_points
            self.split_dir = SPLITS_PATH / split
            assert self.split_dir.exists(), "Split directory does not exist!"

            meshes = []
            models = []
            for folder in self.split_dir.iterdir():
                models.append(folder.name)
                obj_file = folder / "models" / "model_normalized.obj"
                # reference for line below: https://pytorch3d.org/docs/io
                meshes.append(IO().load_mesh(obj_file, device=device))

            self.meshes = meshes
            self.models = models
            self.num_models = len(models)
            self.dataset_size = dataset_size
            self.pre_rotate = pre_rotate
            self.re_sample = re_sample

            point_clouds = []
            if not self.re_sample:
                for mesh in self.meshes:
                    point_cloud = sample_points_from_meshes(
                        mesh,
                        num_samples=self.num_points,
                        return_normals=False,
                        return_textures=False,
                    ).squeeze()
                    if center:
                        point_cloud = point_cloud - point_cloud.mean()
                    point_clouds.append(point_cloud)
                self.point_clouds = point_clouds
            else:
                self.point_clouds = None

        def __len__(self) -> int:
            return self.dataset_size

        @torch.no_grad()
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if self.re_sample:
                point_cloud = sample_points_from_meshes(
                    self.meshes[idx % self.num_models],
                    num_samples=self.num_points,
                    return_normals=False,
                    return_textures=False,
                ).squeeze()
            else:
                point_cloud = self.point_clouds[idx % self.num_models]

            rotation_matrix = random_rotation()
            if self.pre_rotate:
                pre_rotation_matrix = random_rotation()
                pre_rotated = point_cloud @ pre_rotation_matrix.T
            else:
                pre_rotation_matrix = torch.eye(3)
                pre_rotated = point_cloud

            return (
                self.models[idx % self.num_models],
                pre_rotated,
                pre_rotated @ rotation_matrix.T,
                pre_rotation_matrix,
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
