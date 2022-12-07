from pathlib import Path

from pytorch3d.datasets.shapenet.shapenet_core import ShapeNetCore
from torch.utils.data import IterableDataset


class PointCloudDataset(IterableDataset):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.shapenet_core = ShapeNetCore(self.path, version=2)

    def __iter__(self):
        # iterate over directory structure
        pass
