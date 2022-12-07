from pathlib import Path

from torch.utils.data import IterableDataset


class PointDataset(IterableDataset):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def __iter__(self):
        # iterate over directory structure
        pass
