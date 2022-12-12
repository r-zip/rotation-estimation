import pytest
import torch
from torch.utils.data import DataLoader

from rotation_estimation.constants import PROCESSED_DATA_PATH
from rotation_estimation.data import ProcessedDataset


@pytest.mark.skipif(not PROCESSED_DATA_PATH.exists(), reason="Processed dataset does not exist.")
@pytest.mark.parametrize("batch_size", [5, 10, 100])
def test_point_cloud_dataloader(batch_size: int):
    pc_dataloader = DataLoader(ProcessedDataset(split="train"), batch_size=batch_size)
    original_point_cloud, rotated_point_clouds, rotations = next(iter(pc_dataloader))
    assert rotated_point_clouds.shape[0] == batch_size
    assert rotated_point_clouds.shape[2] == 3
    assert rotations.size() == (batch_size, 3, 3)
    assert torch.allclose(torch.matmul(original_point_cloud, rotations), rotated_point_clouds)
