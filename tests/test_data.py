from pathlib import Path

import pytest

from rotation_estimation.data import get_point_cloud_data_loader

DATA_PATH = Path(__file__).parents[1] / "data/ShapeNetCore.v2/"


@pytest.mark.skipif(not DATA_PATH.exists(), reason="ShapeNetCore.v2 is not downloaded.")
@pytest.mark.parametrize("batch_size,points_per_sample", [(5, 100), (10, 1000)])
def test_point_cloud_dataloader(batch_size: int, points_per_sample: int):
    pc_dataloader = get_point_cloud_data_loader(DATA_PATH, batch_size=batch_size, points_per_sample=points_per_sample)
    point_clouds, rotations = next(iter(pc_dataloader))
    assert point_clouds.size() == (batch_size, points_per_sample, 3)
    assert rotations.size() == (batch_size, 3, 3)
