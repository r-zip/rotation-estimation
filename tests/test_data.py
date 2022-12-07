from pathlib import Path

import pytest

from rotation_estimation.data import get_point_cloud_data_loader


@pytest.mark.parametrize("batch_size,points_per_sample", [(5, 100), (10, 1000)])
def test_point_cloud_dataloader(batch_size: int, points_per_sample: int):
    pc_dataloader = get_point_cloud_data_loader(
        Path(__file__).parents[1] / "data/ShapeNetCore.v2/", batch_size=batch_size, points_per_sample=points_per_sample
    )
    batch = next(iter(pc_dataloader))
    assert batch.size() == (batch_size, points_per_sample, 3)
