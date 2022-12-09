import json
from enum import Enum
from typing import Callable

import typer
from torch.utils.data import DataLoader

from rotation_estimation.data import RotationData
from rotation_estimation.models import PointNetSVD
from rotation_estimation.train import train


class Model(str, Enum):
    POINT_NET_SVD = "PointNetSVD"


def main(
    model: Model = Model.POINT_NET_SVD,
    lr: float = 1e-4,
    epochs: int = 5,
    layer_norm: bool = True,
    iteration: int = 1,
    batch_size: int = 10,
    svd_projection: bool = False,
):
    model = PointNetSVD(layer_norm=layer_norm, svd_projection=svd_projection)

    # TODO: train, val, test split
    # train_data_loader = get_point_cloud_data_loader(DATASET_PATH)
    # val_data_loader = get_point_cloud_data_loader(DATASET_PATH)
    train_set = RotationData(dataset_size=2000)
    test_set = RotationData(dataset_size=200)
    val_set = RotationData(dataset_size=200)
    # train_set, val_set, test_set = random_split(dataset, [2000, 200, 200])
    # breakpoint()
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    history = train(model, train_loader, val_loader, lr=lr, epochs=epochs)

    with open(f"history{iteration}.json", "w") as f:
        json.dump(history, f)


if __name__ == "__main__":
    typer.run(main)
