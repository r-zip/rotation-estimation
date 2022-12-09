import json
from enum import Enum

import torch.autograd
import typer
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from rotation_estimation.data import RotationData
from rotation_estimation.evaluation import plot_train_test
from rotation_estimation.models import PointNetRotationRegression
from rotation_estimation.sixd import OrthogonalMSELoss
from rotation_estimation.train import train


class Model(str, Enum):
    POINT_NET_SVD = "PointNetSVD"


def train_once(
    six_d: bool = False,
    lr: float = 1e-4,
    epochs: int = 5,
    layer_norm: bool = True,
    iteration: int = 1,
    batch_size: int = 100,
    svd_projection: bool = False,
    regularization: float = 0.1,
    debug: bool = False,
):
    if debug:
        torch.autograd.set_detect_anomaly(True)

    model = PointNetRotationRegression(layer_norm=layer_norm, svd_projection=svd_projection, six_d=six_d)

    # TODO: train, val, test split
    # train_data_loader = get_point_cloud_data_loader(DATASET_PATH)
    # val_data_loader = get_point_cloud_data_loader(DATASET_PATH)
    train_set = RotationData(dataset_size=2000)
    val_set = RotationData(dataset_size=200)

    # if six_d:
    #     loss_fn = OrthogonalMSELoss(regularization)
    # else:
    #     loss_fn = MSELoss()
    loss_fn = MSELoss()

    # train_set, val_set, test_set = random_split(dataset, [2000, 200, 200])
    # breakpoint()
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    history = train(model, train_loader, val_loader, loss_fn=loss_fn, lr=lr, epochs=epochs)

    if six_d:
        with open(f"six_d_{regularization}_history{iteration}.json", "w") as f:
            json.dump(history, f)
    else:
        with open(f"nine_d_history{iteration}.json", "w") as f:
            json.dump(history, f)


def gen_graph(r, rep):
    x = []
    temp = []
    for u in rep:
        with open(f"nine_d_history{u}.json", "r") as f:
            contents = json.load(f)
            temp.append(contents)
    x.append(temp)
    for i in r:
        temp = []
        for j in rep:
            with open(f"six_d_{i}_history{j}.json", "r") as f:
                contents = json.load(f)
                temp.append(contents)
        x.append(temp)
    plot_train_test(x, ["9D Baseline"] + [f"6D r={x}" for x in r], ["mse", "so3", "euler"], 100)


def main(
    lr: float = 1e-4,
    epochs: int = 5,
    layer_norm: bool = True,
    batch_size: int = 10,
    svd_projection: bool = False,
    debug: bool = False,
):
    r = [1000, 500, 0, 0.05, 0.001]
    rep = [x for x in range(3)]
    for i in rep:
        for j in r:
            train_once(
                lr=lr,
                epochs=epochs,
                layer_norm=layer_norm,
                batch_size=batch_size,
                svd_projection=svd_projection,
                debug=debug,
                six_d=True,
                iteration=i,
                regularization=j,
            )
    gen_graph(r, rep)


if __name__ == "__main__":
    typer.run(main)
