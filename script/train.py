import json
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.autograd
import torch.backends
import typer
from torch.utils.data import DataLoader

from rotation_estimation.constants import (DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS,
                                           DEFAULT_LAYER_NORM, DEFAULT_LR,
                                           DEFAULT_REGULARIZATION,
                                           RESULTS_PATH)
from rotation_estimation.data import ProcessedDataset
from rotation_estimation.evaluation import plot_train_test
from rotation_estimation.losses import OrthogonalMSELoss
from rotation_estimation.models import PointNetRotationRegression
from rotation_estimation.train import train


class Model(str, Enum):
    POINT_NET_SVD = "PointNetSVD"


def train_once(
    six_d: bool = False,
    svd_projection: bool = False,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    layer_norm: bool = DEFAULT_LAYER_NORM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    regularization: float = DEFAULT_REGULARIZATION,
    iteration: int = 1,
    device: Optional[str] = None,
    debug: bool = False,
):
    if debug:
        torch.autograd.set_detect_anomaly(True)

    if device is None and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device is None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = PointNetRotationRegression(layer_norm=layer_norm, svd_projection=svd_projection, six_d=six_d).to(device)

    train_set = ProcessedDataset(split="train", device=device)
    val_set = ProcessedDataset(split="val", device=device)
    test_set = ProcessedDataset(split="test", device=device)

    loss_fn = OrthogonalMSELoss(six_d, regularization)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    # TODO: run on test set
    # test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    history = train(model, train_loader, val_loader, loss_fn=loss_fn, lr=lr, epochs=epochs)

    RESULTS_PATH.mkdir(exist_ok=True)
    if six_d:
        with open(RESULTS_PATH / f"six_d_{regularization}_history_{iteration}.json", "w") as f:
            json.dump(history, f)
    else:
        with open(RESULTS_PATH / f"nine_d_{regularization}_history_{iteration}.json", "w") as f:
            json.dump(history, f)


def gen_graph(r, rep):
    x = []
    temp = []
    for u in rep:
        with open(RESULTS_PATH / f"nine_d_history_{u}.json", "r") as f:
            contents = json.load(f)
            temp.append(contents)
    x.append(temp)
    for i in r:
        temp = []
        for j in rep:
            with open(f"six_d_{i}_history_{j}.json", "r") as f:
                contents = json.load(f)
                temp.append(contents)
        x.append(temp)
    plot_train_test(x, ["9D Baseline"] + [f"6D r={x}" for x in r], ["mse", "so3", "euler"], 100)


def main(
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    layer_norm: bool = DEFAULT_LAYER_NORM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cpu",
    runs: int = 10,
    debug: bool = False,
):
    r = [0.0, *np.logspace(-3, 3, 7)]
    for i in range(runs):
        for j in r:
            train_once(
                lr=lr,
                epochs=epochs,
                layer_norm=layer_norm,
                batch_size=batch_size,
                svd_projection=False,
                debug=debug,
                six_d=True,
                iteration=i,
                regularization=j,
                device=device,
            )

            train_once(
                lr=lr,
                epochs=epochs,
                layer_norm=layer_norm,
                batch_size=batch_size,
                svd_projection=True,
                debug=debug,
                six_d=False,
                iteration=i,
                regularization=j,
                device=device,
            )


if __name__ == "__main__":
    typer.run(main)
