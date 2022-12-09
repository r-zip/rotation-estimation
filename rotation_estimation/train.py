from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms.rotation_conversions import matrix_to_euler_angles
from pytorch3d.transforms.so3 import so3_relative_angle
from torch.optim import Adam
from torch.utils.data import DataLoader

METRICS = ["so3", "euler"]


def compute_metrics(pred: torch.Tensor, truth: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        try:
            so3_distance = so3_relative_angle(pred, truth).sum().item()
        except ValueError:
            so3_distance = np.nan
        euler_distance = (
            torch.abs(matrix_to_euler_angles(pred, "XYZ") - matrix_to_euler_angles(truth, "XYZ"))
            .mean(axis=1)
            .sum()
            .item()
        )
        return {"so3": so3_distance, "euler": euler_distance, "n": pred.shape[0]}


def avg_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    output_dict = {}
    for key in METRICS:
        output_dict[key] = sum([m * n for m, n in zip([m[key] for m in metrics], [m["n"] for m in metrics])]) / sum(
            [m["n"] for m in metrics]
        )
    return output_dict


def train(
    model: nn.Module,
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    lr: float,
    epochs: int = 3,
    val_every: int = 100,
    model_path: Path = Path("./saved_model.pt"),
    loss_fn: Callable = F.mse_loss,
) -> Dict[str, Dict[str, List[float]]]:
    optimizer = Adam(model.parameters(), lr=lr)

    # this loop is based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network
    history = {
        "train": {"mse": [], "so3": [], "euler": [], "epoch": [], "step": [], "sum_steps": []},
        "val": {"mse": [], "so3": [], "euler": [], "epoch": [], "step": [], "sum_steps": []},
    }
    sum_steps = 0
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        train_metrics = []
        for step, (inputs, labels) in enumerate(train_data_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_metrics.append(compute_metrics(outputs, labels))

            val_metrics = []
            if (sum_steps + 1) % val_every == 0:
                avg_train_loss = running_loss / val_every
                print(f"[step: {sum_steps + 1:5d}] train loss: {avg_train_loss:.3f}")
                running_loss = 0.0

                model.eval()
                with torch.no_grad():
                    avg_val_loss = 0
                    n_val = 0
                    for val_input, val_label in val_data_loader:
                        val_output = model(val_input)
                        avg_val_loss += loss_fn(val_output, val_label).item()
                        n_val += 1
                        val_metrics.append(compute_metrics(val_output, val_label))

                    avg_val_loss /= n_val

                model.train()

                print(f"[step: {sum_steps + 1:5d}] val loss: {avg_val_loss:.3f}")

                # validation here
                # TODO: rename mse to loss or something
                history["train"]["mse"].append(avg_train_loss)
                history["val"]["mse"].append(avg_val_loss)

                # average metrics
                avg_train_metrics = avg_metrics(train_metrics)
                avg_val_metrics = avg_metrics(val_metrics)

                for key in METRICS:
                    history["train"][key].append(avg_train_metrics[key])
                    history["val"][key].append(avg_val_metrics[key])

                # clear metrics
                train_metrics.clear()
                val_metrics.clear()

            sum_steps += 1

    torch.save(model, model_path)

    return history
